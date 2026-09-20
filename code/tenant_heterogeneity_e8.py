"""
E8 robustness + per-proxy clustering for the Redset calibration.

Two jobs:
  1. BIN-ROBUSTNESS. Recompute matched alpha per proxy under several binnings and several
     min-queries thresholds. The cross-proxy ordering (the reframe's central claim) is robust
     only if each proxy's alpha stays within its CI as the binning changes. Aggregation is done
     ONCE at a fine granularity per proxy, then coarser binnings are formed by merging columns
     and thresholds by filtering instances - so no re-reading the parquet per variant.
  2. PER-PROXY CLUSTERING (fixes the earlier scalar-vs-query_type mismatch). Cluster tenants on
     EACH proxy's own distribution and measure the separation ratio on the SAME distribution.

Heavy aggregation via DuckDB (bounded memory). Run from research/.
Usage:
  python tenant_heterogeneity_e8.py --input data/provisioned_full.parquet \
      --out results/provisioned_e8 --memory-limit 8GB --min-queries-list 1000,2000
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tenant_heterogeneity_mp import calibrate_with_ci
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
LOG2 = np.log(2.0)

def fine_counts(con, src, expr, order=None):
    df = con.execute(f"SELECT instance_id, {expr} AS b, count(*) c FROM {src} GROUP BY 1,2").df()
    df['b'] = df['b'].astype(str)
    tab = df.pivot_table(index='instance_id', columns='b', values='c', aggfunc='sum', fill_value=0).astype(float)
    if order:
        tab = tab.reindex(columns=[c for c in order if c in tab.columns], fill_value=0)
    return tab

SCAN_FINE = ("CASE WHEN coalesce(mbytes_scanned,0)<=0 THEN 's0' WHEN mbytes_scanned<=1 THEN 's1' "
             "WHEN mbytes_scanned<=10 THEN 's2' WHEN mbytes_scanned<=100 THEN 's3' "
             "WHEN mbytes_scanned<=1000 THEN 's4' ELSE 's5' END")
SCAN_ORDER=['s0','s1','s2','s3','s4','s5']
SCAN_VAR={'default6':[['s0'],['s1'],['s2'],['s3'],['s4'],['s5']],
          'coarse3':[['s0','s1'],['s2','s3'],['s4','s5']],
          'coarse2':[['s0','s1','s2'],['s3','s4','s5']]}
JOIN_FINE=("CASE WHEN coalesce(num_joins,0)=0 THEN 'j0' WHEN num_joins=1 THEN 'j1' WHEN num_joins=2 THEN 'j2' "
           "WHEN num_joins=3 THEN 'j3' WHEN num_joins=4 THEN 'j4' ELSE 'j5' END")
JOIN_ORDER=['j0','j1','j2','j3','j4','j5']
JOIN_VAR={'default4':[['j0'],['j1'],['j2'],['j3','j4','j5']],
          'fine6':[['j0'],['j1'],['j2'],['j3'],['j4'],['j5']],
          'coarse2':[['j0'],['j1','j2','j3','j4','j5']]}
HOUR_FINE="printf('h%02d', CAST(extract(hour FROM arrival_timestamp) AS INTEGER))"
HOUR_ORDER=[f'h{h:02d}' for h in range(24)]
HOUR_VAR={'default24':[[h] for h in HOUR_ORDER],
          'blocks8':[HOUR_ORDER[i:i+3] for i in range(0,24,3)],
          'blocks4':[HOUR_ORDER[0:6],HOUR_ORDER[6:12],HOUR_ORDER[12:18],HOUR_ORDER[18:24]]}

def regroup(fine, groups):
    out={}
    for g in groups:
        present=[c for c in g if c in fine.columns]
        out['+'.join(g)] = fine[present].sum(axis=1) if present else 0.0
    return pd.DataFrame(out, index=fine.index)

def calc(tab, min_q, n_repeats, n_boot):
    n = tab.sum(axis=1); keep = n[n>=min_q].index
    if len(keep) < 3: return None
    sub = tab.loc[keep]; ssum = sub.sum(axis=0)
    dist = sub.div(sub.sum(axis=1), axis=0).values
    pooled = (ssum/ssum.sum()).values
    r = calibrate_with_ci(dist, pooled, n.loc[keep].values, n_repeats, n_boot)
    return {'n':int(len(keep)),'alpha':round(r['matched_alpha'],3),
            'ci':[round(r['alpha_ci95'][0],3),round(r['alpha_ci95'][1],3)],
            'obs_js':round(r['observed_mean_js'],4)}

def cluster_on_dist(dist, kmax=8):
    X=np.clip(dist,1e-12,None); n=X.shape[0]; res=[]
    for k in range(2, min(kmax,n-1)+1):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); lab=km.labels_
        sil=float(silhouette_score(X,lab)) if len(set(lab))>1 else float('nan')
        within=[]; between=[]
        for i in range(n):
            p=X[i]
            for j in range(i+1,n):
                q=X[j]; m=0.5*(p+q)
                d=0.5*np.sum(p*(np.log(p)-np.log(m)))/LOG2+0.5*np.sum(q*(np.log(q)-np.log(m)))/LOG2
                (within if lab[i]==lab[j] else between).append(d)
        sr=float(np.mean(between)/np.mean(within)) if within and np.mean(within)>0 else float('nan')
        res.append({'k':int(k),'silhouette':round(sil,3),'separation_ratio':round(sr,3)})
    best=max(res,key=lambda r:(r['silhouette'] if not np.isnan(r['silhouette']) else -1)) if res else {}
    return {'per_k':res,'best':best}

def run(input_path, out_dir, memory_limit, thresholds, n_repeats, n_boot):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    con=duckdb.connect(); con.execute(f"SET memory_limit='{memory_limit}'"); con.execute("PRAGMA threads=4")
    (out/"duckdb_tmp").mkdir(exist_ok=True); con.execute(f"SET temp_directory='{out}/duckdb_tmp'")
    src="read_parquet('"+input_path.replace("'","''")+"')"

    print("[agg] fine per-proxy counts (one pass each) ...")
    qt   = fine_counts(con, src, "query_type")
    scan = fine_counts(con, src, SCAN_FINE, SCAN_ORDER)
    joins= fine_counts(con, src, JOIN_FINE, JOIN_ORDER)
    hour = fine_counts(con, src, HOUR_FINE, HOUR_ORDER)
    # query_type variants: default all; grouped = keep top-5 by pooled freq, merge rest into 'misc'
    pooled_qt = qt.sum(axis=0).sort_values(ascending=False)
    top5 = list(pooled_qt.index[:5])
    qt_groups_default=[[c] for c in qt.columns]
    qt_groups_grouped=[[c] for c in top5]+[[c for c in qt.columns if c not in top5]]
    PROXIES={
        'query_type': (qt, {'default':qt_groups_default, 'grouped6':qt_groups_grouped}),
        'scan_volume':(scan, SCAN_VAR),
        'joins':      (joins, JOIN_VAR),
        'hour_of_day':(hour, HOUR_VAR),
    }

    robustness={}; clustering={}
    print(f"[calc] bin-robustness across thresholds {thresholds} (n_repeats={n_repeats}, n_boot={n_boot})")
    for pname,(fine,variants) in PROXIES.items():
        robustness[pname]={}
        for vname,groups in variants.items():
            tab=regroup(fine, groups)
            robustness[pname][vname]={}
            for thr in thresholds:
                r=calc(tab, thr, n_repeats, n_boot)
                robustness[pname][vname][f"minq{thr}"]=r
                if r: print(f"  {pname:12} {vname:9} minq{thr:<5} n={r['n']:<4} alpha={r['alpha']:<6} CI{r['ci']}")
        # per-proxy clustering on the DEFAULT binning at the first threshold
        d0=regroup(fine, list(variants.values())[0]); n=d0.sum(axis=1); keep=n[n>=thresholds[0]].index
        if len(keep)>=4:
            dist=d0.loc[keep].div(d0.loc[keep].sum(axis=1),axis=0).values
            clustering[pname]=cluster_on_dist(dist)
            b=clustering[pname]['best']
            print(f"  [cluster] {pname:12} best k={b.get('k')} sil={b.get('silhouette')} sep={b.get('separation_ratio')}")

    results={'input':os.path.basename(input_path),'thresholds':thresholds,
             'n_repeats':n_repeats,'n_boot':n_boot,'bin_robustness':robustness,'per_proxy_clustering':clustering}
    (out/'results_e8.json').write_text(json.dumps(results,indent=2))
    print(f"[done] wrote {out}/results_e8.json")
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--out", default="results/e8")
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--min-queries-list", default="1000,2000")
    p.add_argument("--n-repeats", type=int, default=15); p.add_argument("--n-boot", type=int, default=150)
    a=p.parse_args()
    thr=[int(x) for x in a.min_queries_list.split(",")]
    run(a.input, a.out, a.memory_limit, thr, a.n_repeats, a.n_boot)

if __name__=="__main__":
    main()
