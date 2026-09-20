"""
Multi-proxy, memory-safe Redset heterogeneity analysis with bootstrapped alpha CIs.

Builds on tenant_heterogeneity_full.py. Two additions:
  1. MULTI-PROXY calibration. Recovers the Dirichlet alpha independently under four
     per-tenant histogram proxies: query_type mix, scan-volume bins, join-count
     buckets, and hour-of-day. Agreement across proxies defends the "is query_type a
     fair label proxy?" objection; disagreement is itself a finding.
  2. BOOTSTRAPPED INTERVAL on matched_alpha (never report a point estimate). The
     expensive DuckDB aggregation runs once; calibration + bootstrap run on the small
     per-instance histograms, so this is cheap post-processing.

Heavy aggregation via DuckDB (column pushdown, disk spill) so the 17 GB file streams
in bounded memory. Clustering (separation ratio) is unchanged from the base pipeline,
computed on the 12 scalar signatures vs the query_type distribution.

Usage (run from research/):
    python tenant_heterogeneity_mp.py --input data/full.parquet \
        --out results/full_provisioned_mp --min-queries 1000 --memory-limit 8GB \
        --n-repeats 20 --n-boot 300
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tenant_heterogeneity import pairwise_js, summarise, cluster_structure, RNG

LOG2 = np.log(2.0)

def _rate(col):
    return f"avg(CASE WHEN try_cast({col} AS INTEGER) <> 0 THEN 1.0 ELSE 0.0 END)"

# ------------------------------------------------------------------ metric
def mean_pairwise_js(dist: np.ndarray) -> float:
    """Mean over i<j of the squared Jensen-Shannon distance (base 2), vectorized.
    Equals scipy jensenshannon(p,q,base=2)**2 averaged over pairs, but far faster."""
    P = np.clip(dist, 1e-12, None)
    n = P.shape[0]
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n - 1):
        p = P[i]; Q = P[i + 1:]
        M = 0.5 * (p + Q)
        klp = np.sum(p * (np.log(p) - np.log(M)), axis=1) / LOG2
        klq = np.sum(Q * (np.log(Q) - np.log(M)), axis=1) / LOG2
        jsd = 0.5 * klp + 0.5 * klq
        tot += float(jsd.sum()); cnt += jsd.shape[0]
    return tot / cnt if cnt else 0.0

# ------------------------------------------------------------------ calibration + CI
def _alpha_grid():
    return np.concatenate([np.geomspace(0.01, 1.0, 25), np.geomspace(1.0, 100.0, 25)[1:]])

def calibration_curve(pooled, n_clients, client_sizes, alphas, n_repeats):
    """Simulated mean pairwise JS at each alpha, with the finite-sample correction
    (draw a multinomial of each real client's size, then re-estimate)."""
    client_sizes = np.asarray(client_sizes, dtype=np.int64)
    curve = []
    for alpha in alphas:
        vals = []
        for _ in range(n_repeats):
            latent = RNG.dirichlet(alpha * pooled + 1e-12, size=n_clients)
            sim = np.empty_like(latent)
            for c in range(n_clients):
                counts = RNG.multinomial(int(client_sizes[c]), latent[c])
                t = counts.sum()
                sim[c] = counts / t if t else latent[c]
            vals.append(mean_pairwise_js(sim))
        curve.append((float(alpha), float(np.mean(vals))))
    return curve

def _invert(curve, observed):
    """Map an observed mean-JS to alpha by interpolation. mean-JS decreases with alpha."""
    a = np.array([c[0] for c in curve]); js = np.array([c[1] for c in curve])
    order = np.argsort(js)                    # js ascending -> alpha descending
    o = float(np.clip(observed, js.min(), js.max()))
    return float(np.interp(o, js[order], a[order]))

def _mean_pairwise_js_masked(dist, idx):
    """Mean pairwise squared-JS over a bootstrap resample, SKIPPING self-pairs
    (both draws the same original instance) which would otherwise inject spurious
    JS=0 values and bias observed-JS down / recovered alpha up (worse at small n)."""
    D = np.clip(dist[idx], 1e-12, None)
    n = len(idx); tot = 0.0; cnt = 0
    for i in range(n - 1):
        p = D[i]; Q = D[i + 1:]
        keep = idx[i + 1:] != idx[i]
        if not keep.any():
            continue
        Q = Q[keep]; M = 0.5 * (p + Q)
        klp = np.sum(p * (np.log(p) - np.log(M)), axis=1) / LOG2
        klq = np.sum(Q * (np.log(Q) - np.log(M)), axis=1) / LOG2
        jsd = 0.5 * klp + 0.5 * klq
        tot += float(jsd.sum()); cnt += jsd.shape[0]
    return tot / cnt if cnt else 0.0

def calibrate_with_ci(dist, pooled, client_sizes, n_repeats=20, n_boot=300):
    observed = mean_pairwise_js(dist)
    curve = calibration_curve(pooled, dist.shape[0], client_sizes, _alpha_grid(), n_repeats)
    matched = _invert(curve, observed)
    n = dist.shape[0]
    boot_obs, boot_alpha = [], []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)      # resample tenants with replacement
        ob = _mean_pairwise_js_masked(dist, idx)
        boot_obs.append(ob); boot_alpha.append(_invert(curve, ob))
    lo_a, hi_a = np.percentile(boot_alpha, [2.5, 97.5])
    lo_o, hi_o = np.percentile(boot_obs, [2.5, 97.5])
    return {
        "observed_mean_js": observed,
        "observed_ci95": [float(lo_o), float(hi_o)],
        "matched_alpha": matched,
        "alpha_ci95": [float(lo_a), float(hi_a)],
        "curve": [{"alpha": a, "mean_js": j} for a, j in curve],
    }

# ------------------------------------------------------------------ proxy histograms
def _hist(con, src, expr, keep):
    df = con.execute(f"SELECT instance_id, {expr} AS bin, count(*) c FROM {src} "
                     f"GROUP BY instance_id, bin").df()
    df = df[df["instance_id"].isin(keep)]
    bins = sorted(df["bin"].dropna().astype(str).unique().tolist())
    df["bin"] = df["bin"].astype(str)
    tab = (df.pivot_table(index="instance_id", columns="bin", values="c", aggfunc="sum",
                          fill_value=0).reindex(columns=bins, fill_value=0).astype(float))
    dist_df = tab.div(tab.sum(axis=1), axis=0)
    pooled_counts = tab.sum(axis=0)
    pooled = (pooled_counts / pooled_counts.sum()).reindex(bins, fill_value=0).values
    return dist_df, pooled, bins

SCANV = ("CASE WHEN coalesce(mbytes_scanned,0)<=0 THEN 'b0_zero' "
         "WHEN mbytes_scanned<=1 THEN 'b1_le1' WHEN mbytes_scanned<=10 THEN 'b2_le10' "
         "WHEN mbytes_scanned<=100 THEN 'b3_le100' WHEN mbytes_scanned<=1000 THEN 'b4_le1k' "
         "ELSE 'b5_gt1k' END")
JOINS = ("CASE WHEN coalesce(num_joins,0)=0 THEN 'j0' WHEN num_joins=1 THEN 'j1' "
         "WHEN num_joins=2 THEN 'j2' ELSE 'j3plus' END")
HOUR  = "printf('h%02d', CAST(extract(hour FROM arrival_timestamp) AS INTEGER))"

def build(con, parquet, min_queries):
    src = "read_parquet('" + parquet.replace("'", "''") + "')"
    kept = con.execute(f"SELECT instance_id FROM {src} GROUP BY instance_id "
                       f"HAVING count(*) >= {min_queries}").df()
    keep = set(kept["instance_id"].tolist())
    if len(keep) < 2:
        raise SystemExit(f"only {len(keep)} instances have >= {min_queries} queries")

    qt = _hist(con, src, "query_type", keep)
    sv = _hist(con, src, SCANV, keep)
    jn = _hist(con, src, JOINS, keep)
    hr = _hist(con, src, HOUR, keep)

    scal = con.execute(f"""
        SELECT instance_id, count(*) AS n_queries,
            {_rate('was_aborted')} AS abort_rate, {_rate('was_cached')} AS cache_rate,
            avg(CASE WHEN coalesce(mbytes_spilled,0)>0 THEN 1.0 ELSE 0.0 END) AS spill_rate,
            avg(coalesce(num_joins,0)) AS mean_joins, avg(coalesce(num_scans,0)) AS mean_scans,
            avg(coalesce(num_aggregations,0)) AS mean_aggs,
            avg(coalesce(num_permanent_tables_accessed,0)) AS mean_tables,
            avg(ln(1+coalesce(mbytes_scanned,0))) AS log_mean_scanned,
            coalesce(stddev_samp(ln(1+coalesce(mbytes_scanned,0))),0.0) AS log_std_scanned,
            approx_count_distinct(feature_fingerprint) AS approx_distinct_fp
        FROM {src} GROUP BY instance_id""").df()
    scal = scal[scal["instance_id"].isin(keep)].set_index("instance_id")
    scal["repetition_rate"] = (1.0 - scal["approx_distinct_fp"] / scal["n_queries"]).clip(0, 1)
    burst = con.execute(f"""
        WITH h AS (SELECT instance_id, date_trunc('hour',arrival_timestamp) t, count(*) c
                   FROM {src} GROUP BY 1,2)
        SELECT instance_id, CASE WHEN avg(c)>0 THEN coalesce(stddev_samp(c),0)/avg(c) ELSE 0 END burstiness
        FROM h GROUP BY instance_id""").df().set_index("instance_id")
    scal = scal.join(burst, how="left"); scal["burstiness"] = scal["burstiness"].fillna(0.0)
    sig_cols = ["n_queries","repetition_rate","abort_rate","cache_rate","mean_joins","mean_scans",
                "mean_aggs","mean_tables","log_mean_scanned","log_std_scanned","spill_rate","burstiness"]
    sig = scal.reindex(columns=sig_cols).loc[qt[0].index]
    proxies = {"query_type": qt, "scan_volume": sv, "joins": jn, "hour_of_day": hr}
    return proxies, sig

def run(input_path, out_dir, min_queries=1000, memory_limit="8GB", threads=4, n_repeats=20, n_boot=300):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}"); (out/"duckdb_tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{out}/duckdb_tmp'")
    print(f"[build] aggregating {input_path} (4 proxies) ...")
    proxies, sig = build(con, input_path, min_queries)
    qt_dist = proxies["query_type"][0]
    n_inst = qt_dist.shape[0]
    print(f"[build] {n_inst} instances kept (>= {min_queries} queries)")

    proxy_results = {}
    for name, (dist_df, pooled, bins) in proxies.items():
        dist = dist_df.loc[qt_dist.index].values
        res = calibrate_with_ci(dist, pooled, sig["n_queries"].values, n_repeats, n_boot)
        proxy_results[name] = {"bins": bins, **{k: res[k] for k in
                               ["observed_mean_js","observed_ci95","matched_alpha","alpha_ci95"]}}
        a, lo, hi = res["matched_alpha"], *res["alpha_ci95"]
        print(f"[calib] {name:12} observed_JS={res['observed_mean_js']:.4f}  "
              f"matched_alpha={a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    clusters = cluster_structure(sig, qt_dist.values)
    b = clusters["best"]
    print(f"[cluster] best k={clusters['best_k']} silhouette={b['silhouette']:.3f} "
          f"separation_ratio={b['separation_ratio']:.3f}")

    results = {"input": os.path.basename(input_path), "min_queries": min_queries,
               "n_instances": int(n_inst), "n_queries": int(sig["n_queries"].sum()),
               "n_repeats": n_repeats, "n_boot": n_boot,
               "proxies": proxy_results, "cluster_structure": clusters}
    (out/"results.json").write_text(json.dumps(results, indent=2))
    sig.to_csv(out/"instance_signatures.csv")
    for name,(d,_,_) in proxies.items(): d.to_csv(out/f"dist_{name}.csv")
    print(f"[done] wrote results to {out}/")
    return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--out", default="results/mp")
    p.add_argument("--min-queries", type=int, default=1000)
    p.add_argument("--memory-limit", default="8GB"); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-repeats", type=int, default=20); p.add_argument("--n-boot", type=int, default=300)
    a = p.parse_args()
    run(a.input, a.out, a.min_queries, a.memory_limit, a.threads, a.n_repeats, a.n_boot)

if __name__ == "__main__":
    main()
