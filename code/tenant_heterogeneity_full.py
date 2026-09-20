"""
Full-scale, memory-safe version of the Redset tenant-heterogeneity analysis.

Why this file exists
--------------------
tenant_heterogeneity.py loads the whole parquet into pandas. Fine for the sample
files, impossible for the 17 GB full.parquet (it would expand to well over 100 GB
in RAM). This version does the heavy per-instance aggregation with DuckDB, which
streams the parquet with column pushdown and spills to disk, so memory stays
bounded (a few GB) regardless of file size. The light downstream steps
(divergence, Dirichlet calibration, clustering) are unchanged and reuse the exact
functions from tenant_heterogeneity.py, so results match the original pipeline.

Notes
-----
- repetition_rate uses DuckDB approx_count_distinct (about 2% error) instead of an
  exact per-instance distinct count, so it stays memory-bounded on the full file.
  Everything else is exact.
- Run from the research/ folder so the import below resolves.

Usage
-----
    python tenant_heterogeneity_full.py --input data/full.parquet \
        --out results/full_serverless --min-queries 1000 --memory-limit 8GB
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tenant_heterogeneity import pairwise_js, summarise, dirichlet_calibration, cluster_structure

SIG_COLUMNS = [
    "n_queries", "repetition_rate", "abort_rate", "cache_rate",
    "mean_joins", "mean_scans", "mean_aggs", "mean_tables",
    "log_mean_scanned", "log_std_scanned", "spill_rate", "burstiness",
]

def _rate(col):
    return f"avg(CASE WHEN try_cast({col} AS INTEGER) <> 0 THEN 1.0 ELSE 0.0 END)"

def build_profiles(con, parquet, min_queries):
    src = "read_parquet('" + parquet.replace("'", "''") + "')"

    kept = con.execute(f"SELECT instance_id, count(*) n FROM {src} "
                       f"GROUP BY instance_id HAVING count(*) >= {min_queries}").df()
    keep = set(kept["instance_id"].tolist())
    if len(keep) < 2:
        raise SystemExit(f"only {len(keep)} instances have >= {min_queries} queries; lower --min-queries")

    qt = con.execute(f"SELECT instance_id, query_type, count(*) c FROM {src} "
                     f"GROUP BY instance_id, query_type").df()
    qt = qt[qt["instance_id"].isin(keep)]
    types = sorted(qt["query_type"].dropna().unique().tolist())
    tab = (qt.pivot_table(index="instance_id", columns="query_type", values="c",
                          aggfunc="sum", fill_value=0)
             .reindex(columns=types, fill_value=0).astype(float))
    dist_df = tab.div(tab.sum(axis=1), axis=0)
    pooled_counts = tab.sum(axis=0)
    pooled = (pooled_counts / pooled_counts.sum()).reindex(types, fill_value=0).values

    scal = con.execute(f"""
        SELECT instance_id,
            count(*) AS n_queries,
            {_rate('was_aborted')} AS abort_rate,
            {_rate('was_cached')}  AS cache_rate,
            avg(CASE WHEN coalesce(mbytes_spilled,0) > 0 THEN 1.0 ELSE 0.0 END) AS spill_rate,
            avg(coalesce(num_joins,0)) AS mean_joins,
            avg(coalesce(num_scans,0)) AS mean_scans,
            avg(coalesce(num_aggregations,0)) AS mean_aggs,
            avg(coalesce(num_permanent_tables_accessed,0)) AS mean_tables,
            avg(ln(1 + coalesce(mbytes_scanned,0))) AS log_mean_scanned,
            coalesce(stddev_samp(ln(1 + coalesce(mbytes_scanned,0))), 0.0) AS log_std_scanned,
            approx_count_distinct(feature_fingerprint) AS approx_distinct_fp
        FROM {src} GROUP BY instance_id""").df()
    scal = scal[scal["instance_id"].isin(keep)].set_index("instance_id")
    scal["repetition_rate"] = (1.0 - (scal["approx_distinct_fp"] / scal["n_queries"])).clip(0, 1)

    burst = con.execute(f"""
        WITH hourly AS (
            SELECT instance_id, date_trunc('hour', arrival_timestamp) h, count(*) c
            FROM {src} GROUP BY 1, 2)
        SELECT instance_id,
               CASE WHEN avg(c) > 0 THEN coalesce(stddev_samp(c),0)/avg(c) ELSE 0 END AS burstiness
        FROM hourly GROUP BY instance_id""").df().set_index("instance_id")
    scal = scal.join(burst, how="left")
    scal["burstiness"] = scal["burstiness"].fillna(0.0)

    sig = scal.reindex(columns=SIG_COLUMNS).loc[dist_df.index]
    return dist_df, sig, types, pooled

def run(input_path, out_dir, min_queries=1000, memory_limit="8GB", threads=4):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    (out / "duckdb_tmp").mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{out}/duckdb_tmp'")

    print(f"[build] aggregating {input_path} with DuckDB (memory_limit={memory_limit}) ...")
    dist_df, sig, types, pooled = build_profiles(con, input_path, min_queries)
    dist = dist_df.values
    print(f"[build] {dist.shape[0]} instances kept (>= {min_queries} queries), {len(types)} query types")

    js = pairwise_js(dist); observed = summarise(js)
    print(f"[divergence] mean pairwise JS = {observed['mean']:.4f}")
    calib = dirichlet_calibration(pooled, n_clients=dist.shape[0],
                                  observed_mean_js=observed["mean"], client_sizes=sig["n_queries"].values)
    print(f"[calibration] matched Dirichlet alpha ~= {calib['matched_alpha']:.3f}")
    clusters = cluster_structure(sig, dist)
    print(f"[clustering] best k = {clusters['best_k']}, separation ratio = {clusters['best']['separation_ratio']:.3f}")

    results = {"input": os.path.basename(input_path), "min_queries": min_queries,
               "n_instances": int(dist.shape[0]), "n_queries": int(sig["n_queries"].sum()),
               "query_types": types, "observed_divergence": observed,
               "dirichlet_calibration": calib, "cluster_structure": clusters}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    sig.to_csv(out / "instance_signatures.csv")
    dist_df.to_csv(out / "query_type_distributions.csv")
    print(f"[done] wrote results to {out}/")
    return results

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/full")
    p.add_argument("--min-queries", type=int, default=1000)
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    run(a.input, a.out, a.min_queries, a.memory_limit, a.threads)

if __name__ == "__main__":
    main()
