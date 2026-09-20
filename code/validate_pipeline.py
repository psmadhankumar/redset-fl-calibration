"""
Pipeline validation for tenant_heterogeneity.py.

IMPORTANT
---------
This script generates SYNTHETIC data in the Redset schema. It produces NO evidence
whatsoever about the real Redshift fleet. Its only purpose is twofold:

  1. Smoke test: confirm the pipeline runs end to end on the exact Redset schema.
  2. Recovery test: confirm the Dirichlet calibration estimator is unbiased, by
     generating clients from a KNOWN alpha and checking the estimator recovers it.

The recovery test is the one that matters. If the estimator cannot recover an alpha
it was given, the headline claim of the paper would be unsound, and we would want to
know that before touching the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenant_heterogeneity import (
    dirichlet_calibration,
    pairwise_js,
    query_type_distributions,
    run,
)

RNG = np.random.default_rng(7)

QUERY_TYPES = ["select", "insert", "copy", "update", "delete", "ctas", "unload", "analyze"]
# Fleet-level mix loosely echoing the published Redshift distribution
# (select 48.9%, insert 19.5%, copy 7.4%, ctas 1.9%, remainder spread).
POOLED = np.array([0.489, 0.195, 0.074, 0.060, 0.045, 0.019, 0.068, 0.050])
POOLED = POOLED / POOLED.sum()


def synth_redset(n_instances: int, alpha: float, queries_per_instance: int = 3000) -> pd.DataFrame:
    """Emit a Redset-schema frame whose clients are drawn from a known Dirichlet alpha."""
    client_dists = RNG.dirichlet(alpha * POOLED, size=n_instances)
    frames = []

    for i in range(n_instances):
        n = queries_per_instance
        qtypes = RNG.choice(QUERY_TYPES, size=n, p=client_dists[i])
        # Repetition: a small fingerprint vocabulary reused, per-instance rate varies.
        vocab = max(5, int(n * RNG.uniform(0.02, 0.9)))
        frames.append(
            pd.DataFrame(
                {
                    "instance_id": i,
                    "cluster_size": RNG.choice([2, 4, 8, 16]),
                    "user_id": RNG.integers(0, 20, size=n),
                    "database_id": RNG.integers(0, 5, size=n),
                    "query_id": np.arange(n),
                    "arrival_timestamp": pd.Timestamp("2024-03-01")
                    + pd.to_timedelta(np.sort(RNG.uniform(0, 90 * 24, size=n)), unit="h"),
                    "compile_duration_ms": RNG.exponential(50, size=n),
                    "queue_duration_ms": RNG.exponential(20, size=n),
                    "execution_duration_ms": RNG.exponential(500, size=n),
                    "feature_fingerprint": RNG.integers(0, vocab, size=n),
                    "was_aborted": RNG.random(n) < RNG.uniform(0.001, 0.05),
                    "was_cached": RNG.random(n) < RNG.uniform(0.0, 0.4),
                    "cache_source_query_id": -1,
                    "query_type": qtypes,
                    "num_permanent_tables_accessed": RNG.poisson(2, size=n),
                    "num_external_tables_accessed": RNG.poisson(0.2, size=n),
                    "num_system_tables_accessed": RNG.poisson(0.3, size=n),
                    "read_table_ids": "1,2",
                    "write_table_ids": "3",
                    "mbytes_scanned": RNG.lognormal(RNG.uniform(1, 6), 1.5, size=n),
                    "mbytes_spilled": RNG.lognormal(0, 2, size=n) * (RNG.random(n) < 0.1),
                    "num_joins": RNG.poisson(RNG.uniform(0.5, 5), size=n),
                    "num_scans": RNG.poisson(RNG.uniform(1, 8), size=n),
                    "num_aggregations": RNG.poisson(RNG.uniform(0.2, 3), size=n),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def recovery_test(alphas_true=(0.1, 0.5, 2.0, 10.0), n_instances: int = 60) -> pd.DataFrame:
    """Generate clients at a known alpha, check the estimator recovers it."""
    rows = []
    for a_true in alphas_true:
        df = synth_redset(n_instances, alpha=a_true, queries_per_instance=2000)
        dist_df, types = query_type_distributions(df)
        observed = float(np.mean(pairwise_js(dist_df.values)))
        pooled = (
            df["query_type"].value_counts(normalize=True).reindex(types, fill_value=0).values
        )
        sizes = df.groupby("instance_id").size().loc[dist_df.index].values
        calib = dirichlet_calibration(
            pooled,
            n_clients=dist_df.shape[0],
            observed_mean_js=observed,
            client_sizes=sizes,
            n_repeats=10,
        )
        a_hat = calib["matched_alpha"]
        rows.append(
            {
                "alpha_true": a_true,
                "alpha_recovered": round(a_hat, 3),
                "mean_js": round(observed, 4),
                "rel_error": round(abs(a_hat - a_true) / a_true, 3),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 68)
    print("RECOVERY TEST: can the estimator recover a known Dirichlet alpha?")
    print("=" * 68)
    res = recovery_test()
    print(res.to_string(index=False))
    print()

    print("=" * 68)
    print("SMOKE TEST: full pipeline on Redset-schema synthetic data")
    print("=" * 68)
    df = synth_redset(n_instances=40, alpha=0.5)
    df.to_parquet("/tmp/synthetic_redset.parquet", index=False)
    run("/tmp/synthetic_redset.parquet", "/tmp/synth_results", min_queries=500)
