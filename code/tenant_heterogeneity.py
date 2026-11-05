"""
Tenant heterogeneity analysis for Redset (Amazon Redshift fleet query metadata).

Research question this answers
-----------------------------
Federated learning papers manufacture non-IID clients by Dirichlet-partitioning a
centralised dataset, and then report results at alpha = 0.1 / 0.5 / 1.0 without any
evidence about which alpha reflects reality. Redset contains 200 provisioned and 200
serverless instances whose workloads were never pooled and never partitioned, so the
heterogeneity between them is real rather than manufactured.

This module measures that natural heterogeneity and calibrates it against the
Dirichlet knob, producing the headline claim:

    "Natural cross-tenant heterogeneity in a production analytics fleet corresponds
     to Dirichlet alpha ~= X, which is [inside / outside] the range the FL
     literature routinely evaluates."

Usage
-----
    python tenant_heterogeneity.py --input serverless/full.parquet --out results/

Data
----
Redset, CC BY-NC 4.0, https://github.com/amazon-science/redset
    aws s3 cp --no-sign-request s3://redshift-downloads/redset/serverless/full.parquet .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260816)

# Columns this analysis depends on. Kept explicit so a schema drift fails loudly
# rather than silently producing a different result.
REQUIRED_COLUMNS = [
    "instance_id",
    "arrival_timestamp",
    "query_type",
    "feature_fingerprint",
    "was_aborted",
    "was_cached",
    "num_joins",
    "num_scans",
    "num_aggregations",
    "mbytes_scanned",
    "mbytes_spilled",
    "num_permanent_tables_accessed",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load(path: str, min_queries: int = 1000) -> pd.DataFrame:
    """Load Redset parquet and drop instances too small to characterise."""
    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Redset schema mismatch, missing columns: {missing}")

    df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])

    counts = df.groupby("instance_id").size()
    keep = counts[counts >= min_queries].index
    dropped = len(counts) - len(keep)
    if dropped:
        print(f"[load] dropping {dropped} instances with < {min_queries} queries")

    df = df[df["instance_id"].isin(keep)].copy()
    print(f"[load] {len(df):,} queries across {df['instance_id'].nunique()} instances")
    return df


# --------------------------------------------------------------------------
# Per-instance workload signatures
# --------------------------------------------------------------------------

def query_type_distributions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalised query-type histogram per instance.

    This is the direct analogue of the label distribution that FL papers skew
    with a Dirichlet prior, which is what makes the calibration in
    `dirichlet_calibration` an apples-to-apples comparison.
    """
    types = sorted(df["query_type"].dropna().unique().tolist())
    tab = (
        df.pivot_table(
            index="instance_id",
            columns="query_type",
            values="query_id" if "query_id" in df.columns else "num_scans",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=types, fill_value=0)
        .astype(float)
    )
    dist = tab.div(tab.sum(axis=1), axis=0)
    return dist, types


def _burstiness(ts: pd.Series) -> float:
    """Coefficient of variation of hourly query counts, a scale-free spikiness measure."""
    hourly = ts.dt.floor("h").value_counts().sort_index()
    if len(hourly) < 2 or hourly.mean() == 0:
        return 0.0
    return float(hourly.std() / hourly.mean())


def _repetition_rate(fingerprints: pd.Series) -> float:
    """Fraction of queries whose fingerprint was already seen on that instance."""
    n = len(fingerprints)
    if n == 0:
        return 0.0
    return float(1.0 - fingerprints.nunique() / n)


def behavioural_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Scalar workload descriptors per instance, used for clustering."""
    rows = []
    for iid, g in df.groupby("instance_id"):
        scanned = g["mbytes_scanned"].fillna(0.0)
        rows.append(
            {
                "instance_id": iid,
                "n_queries": len(g),
                "repetition_rate": _repetition_rate(g["feature_fingerprint"]),
                "abort_rate": float(g["was_aborted"].fillna(0).astype(float).mean()),
                "cache_rate": float(g["was_cached"].fillna(0).astype(float).mean()),
                "mean_joins": float(g["num_joins"].fillna(0).mean()),
                "mean_scans": float(g["num_scans"].fillna(0).mean()),
                "mean_aggs": float(g["num_aggregations"].fillna(0).mean()),
                "mean_tables": float(g["num_permanent_tables_accessed"].fillna(0).mean()),
                "log_mean_scanned": float(np.log1p(scanned).mean()),
                "log_std_scanned": float(np.log1p(scanned).std()),
                "spill_rate": float((g["mbytes_spilled"].fillna(0) > 0).mean()),
                "burstiness": _burstiness(g["arrival_timestamp"]),
            }
        )
    return pd.DataFrame(rows).set_index("instance_id")


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

def pairwise_js(dist: np.ndarray, max_pairs: int = 200_000) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence (squared JS distance, in [0, 1], base 2).

    Subsamples pairs when the client count makes the full matrix expensive.
    """
    n = dist.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        idx = RNG.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    out = np.empty(len(pairs), dtype=float)
    for k, (i, j) in enumerate(pairs):
        d = jensenshannon(dist[i], dist[j], base=2)
        # jensenshannon can emit nan from a marginally negative value under sqrt
        out[k] = 0.0 if np.isnan(d) else float(np.clip(d, 0.0, 1.0)) ** 2
    return out


def summarise(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "std": float(np.std(values)),
    }


# --------------------------------------------------------------------------
# Dirichlet calibration: the headline result
# --------------------------------------------------------------------------

def dirichlet_calibration(
    pooled: np.ndarray,
    n_clients: int,
    observed_mean_js: float,
    client_sizes: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    n_repeats: int = 20,
) -> dict:
    """Find the Dirichlet alpha whose synthetic client heterogeneity matches reality.

    For each alpha we draw `n_clients` label distributions from Dir(alpha * pooled),
    exactly as the FL literature does, and compute the same mean pairwise JS
    divergence we measured on the real fleet. The alpha minimising the gap is the
    calibration point.

    Sampling correction
    -------------------
    The observed side measures EMPIRICAL query-type distributions estimated from a
    finite number of queries per instance. Finite-sample noise inflates divergence,
    especially for rare query types. Simulating Dirichlet distributions directly and
    comparing them to empirical ones is therefore not like-for-like, and biases the
    recovered alpha downwards (measured at up to 74% relative error on synthetic
    ground truth). We correct this by drawing a multinomial sample of the same size
    as each real instance and measuring the simulated clients' empirical
    distributions, so both sides of the comparison undergo identical estimation.
    """
    if alphas is None:
        alphas = np.concatenate([
            np.geomspace(0.01, 1.0, 25),
            np.geomspace(1.0, 100.0, 25)[1:],
        ])

    if client_sizes is None:
        client_sizes = np.full(n_clients, 5000)
    client_sizes = np.asarray(client_sizes, dtype=int)

    curve = []
    for alpha in alphas:
        means = []
        for _ in range(n_repeats):
            latent = RNG.dirichlet(alpha * pooled + 1e-12, size=n_clients)
            # Match the observed estimator: sample, then re-estimate.
            sim = np.empty_like(latent)
            for c in range(n_clients):
                counts = RNG.multinomial(client_sizes[c], latent[c])
                total = counts.sum()
                sim[c] = counts / total if total else latent[c]
            means.append(float(np.mean(pairwise_js(sim, max_pairs=20_000))))
        curve.append({"alpha": float(alpha), "mean_js": float(np.mean(means)),
                      "std_js": float(np.std(means))})

    gaps = [abs(c["mean_js"] - observed_mean_js) for c in curve]
    best = curve[int(np.argmin(gaps))]

    return {
        "observed_mean_js": observed_mean_js,
        "matched_alpha": best["alpha"],
        "matched_mean_js": best["mean_js"],
        "curve": curve,
    }


# --------------------------------------------------------------------------
# Cluster structure: does heterogeneity have exploitable structure?
# --------------------------------------------------------------------------

def cluster_structure(sig: pd.DataFrame, dist: np.ndarray, k_range=range(2, 9)) -> dict:
    """Test whether tenants fall into stable workload groups.

    Clustered FL only helps if between-cluster divergence exceeds within-cluster
    divergence. If that gap is absent, the honest finding is that clustering does
    not help on this fleet, and that is a publishable negative result.
    """
    X = StandardScaler().fit_transform(sig.drop(columns=["n_queries"]).values)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        labels = km.labels_
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")

        within, between = [], []
        n = dist.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = jensenshannon(dist[i], dist[j], base=2)
                d = 0.0 if np.isnan(d) else d**2
                (within if labels[i] == labels[j] else between).append(d)

        results.append({
            "k": int(k),
            "silhouette": sil,
            "within_js": float(np.mean(within)) if within else float("nan"),
            "between_js": float(np.mean(between)) if between else float("nan"),
            "separation_ratio": (float(np.mean(between) / np.mean(within))
                                 if within and np.mean(within) > 0 else float("nan")),
        })

    best = max(results, key=lambda r: (r["silhouette"] if not np.isnan(r["silhouette"]) else -1))
    return {"per_k": results, "best_k": best["k"], "best": best}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(input_path: str, out_dir: str, min_queries: int = 1000) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(input_path, min_queries=min_queries)

    dist_df, types = query_type_distributions(df)
    dist = dist_df.values
    sig = behavioural_signatures(df).loc[dist_df.index]

    js = pairwise_js(dist)
    observed = summarise(js)
    print(f"[divergence] mean pairwise JS across tenants = {observed['mean']:.4f}")

    pooled = df["query_type"].value_counts(normalize=True).reindex(types, fill_value=0).values
    calib = dirichlet_calibration(pooled, n_clients=dist.shape[0],
                                  observed_mean_js=observed["mean"],
                                  client_sizes=sig["n_queries"].values)
    print(f"[calibration] natural heterogeneity corresponds to Dirichlet alpha "
          f"~= {calib['matched_alpha']:.3f}")

    clusters = cluster_structure(sig, dist)
    print(f"[clustering] best k = {clusters['best_k']}, "
          f"separation ratio = {clusters['best']['separation_ratio']:.3f}")

    results = {
        "n_instances": int(dist.shape[0]),
        "n_queries": int(len(df)),
        "query_types": types,
        "observed_divergence": observed,
        "dirichlet_calibration": calib,
        "cluster_structure": clusters,
    }

    (out / "results.json").write_text(json.dumps(results, indent=2))
    sig.to_csv(out / "instance_signatures.csv")
    dist_df.to_csv(out / "query_type_distributions.csv")
    print(f"[done] wrote results to {out}/")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Redset parquet path")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--min-queries", type=int, default=1000)
    args = p.parse_args()
    run(args.input, args.out, args.min_queries)


if __name__ == "__main__":
    main()
