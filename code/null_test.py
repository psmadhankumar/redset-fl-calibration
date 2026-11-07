#!/usr/bin/env python3
# null_test.py - Null-model cluster test (reproduction script)
#
# WHAT THIS IS
# This script reproduces the null-model cluster test reported in the paper
# (Section "Method", paragraph "Testing for real clusters", and the null-test
# table). For each workload proxy and fleet it:
#   1. clusters the real tenants on the proxy's own distribution with k-means
#      (n_init=10, random_state=42), sweeping k from 2..8 and selecting the k
#      with the highest silhouette, then measures the separation ratio
#      (mean between-cluster divergence / mean within-cluster divergence, using
#      base-2 Jensen-Shannon divergence);
#   2. draws 200 null fleets of independent Dirichlet clients at the fleet's
#      recovered alpha, with per-client query counts matched to the real tenants
#      and the same finite-sample multinomial step, clusters each null fleet at
#      the SAME k chosen for the real fleet (not re-optimised), and measures its
#      separation ratio;
#   3. reports p = fraction of null fleets whose separation ratio meets or
#      exceeds the real fleet's. p < 0.05 means tenants cluster more than
#      independent Dirichlet draws would.
#
# INPUTS (all shipped in this repo; the raw 17 GB Redset trace is NOT needed):
#   partitions/<fleet>/dist_<proxy>.csv         per-tenant proxy distributions
#   partitions/<fleet>/instance_signatures.csv  per-tenant n_queries
#   results/<fleet>_results.json                recovered alpha (matched_alpha)
# OUTPUT:
#   results/null_test_reproduction.json
#
# NOTE ON THE NUMBERS
# The recorded null-test outputs behind the paper live in
# results/null_and_fixedci.json; that file is the canonical record for the
# published table. This script regenerates the test from the released
# partitions. Because the original was produced in a separate seeded run, a
# fresh run can differ in the last digit, but the conclusion reproduces: seven
# of eight proxy-fleet combinations are significant, with serverless query_type
# the lone exception. This script does NOT overwrite null_and_fixedci.json.

import json, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

LOG2 = np.log(2.0)
SEED = 20260816          # matches the base module RNG seed; run is deterministic
B = 200                  # number of null draws (as in the paper)
PROXIES = ["query_type", "scan_volume", "joins", "hour_of_day"]
FLEETS = ["provisioned", "serverless"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def js_matrix(X):
    # Full pairwise base-2 Jensen-Shannon divergence matrix for rows of X.
    P = X[:, None, :]; Q = X[None, :, :]; M = 0.5 * (P + Q)
    klp = np.sum(P * (np.log(P) - np.log(M)), axis=2)
    klq = np.sum(Q * (np.log(Q) - np.log(M)), axis=2)
    return (0.5 * klp + 0.5 * klq) / LOG2

def separation(JS, lab):
    n = len(lab); iu = np.triu_indices(n, 1)
    d = JS[iu]; same = lab[iu[0]] == lab[iu[1]]
    within = d[same]; between = d[~same]
    if within.size == 0 or within.mean() <= 0:
        return float("nan")
    return float(between.mean() / within.mean())

def best_clustering(X):
    JS = js_matrix(X); n = X.shape[0]; best = None
    for k in range(2, min(8, n - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_
        sil = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else float("nan")
        sr = separation(JS, lab)
        cur = {"k": k, "silhouette": sil, "separation_ratio": sr}
        cmp = best["silhouette"] if (best and not np.isnan(best["silhouette"])) else -1
        if best is None or (not np.isnan(sil) and sil > cmp):
            best = cur
    return best

def sep_at_k(X, k):
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_
    return separation(js_matrix(X), lab)

def load(fleet, proxy):
    dd = pd.read_csv(f"{ROOT}/partitions/{fleet}/dist_{proxy}.csv")
    sig = pd.read_csv(f"{ROOT}/partitions/{fleet}/instance_signatures.csv")[["instance_id", "n_queries"]]
    df = dd.merge(sig, on="instance_id")
    bins = [c for c in dd.columns if c != "instance_id"]
    return df[bins].values.astype(float), df["n_queries"].values.astype(np.int64)

def alpha_for(fleet, proxy):
    r = json.load(open(f"{ROOT}/results/{fleet}_results.json"))
    return float(r["proxies"][proxy]["matched_alpha"])

def main():
    out = {}
    for fleet in FLEETS:
        out[fleet] = {}
        for proxy in PROXIES:
            dist, nq = load(fleet, proxy)
            X = np.clip(dist, 1e-12, None); n = X.shape[0]
            alpha = alpha_for(fleet, proxy)
            pooled = (nq[:, None] * X).sum(axis=0); pooled = pooled / pooled.sum()
            best = best_clustering(X); k = best["k"]
            real_sep = sep_at_k(X, k)
            rng = np.random.default_rng(SEED)
            seps = []
            for _ in range(B):
                latent = rng.dirichlet(alpha * pooled + 1e-12, size=n)
                ph = np.empty_like(latent)
                for c in range(n):
                    ph[c] = rng.multinomial(nq[c], latent[c]) / nq[c]
                seps.append(sep_at_k(np.clip(ph, 1e-12, None), k))
            seps = np.array([s for s in seps if not np.isnan(s)])
            p = float(np.mean(seps >= real_sep))
            out[fleet][proxy] = {
                "alpha": round(alpha, 3), "n": int(n), "best_k": int(k),
                "silhouette": round(best["silhouette"], 3),
                "null_test": {
                    "k": int(k), "real_sep": round(real_sep, 4),
                    "null_mean": round(float(seps.mean()), 4),
                    "null_p95": round(float(np.percentile(seps, 95)), 4),
                    "B": B, "p_value": round(p, 4),
                    "p_report": f"< {1.0 / B:.4f}" if p == 0.0 else round(p, 4),
                },
            }
            print(f"{fleet:12} {proxy:12} k={k} real_sep={real_sep:.3f} "
                  f"null_p95={np.percentile(seps,95):.3f} p={p:.4f}")
    json.dump(out, open(f"{ROOT}/results/null_test_reproduction.json", "w"), indent=2)
    print("\nwrote results/null_test_reproduction.json")

if __name__ == "__main__":
    main()
