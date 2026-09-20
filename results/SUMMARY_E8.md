# E8 robustness + per-proxy clustering - 2026-08-31

## 1. Bin robustness: what is real vs a binning artifact
matched_alpha across binnings (provisioned | serverless), LOWER = more heterogeneous:
- query_type: prov ~3.2-3.5 (stable across default/grouped) | serverless ~1.0-1.3 (stable). ROBUST.
- scan_volume: prov ~2.1-3.1 | serverless ~1.9-2.9. ROBUST (heterogeneous).
- joins: prov ~7.5-9.9 | serverless ~5.5-7.8. ROBUST (homogeneous; stays high at every binning).
- hour_of_day: prov 11.4 (24 bins) -> 4.7 (8 blocks) -> 2.9 (4 blocks); serverless similar. NOT ROBUST:
  the high alpha was largely a BIN-COUNT ARTIFACT. At comparable bin resolution hour is ~3-5, similar to
  scan/query_type, NOT the most homogeneous.

### Corrected ordering (must be stated at comparable bin resolution)
- JOINS are robustly the MOST homogeneous (tenants have similar join-complexity mixes; alpha ~7-9 at all binnings).
- scan_volume and query_type are robustly HETEROGENEOUS (alpha ~2-3).
- hour_of_day is heterogeneous-to-moderate once binning is comparable. DROP the earlier "near-IID in daily
  timing" claim - it was a 24-bin artifact.
- FLEET difference on query_type (prov ~3.3 vs serverless ~1.0) is ROBUST across binnings and non-overlapping -> solid E7 result.

## 2. Per-proxy clustering (fixed: cluster ON each proxy's own distribution)
Separation ratio / silhouette at best k (provisioned): query_type 2.26/0.33; scan_volume 4.23/0.40; joins 3.00/0.49; hour 2.86/0.59.
Serverless similar (joins 3.29/0.60).
=> The earlier "no clusters" (scalar method, sep ~0.97) was a MEASUREMENT ARTIFACT (clustered on scalars,
   measured separation on query_type). Clustering in each proxy's OWN space shows MODERATE structure
   (silhouette 0.3-0.6).
CAVEAT: separation measured in the same space you cluster in rises mechanically with k, so a high ratio is
   NOT by itself proof of natural clusters. Need a NULL-MODEL test: do real tenants cluster more than
   Dirichlet-simulated clients at the matched alpha? Run that before claiming clustered FL helps.
=> This REOPENS Track 2's clustered-vs-personalized FL question (was leaning personalized).

## 3. Net for the paper
- Reframe thesis (heterogeneity is MULTI-DIMENSIONAL and FLEET-DEPENDENT) STANDS, now bin-robust for the
  within-proxy and fleet claims.
- Correct the ordering: joins robustly homogeneous; hour_of_day was inflated by fine binning.
- Clustering upgraded from "none" to "moderate, pending a null-model test."

## 4. Next
1. Null-model cluster test (real vs Dirichlet-at-matched-alpha separation, per proxy) -> decides Track 2.
2. Then DRAFT Track 1 around: per-proxy calibration with CIs, bin-robustness, fleet comparison, per-proxy structure.
3. Optional forest-plot figure (can also show the bin-robustness sweep).
