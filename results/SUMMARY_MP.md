# Multi-proxy calibration + provisioned vs serverless (E7) - 2026-08-31

## Setup
- provisioned_full.parquet (17 GB): 143 tenants (>=1000 queries), 433,097,336 queries.
- serverless_full.parquet (308 MB): 39 tenants (>=1000 queries), 8,231,936 queries. Serverless is a genuinely smaller/sparser fleet (only 39 of ~200 instances clear the threshold).
- 4 histogram proxies; matched Dirichlet alpha with bootstrapped 95% CI (n_boot=300). LOWER alpha = MORE heterogeneous.

## Matched alpha by proxy (95% CI)
| proxy       | provisioned          | serverless            |
|-------------|----------------------|-----------------------|
| query_type  | 3.34 [2.80, 4.07]    | 1.04 [0.86, 1.69]     |
| scan_volume | 2.56 [2.14, 3.15]    | 2.95 [2.27, 4.21]     |
| joins       | 8.33 [6.73, 10.58]   | 7.88 [5.03, 12.92]    |
| hour_of_day | 11.57 [8.64, 15.36]  | 13.77 [10.03, 22.46]  |

## Two headline findings
1. HETEROGENEITY IS DIMENSION-DEPENDENT. Within a fleet, alpha varies a lot by proxy: tenants differ most in scan volume and query-type mix (alpha ~1-3, heterogeneous) and are much more alike in join complexity and hour-of-day (alpha ~8-14, near-IID). The ordering (scan_volume ~ query_type < joins < hour_of_day) holds in BOTH splits. => A single alpha cannot summarize a fleet; the FL literature's single-knob abstraction hides this. Stronger message than one number.
2. PROVISIONED vs SERVERLESS DIFFER, especially on query_type. Serverless query-type alpha 1.04 (CI [0.86,1.69]) lands right IN the literature's tested range (0.1-1.0 area); provisioned 3.34 (CI [2.80,4.07]) is ABOVE it. Non-overlapping CIs = a real difference. Other three proxies overlap between splits.

## Caveats / threats to validity
- Proxies have different bin counts (query_type ~10, scan 6, joins 4, hour 24) and sparsity (joins concentrated at 0 -> obs_JS 0.08). Absolute alphas are NOT perfectly comparable across proxies; the ORDERING is the robust signal. E8 (vary bins) must test this before any cross-proxy claim.
- Serverless n=39 tenants is small; CIs are wider there.

## Cluster structure
- Provisioned: separation_ratio 0.97 (best k=4), no structure.
- Serverless: sep 0.70 @ k=2 with silhouette 0.635 - BUT the silhouette is on the 12 SCALAR features (KMeans optimizes those) while separation is measured on the query_type distribution, so the scalar-clusters do NOT align with query-type separation. Still no clean query-type clusters. The scalar-space structure is a hint to chase with proper per-proxy clustering (the known proxy-mismatch to fix).
- Implication: Track 2 still leans PERSONALIZED FL over clustered FL; do per-proxy clustering before finalizing.

## Next
- Fix clustering proxy-mismatch: cluster ON each proxy's distribution and report separation per proxy (esp. chase the serverless scalar-space hint).
- E8 robustness: vary bin edges + min-queries; confirm the alpha ordering is stable.
- Reframe the paper around multi-dimensional, fleet-dependent calibration (stronger than a single alpha).
- Optional later: a secondary public trace (Alibaba/Borg) for a generality claim.
