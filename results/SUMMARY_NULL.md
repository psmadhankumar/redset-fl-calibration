# Null-model cluster test + bias-corrected CIs - 2026-09-01

## Bug fixed: self-pair bootstrap bias (caught by Claude Code)
calibrate_with_ci resampled tenants WITH replacement; duplicate tenants create JS=0 self-pairs
that biased observed-JS down and recovered alpha UP (worse at small n). FIXED in
tenant_heterogeneity_mp.py via _mean_pairwise_js_masked (skips same-original-instance pairs in
the bootstrap). Point estimates unchanged; CIs corrected (shifted down/tighter on the high end).
NOTE: the mp/e8 result files already on disk used the OLD biased CIs. Re-run mp.py + e8.py with the
fixed code to refresh the published tables (shift is small; hour_of_day binning finding unaffected).

## Bias-corrected matched alpha [95% CI]
| proxy       | provisioned          | serverless           |
|-------------|----------------------|----------------------|
| query_type  | 3.32 [2.77, 3.90]    | 1.10 [0.72, 1.60]    |
| scan_volume | 2.54 [2.09, 3.05]    | 2.83 [2.19, 3.82]    |
| joins       | 8.67 [6.43, 10.56]   | 7.56 [4.63, 12.39]   |
| hour_of_day | 11.51 [8.76, 15.76]  | 13.83 [9.51, 21.01]  |
Fleet difference on query_type still non-overlapping (prov 2.77-3.90 vs srv 0.72-1.60). Story unchanged.

## Null-model cluster test (real vs Dirichlet-at-matched-alpha, per proxy, B=200)
Q: do real tenants cluster MORE than independent Dirichlet clients at the same alpha? p = fraction of
null separation >= real separation. p<0.05 => genuine structure beyond Dirichlet.
| proxy       | provisioned                         | serverless                          |
|-------------|-------------------------------------|-------------------------------------|
| query_type  | p=0.000 (real 2.26 vs null95 1.93) SIG | p=0.835 (real 2.49 vs null95 4.12) NOT SIG |
| scan_volume | p=0.000 SIG                          | p=0.015 SIG                          |
| joins       | p=0.025 SIG                          | p=0.010 SIG                          |
| hour_of_day | p=0.000 (real 2.75 vs null95 1.22) STRONG | p=0.000 STRONG                    |

## Verdict
- The earlier "no clusters" conclusion is OVERTURNED (it was a measurement artifact of clustering on
  scalar features). 7 of 8 proxy/fleet combos show statistically-significant cluster structure BEYOND
  independent Dirichlet clients. hour_of_day is strongest (tenants form clear time-of-day groups on both fleets).
- The ONE exception: SERVERLESS query_type shows NO extra structure (p=0.835) - serverless tenants'
  query-type mixes really do look like independent Dirichlet draws (the literature's assumption holds
  there, and only there). A clean honest counterpoint / negative control.
- IMPLICATION for Track 2: clustered FL is JUSTIFIED - there is real, exploitable workload structure
  (esp. provisioned, and time-of-day everywhere). Track 2 = clustered FL, with serverless-query_type
  as the negative control.

## Deliverables
- alpha_forest.png (Figure: matched alpha CIs by proxy x fleet; * = significant cluster structure vs
  Dirichlet null). Paper Figure candidate.
- null_and_fixedci.json (all numbers).

## Next
1. Re-run mp.py + e8.py with the fixed bootstrap to refresh on-disk tables (for the published record).
2. DRAFT Track 1 (calibration + multi-dimensional + fleet + cluster-structure-is-real).
3. Track 2 framed as clustered FL (null test supports it); serverless query_type = negative control.
