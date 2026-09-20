# Redset FL Calibration

Realistic, never-pooled client partitions for federated-learning experiments,
calibrated against the Amazon Redset trace, plus the code that produced them.

This is the companion repository for the paper *How Non-IID Is Real Data? A
Multi-Dimensional Calibration of Federated Learning's Heterogeneity Assumptions
on a Production Fleet* (under review). Link and citation will be added once the
paper is public.

## What this is, in one paragraph

Federated-learning papers almost never have naturally separated clients, so they
manufacture them by splitting one pooled dataset with a Dirichlet distribution
at some concentration alpha (usually 0.1, 0.5, or 1.0). Nobody had checked which
of those settings, if any, matches real heterogeneity. We use Redset, a trace of
400 Amazon Redshift instances whose workloads were never pooled, treat each
instance as a client, measure how different the clients really are along four
workload dimensions, and recover the Dirichlet alpha that reproduces that
difference. This repository ships those recovered partitions so you can evaluate
a method against production heterogeneity instead of a Dirichlet guess, and the
code so you can reproduce or extend the calibration.

## Layout

```
partitions/        the released client partitions (per-tenant histograms)
  provisioned/     143 provisioned instances
  serverless/       39 serverless instances
code/              the analysis scripts
results/           computed numbers and figures, so you can check without rerunning
```

## Using the partitions

Each fleet folder holds one file per workload dimension plus a signatures file.
Every row is one instance (one federated client), keyed by `instance_id`.

The four `dist_*.csv` files are per-client histograms; each row sums to 1:

- `dist_query_type.csv` - share of each SQL statement type
  (`analyze, copy, ctas, delete, insert, other, select, unload, update, vacuum`)
- `dist_scan_volume.csv` - share of queries by bytes scanned, log-bucketed
  (`b0_zero, b1_le1, b2_le10, b3_le100, b4_le1k, b5_gt1k`, in MB)
- `dist_joins.csv` - share of queries by join count (`j0, j1, j2, j3plus`)
- `dist_hour_of_day.csv` - share of queries by arrival hour (`h00 ... h23`)

`instance_signatures.csv` carries per-client scalars we used as descriptive
context (query count, abort rate, mean joins, burstiness, and so on). These are
not calibration proxies; they are there if you want to characterize a client.

To use a dimension as a set of federated clients, load the matching `dist_*.csv`
and treat each row as one client's label distribution, in place of a Dirichlet
draw. The recovered alpha for each dimension and fleet is in
`results/provisioned_results.json` and `results/serverless_results.json`, so you
can report where your assumed alpha sits relative to the real one.

## Reproducing from Redset

The raw Redset traces are large (about 17 GB provisioned, 320 MB serverless) and
are distributed by Amazon, so they are not included here. Download them first:

```
aws s3 cp --no-sign-request s3://redshift-downloads/redset/provisioned/full.parquet provisioned_full.parquet
aws s3 cp --no-sign-request s3://redshift-downloads/redset/serverless/full.parquet  serverless_full.parquet
```

Then run the calibration (this rebuilds the partitions and the recovered alphas):

```
pip install -r requirements.txt
python3 code/tenant_heterogeneity_mp.py --input provisioned_full.parquet --out results/provisioned
python3 code/tenant_heterogeneity_mp.py --input serverless_full.parquet  --out results/serverless
```

`tenant_heterogeneity_e8.py` runs the bin-robustness and clustering checks,
`validate_pipeline.py` runs the estimator recovery test on synthetic data, and
`make_figures.py` regenerates the calibration curve and the forest plot from the
files already in `results/`:

```
python3 code/make_figures.py
```

## Data and licensing

The code (everything under `code/`) is released under the MIT license; see
`LICENSE`.

The partition files under `partitions/` are derived from Amazon Redset, which is
released under CC BY-NC 4.0. They inherit that license: you may use and share
them for non-commercial purposes with attribution. Redset itself is described in
van Renen et al., "Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet," VLDB 2024, and lives at https://github.com/amazon-science/redset. Note
that Redset is a busy-ness-weighted sample and an instance does not necessarily
map to a single customer, so treat the partitions as a realistic but partial
picture, not a census of the fleet.

## Citation

Citation will be finalized when the paper is public. For now:

```
@misc{subramanian2026noniid,
  title  = {How Non-IID Is Real Data? A Multi-Dimensional Calibration of
            Federated Learning's Heterogeneity Assumptions on a Production Fleet},
  author = {Madhankumar Ponmudi Subramanian},
  year   = {2026},
  note   = {Under review}
}
```
