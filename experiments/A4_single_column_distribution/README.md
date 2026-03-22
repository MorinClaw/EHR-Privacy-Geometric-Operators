# A.4 Single-column distribution comparison (KS / normality / plots)

Compares **raw vs perturbed** single-column distributions: KS tests, normality checks, histograms and Q–Q plots for selected vitals.

## Prerequisites

A.2 has been run; `A2_operator_grid/results/perturbed/` exists.

## Run

```bash
cd A4_single_column_distribution/code
python3 exp_a4_distribution.py
# Optional:
python3 exp_a4_distribution.py --data-dir /path/to/ts_48h --perturbed-dir /path/to/perturbed --out-dir /path/to/A4/results
```

Outputs: `results/tables/table_ks.csv`, figures under `results/figs/`.
