# A.5 Multivariate correlation structure

Builds multivariate matrices from 24h snapshots and compares **correlation matrices** before vs after perturbation (operator-wise Frobenius / ∞ norms). Optional scatter plots for variable pairs.

## Prerequisites

A.2 perturbations and `ts_48h` / snapshot inputs as described in `exp_a5_correlation.py`.

## Run

```bash
cd A5_multivariate_correlation/code
python3 exp_a5_correlation.py
```

Outputs under `results/tables/` and optional `results/figs/`.
