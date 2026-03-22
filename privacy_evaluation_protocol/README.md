# Privacy Evaluation Protocol

Systematic **privacy evaluation** and **attack suite** for numeric operators T1/T2/T3, aligned with game-based / leakage-based views (paper section *Privacy Evaluation and Attack Suite*).

## Goals

| Layer | Focus |
|-------|--------|
| **Sanity** | Means/variances, ℓ∞ vs α, KS, correlation, ACF/Ljung–Box |
| **Average-case hardness** | Reconstruction, linkage, attribute inference under stated leakage |
| **Games** | IND-style distinguishing tasks (optional extensions) |

## Threat model & leakage levels

**Kerckhoffs:** attacker knows operators, α, code; no secret per-record key.

| Level | Meaning |
|-------|---------|
| **L0** | Many y=T(x), no (x,y) pairs |
| **L1** | Very few pairs (~0.01%) |
| **L2** | Many pairs (~20%) |
| **L3** | Almost all pairs |

## Attacks A–D (summary)

- **A:** Reconstruct x̂ from y (linear / MLP, …) — R², MAE_z, correlation.
- **B:** Record linkage among candidates.
- **C:** Membership inference.
- **D:** Attribute inference & pairwise distinguishing.

## Running

**Prerequisites:** A2 `perturbed/` outputs and `ts_48h` matrices. **`--alpha` must match** the `max_diff` used when generating perturbations.

```bash
cd privacy_evaluation_protocol/code
python run_privacy_protocol.py \
  --data-dir /path/to/experiment_extracted/ts_48h \
  --perturbed-dir /path/to/A2_operator_grid/results/perturbed \
  --out-dir /path/to/privacy_evaluation_protocol/results \
  --attack A B C D \
  --variables HR Glucose \
  --alpha 1.0 \
  --seed 42
```

Subset attacks / leakage:

```bash
python run_privacy_protocol.py --attack A --leakage L0 L1 L2 --alpha 1.0
```

See `code/run_privacy_protocol.py --help` for full options.
