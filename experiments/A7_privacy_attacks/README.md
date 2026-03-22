# A.7 Privacy & Numeric Reconstruction Attacks (PDF §1.2.4)

Evaluates A.2 perturbations via: **perturbation statistics**, **regression reconstruction attacks**, and **multi-run variability**.

## Dependencies

- Python 3, numpy, pandas, scipy, matplotlib  
- **Reconstruction attacks** require: `pip install scikit-learn`

## Usage

**Full run (all steps, all variables, all tables and figures):**

```bash
cd A7_privacy_attacks/code && python3 exp_a7_privacy.py
```

(From the **repository root**: `cd A7_privacy_attacks/code && python3 exp_a7_privacy.py`.)

The reconstruction block uses **two fixed training regimes** — **20% paired** and **0.01% paired** — with tables and comparison plots. Without `--skip-delta`, `results/figs/boxplot_delta_*.pdf` are produced.

**Expected outputs — tables:** `table_delta_stats.csv`, `table_reconstruction_20pct.csv`, `table_reconstruction_001pct.csv`, **`table_no_pairs_attack.csv`** (no-pairs attack), `table_multi_run_KS_rho.csv`.  
**Figures:** `boxplot_delta_{var}.pdf`, `plot_reconstruction_R2_*.pdf`, `plot_reconstruction_R2_compare_z.pdf` / `plot_reconstruction_R2_compare_phys.pdf` (20% vs 0.01% side-by-side), **`plot_no_pairs_attack_z.pdf` / `plot_no_pairs_attack_phys.pdf`** (MAE / corr for no-pairs attack).

```bash
cd code
python3 exp_a7_privacy.py
```

- `--data-dir`: `ts_48h` directory (default: discovered `.../experiment_extracted/ts_48h`)
- `--perturbed-dir`: A.2 `results/perturbed`
- `--out-dir`: this experiment’s `results`
- `--variables`: subset, e.g. `--variables HR SBP Creatinine`
- `--K 5`: multi-run count (default 5)
- `--skip-delta` / `--skip-recon` / `--skip-no-pairs` / `--skip-multi`: skip blocks
- `--figs-only`: regenerate figures from existing tables only
- `--recon-max-n 100000`: max points per column for reconstruction (0 = all)
- `--recon-train-n`: fixed train size (e.g. 100) for ablations vs 20% training
- `--run-pipeline-strong`: run **numeric strong** pipeline (T1→T2→T3) reconstruction → `table_reconstruction_pipeline_strong.csv` and comparison plots
- `--multi-subsample-n 30000`: subsample first N points per column for multi-run (0 = all)

When **n ≥ 10,000**, both **20%** and **0.01%** runs are executed; for smaller cohorts (e.g. cross-section n≈5000), only **20%** is run automatically.

---

## Cross-section data (harder to attack)

To test whether poor operator scores are due to “easy” temporal structure, repeat A.2 + A.7 on **cross-section** data (no temporal autocorrelation).

**1. Build cross-section** (requires `ts_48h/snapshot_24h_multicol.csv`):

```bash
cd data_preparation
python3 experiment_data_extraction/build_cross_section_for_a7.py
```

Output: `experiment_extracted/ts_cross_section/` (same schema as `ts_48h`, n≈5000 per variable).

**2. Run A.2 on cross-section:**

```bash
cd A2_operator_grid/code
python3 exp_operators_mimic.py \
  --data-dir /path/to/data_preparation/experiment_extracted/ts_cross_section \
  --out-dir /path/to/A2_operator_grid/results_cross_section
```

**3. Run A.7 on cross-section:**

```bash
cd A7_privacy_attacks/code
python3 exp_a7_privacy.py \
  --data-dir /path/to/data_preparation/experiment_extracted/ts_cross_section \
  --perturbed-dir /path/to/A2_operator_grid/results_cross_section/perturbed \
  --out-dir /path/to/A7_privacy_attacks/results_cross_section
```

**4. Compare 20% R² (ts_48h vs cross-section):**

```bash
cd A7_privacy_attacks/code
python3 compare_20pct_datasets.py --main-dir ../results --cross-dir ../results_cross_section
```

Outputs: `table_compare_20pct_ts48h_vs_cross_section.csv`, scatter plots of R².

**5. Fixed 100 training samples vs 20%:**

```bash
cd A7_privacy_attacks/code
python3 exp_a7_privacy.py --recon-train-n 100 --out-dir ../results_100 --skip-delta --skip-no-pairs --skip-multi
python3 compare_20pct_datasets.py --main-dir ../results --cross-dir ../results_100 --suffix full_vs_100 --label-left "20pct" --label-right "train100"
```

**6. Strong pipeline (T1→T2→T3) reconstruction**

```bash
cd A7_privacy_attacks/code
python3 exp_a7_privacy.py --run-pipeline-strong --skip-delta --skip-recon --skip-no-pairs --skip-multi
```

Produces `table_reconstruction_pipeline_strong.csv` and `plot_reconstruction_R2_with_pipeline_*.pdf`.

---

## Outputs (summary)

- **Perturbation stats:** `table_delta_stats.csv`, `boxplot_delta_{var}.pdf`
- **Reconstruction:** `table_reconstruction_20pct.csv`, `table_reconstruction_001pct.csv`, R² plots and 20% vs 0.01% comparisons; optional `table_reconstruction_pipeline_strong.csv`
- **No-pairs attack:** `table_no_pairs_attack.csv`, `plot_no_pairs_attack_*.pdf`
- **Multi-run:** `table_multi_run_KS_rho.csv`, `plot_multi_run_KS_rho_*.pdf`

## Operator invertibility (principle)

| Operator | Principle | Few / no pairs | Many pairs |
|----------|-------------|------------------|------------|
| **T1** | Triplet micro-rotation, ℓ∞-bounded, seed-dependent | Hard to invert without seed | Lower R², harder to generalize |
| **T2** | Constrained noise + renorm | Simple structure | Easier to learn, high R² |
| **T3** | Householder reflection, deterministic | Need direction **u** to invert | **Very easy**: near-linear, R²≈1 |

## Why paired attacks vs demo “numeric reconstruction”?

- **Demo (`demo_privacy_attacks_synthetic.py`)**: attacker sees **y only**; naive x̂=y.
- **A.7**: attacker has **some (x,y) pairs** (e.g. 20% / 0.01%) to train f(y)≈x — a **stronger** threat when pairs leak.

Both are complementary: demo measures error under **no pairs**; A.7 measures **learnability** under **pair leakage**.

## Does A.7 leak raw data?

**No.** Only aggregate metrics (R², RMSE, ρ, KS, etc.) and PDF figures are written — no raw (x,y) series in outputs.

---

## Reading results

1. **Delta stats:** check `delta_mean`≈0, spread vs `max_diff`.
2. **Reconstruction:** higher **R²** ⇒ easier to reconstruct from y ⇒ weaker privacy under paired attack.
3. **No-pairs:** large MAE / low corr ⇒ naive x̂=y is poor.
4. **Multi-run:** high KS / low ρ between runs ⇒ more stochastic; T3 tends to be deterministic (high ρ).

## Goals vs operators (summary)

| Goal | T1 | T2 | T3 |
|------|----|----|-----|
| Keep distribution / meaning | Partial | Good | **Best** |
| Resist reconstruction | **Best** among three | Medium | **Poor** (near-invertible) |
| Low compute | ✓ | ✓ | ✓ |
| Explainable / auditable | ✓ | ✓ | ✓ |

**Takeaway:** no single operator maximizes both **fidelity** and **privacy**. Prefer **T1** (or T1+T2) for external release; reserve **T3** for controlled / invertible settings, not public privacy claims.

**Practice:** for de-identified EHR release, **prefer T1** (optionally stronger parameters) and monitor R² in A.7; document that **T3 is not for external privacy** — only for reversible / high-fidelity controlled workflows.
