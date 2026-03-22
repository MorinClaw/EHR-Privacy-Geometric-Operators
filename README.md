# EHR Privacy Geometric Operators

A lightweight, CPU-only framework for privacy-preserving transformation of electronic health record (EHR) time series, using geometric operators on the mean–variance manifold.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

This framework applies geometric operators to EHR time series data to achieve privacy-preserving transformation while maintaining clinical utility. Three core operators are provided: **T1** (local triplet rotation — random orthogonal rotation over 3-point windows), **T2** (Gaussian noise with manifold re-projection), and **T3** (Householder reflection in mean-zero subspace, included as a negative case due to its known invertibility). All operators are controlled by a single perturbation parameter **α** — a unified ℓ∞ upper bound in z-score space that applies consistently across all clinical variables. The **Q-mix** extension wraps any operator with per-stay orthogonal mixing, collapsing reconstruction R² from ~0.9 to ~0 at fixed α=1.0 — a one-step privacy cliff. The entire pipeline is CPU-only and suitable for nightly batch runs in hospital ETL environments.

---

## Key Features

- **Mean & variance preserved to machine precision** — satisfies constraint C1; downstream statistics are unaffected
- **Unified ℓ∞ bound α in z-score space** — single privacy knob across all variables (C2); no per-variable tuning needed
- **Full variability** — virtually all time points are perturbed (C3); no silent pass-through
- **O(n) per-column complexity, CPU-only, no GPU required** (C4); runs on standard hospital infrastructure
- **Per-stay Q-mixing** — one-step privacy cliff that drives reconstruction R² to ~0 at fixed α=1.0
- **EHR-Privacy-Agent** — rule-based skill system for automated nightly de-identification pipelines
- **Privacy Evaluation Protocol** — structured attack suite covering attacks A (reconstruction) / B (record linkage) / C (membership inference) / D (attribute inference), with L0/L1/L2 leakage levels

---

## Operators at a Glance

| Operator | Mechanism | Recon R² (HR, α=1.0, L2) | Notes |
|---|---|---|---|
| T1 — Triplet Rotation | Local 3-point random orthogonal rotation | ~0.83 | Good utility, moderate privacy |
| T2 — Noise + Projection | Gaussian noise re-projected onto manifold | ~0.97 | Best utility, weaker reconstruction resistance |
| T3 — Householder | Global reflection in mean-zero subspace | ~0.9999 | **Negative case** — highly invertible |
| Q-mix + T1/T2 | Per-stay orthogonal mixing wrapper | **~0.000** | Strong privacy, acceptable utility |

---

## Repository Structure

```
EHR-Privacy-Geometric-Operators/
├── ehr_privacy/                        # Core framework package
│   ├── numeric_operators.py            # T1, T2, T3, Q-mix
│   ├── non_numeric_operators.py        # Text, categorical, ID operators
│   └── agent.py                        # PrivacyAgent + SkillRegistry
│
├── agent_demo/                         # Runnable demos (synthetic + MIMIC-style data)
│
├── privacy_evaluation_protocol/        # Attack suite CLI (A/B/C/D attacks)
│   └── code/
│       └── run_privacy_protocol.py
│
├── experiments/                        # All experiments
│   ├── A2_operator_grid/               # Operator application & parameter grid
│   ├── A3_theory_validation/           # C1–C4 constraint verification
│   ├── A4_single_column_distribution/  # KS / marginal distribution
│   ├── A5_multivariate_correlation/    # Correlation structure
│   ├── A6_temporal_structure/          # ACF / spectral analysis
│   ├── A7_privacy_attacks/             # Reconstruction & privacy attacks
│   ├── B2_privacy_utility/             # Privacy grid search & pipeline planning
│   ├── B3_theory_e2e/                  # End-to-end theory checks
│   ├── B4_privacy_utility_tradeoff/    # Downstream tasks (LOS / mortality / readmission)
│   ├── B5_complexity_compute/          # Runtime & scaling
│   ├── B6_agent_ablation/              # Operator & agent ablation
│   ├── P3_weak_interpolation/          # Weak interpolation pipeline
│   ├── P6_alpha_hierarchy/             # α-hierarchy Pareto plots
│   └── P7_qmix_pilot/                  # Q-mix pilot experiments
│
├── docs/                               # Documentation
├── repo_discovery.py                   # Path resolution (no hardcoded paths)
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/MorinClaw/EHR-Privacy-Geometric-Operators.git
cd EHR-Privacy-Geometric-Operators
pip install -r requirements.txt
```

**Optional dependencies:**

```bash
pip install ctgan xgboost  # CTGAN baseline (B5) and gradient boosting (B4)
```

---

## Data Setup

Experiments expect MIMIC-IV-style preprocessed data. Place your data under an `experiment_extracted/` directory anywhere on disk, then pass the path via `--data-dir`. The expected layout is:

```
data_preparation/
└── experiment_extracted/
    ├── ts_48h/
    │   ├── ts_single_column_HR_zscore.csv
    │   ├── ts_single_column_Glucose_zscore.csv
    │   └── ...
    ├── patient_profile.csv
    ├── timeline_events.csv
    └── cohort_icu_stays.csv
```

No hardcoded paths — `repo_discovery.py` resolves all paths at runtime.

---

## Quick Start

### Try the agent demo (no real data needed)

```bash
cd agent_demo
python demo_numeric_pipeline.py
python demo_privacy_attacks_synthetic.py
```

### Run operator experiments

```bash
cd experiments/A2_operator_grid/code
python exp_operators_mimic.py --variables HR Glucose SBP
```

### Theory validation (C1–C4)

```bash
cd experiments/A3_theory_validation/code
python exp_a3_sanity_check.py
```

### Privacy attack evaluation

```bash
cd experiments/A7_privacy_attacks/code
python exp_a7_privacy.py --variables HR Glucose --K 5
```

### Q-mix pilot

```bash
cd experiments/P7_qmix_pilot/code
python run_qmix_pilot.py --variables HR Glucose --alphas 1.0 --seed 42
```

### Full attack protocol (A/B/C/D)

```bash
cd privacy_evaluation_protocol/code
python run_privacy_protocol.py \
  --data-dir ../../data_preparation/experiment_extracted/ts_48h \
  --perturbed-dir ../../experiments/A2_operator_grid/results/perturbed \
  --attack A B C D \
  --variables HR Glucose \
  --alpha 1.0
```

### End-to-end downstream tasks

```bash
PYTHONPATH=. python experiments/B4_privacy_utility_tradeoff/code/exp_b4_timeline_icu_tasks.py \
  --max-stays 500
```

---

## Privacy Evaluation Protocol

The protocol evaluates transformed data across four attack types and three leakage levels:

| Attack | Type | Description |
|---|---|---|
| A | Reconstruction | Can an adversary recover the original time series from perturbed output? |
| B | Record Linkage | Can perturbed records be re-linked to original patient records? |
| C | Membership Inference | Can an adversary determine if a specific patient was in the dataset? |
| D | Attribute Inference | Can an adversary infer sensitive attributes from perturbed data? |

| Leakage Level | Description |
|---|---|
| L0 | No auxiliary information — black-box setting |
| L1 | Adversary knows the operator type but not parameters |
| L2 | Adversary knows operator type and perturbation bound α |

Run `python privacy_evaluation_protocol/code/run_privacy_protocol.py --help` for the full list of options.

---

## License

MIT
