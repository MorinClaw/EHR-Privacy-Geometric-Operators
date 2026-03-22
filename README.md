# Privacy-Preserving EHR Data Transformation via Geometric Operators

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Paper**: *Privacy-Preserving EHR Data Transformation via Geometric Operators: A Human–AI Co-Design Technical Report*

**Authors**: Maolin Wang†, Beining Bao†, SciencePal, Gan Yuan, Hongyu Chen, Bingkun Zhao, Baoshuo Kan, Jiming Xu, Qi Shi, Yinggong Zhao, Yao Wang, Wei-Ying Ma, Jun Yan

**Affiliation**: Hong Kong Institute of AI for Science (HKAI-Sci), City University of Hong Kong  
† Equal Contribution

---

## Abstract

We present a geometric framework for column-wise EHR data transformation that is *usable and visible* inside the hospital, yet *hard to reconstruct* under a no-key, structure-aware threat model. All operators are formalized as constrained motions on a mean–variance manifold in z-score space, controlled by a unified ℓ∞ bound α. We propose three base operators — **T1** (local triplet rotations), **T2** (noise + manifold projection), and **T3** (Householder reflection, negative case) — along with a per-stay **Q-mixing** extension that sharply reduces linear reconstruction R² for high-risk variables. We evaluate these operators under a **Privacy Evaluation Protocol** comprising L0/L1/L2 leakage levels and four attack families (A: reconstruction, B: record linkage, C: membership inference, D: attribute inference). The system is deployed as **EHR-Privacy-Agent**, running nightly CPU-only de-identification over in-hospital caches with configurable privacy profiles.

---

## Framework Overview

### Mean–Variance Manifold

Each column operator decomposes into three stages:
1. **Standardize**: x → z(x) ∈ M(0,1) — project to zero-mean, unit-variance manifold
2. **Transform on manifold**: z → z' = T̃α(z; ω), preserving M(0,1)
3. **De-standardize**: z' → y = μ(x)·1 + σ(x)·z'

This guarantees exact mean/variance preservation (Constraint C1) and enables a unified perturbation scale α across all clinical variables.

### Design Constraints

| Constraint | Requirement |
|---|---|
| **C1** | Exact column-wise mean & variance preservation |
| **C2** | Unified z-score ℓ∞ bound: ‖z' − z‖∞ ≤ α |
| **C3** | Full variability: most time points are genuinely perturbed |
| **C4** | O(n) CPU complexity, streaming-compatible |
| **C5** | No-key threat model (Kerckhoffs-style transparency) |

### Core Operators

| Operator | Description | Reconstruction R² (L2, α=1.0) |
|---|---|---|
| **T1** (Triplet Rotation) | Local 3-point random orthogonal rotation in mean-zero subspace | ~0.83 (HR) |
| **T2** (Noise + Projection) | Gaussian noise + re-projection onto mean-variance manifold | ~0.97 (HR) |
| **T3** (Householder) | Global Householder reflection — negative case, highly invertible | ~0.9999 |
| **Q-mix + T1/T2** | Per-stay orthogonal mixing before T1/T2 | **~0.000** (HR) |

---

## Repository Structure

```
.
├── agent_demo/                     # Core operators, agent, skills, and demo scripts
│   ├── numeric_operators.py        # T1, T2, T3, Q-mix implementations
│   ├── non_numeric_operators.py    # Text, categorical, ID field operators
│   ├── skills_and_agent.py         # PrivacyAgent + SkillRegistry + pipeline rules
│   ├── demo_numeric_pipeline.py
│   ├── demo_patient_and_timeline.py
│   ├── demo_privacy_attacks_synthetic.py
│   └── ...
│
├── privacy_evaluation_protocol/    # Full attack suite A–D (CLI)
│   ├── code/
│   │   ├── run_privacy_protocol.py
│   │   ├── attack_a_reconstruction.py
│   │   ├── attack_b_linkage.py
│   │   ├── attack_c_membership.py
│   │   └── attack_d_attribute.py
│   └── PAPER_Privacy_Evaluation_Protocol.md
│
├── A2_operator_grid/               # §5.2 — Operator application & parameter grid
├── A3_theory_validation/           # §5.3 — Theory property validation (C1–C4)
├── A4_single_column_distribution/  # §5.4 — KS / distribution analysis
├── A5_multivariate_correlation/    # §5.5 — Correlation structure preservation
├── A6_temporal_structure/          # §5.6 — ACF / spectral analysis
├── A7_privacy_attacks/             # §5.7 — Reconstruction & privacy attacks
│
├── B2_privacy_utility/             # §7 — Privacy grid search & pipeline planning
├── B3_theory_e2e/                  # §7 — End-to-end theory constraint checks
├── B4_privacy_utility_tradeoff/    # §7 — Downstream tasks (LOS, mortality, readmission)
├── B5_complexity_compute/          # §7 — Runtime & scaling analysis
├── B6_agent_ablation/              # §7 — Operator & agent ablation
│
├── p7_qmix_pilot/                  # §5.8 — Q-mix pilot (privacy cliff experiment)
├── P3_weak_interpolation/          # Supporting: weak interpolation pipeline
├── P6_alpha_hierarchy/             # Supporting: α-hierarchy Pareto plots
│
├── docs/
│   ├── DIRECTORY_LAYOUT.md
│   ├── SECTION_CODE_MAP.md
│   └── ENGLISH_LOCALIZATION.md
│
├── repo_discovery.py               # Path discovery utilities (no hardcoded paths)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Paper–Code Mapping

| Paper Section | Directory | Key Script(s) |
|---|---|---|
| §4 — Operators (T1, T2, T3, Q-mix) | `agent_demo/` | `numeric_operators.py` |
| §4 — Non-numeric operators | `agent_demo/` | `non_numeric_operators.py` |
| §5.2 — Operator grid & parameter sweep | `A2_operator_grid/` | `exp_operators_mimic.py` |
| §5.3 — Theory validation (C1–C4) | `A3_theory_validation/` | `exp_a3_sanity_check.py` |
| §5.4 — Single-column distribution (KS) | `A4_single_column_distribution/` | `exp_a4_distribution.py` |
| §5.5 — Multivariate correlation | `A5_multivariate_correlation/` | `exp_a5_correlation.py` |
| §5.6 — Temporal structure (ACF/PACF) | `A6_temporal_structure/` | `exp_a6_temporal.py` |
| §5.7 — Privacy attacks (A-family) | `A7_privacy_attacks/` | `exp_a7_privacy.py` |
| §5.8 — Q-mix pilot | `p7_qmix_pilot/` | `run_qmix_pilot.py` |
| §6 — EHR-Privacy-Agent system | `agent_demo/` | `skills_and_agent.py` |
| §7 — Privacy grid search | `B2_privacy_utility/` | `exp_b2_grid_search_privacy.py` |
| §7 — End-to-end theory checks | `B3_theory_e2e/` | `exp_b3_theory_constraints.py` |
| §7 — Privacy-utility tradeoff | `B4_privacy_utility_tradeoff/` | `exp_b4_timeline_icu_tasks.py` |
| §7 — Runtime analysis | `B5_complexity_compute/` | `exp_b5_compute_metrics.py` |
| §7 — Agent ablation | `B6_agent_ablation/` | `exp_b6_ablation.py` |
| Attack suite A–D (full) | `privacy_evaluation_protocol/` | `run_privacy_protocol.py` |

---

## Quick Start

### Installation

```bash
git clone https://github.com/MorinClaw/EHR-Privacy-Geometric-Operators.git
cd EHR-Privacy-Geometric-Operators
pip install -r requirements.txt
```

Optional dependencies:
```bash
pip install ctgan xgboost  # for CTGAN baseline (B5) and gradient boosting (B4)
```

### Data Setup

This repository contains **code only**. Place preprocessed MIMIC-IV-style data under any top-level directory containing `experiment_extracted/`:

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

Alternatively, pass explicit `--data-dir` / `--input` paths to each script. `repo_discovery.py` at the repo root auto-locates paths without hardcoded directories.

### Running Core Experiments

**Operator grid (A2):**
```bash
cd A2_operator_grid/code
python exp_operators_mimic.py --variables HR Glucose SBP
```

**Theory validation (A3):**
```bash
cd A3_theory_validation/code
python exp_a3_sanity_check.py
```

**Privacy attacks (A7):**
```bash
cd A7_privacy_attacks/code
python exp_a7_privacy.py --variables HR Glucose --K 5
```

**Q-mix pilot:**
```bash
cd p7_qmix_pilot/code
python run_qmix_pilot.py \
  --raw-ts-dir data_preparation/experiment_extracted/ts_48h \
  --variables HR Glucose \
  --alphas 1.0 2.0 3.0 \
  --secret-seed 42 \
  --out-root p7_qmix_pilot/results
```

**End-to-end system (B4):**
```bash
PYTHONPATH=agent_demo python B4_privacy_utility_tradeoff/code/exp_b4_timeline_icu_tasks.py --max-stays 500
```

---

## Privacy Evaluation Protocol

Run the full attack suite (A: reconstruction, B: record linkage, C: membership inference, D: attribute inference) via a single CLI entry point:

```bash
cd privacy_evaluation_protocol/code
python run_privacy_protocol.py \
  --data-dir data_preparation/experiment_extracted/ts_48h \
  --perturbed-dir A2_operator_grid/results/perturbed \
  --out-dir privacy_evaluation_protocol/results \
  --attack A B C D \
  --variables HR Glucose \
  --alpha 1.0 \
  --seed 42
```

### Leakage Levels

| Level | Description | Example scenario |
|---|---|---|
| **L0** | Only de-identified outputs {y}, no (x,y) pairs | External researcher |
| **L1** | Very few pairs (~0.01% of stays) | Accidental audit leak |
| **L2** | Many pairs (~20% of stays) | Insider with partial access |

### Attack Families

| Family | Task | Primary metric |
|---|---|---|
| **A** | Pointwise reconstruction: predict x̂ from y | R², MAE_z, Pearson ρ |
| **B** | Record linkage / re-identification | Re-id@1 (random baseline = 1/m) |
| **C** | Membership inference | AUC (random baseline = 0.5) |
| **D** | Attribute inference & IND distinguishing | R²_attr, \|acc − 0.5\| |

See [`privacy_evaluation_protocol/PAPER_Privacy_Evaluation_Protocol.md`](privacy_evaluation_protocol/PAPER_Privacy_Evaluation_Protocol.md) for the full protocol specification.

---

## Key Results (from paper)

| Setting | HR Recon R² | Glucose Recon R² | Downstream AUROC (LOS) |
|---|---|---|---|
| Raw (no perturbation) | — | — | 0.70 |
| T1+T2 @ α=0.5 | ~0.70 | ~0.80 | 0.685 |
| T1+T2 @ α=1.0 | ~0.83 | ~0.96 | 0.680 |
| Q-mix+T1 @ α=1.0 | **~0.000** | **~0.000** | 0.647 |
| T3 (negative case) | ~0.9999 | ~0.9999 | 0.695 |

Q-mix introduces a one-step **privacy cliff**: reconstruction R² collapses from ~0.8–0.98 to ~0 at fixed α=1.0, while downstream LOS prediction AUROC remains within 3 percentage points of baseline.

---

## Citation

```bibtex
@techreport{wang2025ehr,
  title     = {Privacy-Preserving EHR Data Transformation via Geometric Operators:
               A Human--AI Co-Design Technical Report},
  author    = {Wang, Maolin and Bao, Beining and {SciencePal} and Yuan, Gan
               and Chen, Hongyu and Zhao, Bingkun and Kan, Baoshuo
               and Xu, Jiming and Shi, Qi and Zhao, Yinggong
               and Wang, Yao and Ma, Wei-Ying and Yan, Jun},
  institution = {Hong Kong Institute of AI for Science (HKAI-Sci),
                City University of Hong Kong},
  year      = {2025},
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This research was supported by Hong Kong Institute of AI for Science (HKAI-Sci) at City University of Hong Kong. The geometric operator framework was co-designed with SciencePal as a constrained tool-inventor, with all scientific judgment retained by human researchers.
