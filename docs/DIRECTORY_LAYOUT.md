# Repository layout (English top-level names)

All top-level directories use **ASCII names** suitable for GitHub clones on any OS.

| Directory | Role |
|-----------|------|
| `agent_demo/` | EHR Privacy Agent demo: `skills_and_agent.py`, numeric/non-numeric operators, small demos |
| `A2_operator_grid/` | Sec. A.2 — apply numeric operators (T1/T2/T3) on preprocessed `ts_48h` |
| `A3_theory_validation/` | Sec. A.3 — sanity metrics on perturbed vs raw columns |
| `A4_single_column_distribution/` | Sec. A.4 — KS / distribution plots |
| `A5_multivariate_correlation/` | Sec. A.5 — correlation structure |
| `A6_temporal_structure/` | Sec. A.6 — ACF/PACF / spectral checks |
| `A7_privacy_attacks/` | Sec. A.7 — reconstruction & privacy attack tables |
| `B2_privacy_utility/` | Sec. B.2 — pipeline planning & privacy grid search |
| `B3_theory_e2e/` | Sec. B.3 — end-to-end theory checks on tabular exports |
| `B4_privacy_utility_tradeoff/` | Sec. B.4 — patient profile & ICU timeline tasks |
| `B5_complexity_compute/` | Sec. B.5 — runtime / scaling |
| `B6_agent_ablation/` | Sec. B.6 — operator & agent ablations |
| `P3_weak_interpolation/` | Weak interpolation utilities + glue to A2 runners |
| `P6_alpha_hierarchy/` | Alpha / tradeoff plotting helpers |
| `p7_qmix_pilot/` | Q-mixing pilot orchestration + B/C/D on outputs |
| `privacy_evaluation_protocol/` | Attack suite A–D entry points |
| `repo_discovery.py` | Resolve `A2_*`, `experiment_extracted`, `agent_demo`, etc. without localized path strings |

## Data layout

Place preprocessed MIMIC-style exports under **any** top-level folder that contains `experiment_extracted/` (this repo uses the placeholder name `data_preparation/` in docs), or pass explicit `--data-dir` / `--input` to each script.

## Markers for `find_repo_root()`

The repository root is detected by the presence of both:

- `privacy_evaluation_protocol/`
- `p7_qmix_pilot/`
