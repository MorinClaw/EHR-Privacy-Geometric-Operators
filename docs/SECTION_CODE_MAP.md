# Paper Section to Code Mapping (English)

Maps the paper outline (Sections 0–11) to main scripts in this repository. Only basenames and structural markers are referenced.

## 0–4

No dedicated code.

## 5. Operators

- `numeric_operators.py`, `non_numeric_operators.py`, `operator_validation.py`
- `build_qmix_ts_dir.py`

## 6. Single-column and related

- `exp_operators_mimic.py`, `exp_a4_distribution.py`, `exp_a3_sanity_check.py`
- `exp_a5_correlation.py`, `exp_a6_temporal.py`
- summary/plot helpers: `make_a3_summary_and_plots.py`, `make_a4_summary_and_plots.py`

## 7. Agent and attacks

- `skills_and_agent.py`
- `run_privacy_protocol.py`, `attack_*.py` in `privacy_evaluation_protocol/code/`

## 8. End-to-end

- `run_qmix_pilot.py`, `run_bcd_for_qmix_results.py`
- `exp_b2_grid_search_privacy.py`, `exp_b2_plan_pipeline.py`
- `exp_b3_theory_constraints.py`, `plot_b3_figures.py`
- `exp_b4_timeline_icu_tasks.py`, `exp_b4_patient_profile_tasks.py`, plotting scripts under B4
- `exp_b5_compute_metrics.py`, `plot_b5_complexity_figures.py`
- `exp_b6_ablation.py`
- P3 glue: `run_a2_on_ts_zonly_np5.py`, `run_a2_on_weak.py`, `run_a2_on_weak_zonly_np5.py`

## 9–11

No new entry points.

## Path discovery

Use `repo_discovery` at repo root: `find_a2_operator_code`, `default_ts48h_dir`, `find_agent_demo_dir`, `find_make_tradeoff_plots`, etc.
