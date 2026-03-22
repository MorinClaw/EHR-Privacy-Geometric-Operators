# B.4 Privacy vs Utility (paper §B.2 — empirical core)

Implements **B.4 / B2** with **patient_profile** tasks and **timeline** tasks.

**Data:** by default `data_preparation/experiment_extracted/` (`patient_profile.csv`, `timeline_events.csv`, `cohort_icu_stays.csv`, …).

---

## B2.1 `patient_profile`: risk + fairness

- Loads `patient_profile.csv`; synthetic outcomes `y_1y`, `y_30d` via `add_synthetic_outcomes`.
- De-identification: `deidentify_patient_profile_df` (`plan_pipeline("patient_profile", privacy_level)`).
- **Task 1:** logistic regression on raw vs de-id features — **AUC, Brier**, calibration curves.
- **Task 2:** stratified 30d event rates by **ethnicity / insurance** with Wilson CIs — bar + forest plots.

### Run

```bash
PYTHONPATH=agent_demo python3 B4_privacy_utility_tradeoff/code/exp_b4_patient_profile_tasks.py
PYTHONPATH=agent_demo python3 B4_privacy_utility_tradeoff/code/exp_b4_patient_profile_tasks.py \
  --data-dir data_preparation/experiment_extracted --privacy-level strong --out-dir B4_privacy_utility_tradeoff/results
```

### Outputs

- `results/tables/table_b4_task1_auc_brier.csv`
- `results/tables/table_b4_task2_stratified_*.csv`
- `results/figs/task1_calibration_raw_vs_deid.png`, `task2_stratified_rates.png`, `task2_forest_plot.png`

---

## B2.2 `timeline_events`: ICU task + privacy metrics

- **Task:** 24h summary features → synthetic `y_icu`; same feature extraction on **raw** vs **de-id** timeline (numeric column uses T1+T2+T3).
- **Models:** logistic regression or XGBoost (`--xgb`).
- **Privacy metrics:** numeric reconstruction, quasi-ID uniqueness, membership-style separation.

### Run

```bash
python3 B4_privacy_utility_tradeoff/code/exp_b4_timeline_icu_tasks.py
python3 B4_privacy_utility_tradeoff/code/exp_b4_timeline_icu_tasks.py --max-stays 500 --max-rows 100000 --out-dir B4_privacy_utility_tradeoff/results
```

### Plots

```bash
python3 B4_privacy_utility_tradeoff/code/plot_b4_b22_figures.py
```

---

## Dependencies

- `PYTHONPATH` must include `agent_demo`.
- pandas, numpy, scikit-learn; optional **matplotlib**, **xgboost**.
