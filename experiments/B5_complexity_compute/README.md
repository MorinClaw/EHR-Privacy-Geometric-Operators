# B.5 Complexity & compute (CPU-friendly deployment)

Measures **wall-clock time**, **per-operator time**, **numeric scaling in N**, and optional **CTGAN** baseline to show the pipeline is **lightweight** (no heavy training for the default path).

**Data:** `data_preparation/experiment_extracted/` (`patient_profile.csv`, `timeline_events.csv`, …).

## Metrics

- Per-table / per-`privacy_level` total time and per-skill timing.
- Numeric operators at N ∈ {1e4, 1e5, 5e5, 1e6} (expect ~O(N)).
- Optional CTGAN train+generate times vs pipeline time.

## Outputs

- `table_b5_summary_time_hardware.csv`, `table_b5_pipeline_timings.csv`, `table_b5_numeric_scaling.csv`
- Figures: `b5_numeric_scaling.png`, `b5_pipeline_operator_breakdown.png`, `b5_runtime_summary_pipeline_vs_ctgan.png`, …

## Run

```bash
PYTHONPATH=agent_demo python3 B5_complexity_compute/code/exp_b5_compute_metrics.py
PYTHONPATH=agent_demo python3 B5_complexity_compute/code/exp_b5_compute_metrics.py --data-dir data_preparation/experiment_extracted --out-dir B5_complexity_compute/results
```

Plots: `python3 B5_complexity_compute/code/plot_b5_complexity_figures.py`

**Optional:** `pip install ctgan` for the CTGAN baseline.
