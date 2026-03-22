## P7: Q-mixing pilot (minimal end-to-end loop)

Evaluates **per-stay orthogonal Q-mixing** on 48h vectors (secret seed) vs **Attack A** (linear/L2 reconstruction toward **raw x**), plus downstream utility and shape metrics.

Start with 1–2 variables (e.g. **HR**, **Glucose**); operators: `T1_uniform`, `T1_weighted`, `T2` only; setting `z`.

### Layout

- `code/build_qmix_ts_dir.py` — build q-mixed `ts_single_column_*_zscore.csv` from wide `ts_48h_*`
- `code/run_qmix_pilot.py` — build qmix → run A2 (z-only) → Attack A → utility → write tables
- `code/run_bcd_for_qmix_results.py` — run attacks B/C/D on saved pilot outputs

### Quick start

```bash
PYTHONUNBUFFERED=1 python -u p7_qmix_pilot/code/run_qmix_pilot.py \
  --raw-ts-dir "data_preparation/experiment_extracted/ts_48h" \
  --weak-ts-dir "P3_weak_interpolation/ts_48h_weak" \
  --cohort-csv "data_preparation/experiment_extracted/cohort_icu_stays.csv" \
  --variables HR Glucose \
  --alphas 1 2 3 \
  --secret-seed 20260318 \
  --out-root "p7_qmix_pilot/results"
```

Artifacts land under `p7_qmix_pilot/results/...` with per-`alpha` / variable / regime folders and tradeoff plots.
