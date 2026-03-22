# P3: Weak interpolation (forward-fill ≤ K, no linear interpolation)

Unifies a **weak interpolation** path: after hourly alignment, only **forward-fill** up to **K** hours; gaps larger than K stay **NaN** (no linear interpolation across gaps).

Outputs wide tables compatible with `ts_48h` (including z-score and single-column extracts) for **A2** and **privacy_evaluation_protocol**.

## Layout

- `code/config_weak.py` — constants and `itemid` maps  
- `code/build_ts48h_weak.py` — build `ts_48h_weak/`  
- `code/run_a2_on_weak.py` — run A2 on weak series → `results_weak/perturbed/`

## Usage

```bash
cd P3_weak_interpolation/code

python build_ts48h_weak.py \
  --data-dir /path/to/experiment_extracted \
  --out-dir ../ts_48h_weak \
  --max-ffill-hours 2

python run_a2_on_weak.py \
  --ts-dir ../ts_48h_weak \
  --out-dir ../../A2_operator_grid/results_weak \
  --variables HR MAP Glucose \
  --max-diff 1.0
```

Point **privacy_evaluation_protocol** at the weak `ts` directory and matching A2 `perturbed/` paths when evaluating weak-regime experiments.

Optional extensions (downsampling, per-variable K) can be added in `build_ts48h_weak.py`.
