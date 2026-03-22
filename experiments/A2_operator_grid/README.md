# A.2 算子应用与参数网格

- **code/**：`exp_operators_mimic.py`、`numeric_operators.py`（A2 用副本）
- **results/**：运行后生成 `perturbed/`、`tables/`、`figs/operators/`

数据来源：`data_preparation/experiment_extracted/ts_48h`（ts_single_column_*.csv）

运行（在 code 下）：
```bash
python3 exp_operators_mimic.py
# 或指定路径
python3 exp_operators_mimic.py --data-dir /path/to/ts_48h --out-dir /path/to/A2/results
```
