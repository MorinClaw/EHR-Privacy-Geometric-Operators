# A.3 理论性质的数值验证（PDF 2.3）

对 A.2 的每个扰动结果 y 与对应原始列 x 计算 sanity 指标，验证算子满足理论约束。

## 指标

- **delta_mean** = |mean(x) − mean(y)|（期望 ≈ 1e-10 量级）
- **delta_var**  = |Var(x) − Var(y)|
- **max_abs_delta** = max_i |y_i − x_i|（应 ≤ max_diff）
- **min_abs_delta** = min_i |y_i − x_i|（应 > 0）
- **unchanged_ratio** = (# |y_i − x_i| ≤ 1e-8) / n（理想接近 0）

## 输出

- `results/tables/table_operator_sanity.csv`：列 var, setting, operator, params, delta_mean, delta_var, max_abs_delta, min_abs_delta, unchanged_ratio

## 运行

```bash
cd A3_theory_validation/code
python3 exp_a3_sanity_check.py
# 可选：--data-dir /path/to/ts_48h --perturbed-dir /path/to/A2/results/perturbed --out-dir /path/to/A3/results
```

依赖：A.2 已跑完，且 `A2_operator_grid/results/perturbed/` 存在。
