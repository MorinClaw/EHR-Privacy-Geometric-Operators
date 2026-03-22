#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a3_sanity_check.py — 实验 A.3：理论性质的数值验证（PDF 2.3）

读取 A.2 的原始列 x 与扰动结果 y，对每个 (var, setting, operator, params) 计算：
  - delta_mean = |mean(x) - mean(y)|
  - delta_var  = |Var(x) - Var(y)|
  - max_abs_delta = max_i |y_i - x_i|
  - min_abs_delta = min_i |y_i - x_i|
  - unchanged_ratio = (# |y_i - x_i| ≤ 1e-8) / n

输出表：table_operator_sanity.csv（列：var, setting, operator, params, delta_mean, delta_var,
       max_abs_delta, min_abs_delta, unchanged_ratio）。

用法（在 A3/code 下）：
  python3 exp_a3_sanity_check.py
  python3 exp_a3_sanity_check.py --data-dir /path/to/ts_48h --perturbed-dir /path/to/A2/results/perturbed --out-dir /path/to/A3/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from repo_discovery import default_a2_perturbed_dir, default_ts48h_dir

EPS = 1e-8  # 与 A.2 一致，用于判定「未变化」


def load_single_column(data_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载原始单列。setting in ('z', 'phys')。"""
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def parse_perturbed_filename(stem: str) -> tuple[str, str]:
    """从文件名 stem 解析 operator 与 params。例如 T1_uniform_n_passes=2_max_diff=0.5 -> ('T1_uniform', 'n_passes=2,max_diff=0.5')。"""
    if stem.startswith("T1_uniform_"):
        op = "T1_uniform"
        rest = stem.replace("T1_uniform_", "")
    elif stem.startswith("T1_weighted_"):
        op = "T1_weighted"
        rest = stem.replace("T1_weighted_", "")
    elif stem.startswith("T2_"):
        op = "T2"
        rest = stem.replace("T2_", "")
    elif stem.startswith("T3_"):
        op = "T3"
        rest = stem.replace("T3_", "")
    else:
        op = stem.split("_")[0] if "_" in stem else stem
        rest = stem
    # rest 形如 n_passes=2_max_diff=0.5 或 max_diff=0.5；仅在 param 边界 "_key=" 处改为 ",key="
    params = rest.replace("_max_diff=", ",max_diff=").replace("_n_passes=", ",n_passes=")
    return op, params


def sanity_one(x: np.ndarray, y: np.ndarray) -> dict:
    """对一对 (x, y) 计算 A.3 的五个指标。仅在 x、y 均非 NaN 的位置上计算。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() == 0:
        return {
            "delta_mean": np.nan,
            "delta_var": np.nan,
            "max_abs_delta": np.nan,
            "min_abs_delta": np.nan,
            "unchanged_ratio": np.nan,
        }
    xv = np.asarray(x[mask], dtype=float)
    yv = np.asarray(y[mask], dtype=float)
    n = len(xv)
    delta = yv - xv
    abs_delta = np.abs(delta)

    mean_x = float(np.mean(xv))
    mean_y = float(np.mean(yv))
    var_x = float(np.var(xv, ddof=0))
    var_y = float(np.var(yv, ddof=0))

    unchanged = np.sum(abs_delta <= EPS)
    return {
        "delta_mean": float(abs(mean_x - mean_y)),
        "delta_var": float(abs(var_x - var_y)),
        "max_abs_delta": float(np.max(abs_delta)),
        "min_abs_delta": float(np.min(abs_delta)),
        "unchanged_ratio": unchanged / n,
    }


def run_a3(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    *,
    variables: list[str] | None = None,
) -> None:
    """扫描 A.2 的 perturbed 目录，对每个扰动文件与对应原始列做 sanity，写出 table_operator_sanity.csv。"""
    data_dir = Path(data_dir)
    perturbed_dir = Path(perturbed_dir)
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    settings = ["z", "phys"]

    for setting in settings:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables is not None:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError as e:
                print(f"[跳过] 原始列 {var} / {setting}: {e}")
                continue
            var_path = setting_path / var
            for csv_path in sorted(var_path.glob("*.csv")):
                stem = csv_path.stem
                operator, params = parse_perturbed_filename(stem)
                df_y = pd.read_csv(csv_path)
                col = "value" if "value" in df_y.columns else df_y.columns[0]
                y = pd.to_numeric(df_y[col], errors="coerce").to_numpy(dtype=float)
                if len(y) != len(x):
                    print(f"[警告] 长度不一致 {var}/{setting}/{stem}: x={len(x)} y={len(y)}，跳过")
                    continue
                metrics = sanity_one(x, y)
                rows.append({
                    "var": var,
                    "setting": setting,
                    "operator": operator,
                    "params": params,
                    "delta_mean": metrics["delta_mean"],
                    "delta_var": metrics["delta_var"],
                    "max_abs_delta": metrics["max_abs_delta"],
                    "min_abs_delta": metrics["min_abs_delta"],
                    "unchanged_ratio": metrics["unchanged_ratio"],
                })
                print(f"[OK] {var} / {setting} / {operator} {params}")

    out_table = tables_dir / "table_operator_sanity.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)
    print(f"\nA.3 完成。输出表: {out_table}")
    print(f"  共 {len(rows)} 行。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A.3 理论性质数值验证：sanity check 表。")
    parser.add_argument("--data-dir", type=str, default=None, help="ts_48h 目录（原始单列）")
    parser.add_argument("--perturbed-dir", type=str, default=None, help="A.2 的 results/perturbed 目录")
    parser.add_argument("--out-dir", type=str, default=None, help="A.3 的 results 目录")
    parser.add_argument("--variables", type=str, nargs="*", default=None)
    args = parser.parse_args()

    root = _REPO_ROOT
    if args.data_dir is None:
        try:
            args.data_dir = default_ts48h_dir(root)
        except FileNotFoundError as e:
            raise SystemExit(f"错误：{e} 请指定 --data-dir。") from e
    else:
        args.data_dir = Path(args.data_dir)
    if args.perturbed_dir is None:
        args.perturbed_dir = default_a2_perturbed_dir(root)
    else:
        args.perturbed_dir = Path(args.perturbed_dir)
    if args.out_dir is None:
        args.out_dir = SCRIPT_DIR.parent / "results"
    else:
        args.out_dir = Path(args.out_dir)
    return args


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"错误：数据目录不存在: {args.data_dir}")
        sys.exit(1)
    if not args.perturbed_dir.exists():
        print(f"错误：A.2 扰动目录不存在: {args.perturbed_dir}")
        sys.exit(1)
    run_a3(
        args.data_dir,
        args.perturbed_dir,
        args.out_dir,
        variables=args.variables,
    )


if __name__ == "__main__":
    main()
