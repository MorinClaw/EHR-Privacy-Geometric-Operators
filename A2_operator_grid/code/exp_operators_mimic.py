#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_operators_mimic.py — 实验 A.2：算子应用与参数网格（MIMIC-IV）

从预处理好的 ts_48h 单列读取数据，对每变量、每 Setting 应用 T1（均匀/加权）、T2、T3，
结果写入本实验 results 目录（perturbed/, tables/, figs/operators/）。

用法（在 A2/code 下）：
  python3 exp_operators_mimic.py
  python3 exp_operators_mimic.py --variables HR SBP --no-skip-existing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 同目录下的 numeric_operators（A2 自包含）
SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from repo_discovery import default_ts48h_dir

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from numeric_operators import (
    NORMALIZED_MAX_DIFF,
    normalize_zscore,
    denormalize_zscore,
    triplet_micro_rotation,
    triplet_micro_rotation_weighted,
    constrained_noise_projection,
    householder_reflection,
)


DEFAULT_VARIABLES = [
    "HR", "SBP", "DBP", "MAP", "TempC", "RespRate", "SpO2",
    "Creatinine", "Lactate", "Glucose",
]


def discover_variables(data_dir: Path) -> list[str]:
    """从 data_dir 的 ts_single_column_*.csv 推断变量名。"""
    seen = set()
    for f in data_dir.glob("ts_single_column_*.csv"):
        name = f.stem.replace("ts_single_column_", "")
        base = name.replace("_zscore", "") if name.endswith("_zscore") else name
        seen.add(base)
    return sorted(seen) if seen else DEFAULT_VARIABLES


def load_single_column(data_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载单列。setting in ('z', 'phys')。"""
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def seed_from_var_setting(var: str, setting: str, global_seed: int = 42) -> int:
    """(var, setting) 固定种子。"""
    h = hashlib.sha256(f"{var}_{setting}_{global_seed}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def run_a2(
    data_dir: Path,
    out_dir: Path,
    *,
    variables: list[str] | None = None,
    global_seed: int = 42,
    max_diff_normalized: float = NORMALIZED_MAX_DIFF,
    n_passes_grid: tuple[int, ...] = (2, 5, 10),
    skip_existing: bool = True,
) -> None:
    """执行 A.2：方案一「先归一化再用固定阈值」。z 空间已是标准化；phys 先 z-score 再算子再反变换。"""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    perturbed_dir = out_dir / "perturbed"
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs" / "operators"
    for d in (perturbed_dir, tables_dir, figs_dir):
        d.mkdir(parents=True, exist_ok=True)

    vars_to_run = variables or discover_variables(data_dir)
    settings = ["z", "phys"]
    t3_trials_records: list[dict] = []
    max_diff = max_diff_normalized  # 归一化空间内统一阈值

    for var in vars_to_run:
        for setting in settings:
            seed = seed_from_var_setting(var, setting, global_seed)
            rng = np.random.default_rng(seed)
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError as e:
                print(f"[跳过] {var} / {setting}: {e}")
                continue
            x = np.asarray(x, dtype=float)
            nan_mask = np.isnan(x)
            x_filled = x.copy()
            if np.any(nan_mask):
                x_filled[nan_mask] = np.nanmean(x[~nan_mask])

            # 方案一：phys 先归一化到 z，算子在归一化空间用固定阈值，再反归一化
            if setting == "phys":
                x_work, mu, sigma = normalize_zscore(x_filled)
            else:
                x_work = x_filled.copy()
                mu, sigma = 0.0, 1.0

            base_dir = perturbed_dir / setting / var
            base_dir.mkdir(parents=True, exist_ok=True)

            def to_output(y_work: np.ndarray) -> np.ndarray:
                if setting == "phys":
                    y = denormalize_zscore(y_work, mu, sigma)
                else:
                    y = np.asarray(y_work, dtype=float)
                if np.any(nan_mask):
                    y = np.asarray(y, dtype=float)
                    y[nan_mask] = np.nan
                return y

            # T1 均匀
            for n_passes in n_passes_grid:
                out_name = f"T1_uniform_n_passes={n_passes}_max_diff={max_diff}.csv"
                out_path = base_dir / out_name
                if skip_existing and out_path.exists():
                    print(f"[已存在] {var}/{setting} T1_uniform n_passes={n_passes} max_diff={max_diff}")
                    continue
                rng_local = np.random.default_rng(seed)
                y_work = triplet_micro_rotation(
                    x_work.copy(), max_diff=max_diff, n_passes=n_passes,
                    x_original=x_work.copy(), rng=rng_local,
                )
                pd.DataFrame({"value": to_output(y_work)}).to_csv(out_path, index=False)
                print(f"[写出] {var}/{setting} T1_uniform n_passes={n_passes} max_diff={max_diff}")

            # T1 加权
            for n_passes in n_passes_grid:
                out_name = f"T1_weighted_n_passes={n_passes}_max_diff={max_diff}.csv"
                out_path = base_dir / out_name
                if skip_existing and out_path.exists():
                    print(f"[已存在] {var}/{setting} T1_weighted n_passes={n_passes} max_diff={max_diff}")
                    continue
                rng_local = np.random.default_rng(seed)
                y_work = triplet_micro_rotation_weighted(
                    x_work.copy(), max_diff=max_diff, n_passes=n_passes,
                    x_original=x_work.copy(), rng=rng_local,
                )
                pd.DataFrame({"value": to_output(y_work)}).to_csv(out_path, index=False)
                print(f"[写出] {var}/{setting} T1_weighted n_passes={n_passes} max_diff={max_diff}")

            # T2
            out_name = f"T2_max_diff={max_diff}.csv"
            out_path = base_dir / out_name
            if skip_existing and out_path.exists():
                print(f"[已存在] {var}/{setting} T2 max_diff={max_diff}")
            else:
                rng_local = np.random.default_rng(seed)
                try:
                    y_work = constrained_noise_projection(x_work.copy(), max_diff=max_diff, rng=rng_local)
                    pd.DataFrame({"value": to_output(y_work)}).to_csv(out_path, index=False)
                    print(f"[写出] {var}/{setting} T2 max_diff={max_diff}")
                except (ValueError, RuntimeError) as e:
                    print(f"[T2 失败] {var}/{setting} max_diff={max_diff}: {e}")

            # T3
            out_name = f"T3_max_diff={max_diff}.csv"
            out_path = base_dir / out_name
            if skip_existing and out_path.exists():
                print(f"[已存在] {var}/{setting} T3 max_diff={max_diff}")
            else:
                rng_local = np.random.default_rng(seed)
                try:
                    y_work, n_trials = householder_reflection(
                        x_work.copy(), max_diff=max_diff, rng=rng_local, return_n_trials=True
                    )
                    t3_trials_records.append({"var": var, "setting": setting, "max_diff": max_diff, "n_trials_used": n_trials, "n": int(np.sum(~nan_mask))})
                    pd.DataFrame({"value": to_output(y_work)}).to_csv(out_path, index=False)
                    print(f"[写出] {var}/{setting} T3 max_diff={max_diff} n_trials={n_trials}")
                except RuntimeError as e:
                    print(f"[T3 失败] {var}/{setting} max_diff={max_diff}: {e}")

    if t3_trials_records:
        pd.DataFrame(t3_trials_records).to_csv(tables_dir / "table_T3_n_trials_used.csv", index=False)
        print(f"[写出] T3 采样记录 -> {tables_dir / 'table_T3_n_trials_used.csv'}")
    print("\nA.2 完成。结果目录:", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A.2 算子应用与参数网格。")
    parser.add_argument("--data-dir", type=str, default=None, help="ts_48h 目录")
    parser.add_argument("--out-dir", type=str, default=None, help="输出根目录（默认 A2/results）")
    parser.add_argument("--max-diff", type=float, default=None, help="归一化空间阈值 α，不设则用 numeric_operators.NORMALIZED_MAX_DIFF")
    parser.add_argument("--variables", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    if args.data_dir is None:
        try:
            args.data_dir = default_ts48h_dir(_REPO_ROOT)
        except FileNotFoundError as e:
            raise SystemExit(f"错误：{e} 请使用 --data-dir。") from e
    else:
        args.data_dir = Path(args.data_dir)
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
    max_diff = float(args.max_diff) if args.max_diff is not None else NORMALIZED_MAX_DIFF
    run_a2(
        args.data_dir,
        args.out_dir,
        variables=args.variables,
        global_seed=args.seed,
        skip_existing=not args.no_skip_existing,
        max_diff_normalized=max_diff,
    )


if __name__ == "__main__":
    main()
