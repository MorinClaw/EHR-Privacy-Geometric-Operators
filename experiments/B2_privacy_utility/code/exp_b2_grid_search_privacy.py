#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.2 隐私强度网格搜索（numeric 算子组合）

在确定 DEFAULT_PIPELINE_RULES 的 light/medium/strong 前，先对 numeric 的各类算子组合
做重建攻击（线性回归 R²），按实际隐私强度（R² 越低越难被反推）排序，再据此建议
或修正规划表。经验上 T1+T2 的隐私强度可能优于 T1+T2+T3（T3 Householder 易导致 y≈x）。

用法（实验根目录 repository root 下）：
  python3 B2_privacy_utility/code/exp_b2_grid_search_privacy.py --out-dir B2_privacy_utility/results
  python3 B2_privacy_utility/code/exp_b2_grid_search_privacy.py --data-dir data_preparation/experiment_extracted/ts_48h --out-dir B2_privacy_utility/results

依赖：A2 numeric_operators、与 A7 相同的数据目录（ts_48h 或 ts_cross_section）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
B2_DIR = SCRIPT_DIR.parent
ROOT = B2_DIR.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import default_ts48h_dir, find_a2_operator_code

A2_CODE = find_a2_operator_code(ROOT)
if str(A2_CODE) not in sys.path:
    sys.path.insert(0, str(A2_CODE))

TRAIN_FRAC = 0.2
DEFAULT_MAX_N = 100_000

# 参与网格搜索的流水线：(pipeline_id, 显示名, 算子序列 ["T1","T2","T3"])
GRID_PIPELINES: List[Tuple[str, str, List[str]]] = [
    ("T1", "T1", ["T1"]),
    ("T2", "T2", ["T2"]),
    ("T3", "T3", ["T3"]),
    ("T1_T2", "T1->T2", ["T1", "T2"]),
    ("T1_T3", "T1->T3", ["T1", "T3"]),
    ("T2_T3", "T2->T3", ["T2", "T3"]),
    ("T1_T2_T3", "T1->T2->T3", ["T1", "T2", "T3"]),
]


def _ensure_numeric_operators():
    from numeric_operators import (
        normalize_zscore,
        denormalize_zscore,
        triplet_micro_rotation,
        constrained_noise_projection,
        householder_reflection,
        NORMALIZED_MAX_DIFF,
    )
    return normalize_zscore, denormalize_zscore, triplet_micro_rotation, constrained_noise_projection, householder_reflection, NORMALIZED_MAX_DIFF


def apply_numeric_pipeline(
    x: np.ndarray,
    ops: List[str],
    seed: int = 42,
) -> np.ndarray:
    """
    对一列 x 依次施加 ops（T1/T2/T3）。方案一：先归一化再固定阈值 NORMALIZED_MAX_DIFF=0.8。
    NaN 用列均值填充，输出中恢复原 NaN 位置。
    """
    norm, denorm, t1, t2, t3, max_diff = _ensure_numeric_operators()
    x = np.asarray(x, dtype=float).ravel()
    nan_mask = np.isnan(x)
    mean_val = np.nanmean(x)
    x_filled = x.copy()
    x_filled[nan_mask] = mean_val
    z_work, mu, sigma = norm(x_filled)
    rng = np.random.default_rng(seed)

    def fill_nan(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y[nan_mask] = np.nan
        return y

    def fill_for_next(y: np.ndarray) -> np.ndarray:
        out = y.copy()
        out[nan_mask] = np.nanmean(y[~nan_mask]) if np.any(~nan_mask) else 0.0
        return out

    current = z_work.copy()
    x_orig_z = z_work.copy()
    for op in ops:
        if op == "T1":
            current = t1(
                current.copy(),
                max_diff=max_diff,
                n_passes=5,
                x_original=x_orig_z.copy(),
                rng=rng,
            )
        elif op == "T2":
            current = t2(current.copy(), max_diff=max_diff, rng=rng)
        elif op == "T3":
            current, _ = t3(
                current.copy(),
                max_diff=max_diff,
                rng=rng,
                return_n_trials=True,
            )
        else:
            raise ValueError(f"未知算子: {op}")
        current = fill_for_next(current)
    return fill_nan(denorm(current, mu, sigma))


def load_single_column(data_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载原始单列（与 A7 一致）。"""
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _linear_regression_fit_predict(
    Y_train: np.ndarray,
    X_train: np.ndarray,
    Y_test: np.ndarray,
) -> np.ndarray:
    """用 Y 预测 X（与 A7 一致）。"""
    n_tr = Y_train.shape[0]
    ones = np.ones((n_tr, 1), dtype=float)
    Y_aug = np.hstack([Y_train.reshape(-1, 1), ones])
    beta, _, _, _ = np.linalg.lstsq(Y_aug, X_train.ravel(), rcond=None)
    n_te = Y_test.shape[0]
    Y_te_aug = np.hstack([Y_test.reshape(-1, 1), np.ones((n_te, 1), dtype=float)])
    return (Y_te_aug @ beta).reshape(-1, 1)


def run_grid_search(
    data_dir: Path,
    out_dir: Path,
    variables: List[str] | None = None,
    seed: int = 42,
    train_frac: float = TRAIN_FRAC,
    max_n: int | None = DEFAULT_MAX_N,
) -> None:
    """
    对 GRID_PIPELINES 中每种组合：在 data_dir 的每个变量、z/phys 上施加流水线，
    做 20% 训练线性回归重建攻击，记录 R²。输出明细表、汇总表与按隐私强度排序的建议。
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []

    # 变量列表：从 data_dir 下 ts_single_column_*.csv 解析（与 A7 数据布局一致）
    data_dir_p = Path(data_dir)
    all_vars = set()
    for f in data_dir_p.glob("ts_single_column_*.csv"):
        stem = f.stem.replace("ts_single_column_", "")
        if stem.endswith("_zscore"):
            all_vars.add(stem.replace("_zscore", ""))
        else:
            all_vars.add(stem)
    vars_here = sorted(all_vars)
    if variables:
        vars_here = [v for v in vars_here if v in variables]
    if not vars_here:
        print("[B2.grid] 未找到变量（需 ts_single_column_*.csv 或 ts_single_column_*_zscore.csv）。")
        return

    for setting in ["z", "phys"]:
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            mask = ~np.isnan(x)
            if mask.sum() < 500:
                continue
            for pipeline_id, params_str, op_list in GRID_PIPELINES:
                try:
                    y = apply_numeric_pipeline(x, op_list, seed=seed)
                except Exception as e:
                    print(f"[B2.grid] {var} / {setting} / {pipeline_id} 应用失败: {e}")
                    continue
                valid = mask & ~np.isnan(y)
                if valid.sum() < 500:
                    continue
                X = x[valid].reshape(-1, 1)
                Y = y[valid].reshape(-1, 1)
                n = X.shape[0]
                if max_n and n > max_n:
                    idx_perm = rng.permutation(n)[:max_n]
                    X, Y = X[idx_perm], Y[idx_perm]
                    n = X.shape[0]
                idx = np.arange(n)
                n_tr = max(2, min(int(n * train_frac), n - 2))
                perm = rng.permutation(n)
                i_train, i_test = perm[:n_tr], perm[n_tr:]
                Y_train, Y_test = Y[i_train], Y[i_test]
                X_train, X_test = X[i_train], X[i_test]
                X_pred = _linear_regression_fit_predict(Y_train, X_train, Y_test)
                var_test = float(np.var(X_test))
                r2 = float(
                    1 - np.sum((X_test - X_pred) ** 2) / (np.sum((X_test - X_test.mean()) ** 2) + 1e-12)
                ) if var_test > 1e-12 else np.nan
                rows.append({
                    "var": var,
                    "setting": setting,
                    "pipeline_id": pipeline_id,
                    "pipeline_label": params_str,
                    "R2": r2,
                })
                print(f"[B2.grid] {var} / {setting} / {pipeline_id} R2={r2:.4f}")

    if not rows:
        print("[B2.grid] 无有效数据，请检查 --data-dir 是否包含 ts_48h 或 ts_single_column_* 数据。")
        return

    df = pd.DataFrame(rows)
    detail_path = tables_dir / "table_b2_grid_search_privacy.csv"
    df.to_csv(detail_path, index=False)
    print(f"[B2.grid] 明细表: {detail_path}")

    # 汇总：按 pipeline_id 对 R2 求均值（R² 越低隐私越强）
    summary = (
        df.groupby("pipeline_id", as_index=False)
        .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"), n=("R2", "count"))
    )
    summary = summary.sort_values("R2_mean", ascending=True).reset_index(drop=True)
    summary["privacy_rank"] = summary.index + 1  # 1 = 最强（R² 最低）
    summary_path = tables_dir / "table_b2_grid_search_privacy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[B2.grid] 汇总表: {summary_path}")

    # 建议：按 R2 升序对应到 strong / medium / light（前三给 strong/medium/light 或按分位）
    n_pl = len(summary)
    recommendation: List[dict] = []
    for _, row in summary.iterrows():
        pl_id = row["pipeline_id"]
        r2_mean = row["R2_mean"]
        if n_pl >= 3:
            if row["privacy_rank"] == 1:
                suggested = "strong"
            elif row["privacy_rank"] == 2:
                suggested = "medium"
            elif row["privacy_rank"] == 3:
                suggested = "light"
            else:
                suggested = "—"
        else:
            suggested = "strong" if row["privacy_rank"] == 1 else "medium"
        recommendation.append({
            "pipeline_id": pl_id,
            "R2_mean": r2_mean,
            "privacy_rank": row["privacy_rank"],
            "suggested_level": suggested,
        })
    rec_path = tables_dir / "table_b2_grid_search_recommendation.csv"
    pd.DataFrame(recommendation).to_csv(rec_path, index=False)
    print(f"[B2.grid] 建议表: {rec_path}")

    # 文本说明
    txt_path = out_dir / "grid_search_privacy_usage.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("B.2 隐私强度网格搜索说明\n")
        f.write("========================\n\n")
        f.write("R² 越低表示线性重建攻击越难，隐私强度越高。\n\n")
        f.write("汇总排序（按 R2_mean 升序，即隐私从强到弱）：\n")
        for _, r in summary.iterrows():
            f.write(f"  {r['privacy_rank']}. {r['pipeline_id']}  R2_mean={r['R2_mean']:.4f}\n")
        f.write("\n若 T1_T2 排在 T1_T2_T3 前面，说明「T1+T2」隐私优于「T1+T2+T3」，\n")
        f.write("可据此将 DEFAULT_PIPELINE_RULES 的 numeric strong 设为 [num_triplet, num_noise_proj]。\n")
    print(f"[B2.grid] 说明: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B.2 numeric 算子组合隐私强度网格搜索")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Raw ts_48h directory (default: discovered .../experiment_extracted/ts_48h)",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录，默认 B2/results")
    parser.add_argument("--variables", type=str, nargs="*", default=None, help="仅跑指定变量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N, help="每列最多采样点数，0=不限制")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else B2_DIR / "results"
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        try:
            data_dir = default_ts48h_dir(ROOT)
        except FileNotFoundError as e:
            print(f"[B2.grid] {e} Specify --data-dir.")
            return
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not data_dir.exists():
        print(f"[B2.grid] 数据目录不存在: {data_dir}，请指定 --data-dir。")
        return

    run_grid_search(
        data_dir=data_dir,
        out_dir=out_dir,
        variables=args.variables,
        seed=args.seed,
        train_frac=args.train_frac,
        max_n=args.max_n if args.max_n > 0 else None,
    )
    print("\nB.2 隐私强度网格搜索完成。建议先根据汇总表与建议表再确定 DEFAULT_PIPELINE_RULES。")


if __name__ == "__main__":
    main()
