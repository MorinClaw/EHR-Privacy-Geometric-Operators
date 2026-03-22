#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a5_correlation.py — 实验 A.5：多列相关结构比较（PDF 1.2.2）

1) 构造多变量矩阵 X：取 24h 截面（snapshot_24h_multicol），N×d（N≈5000, d≈10）。
2) 对每种算子/参数，从 A.2 扰动单列中提取 24h 切片，逐列拼成 Y(O)。
3) 计算 R = corr(X)，R(O) = corr(Y(O))；
   指标：||R - R(O)||_∞，||R - R(O)||_F。
4) 可选：对几对典型变量画散点图 before/after。

输出：
  - results/tables/table_correlation_norms.csv（setting, operator, params, norm_inf, norm_fro）
  - results/figs/scatter_*_before_after.pdf（可选）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from repo_discovery import default_a2_perturbed_dir, default_ts48h_dir

HOUR_24 = 24
WINDOW_HOURS = 48

# 作散点图用的变量对（before/after）
SCATTER_PAIRS = [("HR", "SBP"), ("SBP", "DBP")]
# 作散图时选的算子（方案一：归一化空间 max_diff；可由 --max-diff 覆盖）
SCATTER_OPERATORS = [
    ("T1_uniform", "n_passes=5,max_diff=0.8"),
    ("T2", "max_diff=0.8"),
    ("T3", "max_diff=0.8"),
]


def set_scatter_operators_max_diff(alpha: float) -> None:
    """用给定 α 更新 SCATTER_OPERATORS（供阿尔法扫掠时调用）。"""
    global SCATTER_OPERATORS
    SCATTER_OPERATORS = [
        ("T1_uniform", f"n_passes=5,max_diff={alpha}"),
        ("T2", f"max_diff={alpha}"),
        ("T3", f"max_diff={alpha}"),
    ]


def load_snapshot(ts_dir: Path, setting: str) -> pd.DataFrame:
    """加载 24h 截面：z 用 _zscore，phys 不用。"""
    if setting == "z":
        path = ts_dir / "snapshot_24h_multicol_zscore.csv"
    else:
        path = ts_dir / "snapshot_24h_multicol.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def extract_hour24_from_single_column(full_column: np.ndarray, n_stays: int) -> np.ndarray:
    """从「按 stay 展开」的单列中取出第 24 小时一列。full_column 长度 = n_stays * 48。"""
    return np.asarray(
        [full_column[HOUR_24 + k * WINDOW_HOURS] for k in range(n_stays)],
        dtype=float,
    )


def params_to_stem(operator: str, params: str) -> str:
    """operator + params -> A.2 文件名 stem。"""
    p = params.replace(",", "_")
    if operator == "T1_uniform":
        return f"T1_uniform_{p}"
    if operator == "T1_weighted":
        return f"T1_weighted_{p}"
    if operator == "T2":
        return f"T2_{p}"
    if operator == "T3":
        return f"T3_{p}"
    return ""


def load_perturbed_24h(perturbed_dir: Path, setting: str, var: str, stem: str, n_stays: int) -> np.ndarray:
    """加载 A.2 扰动单列并返回 24h 切片。"""
    path = perturbed_dir / setting / var / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    full = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return extract_hour24_from_single_column(full, n_stays)


def run_correlation_norms(
    ts_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
) -> None:
    """构造 X，对每种 (setting, operator, params) 构造 Y(O)，算 ||R-R(O)||_∞ 与 ||R-R(O)||_F，写表。"""
    ts_dir = Path(ts_dir)
    perturbed_dir = Path(perturbed_dir)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 数值列（排除 stay_id）
    num_cols = ["HR", "SBP", "DBP", "MAP", "TempC", "RespRate", "SpO2", "Creatinine", "Lactate", "Glucose"]
    rows: list[dict] = []

    for setting in ["z", "phys"]:
        try:
            snap = load_snapshot(ts_dir, setting)
        except FileNotFoundError:
            continue
        cols = [c for c in num_cols if c in snap.columns]
        if not cols:
            continue
        X = snap[cols].astype(float)
        n_stays = len(X)
        # 去掉全 NaN 行再算相关
        mask = ~(X.isna().all(axis=1))
        X_clean = X.loc[mask].dropna(how="all", axis=1)
        if X_clean.shape[0] < 10:
            continue
        R = X_clean.corr().to_numpy()

        # 枚举该 setting 下存在的 (operator, params)
        var_dir = perturbed_dir / setting / cols[0]
        if not var_dir.exists():
            continue
        stems = [f.stem for f in var_dir.glob("*.csv")]
        for stem in sorted(stems):
            # stem -> operator, params（与 A3/A4 一致）
            if stem.startswith("T1_uniform_"):
                op, rest = "T1_uniform", stem.replace("T1_uniform_", "")
            elif stem.startswith("T1_weighted_"):
                op, rest = "T1_weighted", stem.replace("T1_weighted_", "")
            elif stem.startswith("T2_"):
                op, rest = "T2", stem.replace("T2_", "")
            elif stem.startswith("T3_"):
                op, rest = "T3", stem.replace("T3_", "")
            else:
                continue
            params = rest.replace("_max_diff=", ",max_diff=").replace("_n_passes=", ",n_passes=")

            Y_list = []
            for var in cols:
                try:
                    y24 = load_perturbed_24h(perturbed_dir, setting, var, stem, n_stays)
                except FileNotFoundError:
                    break
                Y_list.append(y24)
            if len(Y_list) != len(cols):
                continue
            Y = pd.DataFrame(np.column_stack(Y_list), columns=cols)
            Y_clean = Y.loc[mask].dropna(how="all", axis=1)
            if Y_clean.shape[0] < 10 or Y_clean.shape[1] != X_clean.shape[1]:
                continue
            R_O = Y_clean.corr().to_numpy()
            diff = R - R_O
            norm_inf = float(np.max(np.abs(diff)))
            norm_fro = float(np.linalg.norm(diff, "fro"))
            rows.append({
                "setting": setting,
                "operator": op,
                "params": params,
                "norm_inf": norm_inf,
                "norm_fro": norm_fro,
                "n_stays": int(mask.sum()),
                "d": len(cols),
            })
            print(f"[OK] {setting} / {op} / {params}  norm_inf={norm_inf:.6f}  norm_fro={norm_fro:.6f}")

    pd.DataFrame(rows).to_csv(tables_dir / "table_correlation_norms.csv", index=False)
    print(f"[写出] {tables_dir / 'table_correlation_norms.csv'}  共 {len(rows)} 行")


def run_scatter_plots(
    ts_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
) -> None:
    """对典型变量对画散点图 before（X）vs after（Y(O)）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[跳过] 未安装 matplotlib，不生成散点图")
        return

    ts_dir = Path(ts_dir)
    perturbed_dir = Path(perturbed_dir)
    figs_dir = Path(out_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    setting = "z"
    num_cols = ["HR", "SBP", "DBP", "MAP", "TempC", "RespRate", "SpO2", "Creatinine", "Lactate", "Glucose"]

    try:
        snap = load_snapshot(ts_dir, setting)
    except FileNotFoundError:
        return
    cols = [c for c in num_cols if c in snap.columns]
    X = snap[cols].astype(float)
    n_stays = len(X)

    for (v1, v2) in SCATTER_PAIRS:
        if v1 not in cols or v2 not in cols:
            continue
        for op, params in SCATTER_OPERATORS:
            stem = params_to_stem(op, params)
            try:
                y1 = load_perturbed_24h(perturbed_dir, setting, v1, stem, n_stays)
                y2 = load_perturbed_24h(perturbed_dir, setting, v2, stem, n_stays)
            except FileNotFoundError:
                continue
            mask = ~(np.isnan(X[v1]) | np.isnan(X[v2]) | np.isnan(y1) | np.isnan(y2))
            if mask.sum() < 50:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax = axes[0]
            ax.scatter(X.loc[mask, v1], X.loc[mask, v2], alpha=0.3, s=5, c="C0", label="Original")
            ax.set_xlabel(v1)
            ax.set_ylabel(v2)
            ax.set_title(f"{v1} vs {v2} — Original")
            ax.legend()

            ax = axes[1]
            ax.scatter(y1[mask], y2[mask], alpha=0.3, s=5, c="C1", label=f"{op} ({params})")
            ax.set_xlabel(f"{v1} (perturbed)")
            ax.set_ylabel(f"{v2} (perturbed)")
            ax.set_title(f"{v1} vs {v2} — {op}")
            ax.legend()

            plt.tight_layout()
            out_path = figs_dir / f"scatter_{v1}_{v2}_{op.replace(' ', '_')}_before_after.pdf"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[写出] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A.5 多列相关结构比较。")
    parser.add_argument("--ts-dir", type=str, default=None, help="ts_48h 目录（含 snapshot_24h_multicol*.csv）")
    parser.add_argument("--perturbed-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-diff", type=float, default=None, help="归一化空间 α，与扰动数据一致（散点图用此参数选 stem）")
    parser.add_argument("--no-scatter", action="store_true", help="不生成散点图")
    args = parser.parse_args()
    root = _REPO_ROOT
    if args.max_diff is not None:
        set_scatter_operators_max_diff(args.max_diff)
    if args.ts_dir:
        args.ts_dir = Path(args.ts_dir)
    else:
        try:
            args.ts_dir = default_ts48h_dir(root)
        except FileNotFoundError as e:
            raise SystemExit(f"错误：{e} 请指定 --ts-dir。") from e
    args.perturbed_dir = Path(
        args.perturbed_dir or default_a2_perturbed_dir(root)
    )
    args.out_dir = Path(args.out_dir or SCRIPT_DIR.parent / "results")
    return args


def main() -> None:
    args = parse_args()
    if not args.ts_dir.exists():
        print(f"错误：ts 目录不存在 {args.ts_dir}")
        sys.exit(1)
    if not args.perturbed_dir.exists():
        print(f"错误：扰动目录不存在 {args.perturbed_dir}")
        sys.exit(1)

    run_correlation_norms(args.ts_dir, args.perturbed_dir, args.out_dir)
    if not args.no_scatter:
        run_scatter_plots(args.ts_dir, args.perturbed_dir, args.out_dir)
    print("\nA.5 完成。")


if __name__ == "__main__":
    main()
