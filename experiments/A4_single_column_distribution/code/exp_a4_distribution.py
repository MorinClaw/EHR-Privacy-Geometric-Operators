#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a4_distribution.py — 实验 A.4：单列分布比较（CDF equivalence，PDF 1.2.1）

1) KS 检验：对每个 (var, setting, operator, params)，两样本 KS 检验 x vs y，保存 K、p-value。
2) 正态性检验：对 HR 等近似正态列，做 D'Agostino-Pearson normaltest（或 Cramér–von Mises），
   比较扰动前后“接近正态程度”。
3) 可视化：选代表变量 HR, SBP, 一实验室值（Creatinine），画直方图（原始 vs 各算子）、Q–Q 图。

输出：
  - results/tables/table_ks.csv
  - results/tables/table_normality.csv（可选，对 HR/SBP/Creatinine 等）
  - results/figs/hist_qq_HR.pdf, hist_qq_SBP.pdf, hist_qq_Creatinine.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from repo_discovery import default_a2_perturbed_dir, default_ts48h_dir

# 代表变量：HR, SBP, 一实验室值（用于正态性 + 作图）
REP_VARS = ("HR", "SBP", "Creatinine")
# 作图中每个算子选一组代表参数（方案一：归一化空间 max_diff；可由 --max-diff 覆盖）
def _default_plot_max_diff() -> float:
    return 0.8


PLOT_PARAMS = {
    "T1_uniform": "n_passes=5,max_diff=0.8",
    "T1_weighted": "n_passes=5,max_diff=0.8",
    "T2": "max_diff=0.8",
    "T3": "max_diff=0.8",
}


def set_plot_params_max_diff(alpha: float) -> None:
    """用给定 α 更新 PLOT_PARAMS（供阿尔法扫掠时调用）。"""
    global PLOT_PARAMS
    PLOT_PARAMS = {
        "T1_uniform": f"n_passes=5,max_diff={alpha}",
        "T1_weighted": f"n_passes=5,max_diff={alpha}",
        "T2": f"max_diff={alpha}",
        "T3": f"max_diff={alpha}",
    }


def load_single_column(data_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载原始单列。"""
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def load_perturbed(perturbed_dir: Path, setting: str, var: str, stem: str) -> np.ndarray:
    """加载 A.2 扰动结果。stem 如 T1_uniform_n_passes=5_max_diff=0.8。"""
    path = perturbed_dir / setting / var / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def parse_perturbed_filename(stem: str) -> tuple[str, str]:
    """从文件名 stem 解析 operator 与 params。"""
    if stem.startswith("T1_uniform_"):
        op, rest = "T1_uniform", stem.replace("T1_uniform_", "")
    elif stem.startswith("T1_weighted_"):
        op, rest = "T1_weighted", stem.replace("T1_weighted_", "")
    elif stem.startswith("T2_"):
        op, rest = "T2", stem.replace("T2_", "")
    elif stem.startswith("T3_"):
        op, rest = "T3", stem.replace("T3_", "")
    else:
        op = stem.split("_")[0] if "_" in stem else stem
        rest = stem
    params = rest.replace("_max_diff=", ",max_diff=").replace("_n_passes=", ",n_passes=")
    return op, params


def run_ks_and_normality(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    rep_vars: tuple[str, ...],
    variables: list[str] | None,
) -> None:
    """KS 检验 + 正态性检验，写出 table_ks.csv 与 table_normality.csv。"""
    data_dir = Path(data_dir)
    perturbed_dir = Path(perturbed_dir)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_ks: list[dict] = []
    rows_norm: list[dict] = []
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
            except FileNotFoundError:
                continue
            mask = ~(np.isnan(x))
            xv = x[mask]
            if len(xv) < 10:
                continue

            # 正态性：仅对代表变量算原始 x 的 normaltest（每个 var/setting 算一次）
            if var in rep_vars:
                try:
                    nt_x = stats.normaltest(xv)
                    norm_stat_x, norm_p_x = float(nt_x.statistic), float(nt_x.pvalue)
                except Exception:
                    norm_stat_x = norm_p_x = np.nan

            for csv_path in sorted(setting_path.joinpath(var).glob("*.csv")):
                stem = csv_path.stem
                operator, params = parse_perturbed_filename(stem)
                df_y = pd.read_csv(csv_path)
                col = "value" if "value" in df_y.columns else df_y.columns[0]
                y = pd.to_numeric(df_y[col], errors="coerce").to_numpy(dtype=float)
                if len(y) != len(x):
                    continue
                mask_both = mask & ~np.isnan(y)
                xb, yb = x[mask_both], y[mask_both]
                n = len(xb)
                if n < 10:
                    continue

                # KS 两样本
                ks_stat, ks_pval = stats.ks_2samp(xb, yb)
                rows_ks.append({
                    "var": var,
                    "setting": setting,
                    "operator": operator,
                    "params": params,
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_pval),
                    "n": n,
                })

                # 正态性：对代表变量算扰动后 y 的 normaltest
                if var in rep_vars:
                    try:
                        nt_y = stats.normaltest(yb)
                        norm_stat_y, norm_p_y = float(nt_y.statistic), float(nt_y.pvalue)
                    except Exception:
                        norm_stat_y = norm_p_y = np.nan
                    rows_norm.append({
                        "var": var,
                        "setting": setting,
                        "operator": operator,
                        "params": params,
                        "norm_stat_orig": norm_stat_x,
                        "norm_p_orig": norm_p_x,
                        "norm_stat_pert": norm_stat_y,
                        "norm_p_pert": norm_p_y,
                    })

    pd.DataFrame(rows_ks).to_csv(tables_dir / "table_ks.csv", index=False)
    print(f"[写出] {tables_dir / 'table_ks.csv'}  共 {len(rows_ks)} 行")
    if rows_norm:
        pd.DataFrame(rows_norm).to_csv(tables_dir / "table_normality.csv", index=False)
        print(f"[写出] {tables_dir / 'table_normality.csv'}  共 {len(rows_norm)} 行")


def stem_for_plot_params(operator: str, params: str) -> str:
    """由 operator 与 params 反推 A.2 文件名 stem。"""
    if operator == "T1_uniform":
        # params = n_passes=5,max_diff=1.0
        return "T1_uniform_" + params.replace(",", "_")
    if operator == "T1_weighted":
        return "T1_weighted_" + params.replace(",", "_")
    if operator == "T2":
        return "T2_" + params.replace(",", "_")
    if operator == "T3":
        return "T3_" + params.replace(",", "_")
    return ""


def run_plots(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    rep_vars: tuple[str, ...],
) -> None:
    """对代表变量画直方图 + Q–Q 图（原始 vs 各算子代表参数），保存到 results/figs/。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[跳过] 未安装 matplotlib，不生成图")
        return

    data_dir = Path(data_dir)
    perturbed_dir = Path(perturbed_dir)
    figs_dir = Path(out_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    setting = "z"  # 作图用 z setting，尺度可比

    for var in rep_vars:
        try:
            x = load_single_column(data_dir, var, setting)
        except FileNotFoundError:
            print(f"[跳过] 无原始列 {var} / {setting}")
            continue
        mask = ~(np.isnan(x))
        xv = np.asarray(x[mask], dtype=float)
        if len(xv) < 100:
            continue

        # 加载各算子代表参数对应的扰动结果
        series: list[tuple[str, np.ndarray]] = [("original", xv)]
        for op, params in PLOT_PARAMS.items():
            stem = stem_for_plot_params(op, params)
            if not stem:
                continue
            try:
                y = load_perturbed(perturbed_dir, setting, var, stem)
            except FileNotFoundError:
                continue
            yv = np.asarray(y[mask & ~np.isnan(y)], dtype=float)
            if len(yv) < 100:
                continue
            series.append((op, yv))

        if len(series) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # 左：直方图（原始 + 各算子，半透明叠加）
        ax = axes[0]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i, (label, arr) in enumerate(series):
            ax.hist(arr, bins=50, density=True, alpha=0.4, label=label, color=colors[i % len(colors)], histtype="stepfilled")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.set_title(f"{var} (z) — Histogram")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=0)

        # 右：Q–Q 图（统一分位网格下：原始 x 分位数 vs 各 y 分位数）
        ax = axes[1]
        q_grid = np.linspace(0.01, 0.99, 300)
        x_quantiles = np.quantile(xv, q_grid)
        ax.plot(x_quantiles, x_quantiles, "k--", alpha=0.7, label="y=x")
        for i, (label, arr) in enumerate(series):
            if label == "original":
                continue
            y_quantiles = np.quantile(arr, q_grid)
            ax.plot(x_quantiles, y_quantiles, alpha=0.7, label=label, color=colors[(i + 1) % len(colors)])
        ax.set_xlabel("Original (x) quantiles")
        ax.set_ylabel("Perturbed (y) quantiles")
        ax.set_title(f"{var} (z) — Q–Q (x vs y)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(bottom=ax.get_ylim()[0])

        plt.tight_layout()
        out_path = figs_dir / f"hist_qq_{var}.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[写出] {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A.4 单列分布比较：KS、正态性、直方图与 Q–Q 图。")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--perturbed-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-diff", type=float, default=None, help="归一化空间 α，与扰动数据一致（作图中用此参数选 stem）")
    parser.add_argument("--variables", type=str, nargs="*", default=None)
    parser.add_argument("--no-plots", action="store_true", help="不生成直方图与 Q–Q 图")
    args = parser.parse_args()

    root = _REPO_ROOT
    if args.max_diff is not None:
        set_plot_params_max_diff(args.max_diff)
    if args.data_dir:
        args.data_dir = Path(args.data_dir)
    else:
        try:
            args.data_dir = default_ts48h_dir(root)
        except FileNotFoundError as e:
            raise SystemExit(f"错误：{e} 请指定 --data-dir。") from e
    args.perturbed_dir = Path(
        args.perturbed_dir or default_a2_perturbed_dir(root)
    )
    args.out_dir = Path(args.out_dir or SCRIPT_DIR.parent / "results")
    return args


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"错误：数据目录不存在 {args.data_dir}")
        sys.exit(1)
    if not args.perturbed_dir.exists():
        print(f"错误：扰动目录不存在 {args.perturbed_dir}")
        sys.exit(1)

    run_ks_and_normality(
        args.data_dir,
        args.perturbed_dir,
        args.out_dir,
        rep_vars=REP_VARS,
        variables=args.variables,
    )
    if not args.no_plots:
        run_plots(args.data_dir, args.perturbed_dir, args.out_dir, rep_vars=REP_VARS)
    print("\nA.4 完成。")


if __name__ == "__main__":
    main()
