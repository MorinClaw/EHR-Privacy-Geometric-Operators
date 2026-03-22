#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.4 B2.2 timeline_events 结果可视化

读取 results/tables 下 B2.2 的 CSV，生成：
- task_b22_auc_auprc.png：ICU 预测 AUC/AUPRC（raw vs deid）
- task_b22_reconstruction.png：数值重构 MAE、RMSE、相关系数
- task_b22_mi_separation.png：MI 正/负样本预测概率分离度（raw vs deid）
- task_b22_quasiid_summary.png：Quasi-ID 唯一化比例（若存在表）

用法（实验根目录）：
  python3 B4_privacy_utility_tradeoff/code/plot_b4_b22_figures.py
  python3 B4_privacy_utility_tradeoff/code/plot_b4_b22_figures.py --out-dir B4_privacy_utility_tradeoff/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
B4_ROOT = SCRIPT_DIR.parent
ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUT = B4_ROOT / "results"


def _fig_dir(out_dir: Path) -> Path:
    d = out_dir / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_auc_auprc(tables_dir: Path, figs_dir: Path) -> None:
    """ICU 预测：AUC / AUPRC raw vs deid 柱状图。"""
    path = tables_dir / "table_b4_timeline_auc_auprc.csv"
    if not path.exists():
        print(f"[B2.2 图] 未找到 {path}，跳过 AUC/AUPRC 图")
        return
    df = pd.read_csv(path)
    if df.empty or "metric" not in df.columns:
        print("[B2.2 图] table_b4_timeline_auc_auprc 格式异常，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B2.2 图] 未安装 matplotlib，跳过")
        return

    metrics = df["metric"].tolist()
    x = np.arange(len(metrics))
    w = 0.35
    raw_vals = df["raw"].values
    deid_vals = df["deid"].values

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - w / 2, raw_vals, width=w, label="Raw", color="steelblue")
    ax.bar(x + w / 2, deid_vals, width=w, label="De-id", color="coral", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title("B2.2 ICU prediction (24h features): AUC / AUPRC")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(figs_dir / "task_b22_auc_auprc.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[B2.2 图] 已保存 {figs_dir / 'task_b22_auc_auprc.png'}")


def plot_reconstruction(tables_dir: Path, figs_dir: Path) -> None:
    """数值重构：MAE、RMSE、相关系数（柱状图，多列时按列分组）。"""
    path = tables_dir / "table_b4_timeline_reconstruction.csv"
    if not path.exists():
        print(f"[B2.2 图] 未找到 {path}，跳过 reconstruction 图")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("[B2.2 图] table_b4_timeline_reconstruction 为空，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B2.2 图] 未安装 matplotlib，跳过")
        return

    cols = df["col"].tolist() if "col" in df.columns else [f"col_{i}" for i in range(len(df))]
    metrics = ["mae", "rmse", "corr"]
    # 单列时：3 个子图 MAE / RMSE / corr；多列时：每个 metric 一组柱
    if len(cols) == 1:
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        row = df.iloc[0]
        for i, m in enumerate(metrics):
            v = row.get(m, np.nan)
            axes[i].bar([0], [v], color="teal", width=0.4)
            axes[i].set_ylabel(m.upper())
            axes[i].set_xticks([])
            if m == "corr":
                axes[i].set_ylim(-0.05, 1.05)
            axes[i].grid(True, alpha=0.3)
        fig.suptitle("B2.2 Numeric reconstruction (valuenum): MAE, RMSE, Corr", fontsize=11)
    else:
        x = np.arange(len(cols))
        w = 0.25
        fig, axes = plt.subplots(1, 3, figsize=(4 + len(cols) * 0.8, 4))
        for i, m in enumerate(metrics):
            vals = df[m].values
            axes[i].bar(x, vals, width=w * 2, color="teal", alpha=0.8)
            axes[i].set_ylabel(m.upper())
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(cols, rotation=45, ha="right")
            if m == "corr":
                axes[i].set_ylim(-0.05, 1.05)
            axes[i].grid(True, alpha=0.3, axis="y")
        fig.suptitle("B2.2 Numeric reconstruction by column", fontsize=11)
    plt.tight_layout()
    fig.savefig(figs_dir / "task_b22_reconstruction.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[B2.2 图] 已保存 {figs_dir / 'task_b22_reconstruction.png'}")


def plot_mi_separation(tables_dir: Path, figs_dir: Path) -> None:
    """MI：正/负样本预测概率分离度 raw vs deid。"""
    path = tables_dir / "table_b4_timeline_mi.csv"
    if not path.exists():
        print(f"[B2.2 图] 未找到 {path}，跳过 MI 图")
        return
    df = pd.read_csv(path)
    if df.empty or "sep_raw" not in df.columns:
        print("[B2.2 图] table_b4_timeline_mi 格式异常，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B2.2 图] 未安装 matplotlib，跳过")
        return

    row = df.iloc[0]
    sep_raw = float(row["sep_raw"])
    sep_deid = float(row["sep_deid"])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0], [sep_raw], width=0.35, label="Raw", color="steelblue")
    ax.bar([0.45], [sep_deid], width=0.35, label="De-id", color="coral", alpha=0.9)
    ax.set_xticks([0.225])
    ax.set_xticklabels(["Separation (mean prob pos − neg)"])
    ax.set_ylabel("Separation")
    ax.legend()
    ax.set_title("B2.2 Membership inference: prediction separation")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(figs_dir / "task_b22_mi_separation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[B2.2 图] 已保存 {figs_dir / 'task_b22_mi_separation.png'}")


def plot_quasiid_summary(tables_dir: Path, figs_dir: Path) -> None:
    """Quasi-ID 唯一化：唯一组合数 / 总行数 比例。"""
    path = tables_dir / "table_b4_quasiid_uniqueness.csv"
    if not path.exists():
        print(f"[B2.2 图] 未找到 {path}，跳过 Quasi-ID 图")
        return
    df = pd.read_csv(path)
    if df.empty or "unique_ratio" not in df.columns:
        print("[B2.2 图] table_b4_quasiid_uniqueness 格式异常，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B2.2 图] 未安装 matplotlib，跳过")
        return

    row = df.iloc[0]
    ratio = float(row["unique_ratio"])
    n_unique = int(row.get("n_unique_quasiid", 0))
    n_rows = int(row.get("n_rows", 0))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0], [ratio], width=0.4, color="darkorange", alpha=0.9)
    ax.set_xticks([0])
    ax.set_xticklabels(["Unique quasi-ID ratio"])
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"B2.2 Quasi-ID uniqueness (n={n_rows}, unique={n_unique})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(figs_dir / "task_b22_quasiid_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[B2.2 图] 已保存 {figs_dir / 'task_b22_quasiid_summary.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="B2.2 timeline 结果可视化")
    p.add_argument("--out-dir", type=str, default=None, help="结果目录，默认 B4_privacy_utility_tradeoff/results")
    args = p.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    tables_dir = out_dir / "tables"
    figs_dir = _fig_dir(out_dir)

    if not tables_dir.exists():
        print(f"[B2.2 图] 未找到 {tables_dir}，请先运行 exp_b4_timeline_icu_tasks.py")
        return

    plot_auc_auprc(tables_dir, figs_dir)
    plot_reconstruction(tables_dir, figs_dir)
    plot_mi_separation(tables_dir, figs_dir)
    plot_quasiid_summary(tables_dir, figs_dir)
    print("B2.2 可视化完成。")


if __name__ == "__main__":
    main()
