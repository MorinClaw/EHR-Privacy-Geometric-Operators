#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.3 理论约束端到端 — 结果可视化

读取 results 下 CSV/JSON，生成：
- plot_b3_numeric_sanity.pdf：数值列（valuenum）理论约束验证（delta_mean, delta_var, max_abs_delta, unchanged_ratio）
- plot_b3_text_phi.pdf：文本 PHI 脱敏前后匹配数对比
- plot_b3_id_time_summary.pdf：ID/时间验证摘要（通过率或状态）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
B3_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS = B3_ROOT / "results"
DEFAULT_FIGS = B3_ROOT / "results" / "figs"


def plot_numeric_sanity(out_dir: Path, results_dir: Path) -> None:
    """数值 sanity：delta_mean, delta_var, max_abs_delta, unchanged_ratio 柱状/指标图。"""
    path = results_dir / "table_agent_sanity_numeric.csv"
    if not path.exists():
        print(f"[B.3 图] 未找到 {path}，跳过 numeric sanity 图")
        return
    df = pd.read_csv(path)
    # 去掉带 error 的行（若有）
    df = df.dropna(subset=["delta_mean"], how="all")
    if df.empty:
        print("[B.3 图] table_agent_sanity_numeric 无有效行，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B.3 图] 未安装 matplotlib，跳过")
        return

    # 每行对应一列（如 valuenum），指标为 delta_mean, delta_var, max_abs_delta, unchanged_ratio
    metrics = ["delta_mean", "delta_var", "max_abs_delta", "unchanged_ratio"]
    labels_en = [r"$|\Delta$ mean|", r"$|\Delta$ var|", r"max $|\Delta|$", "unchanged ratio"]
    for _, row in df.iterrows():
        col_name = row.get("column", "valuenum")
        fig, axes = plt.subplots(2, 2, figsize=(7, 5))
        axes = axes.ravel()
        for i, (m, lb) in enumerate(zip(metrics, labels_en)):
            ax = axes[i]
            val = row.get(m, np.nan)
            if pd.isna(val):
                val = 0.0
            color = "C0" if m != "unchanged_ratio" else "C1"
            ax.bar([0], [val], color=color, width=0.5)
            ax.set_ylabel(lb)
            ax.set_title(lb)
            ax.set_xticks([])
            if m == "unchanged_ratio":
                ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"B.3 Numeric theory constraints ({col_name}, strong pipeline)", fontsize=11)
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "plot_b3_numeric_sanity.pdf", bbox_inches="tight")
        plt.close()
        print(f"[B.3 图] {out_dir / 'plot_b3_numeric_sanity.pdf'}")
        break  # 只画第一列（通常只有 valuenum）


def plot_text_phi(out_dir: Path, results_dir: Path) -> None:
    """文本 PHI：脱敏前后各 pattern 匹配数对比（分组柱状图）。"""
    path = results_dir / "table_agent_text_phi_leakage.csv"
    if not path.exists():
        print(f"[B.3 图] 未找到 {path}，跳过 text PHI 图")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("[B.3 图] table_agent_text_phi_leakage 为空，跳过")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B.3 图] 未安装 matplotlib，跳过")
        return

    # 按 (table, pattern) 聚合或按 pattern 聚合（若多表则取 sum）
    agg = df.groupby("pattern").agg(count_orig=("count_orig", "sum"), count_deid=("count_deid", "sum")).reset_index()
    patterns = agg["pattern"].tolist()
    x = np.arange(len(patterns))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, agg["count_orig"], w, label="Before", color="C0", alpha=0.8)
    ax.bar(x + w / 2, agg["count_deid"], w, label="After", color="C1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=15, ha="right")
    ax.set_ylabel("Match count")
    ax.set_title("B.3 Text PHI patterns: before vs after de-identification")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "plot_b3_text_phi.pdf", bbox_inches="tight")
    plt.close()
    print(f"[B.3 图] {out_dir / 'plot_b3_text_phi.pdf'}")


def plot_id_time_summary(out_dir: Path, results_dir: Path) -> None:
    """ID/时间验证：通过情况汇总（表格式或柱状通过率）。"""
    id_path = results_dir / "table_agent_id_validation.csv"
    time_path = results_dir / "table_agent_time_validation.csv"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[B.3 图] 未安装 matplotlib，跳过 ID/Time 图")
        return

    items = []
    if id_path.exists():
        id_df = pd.read_csv(id_path)
        for _, r in id_df.iterrows():
            items.append({"check": f"{r['table']}.{r['column']}", "pass": r["all_hex_fixed_len"] and r["consistent_mapping"]})
    if time_path.exists():
        time_df = pd.read_csv(time_path)
        for _, r in time_df.iterrows():
            items.append({"check": f"{r['table']}.{r['column']}", "pass": bool(r.get("is_numeric_days", True))})

    if not items:
        print("[B.3 图] 无 ID/Time 表，跳过")
        return

    df = pd.DataFrame(items)
    n_ok = df["pass"].sum()
    n_total = len(df)
    fig, ax = plt.subplots(figsize=(6, max(3, len(items) * 0.35)))
    y = np.arange(len(items))
    colors = ["#2ecc71" if p else "#e74c3c" for p in df["pass"]]
    ax.barh(y, [1] * len(items), color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["check"].tolist(), fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0.5])
    ax.set_xticklabels(["Pass"])
    ax.set_title(f"B.3 ID/Time validation ({n_ok}/{n_total} pass)")
    ax.set_xlabel("")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "plot_b3_id_time_summary.pdf", bbox_inches="tight")
    plt.close()
    print(f"[B.3 图] {out_dir / 'plot_b3_id_time_summary.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B.3 理论约束端到端 — 结果可视化")
    parser.add_argument("--results-dir", type=str, default=None, help="结果目录，默认 B3_theory_e2e/results")
    parser.add_argument("--out-dir", type=str, default=None, help="图输出目录，默认 results/figs")
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_FIGS
    # 相对路径相对于当前工作目录解析（便于在实验根目录下指定 B3_theory_e2e/results）
    if not results_dir.is_absolute():
        results_dir = results_dir.resolve()
    if not out_dir.is_absolute():
        out_dir = out_dir.resolve()

    plot_numeric_sanity(out_dir, results_dir)
    plot_text_phi(out_dir, results_dir)
    plot_id_time_summary(out_dir, results_dir)
    print("B.3 可视化完成。")


if __name__ == "__main__":
    main()
