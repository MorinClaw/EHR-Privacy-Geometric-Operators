#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.5 实验 B3：复杂度 / 算力 可视化脚本

基于 `exp_b5_compute_metrics.py` 生成的表：
- table_b5_pipeline_timings.csv
- table_b5_summary_time_hardware.csv

输出图：
- b5_pipeline_operator_breakdown.png   : 各表 / 各隐私级别下算子耗时分解（堆叠柱状图）
- b5_runtime_summary_pipeline_vs_ctgan.png : pipeline vs CTGAN/CTGAN_gen 运行时间对比（按表拆分）

用法（实验根目录）：
  python3 B5_complexity_compute/code/plot_b5_complexity_figures.py
  python3 B5_complexity_compute/code/plot_b5_complexity_figures.py --tables-dir B5_complexity_compute/results/tables --out-dir B5_complexity_compute/results/figs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover - 纯防御
    HAS_MPL = False


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import find_b5_default_results_dir


def plot_pipeline_operator_breakdown(
    df_pipeline: pd.DataFrame,
    out_path: Path,
) -> None:
    """各表 / 隐私级别的算子耗时分解（去掉 total，只看每个 skill_id）。"""
    if not HAS_MPL or df_pipeline.empty:
        return

    df_ops = df_pipeline[df_pipeline["skill_id"] != "total"].copy()
    if df_ops.empty:
        return

    df_ops["group"] = df_ops["table"].astype(str) + "\n" + df_ops["privacy_level"].astype(str)
    pivot = (
        df_ops.pivot_table(
            index="group",
            columns="skill_id",
            values="time_sec",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    if pivot.empty:
        return

    groups = pivot.index.tolist()
    skill_ids = list(pivot.columns)

    x = np.arange(len(groups))
    width = 0.8

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(groups) * 1.3), 4))
    bottom = np.zeros(len(groups))

    for sid in skill_ids:
        vals = pivot[sid].to_numpy()
        ax.bar(x, vals, width, bottom=bottom, label=sid)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Time (sec)")
    ax.set_title("B.5 Pipeline operator breakdown (per table / privacy_level)")
    ax.legend(title="skill_id", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_runtime_summary_pipeline_vs_ctgan(
    df_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """pipeline 与 CTGAN / CTGAN_gen 的运行时间对比（按 table 拆分，log 纵轴）。"""
    if not HAS_MPL or df_summary.empty:
        return

    df_time = df_summary[~df_summary["time_sec"].isna()].copy()
    if df_time.empty:
        return

    def _make_label(row: pd.Series) -> str:
        pl = str(row.get("privacy_level", ""))
        method = str(row.get("method", ""))
        if pl in ("-", "", "nan"):
            return method
        return f"{method} ({pl})"

    df_time["label"] = df_time.apply(_make_label, axis=1)
    df_time["hardware_clean"] = df_time["hardware"].fillna("unknown").astype(str)

    tables = sorted(df_time["table"].unique())
    n_tbl = len(tables)
    if n_tbl == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_tbl,
        figsize=(max(5, 4 * n_tbl), 4),
        sharey=True if n_tbl > 1 else False,
    )
    if n_tbl == 1:
        axes = [axes]

    hardware_types = df_time["hardware_clean"].unique().tolist()
    cmap = plt.get_cmap("Set2")
    colors = {h: cmap(i % cmap.N) for i, h in enumerate(hardware_types)}

    for ax, tbl in zip(axes, tables):
        sub = df_time[df_time["table"] == tbl].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(sub))
        ax.bar(
            x,
            sub["time_sec"].to_numpy(),
            color=[colors[h] for h in sub["hardware_clean"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Time (sec)")
        ax.set_title(tbl)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[h]) for h in hardware_types]
    fig.legend(handles, hardware_types, title="Hardware", loc="upper right")

    auc_row = df_summary[df_summary["method"] == "CTGAN_downstream_auc"]
    if not auc_row.empty:
        try:
            auc = float(auc_row.iloc[0]["hardware"])
            suptitle = f"B.5 Runtime: pipeline vs CTGAN (CTGAN downstream AUC = {auc:.3f})"
        except Exception:
            suptitle = "B.5 Runtime: pipeline vs CTGAN"
    else:
        suptitle = "B.5 Runtime: pipeline vs CTGAN"

    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.5 复杂度 / 算力 可视化（CPU / baseline 对比）")
    p.add_argument(
        "--tables-dir",
        type=str,
        default=None,
        help="包含 table_b5_*.csv 的目录（默认使用 B5_complexity_compute/results/tables）",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出图像目录（默认 B5_complexity_compute/results/figs）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b5_res = find_b5_default_results_dir(ROOT)
    default_tables = b5_res / "tables"
    default_figs = b5_res / "figs"

    tables_dir = Path(args.tables_dir) if args.tables_dir else default_tables
    out_dir = Path(args.out_dir) if args.out_dir else default_figs
    out_dir.mkdir(parents=True, exist_ok=True)

    path_pipeline = tables_dir / "table_b5_pipeline_timings.csv"
    path_summary = tables_dir / "table_b5_summary_time_hardware.csv"

    if path_pipeline.exists():
        df_pipeline = pd.read_csv(path_pipeline)
        plot_pipeline_operator_breakdown(
            df_pipeline,
            out_dir / "b5_pipeline_operator_breakdown.png",
        )
    if path_summary.exists():
        df_summary = pd.read_csv(path_summary)
        plot_runtime_summary_pipeline_vs_ctgan(
            df_summary,
            out_dir / "b5_runtime_summary_pipeline_vs_ctgan.png",
        )


if __name__ == "__main__":
    main()

