#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
隐私–效用“帕累托式”前沿可视化

从现有 B4 / A7 结果表中抽点，生成 1–2 张示意性的隐私–效用前沿图：

1. B4 timeline 级别（ICU 预测任务）：
   - 使用 `table_b4_timeline_auc_auprc.csv` 与 `table_b4_timeline_reconstruction.csv`
   - 点：
     * Raw：隐私≈0（没有脱敏），效用=AUC_raw
     * De-id（当前实现为 strong 流水线）：隐私=数值重构 RMSE，效用=AUC_deid
   - 输出：`frontier_b4_timeline_auc_vs_recon.png`

2. A7 数值算子级别（T1/T2/T3 + pipeline_strong）：
   - 使用 `A7_privacy_attacks/results/tables/table_reconstruction_20pct.csv`
     与 `table_reconstruction_pipeline_strong.csv`、`table_delta_stats.csv`
   - 对每个算子 operator，构造：
       隐私 P ≈ 1 - mean_R2_20pct_linear（重建 R² 越低，越难被配对攻击反推）
       效用 U ≈ 1 / (1 + mean_delta_std)（Δ 标准差越小，扰动越小、效用越高）
   - 输出：`frontier_a7_numeric_privacy_utility.png`

注意：这里的“帕累托前沿”是示意性质，用于在一张图上同时展示
「隐私指标」与「效用/失真指标」之间的大致权衡关系，具体指标可按需要调整。

用法（实验根目录）：
  python3 B4_privacy_utility_tradeoff/code/plot_privacy_utility_frontier.py
  python3 B4_privacy_utility_tradeoff/code/plot_privacy_utility_frontier.py \\
      --b4-dir B4_privacy_utility_tradeoff/results \\
      --a7-dir A7_privacy_attacks/results
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
except ImportError:  # pragma: no cover - 防御性分支
    HAS_MPL = False


SCRIPT_DIR = Path(__file__).resolve().parent
B4_ROOT = SCRIPT_DIR.parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import find_prefixed_section_dir

A7_ROOT = find_prefixed_section_dir(ROOT, "A7_")


def _ensure_fig_dir(base: Path) -> Path:
    d = base / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_b4_timeline_frontier(b4_results_dir: Path) -> None:
    """B4 timeline：AUC vs reconstruction RMSE 的 2 点前沿（Raw vs De-id）。"""
    if not HAS_MPL:
        return
    tables_dir = b4_results_dir / "tables"
    figs_dir = _ensure_fig_dir(b4_results_dir)

    path_auc = tables_dir / "table_b4_timeline_auc_auprc.csv"
    path_recon = tables_dir / "table_b4_timeline_reconstruction.csv"
    if not path_auc.exists() or not path_recon.exists():
        print("[frontier] 未找到 B4 timeline 的表，跳过 B4 前沿图")
        return

    df_auc = pd.read_csv(path_auc)
    df_recon = pd.read_csv(path_recon)
    if df_auc.empty or df_recon.empty:
        print("[frontier] B4 timeline 表为空，跳过 B4 前沿图")
        return

    # 仅取 AUC 一行；若没有 AUC，则用第一行作为示意
    if "metric" in df_auc.columns:
        if "AUC" in df_auc["metric"].values:
            row_auc = df_auc[df_auc["metric"] == "AUC"].iloc[0]
        else:
            row_auc = df_auc.iloc[0]
    else:
        row_auc = df_auc.iloc[0]

    auc_raw = float(row_auc.get("raw", np.nan))
    auc_deid = float(row_auc.get("deid", np.nan))

    row_rec = df_recon.iloc[0]
    rmse = float(row_rec.get("rmse", np.nan))

    if not np.isfinite(auc_raw) or not np.isfinite(auc_deid) or not np.isfinite(rmse):
        print("[frontier] B4 timeline 指标存在 NaN，跳过 B4 前沿图")
        return

    # 两个点：Raw (privacy≈0)，De-id (privacy≈RMSE)
    x_vals = [0.0, rmse]
    y_vals = [auc_raw, auc_deid]
    labels = ["Raw", "De-id\n(strong pipeline)"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(x_vals, y_vals, c=["steelblue", "coral"], s=60)
    for x, y, lab in zip(x_vals, y_vals, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 5), fontsize=9)

    ax.set_xlabel("Privacy (numeric RMSE, higher = more noise)")
    ax.set_ylabel("Utility (ICU AUC)")
    ax.set_title("B4 timeline privacy–utility frontier (Raw vs De-id)")
    ax.grid(True, alpha=0.4)

    out_path = figs_dir / "frontier_b4_timeline_auc_vs_recon.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[frontier] 已保存 {out_path}")


def _load_a7_tables(a7_results_dir: Path):
    t_recon = a7_results_dir / "tables" / "table_reconstruction_20pct.csv"
    t_recon_pipe = a7_results_dir / "tables" / "table_reconstruction_pipeline_strong.csv"
    t_delta = a7_results_dir / "tables" / "table_delta_stats.csv"
    if not (t_recon.exists() and t_recon_pipe.exists() and t_delta.exists()):
        print("[frontier] 未找到完整 A7 表（reconstruction_20pct / pipeline_strong / delta_stats），跳过 A7 前沿图")
        return None, None, None
    return (
        pd.read_csv(t_recon),
        pd.read_csv(t_recon_pipe),
        pd.read_csv(t_delta),
    )


def plot_a7_numeric_frontier(a7_results_dir: Path) -> None:
    """A7 数值算子：基于重建 R² 与 Δ 标准差构造隐私–效用点。"""
    if not HAS_MPL:
        return

    df_recon, df_recon_pipe, df_delta = _load_a7_tables(a7_results_dir)
    if df_recon is None:
        return

    figs_dir = _ensure_fig_dir(a7_results_dir)

    # 仅考虑 setting="z"、train_frac=0.2、model="linear" 的主结果，参数 max_diff=0.8、n_passes=5（若适用，方案一）
    mask_main = (
        (df_recon["setting"] == "z")
        & (df_recon["train_frac"] == 0.2)
        & (df_recon["model"] == "linear")
    )
    df_r = df_recon[mask_main].copy()
    if df_r.empty:
        print("[frontier] A7 reconstruction_20pct 无满足条件的行，跳过 A7 前沿图")
        return

    # 选出我们关心的一组 operator（可按需要扩展）
    target_ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    df_r = df_r[df_r["operator"].isin(target_ops)]

    # pipeline_strong 的 R² 在单独表中
    df_rp = df_recon_pipe[
        (df_recon_pipe["setting"] == "z")
        & (df_recon_pipe["train_frac"] == 0.2)
        & (df_recon_pipe["model"] == "linear")
    ].copy()

    # 计算每个 operator 的 mean R²（跨变量平均）
    op_to_R2 = (
        df_r.groupby("operator")["R2"].mean().to_dict()
    )  # T1/T2/T3
    if not df_rp.empty:
        op_to_R2["pipeline_strong"] = float(df_rp["R2"].mean())

    # 从 delta_stats 里取对应算子的 Δ 标准差（同样 setting="z"、max_diff=0.8、n_passes=5）
    df_d = df_delta[df_delta["setting"] == "z"].copy()
    # 简单起见：优先选 max_diff=0.8；对 T1 选 n_passes=5
    def _is_main_param(row) -> bool:
        params = str(row.get("params", ""))
        if "max_diff=0.8" not in params:
            return False
        if row["operator"].startswith("T1") and "n_passes=5" not in params:
            return False
        return True

    df_d = df_d[df_d.apply(_is_main_param, axis=1)]
    if df_d.empty:
        print("[frontier] A7 delta_stats 里没有匹配的主参数行，跳过 A7 前沿图")
        return

    op_to_std = df_d.groupby("operator")["delta_std"].mean().to_dict()

    ops = sorted(set(op_to_R2.keys()) & set(op_to_std.keys()))
    if not ops:
        print("[frontier] A7 中 operator 集合为空，跳过 A7 前沿图")
        return

    # 构造隐私–效用指标：
    #   Privacy P = 1 - mean_R2  （越大表示重建越差，隐私越强）
    #   Utility U = 1 / (1 + mean_delta_std) （Δ 标准差越小，U 越接近 1）
    P = []
    U = []
    labels = []
    for op in ops:
        r2 = float(op_to_R2[op])
        std = float(op_to_std[op])
        P.append(max(0.0, 1.0 - r2))
        U.append(1.0 / (1.0 + std))
        labels.append(op)

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(P, U, c=np.linspace(0, 1, len(ops)), cmap="viridis", s=60)
    for x, y, lab in zip(P, U, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 4), fontsize=8)

    ax.set_xlabel("Privacy ~ (1 − mean R²) under 20% paired attack")
    ax.set_ylabel("Utility ~ 1 / (1 + Δ std)")
    ax.set_title("A7 numeric operators: privacy–utility trade-off (z, 20% train, linear)")
    ax.grid(True, alpha=0.4)

    out_path = figs_dir / "frontier_a7_numeric_privacy_utility.png"
    plt.colorbar(sc, ax=ax, label="operator index")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[frontier] 已保存 {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="隐私–效用前沿可视化（B4 + A7）")
    p.add_argument(
        "--b4-dir",
        type=str,
        default=None,
        help="B4 结果目录（包含 tables/figs），默认 B4_privacy_utility_tradeoff/results",
    )
    p.add_argument(
        "--a7-dir",
        type=str,
        default=None,
        help="A7 结果目录（包含 tables/figs），默认 A7_privacy_attacks/results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b4_dir = Path(args.b4_dir) if args.b4_dir else (B4_ROOT / "results")
    a7_dir = Path(args.a7_dir) if args.a7_dir else (A7_ROOT / "results")

    if b4_dir.exists():
        plot_b4_timeline_frontier(b4_dir)
    else:
        print(f"[frontier] 未找到 B4 结果目录 {b4_dir}，跳过 B4 前沿图")

    if a7_dir.exists():
        plot_a7_numeric_frontier(a7_dir)
    else:
        print(f"[frontier] 未找到 A7 结果目录 {a7_dir}，跳过 A7 前沿图")


if __name__ == "__main__":
    main()

