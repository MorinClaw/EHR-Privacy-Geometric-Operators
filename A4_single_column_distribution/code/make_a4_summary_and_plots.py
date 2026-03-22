#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 A4 两个结果表 table_ks.csv、table_normality.csv 生成：
1. summary 汇总表（两表关键指标汇总）
2. KS 总表可视化（热力图、按算子/变量柱状图）
"""
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    path_ks = TABLES_DIR / "table_ks.csv"
    path_norm = TABLES_DIR / "table_normality.csv"
    if not path_ks.exists():
        print(f"未找到: {path_ks}")
        return
    df_ks = pd.read_csv(path_ks)
    df_norm = pd.read_csv(path_norm) if path_norm.exists() else None

    # ---------- 1. Summary 表 ----------
    # 1a) KS 按 setting × operator 汇总
    summary_ks_setting_op = (
        df_ks.groupby(["setting", "operator"], as_index=False)
        .agg(
            ks_statistic_mean=("ks_statistic", "mean"),
            ks_statistic_max=("ks_statistic", "max"),
            ks_statistic_min=("ks_statistic", "min"),
            ks_pvalue_mean=("ks_pvalue", "mean"),
            n_configs=("ks_statistic", "count"),
        )
        .round(6)
    )
    summary_ks_setting_op.to_csv(
        TABLES_DIR / "table_a4_summary_ks_by_setting_operator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 1b) KS 按 var × operator 汇总（便于按变量看分布保持）
    summary_ks_var_op = (
        df_ks.groupby(["var", "operator"], as_index=False)
        .agg(
            ks_statistic_mean=("ks_statistic", "mean"),
            ks_statistic_max=("ks_statistic", "max"),
            n_configs=("ks_statistic", "count"),
        )
        .round(6)
    )
    summary_ks_var_op.to_csv(
        TABLES_DIR / "table_a4_summary_ks_by_var_operator.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 1c) 正态性表汇总（代表变量：HR, SBP, Creatinine）
    if df_norm is not None and len(df_norm) > 0:
        summary_norm = (
            df_norm.groupby(["var", "operator"], as_index=False)
            .agg(
                norm_stat_orig_mean=("norm_stat_orig", "mean"),
                norm_stat_pert_mean=("norm_stat_pert", "mean"),
                norm_p_orig_mean=("norm_p_orig", "mean"),
                norm_p_pert_mean=("norm_p_pert", "mean"),
                n_configs=("norm_stat_orig", "count"),
            )
            .round(6)
        )
        summary_norm.to_csv(
            TABLES_DIR / "table_a4_summary_normality_by_var_operator.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        summary_norm = None

    # 1d) 一张总 summary：两结果要点（KS + 正态性）
    conclusion_rows = [
        {
            "结果表": "table_ks.csv",
            "指标": "ks_statistic",
            "说明": "脱敏前后样本 CDF 差异（两样本 KS）；越小分布越接近",
            "按算子均值范围": f"{df_ks.groupby('operator')['ks_statistic'].mean().min():.4f} ~ {df_ks.groupby('operator')['ks_statistic'].mean().max():.4f}",
            "结论": "T3 等距保持最好(KS 最小)；T1 随 n_passes/max_diff 增大略升；T2 加噪略大",
        },
        {
            "结果表": "table_normality.csv",
            "指标": "norm_stat_orig / norm_stat_pert",
            "说明": "D'Agostino-Pearson 正态性检验统计量（代表变量 HR/SBP/Creatinine）",
            "按算子均值范围": (
                f"orig: {df_norm['norm_stat_orig'].mean():.0f}, pert: {df_norm['norm_stat_pert'].mean():.0f}"
                if df_norm is not None and len(df_norm) > 0
                else "—"
            ),
            "结论": "扰动前后正态性统计量量级接近，形态保持",
        },
    ]
    pd.DataFrame(conclusion_rows).to_csv(
        TABLES_DIR / "table_a4_summary_conclusion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Summary 表已写入:", list(TABLES_DIR.glob("table_a4_summary*.csv")))

    # ---------- 2. KS 总表可视化 ----------
    if not HAS_PLOT:
        print("未安装 matplotlib，跳过作图")
        return

    op_order = [c for c in ["T1_uniform", "T1_weighted", "T2", "T3"] if c in df_ks["operator"].unique()]

    # 图1: 热力图 — var × operator，值为 ks_statistic 均值
    pivot_ks = df_ks.pivot_table(
        values="ks_statistic",
        index="var",
        columns="operator",
        aggfunc="mean",
    )
    pivot_ks = pivot_ks[[c for c in op_order if c in pivot_ks.columns]]
    if pivot_ks.size > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 8))
        im = ax1.imshow(pivot_ks.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.25)
        ax1.set_xticks(range(len(pivot_ks.columns)))
        ax1.set_xticklabels(pivot_ks.columns, rotation=30, ha="right")
        ax1.set_yticks(range(len(pivot_ks.index)))
        ax1.set_yticklabels(pivot_ks.index)
        ax1.set_xlabel("operator")
        ax1.set_ylabel("var")
        ax1.set_title("A4 KS: ks_statistic mean (var × operator)")
        plt.colorbar(im, ax=ax1, label="ks_statistic (mean)")
        plt.tight_layout()
        fig1.savefig(FIGURES_DIR / "fig_a4_ks_heatmap_var_operator.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)

    # 图2: 按 setting 分面热力图 — var × operator（z / phys 各一张或并排）
    for setting in df_ks["setting"].unique():
        sub = df_ks[df_ks["setting"] == setting]
        piv = sub.pivot_table(
            values="ks_statistic",
            index="var",
            columns="operator",
            aggfunc="mean",
        )
        piv = piv[[c for c in op_order if c in piv.columns]]
        if piv.size > 0:
            fig, ax = plt.subplots(figsize=(5, 6))
            im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.25)
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index)
            ax.set_title(f"A4 KS: ks_statistic mean (setting={setting})")
            plt.colorbar(im, ax=ax, label="ks_statistic (mean)")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"fig_a4_ks_heatmap_setting_{setting}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 图3: 柱状图 — 各 operator 的 ks_statistic 均值（全表）
    agg_op = df_ks.groupby("operator")["ks_statistic"].agg(["mean", "std", "count"]).reset_index()
    agg_op = agg_op.sort_values("mean")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(agg_op))
    ax3.bar(x, agg_op["mean"], yerr=agg_op["std"], capsize=4, color="steelblue", edgecolor="navy")
    ax3.set_xticks(x)
    ax3.set_xticklabels(agg_op["operator"], rotation=25, ha="right")
    ax3.set_ylabel("ks_statistic (mean ± std)")
    ax3.set_title("A4 KS: by operator (all vars, all settings)")
    ax3.set_ylim(0, None)
    plt.tight_layout()
    fig3.savefig(FIGURES_DIR / "fig_a4_ks_bar_by_operator.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 图4: 柱状图 — 各 var 的 ks_statistic 均值（按算子分色）
    vars_order = df_ks["var"].unique().tolist()
    op_colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(df_ks["operator"].unique()))))
    op_list = df_ks["operator"].unique().tolist()
    op2color = {op: op_colors[i % len(op_colors)] for i, op in enumerate(op_list)}
    pivot_var_op = df_ks.pivot_table(
        values="ks_statistic",
        index="var",
        columns="operator",
        aggfunc="mean",
    )
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    n_ops = len(pivot_var_op.columns)
    w = 0.8 / n_ops
    for i, op in enumerate(pivot_var_op.columns):
        off = (i - n_ops / 2 + 0.5) * w
        ax4.bar(
            np.arange(len(pivot_var_op.index)) + off,
            pivot_var_op[op],
            width=w,
            label=op,
            color=op2color.get(op, "gray"),
        )
    ax4.set_xticks(np.arange(len(pivot_var_op.index)))
    ax4.set_xticklabels(pivot_var_op.index, rotation=40, ha="right")
    ax4.set_ylabel("ks_statistic (mean)")
    ax4.set_title("A4 KS: each var × operator")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_ylim(0, None)
    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "fig_a4_ks_bar_by_var_operator.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print("Figures 已写入:", list(FIGURES_DIR.glob("fig_a4_*.png")))


if __name__ == "__main__":
    main()
