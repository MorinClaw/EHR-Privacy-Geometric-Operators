#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 A3 总表 table_operator_sanity.csv 生成：
1. summary 汇总表 (table_a3_summary.csv)
2. 总表可视化图 (figures/)
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np

# 尝试导入 matplotlib，若无则只生成表
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


def parse_max_diff(params: str) -> float:
    """从 params 解析 max_diff。"""
    if pd.isna(params):
        return np.nan
    m = re.search(r"max_diff=([\d.]+)", str(params))
    return float(m.group(1)) if m else np.nan


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    path_full = TABLES_DIR / "table_operator_sanity.csv"
    if not path_full.exists():
        print(f"未找到: {path_full}")
        return
    df = pd.read_csv(path_full)

    # 解析 max_diff 便于按算子与设定汇总
    df["max_diff_param"] = df["params"].apply(parse_max_diff)

    # ---------- 1. Summary 表 ----------
    # 1a) 按 setting 汇总：delta_mean / delta_var 的范围（量级）
    summary_setting = (
        df.groupby("setting", as_index=False)
        .agg(
            delta_mean_min=("delta_mean", "min"),
            delta_mean_max=("delta_mean", "max"),
            delta_var_min=("delta_var", "min"),
            delta_var_max=("delta_var", "max"),
            n_rows=("delta_mean", "count"),
        )
    )
    summary_setting["说明"] = summary_setting["setting"].map(
        lambda s: "z-setting: 均值/方差变化量级 10^-17~10^-15"
        if s == "z"
        else "phys-setting: 均值/方差变化量级 10^-16~10^-13"
    )

    # 1b) 按 setting × operator 汇总
    summary_op = (
        df.groupby(["setting", "operator"], as_index=False)
        .agg(
            delta_mean_max=("delta_mean", "max"),
            delta_var_max=("delta_var", "max"),
            max_abs_delta_mean=("max_abs_delta", "mean"),
            max_abs_delta_max=("max_abs_delta", "max"),
            unchanged_ratio_max=("unchanged_ratio", "max"),
            n=("delta_mean", "count"),
        )
    )

    # 1c) 按 operator × max_diff 汇总 max_abs_delta（验证与 max_diff 一致）
    df_t1 = df[df["operator"].str.startswith("T1")].copy()
    df_t1["max_diff_param"] = df_t1["params"].apply(parse_max_diff)
    summary_maxdiff = (
        df.groupby(["operator", "max_diff_param"], as_index=False)
        .agg(
            max_abs_delta_mean=("max_abs_delta", "mean"),
            max_abs_delta_min=("max_abs_delta", "min"),
            max_abs_delta_max=("max_abs_delta", "max"),
            n=("max_abs_delta", "count"),
        )
    ).dropna(subset=["max_diff_param"])

    # 写入 summary 表（多 sheet 用多个 csv 表示）
    summary_setting.to_csv(TABLES_DIR / "table_a3_summary_by_setting.csv", index=False, encoding="utf-8-sig")
    summary_op.to_csv(TABLES_DIR / "table_a3_summary_by_operator.csv", index=False, encoding="utf-8-sig")
    summary_maxdiff.to_csv(TABLES_DIR / "table_a3_summary_by_maxdiff.csv", index=False, encoding="utf-8-sig")

    # 一张总 summary：关键结论表（与您图片中的三点对应）
    conclusion_rows = [
        {
            "指标": "△mean / △var 量级",
            "z-setting": "10^-17 ~ 10^-15",
            "phys-setting": "10^-16 ~ 10^-13",
            "结论": "与机器精度一致，理论均值/方差保持得到满足",
        },
        {
            "指标": "unchanged_ratio",
            "z-setting": "基本为 0，个别 T1_weighted 小 n_passes 为 10^-5~10^-2",
            "phys-setting": "同上",
            "结论": "绝大多数点均被扰动，全变性满足",
        },
        {
            "指标": "max_abs_delta",
            "z-setting": "T1_uniform: max_diff=0.5→约0.5, 1.0→约1.0; T2 稍小; T3 约 0.01~0.04",
            "phys-setting": "与设定 max_diff 一致",
            "结论": "扰动幅度在设计范围内",
        },
    ]
    pd.DataFrame(conclusion_rows).to_csv(
        TABLES_DIR / "table_a3_summary_conclusion.csv", index=False, encoding="utf-8-sig"
    )

    print("Summary 表已写入:", list(TABLES_DIR.glob("table_a3_summary*.csv")))

    # ---------- 2. 可视化 ----------
    if not HAS_PLOT:
        print("未安装 matplotlib，跳过作图")
        return

    # 图1: delta_mean, delta_var 按 setting 分布（箱线图）
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(
        axes,
        ["delta_mean", "delta_var"],
        [r"$\Delta$mean (|mean(x)-mean(y)|)", r"$\Delta$var (|Var(x)-Var(y)|)"],
    ):
        data_by_setting = [df.loc[df["setting"] == s, col].values for s in df["setting"].unique()]
        bp = ax.boxplot(data_by_setting, labels=df["setting"].unique().tolist(), patch_artist=True)
        ax.set_title(title)
        ax.set_xlabel("setting")
        ax.set_ylabel(col)
    plt.suptitle("A3 theory check: delta mean/var by setting")
    plt.tight_layout()
    fig1.savefig(FIGURES_DIR / "fig_a3_delta_mean_var_by_setting.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 图2: max_abs_delta 按 operator 与 max_diff 分组（柱状图，取各组合均值）
    agg_max = (
        df.groupby(["operator", "max_diff_param"])["max_abs_delta"]
        .mean()
        .reset_index()
    )
    agg_max = agg_max.dropna(subset=["max_diff_param"])
    piv = agg_max.pivot(index="operator", columns="max_diff_param", values="max_abs_delta")
    # 列顺序固定便于对比
    piv = piv.reindex(columns=sorted([c for c in piv.columns if not np.isnan(c)]))
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    piv.plot(kind="bar", ax=ax2, width=0.75)
    ax2.set_title("A3 theory check: max_abs_delta mean by operator and max_diff")
    ax2.set_ylabel("max_abs_delta (mean)")
    ax2.set_xlabel("operator")
    ax2.legend(title="max_diff")
    ax2.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig2.savefig(FIGURES_DIR / "fig_a3_max_abs_delta_by_operator.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 图3: unchanged_ratio 分布（仅显示 >0 的以便观察）
    df_nonzero = df[df["unchanged_ratio"] > 0].copy()
    if len(df_nonzero) > 0:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df_nonzero.plot(x="operator", y="unchanged_ratio", kind="bar", ax=ax3, legend=False)
        ax3.set_title("A3 theory check: unchanged_ratio > 0 (some T1_weighted low n_passes)")
        ax3.set_ylabel("unchanged_ratio")
        ax3.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        fig3.savefig(FIGURES_DIR / "fig_a3_unchanged_ratio_nonzero.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

    # 图4: 总表热力图风格——按 (setting, operator) 的 delta_mean 与 max_abs_delta 均值
    pivot_dmean = df.pivot_table(
        values="delta_mean", index="operator", columns="setting", aggfunc="max"
    )
    pivot_mad = df.pivot_table(
        values="max_abs_delta", index="operator", columns="setting", aggfunc="mean"
    )
    fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes4[0].imshow(np.log10(pivot_dmean + 1e-20), aspect="auto", cmap="viridis")
    axes4[0].set_xticks(range(len(pivot_dmean.columns)))
    axes4[0].set_xticklabels(pivot_dmean.columns)
    axes4[0].set_yticks(range(len(pivot_dmean.index)))
    axes4[0].set_yticklabels(pivot_dmean.index, fontsize=8)
    axes4[0].set_title(r"log10($\Delta$mean) max by setting × operator")
    plt.colorbar(im0, ax=axes4[0])
    im1 = axes4[1].imshow(pivot_mad, aspect="auto", cmap="plasma")
    axes4[1].set_xticks(range(len(pivot_mad.columns)))
    axes4[1].set_xticklabels(pivot_mad.columns)
    axes4[1].set_yticks(range(len(pivot_mad.index)))
    axes4[1].set_yticklabels(pivot_mad.index, fontsize=8)
    axes4[1].set_title("max_abs_delta mean by setting × operator")
    plt.colorbar(im1, ax=axes4[1])
    plt.suptitle("A3 summary by setting × operator")
    plt.tight_layout()
    fig4.savefig(FIGURES_DIR / "fig_a3_heatmap_setting_operator.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print(" figures 已写入:", list(FIGURES_DIR.glob("fig_a3_*.png")))


if __name__ == "__main__":
    main()
