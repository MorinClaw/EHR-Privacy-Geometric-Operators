#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 ts_48h（主实验）与 截面数据 在 20% 配对下的 R²，生成对比表与散点图，用于看数据集影响。

用法（在 A7/code 下）：
  python3 compare_20pct_datasets.py
  python3 compare_20pct_datasets.py --main-dir ../results --cross-dir ../results_cross_section
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN = SCRIPT_DIR.parent / "results"
DEFAULT_CROSS = SCRIPT_DIR.parent / "results_cross_section"


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两组 20% 配对 R²")
    parser.add_argument("--main-dir", type=str, default=str(DEFAULT_MAIN), help="主实验 results 目录")
    parser.add_argument("--cross-dir", type=str, default=str(DEFAULT_CROSS), help="对比实验 results 目录")
    parser.add_argument("--suffix", type=str, default="ts48h_vs_cross_section", help="输出表/图文件名后缀")
    parser.add_argument("--label-left", type=str, default="ts_48h", help="主实验在图上的标签")
    parser.add_argument("--label-right", type=str, default="cross_section", help="对比实验在图上的标签")
    args = parser.parse_args()
    main_dir = Path(args.main_dir)
    cross_dir = Path(args.cross_dir)
    col_left, col_right = "R2_left", "R2_right"

    path_main = main_dir / "tables" / "table_reconstruction_20pct.csv"
    path_cross = cross_dir / "tables" / "table_reconstruction_20pct.csv"
    if not path_main.exists():
        print(f"未找到: {path_main}")
        return
    if not path_cross.exists():
        print(f"未找到: {path_cross}")
        return

    df_main = pd.read_csv(path_main)
    df_cross = pd.read_csv(path_cross)
    df_main = df_main[df_main["model"] == "linear"][["var", "setting", "operator", "R2"]].copy()
    df_cross = df_cross[df_cross["model"] == "linear"][["var", "setting", "operator", "R2"]].copy()
    df_main = df_main.rename(columns={"R2": col_left})
    df_cross = df_cross.rename(columns={"R2": col_right})
    merge_cols = ["var", "setting", "operator"]
    df = df_main.merge(df_cross, on=merge_cols, how="inner")
    df["R2_diff"] = df[col_right] - df[col_left]
    lname = args.label_left.replace(" ", "_")
    rname = args.label_right.replace(" ", "_")
    df_out = df.rename(columns={col_left: f"R2_{lname}", col_right: f"R2_{rname}"})

    out_table = main_dir / "tables" / f"table_compare_20pct_{args.suffix}.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_table, index=False)
    print(f"对比表: {out_table} ({len(df_out)} 行)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过对比图")
        return

    figs_dir = main_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    colors = ["C0", "C1", "C2", "C3"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub = df[df["setting"] == setting]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, op in enumerate(ops):
            s = sub[sub["operator"] == op]
            if s.empty:
                continue
            ax.scatter(
                s[col_left],
                s[col_right],
                label=op,
                color=colors[i % len(colors)],
                alpha=0.8,
            )
        lims = [0, 1.02]
        ax.plot(lims, lims, "k--", alpha=0.5, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"R² ({args.label_left} 20%)")
        ax.set_ylabel(f"R² ({args.label_right} 20%)")
        ax.set_title(f"20% paired R² comparison ({title_suffix})\nPoints above diagonal = right easier to reconstruct")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_compare_20pct_{args.suffix}_{setting}.pdf")
        plt.close()
        print(f"图: plot_compare_20pct_{args.suffix}_{setting}.pdf")
    print("完成。")


if __name__ == "__main__":
    main()
