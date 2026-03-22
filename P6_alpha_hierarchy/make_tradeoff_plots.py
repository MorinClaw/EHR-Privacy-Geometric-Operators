#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make privacy–utility trade-off scatter plots under out_v2 hierarchy.

For each leaf dir:
  out_v2/alpha_*/<var>/{strong,weak}/
expects:
  - table_pr.csv  with columns: operator, attackA_R2 (others ignored)
  - table_ut.csv  with columns: operator, utility_score (others ignored)
Outputs:
  - tradeoff_utility.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def plot_one(leaf: Path, y_col: str, out_name: str) -> None:
    pr_path = leaf / "table_pr.csv"
    ut_path = leaf / "table_ut.csv"
    if not pr_path.exists() or not ut_path.exists() or not HAS_MPL:
        return

    pr = pd.read_csv(pr_path)
    ut = pd.read_csv(ut_path)
    if pr.empty or ut.empty:
        return

    df = pr.merge(ut, on="operator", how="inner")
    if df.empty or "attackA_R2" not in df.columns or y_col not in df.columns:
        return

    df["privacy"] = 1.0 - df["attackA_R2"].astype(float)
    df["utility"] = df[y_col].astype(float)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(0.0, color="#888888", lw=1, ls="--", alpha=0.25)
    ax.axvline(0.0, color="#888888", lw=1, ls="--", alpha=0.3)
    ax.scatter(df["privacy"], df["utility"], s=70, alpha=0.9)

    for _, r in df.iterrows():
        ax.annotate(
            str(r["operator"]).replace("_", "\\n"),
            (float(r["privacy"]), float(r["utility"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Privacy = 1 − AttackA R²  (higher = more private)")
    ax.set_ylabel(f"Utility = {y_col} (higher = better)")
    ax.set_title(f"Trade-off ({leaf.parts[-1]})")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    fig.savefig(leaf / out_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate trade-off plots under out_v2")
    ap.add_argument("--out-v2", type=Path, required=True)
    args = ap.parse_args()

    base = Path(args.out_v2)
    leaves = [p for p in base.glob("alpha_*/*/*") if p.is_dir()]
    for leaf in leaves:
        plot_one(leaf, "utility_score", "tradeoff_utility.png")

    print("Done. Plots written under:", base)


if __name__ == "__main__":
    main()

