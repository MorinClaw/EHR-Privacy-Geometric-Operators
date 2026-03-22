#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export tables into hierarchy:
  level1: alpha_*
  level2: var
Inside var folder: privacy + utility (by operator)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _alpha_dirname(alpha: float) -> str:
    # 1.0 -> alpha_1_0
    s = f"{alpha:.1f}".replace(".", "_")
    return f"alpha_{s}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--variables", type=str, nargs="+", default=["HR", "MAP", "Glucose"])
    ap.add_argument("--operators", type=str, nargs="+", default=["T1_uniform", "T1_weighted", "T2", "T3"])
    ap.add_argument("--privacy-p2-alpha1", type=Path, required=True)
    ap.add_argument("--privacy-p2-alpha2", type=Path, required=True)
    ap.add_argument("--utility-strong-varwise", type=Path, required=True)
    ap.add_argument("--utility-weak-varwise", type=Path, required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # privacy source (P2 attackA)
    priv_by_alpha = {}
    priv_by_alpha[1.0] = pd.read_csv(args.privacy_p2_alpha1)
    priv_by_alpha[2.0] = pd.read_csv(args.privacy_p2_alpha2)

    util_strong = pd.read_csv(args.utility_strong_varwise)
    util_weak = pd.read_csv(args.utility_weak_varwise)

    for a in args.alphas:
        alpha_dir = out_root / _alpha_dirname(a)
        for var in args.variables:
            var_dir = alpha_dir / var
            var_dir.mkdir(parents=True, exist_ok=True)

            # privacy (if available)
            if a in priv_by_alpha:
                p = priv_by_alpha[a]
                p = p[(p["var"] == var) & (p["operator"].isin(args.operators))].copy()
                p = p[["operator", "R2", "mse", "num_points", "agg"]].rename(
                    columns={"R2": "attackA_R2", "mse": "attackA_mse"}
                )
                p.to_csv(var_dir / "table_privacy_p2_attackA.csv", index=False)

            # utility (alpha=1 only currently, but export if matches)
            for name, df in [("strong", util_strong), ("weak", util_weak)]:
                sub = df[(df["alpha"] == a) & (df["var"] == var) & (df["operator"].isin(args.operators))].copy()
                if sub.empty:
                    continue
                sub = sub[[
                    "operator", "task",
                    "auroc_raw", "auprc_raw", "auroc_op", "auprc_op",
                    "delta_auroc", "delta_auprc",
                    "n_total", "n_train", "n_test",
                ]]
                sub.to_csv(var_dir / f"table_utility_{name}_varwise.csv", index=False)

    print("Exported to", out_root)


if __name__ == "__main__":
    main()

