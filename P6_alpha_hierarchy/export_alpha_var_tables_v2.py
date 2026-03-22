#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export hierarchy:
  level1: alpha_*
  level2: var
  level3: strong/weak
Each leaf has:
  - table_pr.csv  (Attack A privacy metrics)
  - table_ut.csv  (Utility metrics, delta columns on the left)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _alpha_dirname(alpha: float) -> str:
    s = f"{alpha:.1f}".replace(".", "_")
    return f"alpha_{s}"


def _slice_pr(df: pd.DataFrame, var: str, alpha: float, operators: list[str]) -> pd.DataFrame:
    # privacy_evaluation_protocol Attack A table schema:
    # var,setting,operator,leakage,n_train,n_test,model,R2,MAE_z,rho,Hit_*,alpha
    sub = df[(df["var"] == var) & (df["alpha"] == alpha) & (df["operator"].isin(operators))].copy()
    # keep the most standard view: setting=z, model=linear, leakage=L2
    if {"setting", "model", "leakage"}.issubset(sub.columns):
        sub = sub[(sub["setting"] == "z") & (sub["model"] == "linear") & (sub["leakage"] == "L2")].copy()
    keep = [c for c in ["operator", "R2", "MAE_z", "rho", "n_train", "n_test", "setting", "model", "leakage", "alpha"] if c in sub.columns]
    out = sub[keep].sort_values(["operator"])
    out = out.rename(columns={"R2": "attackA_R2", "MAE_z": "attackA_MAEz"})
    return out


def _slice_ut(df: pd.DataFrame, var: str, alpha: float, operators: list[str]) -> pd.DataFrame:
    # var-wise utility table schema (our P4 runner):
    # alpha,operator,setting,task,var,auroc_raw,auprc_raw,auroc_op,auprc_op,delta_*,...,n_total,n_train,n_test
    sub = df[(df["alpha"] == alpha) & (df["var"] == var) & (df["operator"].isin(operators))].copy()
    # default task focus: los_binary
    if "task" in sub.columns:
        sub = sub[sub["task"] == "los_binary"].copy()
    # reorder: delta on the left
    cols = []
    for c in ["operator", "delta_auroc", "delta_auprc", "auroc_raw", "auroc_op", "auprc_raw", "auprc_op", "n_total", "n_train", "n_test"]:
        if c in sub.columns:
            cols.append(c)
    out = sub[cols].sort_values(["operator"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--variables", type=str, nargs="+", default=["HR", "MAP", "Glucose"])
    ap.add_argument("--operators", type=str, nargs="+", default=["T1_uniform", "T1_weighted", "T2", "T3"])
    ap.add_argument("--pr-strong-table", type=Path, required=True)
    ap.add_argument("--pr-weak-table", type=Path, required=True)
    ap.add_argument("--ut-strong-varwise", type=Path, required=True)
    ap.add_argument("--ut-weak-varwise", type=Path, required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pr_strong = pd.read_csv(args.pr_strong_table)
    pr_weak = pd.read_csv(args.pr_weak_table)
    ut_strong = pd.read_csv(args.ut_strong_varwise)
    ut_weak = pd.read_csv(args.ut_weak_varwise)

    for a in args.alphas:
        alpha_dir = out_root / _alpha_dirname(a)
        for var in args.variables:
            for flavor in ["strong", "weak"]:
                leaf = alpha_dir / var / flavor
                leaf.mkdir(parents=True, exist_ok=True)

                pr_df = pr_strong if flavor == "strong" else pr_weak
                pr_out = _slice_pr(pr_df, var, a, list(args.operators))
                pr_out.to_csv(leaf / "table_pr.csv", index=False)

                ut_df = ut_strong if flavor == "strong" else ut_weak
                ut_out = _slice_ut(ut_df, var, a, list(args.operators))
                ut_out.to_csv(leaf / "table_ut.csv", index=False)

    print("Exported to", out_root)


if __name__ == "__main__":
    main()

