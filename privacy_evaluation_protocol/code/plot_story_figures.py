#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story-driven visualizations for privacy_evaluation_protocol outputs.

Inputs (CSV):
  - results/tables/table_attack_a_reconstruction.csv
  - results/tables/table_attack_b_linkage.csv
  - results/tables/table_attack_c_membership.csv
  - results/tables/table_attack_d_attribute.csv
  - results/tables/table_attack_d_distinguish.csv

Outputs (PDF, English-only text):
  - results/figs/story_attackA_R2_vs_leakage_<var>_<setting>.pdf
  - results/figs/story_attackA_privacy_by_operator_<var>_<setting>.pdf
  - results/figs/story_attackB_Reid1_<var>_m<m>_<setting>.pdf
  - results/figs/story_attackC_MI_AUC_<var>_<setting>.pdf
  - results/figs/story_attackD_attribute_R2_<var>_<setting>.pdf
  - results/figs/story_attackD_distinguish_advantage_<var>.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "results"


LEAKAGE_ORDER = ["L0", "L1", "L2", "L3"]
OP_ORDER = ["T1_uniform", "T1_weighted", "T2", "T3"]
ATTR_ORDER = ["max", "min", "value_t24", "above_p90"]


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figs_dir


def _mpl() -> tuple[object, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Consistent, publication-friendly styling (no Chinese text)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
        }
    )
    return matplotlib, plt


def plot_attack_a_leakage_curve(
    out_dir: Path,
    var: str,
    setting: str,
    model: str = "linear",
) -> None:
    """Story A1: leakage → reconstruction R² (one variable, one setting)."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_a_reconstruction.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[(df["var"] == var) & (df["setting"] == setting) & (df["model"] == model)].copy()
    if sub.empty:
        return
    sub = sub[sub["operator"].isin(OP_ORDER)]
    sub["leakage"] = pd.Categorical(sub["leakage"], LEAKAGE_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for op in OP_ORDER:
        d = sub[sub["operator"] == op].sort_values("leakage")
        if d.empty:
            continue
        ax.plot(d["leakage"], d["R2"], marker="o", linewidth=2, label=op)
    ax.set_xlabel("Leakage level")
    ax.set_ylabel("R² (reconstruction; higher = easier to invert)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Attack A: Reconstruction R² vs leakage ({var}, {setting}, {model})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackA_R2_vs_leakage_{var}_{setting}.pdf")
    plt.close(fig)




def plot_attack_a_privacy_bars(
    out_dir: Path,
    var: str,
    setting: str,
    leakages: list[str] = ["L1", "L2"],
    model: str = "linear",
    metric: str = "R2",
) -> None:
    """Story A2: operator → privacy summary (1-R² or 1-Hit_0.1) under selected leakages."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_a_reconstruction.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[
        (df["var"] == var)
        & (df["setting"] == setting)
        & (df["model"] == model)
        & (df["operator"].isin(OP_ORDER))
        & (df["leakage"].isin(leakages))
    ].copy()
    if sub.empty:
        return

    # privacy = 1 - metric (for R2 / Hit_*). For MAE_z, privacy = MAE_z.
    if metric.startswith("Hit_") or metric == "R2":
        sub["privacy"] = 1.0 - sub[metric].astype(float)
        ylab = f"1 - {metric} (higher = more privacy)"
    else:
        sub["privacy"] = sub[metric].astype(float)
        ylab = f"{metric} (higher = more privacy)"

    sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
    sub["leakage"] = pd.Categorical(sub["leakage"], leakages, ordered=True)
    piv = sub.pivot_table(index="operator", columns="leakage", values="privacy", aggfunc="mean").reindex(OP_ORDER)

    x = np.arange(len(piv.index))
    w = 0.35 if len(leakages) <= 2 else 0.25
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, lk in enumerate(leakages):
        vals = piv[lk].values if lk in piv.columns else np.full(len(x), np.nan)
        ax.bar(x + (i - (len(leakages) - 1) / 2) * w, vals, width=w, label=lk)

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel(ylab)
    ax.set_title(f"Attack A: Privacy summary by operator ({var}, {setting}, {model})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Leakage", frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackA_privacy_by_operator_{var}_{setting}.pdf")
    plt.close(fig)


def plot_attack_b_reid1(
    out_dir: Path,
    var: str,
    setting: str,
    m: int = 10,
) -> None:
    """Story B: re-identification close to random baseline 1/m."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_b_linkage.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[(df["var"] == var) & (df["setting"] == setting) & (df["m"] == m)].copy()
    if sub.empty:
        return
    sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
    sub = sub.sort_values("operator")

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(sub["operator"].astype(str).tolist(), sub["Reid_at_1"].astype(float).values, color="C0", alpha=0.85)
    ax.axhline(1.0 / m, color="black", linestyle="--", linewidth=1.2, label=f"random (1/{m})")
    ax.set_ylabel("Re-id@1")
    ax.set_xlabel("Operator")
    ax.set_ylim(0, max(0.2, float(np.nanmax(sub["Reid_at_1"]) + 0.05)))
    ax.set_title(f"Attack B: Re-identification@1 ({var}, m={m}, {setting})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackB_Reid1_{var}_m{m}_{setting}.pdf")
    plt.close(fig)


def plot_attack_c_membership_auc(
    out_dir: Path,
    var: str,
    setting: str,
) -> None:
    """Story C: membership inference AUC ~ 0.5."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_c_membership.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[(df["var"] == var) & (df["setting"] == setting)].copy()
    if sub.empty:
        return
    sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
    sub = sub.sort_values("operator")

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(sub["operator"].astype(str).tolist(), sub["AUC"].astype(float).values, color="C1", alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="random (0.5)")
    ax.set_ylabel("AUC (membership inference)")
    ax.set_xlabel("Operator")
    ax.set_ylim(0.35, 0.65)
    ax.set_title(f"Attack C: Membership inference AUC ({var}, {setting})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackC_MI_AUC_{var}_{setting}.pdf")
    plt.close(fig)


def plot_attack_d_attribute_r2(
    out_dir: Path,
    var: str,
    setting: str,
) -> None:
    """Story D1: attribute inference (R²) across attributes and operators."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_d_attribute.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[(df["var"] == var) & (df["setting"] == setting)].copy()
    if sub.empty:
        return
    sub = sub[sub["operator"].isin(OP_ORDER) & sub["attr"].isin(ATTR_ORDER)]
    if sub.empty:
        return
    sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
    sub["attr"] = pd.Categorical(sub["attr"], ATTR_ORDER, ordered=True)

    piv = sub.pivot_table(index="attr", columns="operator", values="R2", aggfunc="mean").reindex(ATTR_ORDER)
    x = np.arange(len(piv.index))
    w = 0.18
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    for i, op in enumerate(OP_ORDER):
        vals = piv[op].values if op in piv.columns else np.full(len(x), np.nan)
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=op)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.tolist())
    ax.set_ylabel("R² (attribute inference; higher = more leakage)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Attack D: Attribute inference ({var}, {setting})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackD_attribute_R2_{var}_{setting}.pdf")
    plt.close(fig)


def plot_attack_d_distinguish_advantage(out_dir: Path, var: str) -> None:
    """Story D2: IND-style pairwise distinguishing advantage (|acc-0.5|)."""
    tables_dir, figs_dir = _ensure_dirs(out_dir)
    path = tables_dir / "table_attack_d_distinguish.csv"
    if not path.exists():
        return
    _, plt = _mpl()

    df = pd.read_csv(path)
    sub = df[df["var"] == var].copy()
    if sub.empty:
        return
    sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
    sub["setting"] = pd.Categorical(sub["setting"], ["z", "phys"], ordered=True)
    sub = sub.sort_values(["setting", "operator"])

    labels = [f"{op}-{st}" for op, st in zip(sub["operator"].astype(str), sub["setting"].astype(str))]
    vals = sub["advantage"].astype(float).values

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(labels, vals, color="C2", alpha=0.85)
    ax.set_ylabel("Advantage |acc - 0.5|")
    ax.set_title(f"Attack D: Pairwise indistinguishability advantage ({var})")
    ax.grid(axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(figs_dir / f"story_attackD_distinguish_advantage_{var}.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate story-driven figures (English-only).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Protocol results directory")
    p.add_argument("--var", type=str, default="HR", help="Variable name (default: HR)")
    p.add_argument(
        "--vars-multi",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of variables for multi-variable comparison plots (Attack A/D).",
    )
    p.add_argument("--setting", type=str, default=None, choices=[None, "z", "phys"], help="If set, only plot one setting")
    p.add_argument("--m", type=int, default=10, help="Candidate size for Attack B (default: 10)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    settings = [args.setting] if args.setting else ["z", "phys"]

    # Single-variable story figures (HR by default)
    for st in settings:
        plot_attack_a_leakage_curve(out_dir, var=args.var, setting=st, model="linear")
        plot_attack_a_privacy_bars(
            out_dir, var=args.var, setting=st, leakages=["L1", "L2"], model="linear", metric="R2"
        )
        plot_attack_b_reid1(out_dir, var=args.var, setting=st, m=args.m)
        plot_attack_c_membership_auc(out_dir, var=args.var, setting=st)
        plot_attack_d_attribute_r2(out_dir, var=args.var, setting=st)

    plot_attack_d_distinguish_advantage(out_dir, var=args.var)

    # Multi-variable Attack A comparison: HR + MAP + Glucose (if available)
    vars_multi = args.vars_multi or ["HR", "MAP", "Glucose"]
    try:
        tables_dir, figs_dir = _ensure_dirs(out_dir)
        path_a = tables_dir / "table_attack_a_reconstruction.csv"
        if path_a.exists():
            _, plt = _mpl()
            df_a = pd.read_csv(path_a)
            for st in settings:
                sub = df_a[
                    (df_a["setting"] == st)
                    & (df_a["model"] == "linear")
                    & (df_a["leakage"] == "L2")  # focus on 20% pairs for the main story
                    & (df_a["var"].isin(vars_multi))
                    & (df_a["operator"].isin(OP_ORDER))
                ].copy()
                if sub.empty:
                    continue
                sub["operator"] = pd.Categorical(sub["operator"], OP_ORDER, ordered=True)
                sub["var"] = pd.Categorical(sub["var"], vars_multi, ordered=True)
                # Use 1 - R² as privacy-oriented error metric
                sub["one_minus_R2"] = 1.0 - sub["R2"].astype(float)
                piv = sub.pivot_table(
                    index="operator", columns="var", values="one_minus_R2", aggfunc="mean"
                ).reindex(OP_ORDER)
                x = np.arange(len(piv.index))
                fig, ax = plt.subplots(figsize=(6.0, 3.2))
                for i, vname in enumerate(vars_multi):
                    vals = piv[vname].values if vname in piv.columns else np.full(len(x), np.nan)
                    ax.plot(
                        x,
                        vals,
                        marker="o",
                        linewidth=2,
                        label=vname,
                    )
                ax.set_xticks(x)
                ax.set_xticklabels(piv.index.tolist(), rotation=20, ha="right")
                ax.set_ylabel("1 - R² (reconstruction error; 20% pairs)")
                ax.set_ylim(0.0, 0.3)
                ax.set_xlabel("Operator")
                ax.set_title(f"Attack A: 1 - R² across variables (L2, {st})")
                ax.grid(alpha=0.3)
                ax.legend(frameon=False, ncol=len(vars_multi))
                fig.tight_layout()
                fig.savefig(figs_dir / f"story_attackA_R2_multi_var_L2_{st}.pdf")
                plt.close(fig)

        # Multi-variable attribute inference comparison: HR/MAP/Glucose × (max, above_p90)
        path_d = tables_dir / "table_attack_d_attribute.csv"
        if path_d.exists():
            _, plt = _mpl()
            df_d = pd.read_csv(path_d)
            attrs_focus = ["max", "above_p90"]
            for st in settings:
                sub = df_d[
                    (df_d["setting"] == st)
                    & (df_d["var"].isin(vars_multi))
                    & (df_d["operator"].isin(OP_ORDER))
                    & (df_d["attr"].isin(attrs_focus))
                ].copy()
                if sub.empty:
                    continue
                # index: (var, attr), columns: operator, values: R2
                sub["var_attr"] = sub["var"] + " / " + sub["attr"]
                idx_order = []
                for v in vars_multi:
                    for a in attrs_focus:
                        va = f"{v} / {a}"
                        if va in sub["var_attr"].values:
                            idx_order.append(va)
                if not idx_order:
                    continue
                # Use 1 - R² as privacy-oriented error metric
                sub["one_minus_R2"] = 1.0 - sub["R2"].astype(float)
                piv = sub.pivot_table(
                    index="var_attr", columns="operator", values="one_minus_R2", aggfunc="mean"
                ).reindex(idx_order)
                x = np.arange(len(piv.index))
                w = 0.18
                fig, ax = plt.subplots(figsize=(6.5, 3.4))
                for i, op in enumerate(OP_ORDER):
                    vals = piv[op].values if op in piv.columns else np.full(len(x), np.nan)
                    ax.bar(x + (i - 1.5) * w, vals, width=w, label=op)
                ax.set_xticks(x)
                ax.set_xticklabels(piv.index.tolist(), rotation=25, ha="right")
                ax.set_ylabel("1 - R² (attribute inference error)")
                ax.set_ylim(0.0, 0.3)
                ax.set_title(f"Attack D: Attribute inference across variables ({st})")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(frameon=False, ncol=2)
                fig.tight_layout()
                fig.savefig(figs_dir / f"story_attackD_attribute_R2_multi_var_{st}.pdf")
                plt.close(fig)
    except Exception as e:
        # plotting should be best-effort; do not crash on multi-var figures
        print("Warning while generating multi-variable story figures:", e)

    print("Done. Figures saved to", out_dir / "figs")


if __name__ == "__main__":
    main()

