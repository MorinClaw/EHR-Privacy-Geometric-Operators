#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Privacy Evaluation Protocol entry point.

This script runs the full attack suite (A-D) for privacy evaluation using:
  - a time-series directory (ts_48h, or a directory containing ts_single_column_*.csv)
  - a perturbed outputs directory produced by the operator pipelines (A2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same-directory modules
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    LEAKAGE_LEVELS,
    LINKAGE_CANDIDATE_SIZES,
    ALPHAS,
    DEFAULT_ALPHA,
    ATTACK_A_RECON,
    ATTACK_B_LINKAGE,
    ATTACK_C_MEMBERSHIP,
    ATTACK_D_ATTRIBUTE,
)
from attack_a_reconstruction import run_attack_a
from attack_b_linkage import run_attack_b
from attack_c_membership import run_attack_c
from attack_d_attribute import run_attack_d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Privacy evaluation protocol: attacks A-D with configurable leakage/candidate sizes. Outputs tables and figures."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing ts_48h_* data (ts_48h and/or ts_single_column_*.csv)",
    )
    parser.add_argument(
        "--perturbed-dir",
        type=Path,
        required=True,
        help="Directory containing perturbed outputs (A2 results/perturbed)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for tables and figures (default: <script_parent>/results)",
    )
    parser.add_argument(
        "--attack",
        type=str,
        nargs="+",
        default=[ATTACK_A_RECON, ATTACK_B_LINKAGE, ATTACK_C_MEMBERSHIP, ATTACK_D_ATTRIBUTE],
        choices=[ATTACK_A_RECON, ATTACK_B_LINKAGE, ATTACK_C_MEMBERSHIP, ATTACK_D_ATTRIBUTE],
        help="Which attacks to run",
    )
    parser.add_argument(
        "--leakage",
        type=str,
        nargs="+",
        default=list(LEAKAGE_LEVELS),
        help="Leakage levels used for Attack A reconstruction (L0/L1/L2/L3)",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        nargs="+",
        default=list(LINKAGE_CANDIDATE_SIZES),
        help="Candidate set size m for Attack B record linkage",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="*",
        default=None,
        help="Restrict evaluation to specific variables (default: all available)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Normalization-space max_diff (must match the operator perturbation used to generate perturbed outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    perturbed_dir = Path(args.perturbed_dir)
    out_dir = args.out_dir or (SCRIPT_DIR.parent / "results")

    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}")
        sys.exit(1)
    if not perturbed_dir.exists():
        print(f"Error: perturbed directory does not exist: {perturbed_dir}")
        sys.exit(1)

    variables = args.variables if args.variables else None
    alpha = args.alpha
    seed = args.seed

    if ATTACK_A_RECON in args.attack:
        run_attack_a(
            data_dir,
            perturbed_dir,
            out_dir,
            variables=variables,
            leakages=tuple(args.leakage),
            alpha=alpha,
            seed=seed,
        )
    if ATTACK_B_LINKAGE in args.attack:
        run_attack_b(
            data_dir,
            perturbed_dir,
            out_dir,
            variables=variables,
            candidate_sizes=tuple(args.candidate_size),
            alpha=alpha,
            seed=seed,
        )
    if ATTACK_C_MEMBERSHIP in args.attack:
        run_attack_c(
            data_dir,
            perturbed_dir,
            out_dir,
            variables=variables,
            alpha=alpha,
            seed=seed,
        )
    if ATTACK_D_ATTRIBUTE in args.attack:
        run_attack_d(
            data_dir,
            perturbed_dir,
            out_dir,
            variables=variables,
            alpha=alpha,
            seed=seed,
        )

    print(f"\nDone. Results written to: {out_dir / 'tables'}")


if __name__ == "__main__":
    main()
