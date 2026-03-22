#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run privacy_evaluation_protocol Attack B/C/D for existing Q-mixing pilot outputs.

Given qmix results folder created by run_qmix_pilot.py:
  results_alpha123_full_seedXXXX/alpha_{a}_{b}/a2_qmix_{regime}_alpha={alpha}/perturbed/...

This script will run:
  run_privacy_protocol.py --attack B C D

and write:
  <alpha_dir>/bcd_{regime}/tables/table_attack_b_linkage.csv
  <alpha_dir>/bcd_{regime}/tables/table_attack_c_membership.csv
  <alpha_dir>/bcd_{regime}/tables/table_attack_d_attribute.csv
  <alpha_dir>/bcd_{regime}/tables/table_attack_d_distinguish.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_alpha_dir(alpha: float) -> str:
    a = int(alpha)
    b = int(round((alpha - a) * 10))
    return f"alpha_{a}_{b}"


def run_cmd(cmd: str) -> None:
    import subprocess

    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Attack B/C/D for Q-mixing pilot results")
    ap.add_argument(
        "--qmix-root",
        type=Path,
        required=True,
        help="Path to the qmix pilot results root directory",
    )
    ap.add_argument("--raw-ts-dir", type=Path, required=True, help="Path to the raw time-series directory (ts_48h)")
    ap.add_argument("--weak-ts-dir", type=Path, required=True, help="Path to the weak time-series directory (ts_48h_weak)")
    ap.add_argument(
        "--cohort-csv",
        type=Path,
        required=False,
        help="Optional cohort CSV (accepted for interface compatibility; not used by this script)",
    )
    ap.add_argument("--variables", nargs="+", default=["HR", "Glucose"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=42, help="privacy protocol seed")
    args = ap.parse_args()

    # Repository root for locating privacy_evaluation_protocol scripts.
    repo_root = Path(__file__).resolve().parents[2]
    run_attack = repo_root / "privacy_evaluation_protocol" / "code" / "run_privacy_protocol.py"
    raw_ts = Path(args.raw_ts_dir)
    weak_ts = Path(args.weak_ts_dir)

    for alpha in args.alphas:
        alpha_dir_name = parse_alpha_dir(alpha)
        alpha_dir = args.qmix_root / alpha_dir_name
        if not alpha_dir.exists():
            raise FileNotFoundError(f"Missing alpha_dir: {alpha_dir}")

        for regime, ts_dir in [("strong", raw_ts), ("weak", weak_ts)]:
            a2_out = alpha_dir / f"a2_qmix_{regime}_alpha={alpha}"
            perturbed_root = a2_out / "perturbed"
            if not perturbed_root.exists():
                raise FileNotFoundError(f"Missing perturbed: {perturbed_root}")

            out_dir = alpha_dir / f"bcd_{regime}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Run B/C/D only. Leave Attack A excluded.
            # NOTE: protocol will try both setting z/phys; since we only generated perturbed/z,
            # phys rows should be empty and will not dominate.
            cmd = (
                f"PYTHONUNBUFFERED=1 python -u \"{run_attack}\" "
                f"--attack B C D "
                f"--data-dir \"{ts_dir}\" "
                f"--perturbed-dir \"{perturbed_root}\" "
                f"--out-dir \"{out_dir}\" "
                f"--variables {' '.join(args.variables)} "
                f"--alpha {alpha} "
                f"--seed {args.seed} "
            )
            run_cmd(cmd)

    print("Done. B/C/D tables written under:", args.qmix_root)


if __name__ == "__main__":
    main()

