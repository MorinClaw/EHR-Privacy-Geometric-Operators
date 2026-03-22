#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run A2 operators on weak-interp ts_48h and write to results_weak/perturbed.
This is a thin wrapper that calls A2's exp_operators_mimic.run_a2 with a custom data_dir/out_dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from repo_discovery import find_a2_operator_code

A2_DIR = find_a2_operator_code(REPO_ROOT)
if str(A2_DIR) not in sys.path:
    sys.path.insert(0, str(A2_DIR))
from exp_operators_mimic import run_a2, NORMALIZED_MAX_DIFF


def main() -> None:
    p = argparse.ArgumentParser(description="Run A2 operators on ts_48h_weak")
    p.add_argument("--ts-dir", type=Path, required=True, help="Directory: P3_weak_interpolation/ts_48h_weak")
    p.add_argument("--out-dir", type=Path, required=True, help="Output root (e.g. A2_operator_grid/results_weak)")
    p.add_argument("--variables", nargs="+", default=None)
    p.add_argument("--max-diff", type=float, default=None, help="Normalized-space threshold (default: A2 NORMALIZED_MAX_DIFF)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-skip-existing", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.ts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_diff = float(args.max_diff) if args.max_diff is not None else NORMALIZED_MAX_DIFF

    run_a2(
        data_dir=data_dir,
        out_dir=out_dir,
        variables=args.variables,
        global_seed=args.seed,
        max_diff_normalized=max_diff,
        skip_existing=not args.no_skip_existing,
    )
    print("A2 on weak ts done. Outputs at:", out_dir)


if __name__ == "__main__":
    main()

