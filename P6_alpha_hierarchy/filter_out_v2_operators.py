#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter an existing out_v2 hierarchy by operator list.

Input hierarchy:
  out_v2/alpha_*/<var>/{strong,weak}/
Each leaf contains:
  - table_pr.csv  (privacy, must contain column: operator)
  - table_ut.csv  (utility, must contain column: operator)

This script writes a NEW hierarchy under --out, with filtered CSVs.
Optionally regenerates trade-off plots by calling make_tradeoff_plots.py.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def _filter_one_csv(src: Path, dst: Path, operators: set[str]) -> bool:
    if not src.exists():
        return False
    df = pd.read_csv(src)
    if df.empty or "operator" not in df.columns:
        return False
    df = df[df["operator"].isin(sorted(operators))].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, required=True, help="Input out_v2 dir")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True, help="Output dir (new hierarchy)")
    ap.add_argument(
        "--operators",
        type=str,
        nargs="+",
        default=["T1_uniform", "T1_weighted"],
        help="Operators to keep (default: first two)",
    )
    ap.add_argument(
        "--regen-plots",
        action="store_true",
        help="Regenerate trade-off pngs under --out using make_tradeoff_plots.py",
    )
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    operators = set(args.operators)

    leaves = [p for p in in_dir.glob("alpha_*/*/*") if p.is_dir()]
    if not leaves:
        raise SystemExit(f"No leaves found under: {in_dir}")

    kept = 0
    for leaf in leaves:
        rel = leaf.relative_to(in_dir)
        out_leaf = out_dir / rel
        ok1 = _filter_one_csv(leaf / "table_pr.csv", out_leaf / "table_pr.csv", operators)
        ok2 = _filter_one_csv(leaf / "table_ut.csv", out_leaf / "table_ut.csv", operators)
        if ok1 or ok2:
            kept += 1

    print(f"Filtered leaves written: {kept}  (operators={sorted(operators)})")
    print(f"Output root: {out_dir}")

    if args.regen_plots:
        script = Path(__file__).with_name("make_tradeoff_plots.py")
        if not script.exists():
            raise SystemExit(f"Cannot find plot script: {script}")
        subprocess.check_call(["python", str(script), "--out-v2", str(out_dir)])


if __name__ == "__main__":
    main()

