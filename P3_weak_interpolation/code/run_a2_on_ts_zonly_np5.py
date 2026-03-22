#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic A2 runner (z-only, n_passes=5 only) on a ts_48h directory.

Input:
  ts_dir must contain ts_single_column_{var}_zscore.csv
Output:
  out_dir/perturbed/z/{var}/{stem}.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from repo_discovery import find_a2_operator_code

A2_CODE = find_a2_operator_code(REPO_ROOT)
if str(A2_CODE) not in sys.path:
    sys.path.insert(0, str(A2_CODE))

from numeric_operators import (
    triplet_micro_rotation,
    triplet_micro_rotation_weighted,
    constrained_noise_projection,
)


def seed_from_var_setting(var: str, setting: str, global_seed: int = 42) -> int:
    h = hashlib.sha256(f"{var}_{setting}_{global_seed}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def stem_from_operator(operator: str, alpha: float) -> str:
    if operator in ("T1_uniform", "T1_weighted"):
        return f"{operator}_n_passes=5_max_diff={alpha}"
    return f"{operator}_max_diff={alpha}"


def load_single_column_z(ts_dir: Path, var: str) -> np.ndarray:
    path = ts_dir / f"ts_single_column_{var}_zscore.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run A2 z-only (n_passes=5) on a ts_48h directory")
    ap.add_argument("--ts-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="output root (will create perturbed/z)")
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--operators", nargs="+", default=["T1_uniform", "T1_weighted", "T2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-skip-existing", action="store_true")
    args = ap.parse_args()

    ts_dir = Path(args.ts_dir)
    out_root = Path(args.out_dir)
    pert_dir = out_root / "perturbed" / "z"
    pert_dir.mkdir(parents=True, exist_ok=True)
    skip = not args.no_skip_existing

    for var in args.variables:
        x = load_single_column_z(ts_dir, var)
        x = np.asarray(x, dtype=float)
        nan_mask = np.isnan(x)
        if np.any(~nan_mask):
            x_filled = x.copy()
            x_filled[nan_mask] = np.nanmean(x[~nan_mask])
        else:
            x_filled = np.zeros_like(x)

        base = pert_dir / var
        base.mkdir(parents=True, exist_ok=True)
        seed = seed_from_var_setting(var, "z", args.seed)

        def to_output(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            y[nan_mask] = np.nan
            return y

        if "T1_uniform" in args.operators:
            stem = stem_from_operator("T1_uniform", args.alpha)
            out_path = base / f"{stem}.csv"
            if (not skip) or (not out_path.exists()):
                y = triplet_micro_rotation(
                    x_filled.copy(),
                    max_diff=args.alpha,
                    n_passes=5,
                    x_original=x_filled.copy(),
                    rng=np.random.default_rng(seed),
                )
                pd.DataFrame({"value": to_output(y)}).to_csv(out_path, index=False)
                print(f"[z-only] {var} T1_uniform -> {out_path.name}")

        if "T1_weighted" in args.operators:
            stem = stem_from_operator("T1_weighted", args.alpha)
            out_path = base / f"{stem}.csv"
            if (not skip) or (not out_path.exists()):
                y = triplet_micro_rotation_weighted(
                    x_filled.copy(),
                    max_diff=args.alpha,
                    n_passes=5,
                    x_original=x_filled.copy(),
                    rng=np.random.default_rng(seed),
                )
                pd.DataFrame({"value": to_output(y)}).to_csv(out_path, index=False)
                print(f"[z-only] {var} T1_weighted -> {out_path.name}")

        if "T2" in args.operators:
            stem = stem_from_operator("T2", args.alpha)
            out_path = base / f"{stem}.csv"
            if (not skip) or (not out_path.exists()):
                y = constrained_noise_projection(x_filled.copy(), max_diff=args.alpha, rng=np.random.default_rng(seed))
                pd.DataFrame({"value": to_output(y)}).to_csv(out_path, index=False)
                print(f"[z-only] {var} T2 -> {out_path.name}")

    print("Done. Outputs at:", out_root)


if __name__ == "__main__":
    main()

