#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a q-mixed ts directory for A2 runner input.

We generate per-variable, per-stay orthogonal mixing:
  for each stay_id and variable var:
    x ∈ R^48 (z-scored)  ->  x' = Q(var, stay_id, secret_seed) @ x

Outputs (in out_ts_dir):
  - ts_single_column_{var}_zscore.csv   (required by A2 z-only runner)
Optionally also writes:
  - ts_48h_{var}_zscore.csv (wide) for inspection/debugging

Notes:
  - NaNs are preserved: we mix after imputing NaNs with stay-wise mean, then restore NaN mask.
  - Q is generated deterministically from a seed derived from (secret_seed, stay_id, var).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _seed_u32(secret_seed: int, stay_id: int, var: str) -> int:
    h = hashlib.sha256(f"{secret_seed}|{stay_id}|{var}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _orthogonal_q(rng: np.random.Generator, d: int = 48) -> np.ndarray:
    """Random orthogonal matrix via QR; enforce det=+1 (proper rotation)."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # make diagonal of R positive for determinism
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def load_ts48h_z(raw_ts_dir: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    path = raw_ts_dir / f"ts_48h_{var}_zscore.csv"
    df = pd.read_csv(path)
    stay_ids = df["stay_id"].values.astype(int)
    cols = [c for c in df.columns if c != "stay_id" and str(c).isdigit()]
    cols = sorted(cols, key=int)
    X = df[cols].to_numpy(dtype=float)
    if X.shape[1] != 48:
        raise ValueError(f"Expected 48 columns, got {X.shape} for {path}")
    return X, stay_ids


def qmix_one_var(
    X: np.ndarray,
    stay_ids: np.ndarray,
    var: str,
    secret_seed: int,
) -> np.ndarray:
    n, d = X.shape
    out = np.full_like(X, np.nan, dtype=float)
    for i in range(n):
        sid = int(stay_ids[i])
        x = X[i].astype(float)
        nan_mask = ~np.isfinite(x)
        if nan_mask.all():
            continue
        x_filled = x.copy()
        x_filled[nan_mask] = float(np.nanmean(x_filled[~nan_mask]))
        rng = np.random.default_rng(_seed_u32(secret_seed, sid, var))
        Q = _orthogonal_q(rng, d=d)
        y = Q @ x_filled
        y[nan_mask] = np.nan
        out[i] = y
    return out


def write_single_column(out_ts_dir: Path, var: str, X: np.ndarray, stay_ids: np.ndarray) -> None:
    # flatten in stay order, time order 0..47
    vec = X.reshape(-1)
    out_path = out_ts_dir / f"ts_single_column_{var}_zscore.csv"
    pd.DataFrame({"value": vec}).to_csv(out_path, index=False)

    # also write stay_id order for traceability
    (out_ts_dir / f"stay_ids_{var}.csv").write_text(
        "\n".join(map(str, stay_ids.tolist())) + "\n",
        encoding="utf-8",
    )


def write_wide(out_ts_dir: Path, var: str, X: np.ndarray, stay_ids: np.ndarray) -> None:
    df = pd.DataFrame(X, columns=[str(i) for i in range(48)])
    df.insert(0, "stay_id", stay_ids.astype(int))
    df.to_csv(out_ts_dir / f"ts_48h_{var}_zscore.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build q-mixed ts dir (zscore) for A2 runner")
    ap.add_argument("--raw-ts-dir", type=Path, required=True, help="ts_48h directory containing ts_48h_{var}_zscore.csv")
    ap.add_argument("--out-ts-dir", type=Path, required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--secret-seed", type=int, required=True)
    ap.add_argument("--write-wide", action="store_true", help="also write wide ts_48h_{var}_zscore.csv for inspection")
    args = ap.parse_args()

    out_ts_dir = Path(args.out_ts_dir)
    out_ts_dir.mkdir(parents=True, exist_ok=True)

    for var in args.variables:
        X, stay_ids = load_ts48h_z(Path(args.raw_ts_dir), var)
        Xq = qmix_one_var(X, stay_ids, var=var, secret_seed=int(args.secret_seed))
        write_single_column(out_ts_dir, var, Xq, stay_ids)
        if args.write_wide:
            write_wide(out_ts_dir, var, Xq, stay_ids)
        print(f"[qmix] wrote {var} to {out_ts_dir}")


if __name__ == "__main__":
    main()

