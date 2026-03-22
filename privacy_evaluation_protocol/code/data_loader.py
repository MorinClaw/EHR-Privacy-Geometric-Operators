# -*- coding: utf-8 -*-
"""
Data loading utilities shared by all privacy-evaluation attacks.

The module provides:
  - loaders for both ts_48h matrices and ts_single_column vectors
  - stay-based Train/Val/Test splitting used across experiments
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    LEAKAGE_TRAIN_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    WINDOW_HOURS,
    OPERATORS,
    DEFAULT_ALPHA,
)


def _stem_from_operator_params(operator: str, alpha: float) -> str:
    """Operator-specific filename stem used by A2 (e.g., T1_uniform_n_passes=5_max_diff=0.8)."""
    if operator in ("T1_uniform", "T1_weighted"):
        return f"{operator}_n_passes=5_max_diff={alpha}"
    return f"{operator}_max_diff={alpha}"


def load_ts48h_matrix(data_dir: Path, var: str, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single-variable matrix from a ts_48h directory.

    setting in ('z','phys') maps to:
      - ts_48h_{var}_zscore.csv (z)
      - ts_48h_{var}.csv (phys)

    Returns:
      values: (n_stays, 48)
      stay_ids: (n_stays,)
    """
    data_dir = Path(data_dir)
    fname = f"ts_48h_{var}_zscore.csv" if setting == "z" else f"ts_48h_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "stay_id" in df.columns:
        stay_ids = df["stay_id"].values.astype(int)
        cols = [c for c in df.columns if c != "stay_id" and str(c).isdigit()]
        cols = sorted(cols, key=int)
        values = df[cols].values.astype(float)
    else:
        # If stay_id is missing, use row indices as stay identifiers.
        stay_ids = np.arange(len(df))
        cols = [c for c in df.columns if str(c).isdigit()]
        cols = sorted(cols, key=int)
        values = df[cols].values.astype(float)
    return values, stay_ids


def load_single_column_fallback(data_dir: Path, var: str, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a long vector from ts_single_column_*.csv and reshape it into a matrix
    with 48 points per stay. Returns (values, stay_ids).
    """
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    vec = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n = len(vec)
    n_stays = n // WINDOW_HOURS
    if n_stays * WINDOW_HOURS != n:
        n_stays = n_stays + 1
    # Pad or truncate to match WINDOW_HOURS points per stay.
    need = n_stays * WINDOW_HOURS
    if n < need:
        vec = np.concatenate([vec, np.full(need - n, np.nan)])
    else:
        vec = vec[:need]
    values = vec.reshape(n_stays, WINDOW_HOURS)
    stay_ids = np.arange(n_stays)
    return values, stay_ids


def load_raw_matrix(data_dir: Path, var: str, setting: str) -> tuple[np.ndarray, np.ndarray]:
    """Prefer the ts_48h matrix; fall back to ts_single_column vectors if needed."""
    try:
        return load_ts48h_matrix(data_dir, var, setting)
    except FileNotFoundError:
        return load_single_column_fallback(data_dir, var, setting)


def load_perturbed_matrix(
    perturbed_dir: Path,
    setting: str,
    var: str,
    operator: str,
    alpha: float = DEFAULT_ALPHA,
    n_stays: Optional[int] = None,
) -> np.ndarray:
    """
    Load A2 perturbed outputs with the same shape as the raw data: (n_stays, 48).

    A2 stores a single-column long vector with the same ordering as the raw data.
    We reshape it using n_stays (when provided) or len/48.
    """
    stem = _stem_from_operator_params(operator, alpha)
    path = perturbed_dir / setting / var / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    vec = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n = len(vec)
    if n_stays is None:
        n_stays = n // WINDOW_HOURS
    if n_stays * WINDOW_HOURS != n:
        vec = vec[: n_stays * WINDOW_HOURS] if n >= n_stays * WINDOW_HOURS else np.concatenate(
            [vec, np.full(n_stays * WINDOW_HOURS - n, np.nan)]
        )
    return vec.reshape(n_stays, WINDOW_HOURS)


def train_val_test_split_stays(
    stay_ids: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split stay IDs into train/val/test according to ratios defined in config."""
    n = len(stay_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_test = n - n_train - n_val
    i_train = perm[:n_train]
    i_val = perm[n_train : n_train + n_val]
    i_test = perm[n_train + n_val :]
    return (
        stay_ids[i_train],
        stay_ids[i_val],
        stay_ids[i_test],
    )


def get_cohort_indices(
    stay_ids_all: np.ndarray,
    stay_ids_cohort: np.ndarray,
) -> np.ndarray:
    """Map cohort stay IDs to indices in the full stay_id array."""
    sid2idx = {int(s): i for i, s in enumerate(stay_ids_all)}
    return np.array([sid2idx[int(s)] for s in stay_ids_cohort if int(s) in sid2idx])


class CohortData:
    """
    Holds the raw input X and perturbed outputs Y for one (var, setting) pair,
    split into train/val/test by stay.

    Y_* are operator-indexed dictionaries mapping operator -> (n_stays_subset, 48).
    """

    def __init__(
        self,
        var: str,
        setting: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        Y_train: dict[str, np.ndarray],
        Y_val: dict[str, np.ndarray],
        Y_test: dict[str, np.ndarray],
        stay_train: np.ndarray,
        stay_val: np.ndarray,
        stay_test: np.ndarray,
    ):
        self.var = var
        self.setting = setting
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.Y_test = Y_test
        self.stay_train = stay_train
        self.stay_val = stay_val
        self.stay_test = stay_test

    def flatten_for_recon(self, cohort: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        cohort in ('train','val','test').

        Returns:
          - X flattened to shape (N,)
          - each operator's Y flattened to shape (N,)
        where N = n_stays * 48.
        """
        if cohort == "train":
            X = self.X_train
            Yd = self.Y_train
        elif cohort == "val":
            X = self.X_val
            Yd = self.Y_val
        else:
            X = self.X_test
            Yd = self.Y_test
        x_flat = X.ravel()
        y_flat_by_op = {op: Yd[op].ravel() for op in Yd}
        return x_flat, y_flat_by_op


def load_cohort_data(
    data_dir: Path,
    perturbed_dir: Path,
    var: str,
    setting: str,
    operators: tuple[str, ...] = OPERATORS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> Optional[CohortData]:
    """
    Load raw data and perturbed outputs for each operator for a (var, setting) pair,
    and split them into train/val/test by stay.

    If an operator's perturbed outputs are missing, that operator is skipped.
    """
    data_dir = Path(data_dir)
    perturbed_dir = Path(perturbed_dir)
    X_all, stay_ids_all = load_raw_matrix(data_dir, var, setting)
    n_stays, T = X_all.shape
    if T != WINDOW_HOURS:
        return None

    stay_train, stay_val, stay_test = train_val_test_split_stays(stay_ids_all, seed=seed)
    i_train = get_cohort_indices(stay_ids_all, stay_train)
    i_val = get_cohort_indices(stay_ids_all, stay_val)
    i_test = get_cohort_indices(stay_ids_all, stay_test)

    X_train = X_all[i_train]
    X_val = X_all[i_val]
    X_test = X_all[i_test]

    Y_train: dict[str, np.ndarray] = {}
    Y_val: dict[str, np.ndarray] = {}
    Y_test: dict[str, np.ndarray] = {}

    n_stays = X_all.shape[0]
    for op in operators:
        try:
            Y_all = load_perturbed_matrix(
                perturbed_dir, setting, var, op, alpha, n_stays=n_stays
            )
        except FileNotFoundError:
            continue
        if Y_all.shape[0] != n_stays:
            continue
        Y_train[op] = Y_all[i_train]
        Y_val[op] = Y_all[i_val]
        Y_test[op] = Y_all[i_test]

    if not Y_train:
        return None

    return CohortData(
        var=var,
        setting=setting,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        Y_train=Y_train,
        Y_val=Y_val,
        Y_test=Y_test,
        stay_train=stay_train,
        stay_val=stay_val,
        stay_test=stay_test,
    )
