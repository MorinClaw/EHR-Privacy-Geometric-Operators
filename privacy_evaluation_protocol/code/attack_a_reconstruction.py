# -*- coding: utf-8 -*-
"""
Attack A: pointwise reconstruction.

Leakage levels L0/L1/L2/L3 and metrics: R2, Hit_delta, MAE_z, and correlation.

Attacker families: linear, MLP (CNN/Transformer placeholders).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    LEAKAGE_LEVELS,
    LEAKAGE_TRAIN_FRAC,
    LEAKAGE_L0,
    HIT_DELTAS,
    OPERATORS,
)
from data_loader import CohortData, load_cohort_data


def _linear_fit_predict(
    Y_train: np.ndarray,
    X_train: np.ndarray,
    Y_test: np.ndarray,
) -> np.ndarray:
    """Ordinary least squares: X ~= Y @ coef + intercept. Returns X_pred with shape (n_test, 1)."""
    n_tr = Y_train.shape[0]
    Y_aug = np.hstack([Y_train.reshape(-1, 1), np.ones((n_tr, 1), dtype=float)])
    beta, _, _, _ = np.linalg.lstsq(Y_aug, X_train.ravel(), rcond=None)
    n_te = Y_test.shape[0]
    Y_te_aug = np.hstack([Y_test.reshape(-1, 1), np.ones((n_te, 1), dtype=float)])
    return (Y_te_aug @ beta).reshape(-1, 1)


def _compute_metrics(
    X_test: np.ndarray,
    X_pred: np.ndarray,
    std_train: float,
) -> dict:
    """Compute R2, Hit_delta, MAE_z, and Pearson correlation."""
    x_flat = X_test.ravel()
    pred_flat = X_pred.ravel()
    valid = ~(np.isnan(x_flat) | np.isnan(pred_flat))
    if valid.sum() < 2:
        return {"R2": np.nan, "MAE_z": np.nan, "rho": np.nan, **{f"Hit_{d}": np.nan for d in HIT_DELTAS}}
    x = x_flat[valid]
    pred = pred_flat[valid]
    ss_res = np.sum((x - pred) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    R2 = float(1 - ss_res / (ss_tot + 1e-12))
    sigma = std_train if std_train > 1e-10 else float(np.std(x))
    MAE_z = float(np.mean(np.abs(pred - x)) / (sigma + 1e-12))
    rho = float(np.corrcoef(x, pred)[0, 1]) if np.std(pred) > 1e-12 else np.nan
    hit = {}
    for d in HIT_DELTAS:
        eps = d * (sigma + 1e-12)
        hit[f"Hit_{d}"] = float(np.mean(np.abs(pred - x) <= eps))
    return {"R2": R2, "MAE_z": MAE_z, "rho": rho, **hit}


def run_reconstruction_one(
    cohort: CohortData,
    operator: str,
    leakage: str,
    seed: int = 42,
    max_n: Optional[int] = 100000,
) -> list[dict]:
    """
    Run the reconstruction attack for a single (CohortData, operator, leakage) tuple.

    Returns multiple rows (one per attacker model).

    - L0 (no-pairs): use a baseline predictor (constant mean) and y as a proxy for x.
    - L1/L2/L3: use the configured training fraction, fit linear (and optionally MLP),
      and evaluate on the test set.
    """
    x_flat, y_by_op = cohort.flatten_for_recon("train")
    x_test_flat, y_test_by_op = cohort.flatten_for_recon("test")
    if operator not in y_by_op or operator not in y_test_by_op:
        return []

    Y_train = y_by_op[operator].reshape(-1, 1)
    X_train = x_flat.reshape(-1, 1)
    Y_test = y_test_by_op[operator].reshape(-1, 1)
    X_test = x_test_flat.reshape(-1, 1)

    valid_tr = ~(np.isnan(X_train) | np.isnan(Y_train))
    valid_te = ~(np.isnan(X_test) | np.isnan(Y_test))
    X_train = X_train[valid_tr]
    Y_train = Y_train[valid_tr]
    X_test = X_test[valid_te]
    Y_test = Y_test[valid_te]
    if X_train.size < 10 or X_test.size < 10:
        return []

    n = X_train.shape[0]
    train_frac = LEAKAGE_TRAIN_FRAC.get(leakage, 0.0)
    if leakage == LEAKAGE_L0:
        n_tr = 0
    else:
        n_tr = max(2, min(int(n * train_frac), n - 2))
    rng = np.random.default_rng(seed)
    if n_tr >= 2:
        perm = rng.permutation(n)
        i_tr, _ = perm[:n_tr], perm[n_tr:]
        Y_tr = Y_train[i_tr]
        X_tr = X_train[i_tr]
    std_train = float(np.std(X_train)) if X_train.size > 1 else 1.0

    rows: list[dict] = []
    base = {
        "var": cohort.var,
        "setting": cohort.setting,
        "operator": operator,
        "leakage": leakage,
        "n_train": n_tr,
        "n_test": int(X_test.size),
    }

    # L0 baseline: x̂ = y (and constant-mean prediction).
    if leakage == LEAKAGE_L0:
        X_pred_baseline = Y_test.copy()
        met = _compute_metrics(X_test, X_pred_baseline, std_train)
        rows.append({**base, "model": "no_pairs_y", **met})
        # Constant prediction (mean of X_train)
        X_pred_const = np.full_like(X_test, np.nanmean(X_train))
        met_const = _compute_metrics(X_test, X_pred_const, std_train)
        rows.append({**base, "model": "no_pairs_const", **met_const})
        return rows

    # L1/L2/L3: supervised training
    X_pred_linear = _linear_fit_predict(Y_tr, X_tr, Y_test)
    met_linear = _compute_metrics(X_test, X_pred_linear, std_train)
    rows.append({**base, "model": "linear", **met_linear})

    try:
        from sklearn.neural_network import MLPRegressor

        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed)
        Y_tr_2d = np.asarray(Y_tr).reshape(-1, 1) if np.asarray(Y_tr).ndim == 1 else Y_tr
        mlp.fit(Y_tr_2d, X_tr.ravel())
        Y_te_2d = np.asarray(Y_test).reshape(-1, 1) if np.asarray(Y_test).ndim == 1 else Y_test
        X_pred_mlp = mlp.predict(Y_te_2d).reshape(-1, 1)
        met_mlp = _compute_metrics(X_test, X_pred_mlp, std_train)
        rows.append({**base, "model": "mlp", **met_mlp})
    except ImportError:
        pass

    return rows


def run_attack_a(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: Optional[list[str]] = None,
    leakages: tuple[str, ...] = LEAKAGE_LEVELS,
    operators: tuple[str, ...] = OPERATORS,
    alpha: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full attack A reconstruction matrix and write the output table.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Discover variables available in the perturbed directory
    setting_path = perturbed_dir / "z"
    if not setting_path.exists():
        setting_path = perturbed_dir / "phys"
    vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()]) if setting_path.exists() else []
    if variables:
        vars_here = [v for v in vars_here if v in variables]

    all_rows: list[dict] = []
    for var in vars_here:
        for setting in ("z", "phys"):
            cohort = load_cohort_data(
                data_dir, perturbed_dir, var, setting,
                operators=operators, alpha=alpha, seed=seed,
            )
            if cohort is None:
                continue
            for op in cohort.Y_train.keys():
                for leakage in leakages:
                    rows = run_reconstruction_one(cohort, op, leakage, seed=seed)
                    for r in rows:
                        r["alpha"] = alpha
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = tables_dir / "table_attack_a_reconstruction.csv"
    df.to_csv(out_path, index=False)
    print(f"[Attack A] Wrote {out_path} ({len(df)} rows)")
    return df
