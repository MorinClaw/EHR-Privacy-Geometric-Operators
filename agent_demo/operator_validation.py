# filename: operator_validation.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
operator_validation.py — validation utilities for numeric operators.

This module is used by the Agent integration to sanity-check that a transformation
function x -> y behaves as expected:
  - sanity: preserve mean/variance, bound delta, and unchanged ratio
  - reconstruction: evaluate linear reconstruction attack (R2/RMSE/rho; optional MLP)
  - multi_run: compare variability across multiple runs (KS/rho) to assess randomness

The entry points are:
  - `validate_numeric_operator`
  - `validate_skill`
  - `validate_registry_numeric`

When `x=None`, validation uses a synthetic input vector to run a quick self-check.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Aligned with experiment A.3.
EPS = 1e-8

# Default thresholds: if satisfied, report pass.
DEFAULT_SANITY_TOL_MEAN = 1e-6
DEFAULT_SANITY_TOL_VAR = 1e-4
DEFAULT_RECON_R2_FAIL = 0.98  # reconstruction R2 below this value => harder to reconstruct => pass
DEFAULT_MULTI_RUN_K = 3
DEFAULT_RECON_MAX_N = 20000
DEFAULT_TRAIN_FRAC = 0.8


def _ensure_float1d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a


def run_sanity(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """A.3-style sanity checks on (x, y): delta_mean/delta_var/max_abs_delta/min_abs_delta/unchanged_ratio."""
    x = _ensure_float1d(x)
    y = _ensure_float1d(y)
    if len(x) != len(y):
        return {"error": "length_mismatch", "pass": False}
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return {"error": "too_few_valid", "pass": False}
    xv = x[mask]
    yv = y[mask]
    n = len(xv)
    delta = yv - xv
    abs_delta = np.abs(delta)
    mean_x, mean_y = float(np.mean(xv)), float(np.mean(yv))
    var_x = float(np.var(xv, ddof=0))
    var_y = float(np.var(yv, ddof=0))
    unchanged = int(np.sum(abs_delta <= EPS))
    return {
        "delta_mean": float(abs(mean_x - mean_y)),
        "delta_var": float(abs(var_x - var_y)),
        "max_abs_delta": float(np.max(abs_delta)),
        "min_abs_delta": float(np.min(abs_delta)),
        "unchanged_ratio": unchanged / n,
        "n_valid": n,
        "pass": None,  # Filled by the caller based on thresholds.
    }


def run_reconstruction_check(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    max_n: Optional[int] = DEFAULT_RECON_MAX_N,
    use_mlp: bool = False,
) -> Dict[str, Any]:
    """Reconstruction attack: use Y to predict X; returns linear R2/RMSE/rho; optional MLP (requires sklearn)."""
    x = _ensure_float1d(x)
    y = _ensure_float1d(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return {"error": "too_few_valid", "R2": np.nan, "RMSE": np.nan, "rho": np.nan, "pass": False}
    x = x[mask]
    y = y[mask]
    n = len(x)
    if max_n and n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)[:max_n]
        x, y = x[idx], y[idx]
        n = len(x)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(n * train_frac)
    i_tr, i_te = perm[:n_tr], perm[n_tr:]
    Y_tr = y[i_tr].reshape(-1, 1)
    X_tr = x[i_tr].reshape(-1, 1)
    Y_te = y[i_te].reshape(-1, 1)
    X_te = x[i_te].reshape(-1, 1)
    # OLS: X ~ Y
    ones = np.ones((len(i_tr), 1), dtype=float)
    Y_aug = np.hstack([Y_tr, ones])
    beta, _, _, _ = np.linalg.lstsq(Y_aug, X_tr.ravel(), rcond=None)
    Y_te_aug = np.hstack([Y_te, np.ones((len(i_te), 1), dtype=float)])
    X_pred = (Y_te_aug @ beta).reshape(-1, 1)
    var_te = float(np.var(X_te))
    r2 = float(1 - np.sum((X_te - X_pred) ** 2) / (np.sum((X_te - X_te.mean()) ** 2) + 1e-12)) if var_te > 1e-12 else np.nan
    rmse = float(np.sqrt(np.mean((X_te - X_pred) ** 2)))
    rho = float(np.corrcoef(X_te.ravel(), X_pred.ravel())[0, 1]) if len(X_te) > 1 else np.nan
    out: Dict[str, Any] = {
        "model": "linear",
        "R2": r2,
        "RMSE": rmse,
        "rho": rho,
        "n_train": len(i_tr),
        "n_test": len(i_te),
        "pass": None,
    }
    if use_mlp:
        try:
            from sklearn.neural_network import MLPRegressor
            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=150, random_state=seed)
            mlp.fit(Y_tr, X_tr.ravel())
            X_pred_mlp = mlp.predict(Y_te).reshape(-1, 1)
            r2_mlp = float(1 - np.sum((X_te - X_pred_mlp) ** 2) / (np.sum((X_te - X_te.mean()) ** 2) + 1e-12)) if var_te > 1e-12 else np.nan
            out["R2_mlp"] = r2_mlp
            out["RMSE_mlp"] = float(np.sqrt(np.mean((X_te - X_pred_mlp) ** 2)))
        except ImportError:
            out["R2_mlp"] = None
            out["RMSE_mlp"] = None
    return out


def run_multi_run_check(
    fn: Callable[[Any, Dict[str, Any]], Any],
    x: np.ndarray,
    config: Dict[str, Any],
    K: int = DEFAULT_MULTI_RUN_K,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run fn(·, config) K times on the same x (accounting for internal randomness) and compute pairwise KS and rho."""
    x = _ensure_float1d(x)
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return {"error": "all_nan", "mean_KS": np.nan, "min_rho": np.nan, "pass": None}
    x_filled = np.where(nan_mask, np.nanmean(x[~nan_mask]), x)
    if seeds is None:
        seeds = list(range(42, 42 + K))
    ys: List[np.ndarray] = []
    for _ in range(K):
        try:
            y = fn(x_filled.copy(), config)
            y = np.asarray(y, dtype=float)
            y[nan_mask] = np.nan
            ys.append(y)
        except Exception as e:
            return {"error": str(e), "mean_KS": np.nan, "min_rho": np.nan, "pass": False}
    if len(ys) < 2:
        return {"error": "too_few_runs", "mean_KS": np.nan, "min_rho": np.nan, "pass": None}
    ks_vals: List[float] = []
    rho_vals: List[float] = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            valid = ~(np.isnan(ys[i]) | np.isnan(ys[j]))
            if valid.sum() < 50:
                continue
            a, b = ys[i][valid], ys[j][valid]
            ks_vals.append(float(stats.ks_2samp(a, b).statistic))
            if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                rho_vals.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "mean_KS": float(np.mean(ks_vals)) if ks_vals else np.nan,
        "max_KS": float(np.max(ks_vals)) if ks_vals else np.nan,
        "mean_rho": float(np.mean(rho_vals)) if rho_vals else np.nan,
        "min_rho": float(np.min(rho_vals)) if rho_vals else np.nan,
        "n_pairs": len(ks_vals),
        "pass": None,
    }


def validate_numeric_operator(
    fn: Callable[[Any, Dict[str, Any]], Any],
    x: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    *,
    sanity_tol_mean: float = DEFAULT_SANITY_TOL_MEAN,
    sanity_tol_var: float = DEFAULT_SANITY_TOL_VAR,
    recon_r2_fail_threshold: float = DEFAULT_RECON_R2_FAIL,
    recon_max_n: Optional[int] = DEFAULT_RECON_MAX_N,
    recon_use_mlp: bool = False,
    multi_run_K: int = DEFAULT_MULTI_RUN_K,
    checks: Tuple[str, ...] = ("sanity", "reconstruction", "multi_run"),
) -> Dict[str, Any]:
    """
    Validate a numeric operator fn(x, config) -> y and return per-check results plus an overall pass flag.

    - sanity: delta_mean < sanity_tol_mean and delta_var < sanity_tol_var => pass
    - reconstruction: linear R2 < recon_r2_fail_threshold => pass (harder to reconstruct)
    - multi_run: record mean_KS/min_rho only (no hard pass; helps interpret randomness)
    """
    cfg = dict(config) if config else {}
    x = _ensure_float1d(x)
    try:
        y = fn(x.copy(), cfg)
        y = _ensure_float1d(y)
    except Exception as e:
        return {"error": str(e), "pass": False, "checks": {}}

    results: Dict[str, Any] = {"checks": {}, "pass": True}

    if "sanity" in checks:
        s = run_sanity(x, y)
        if "error" in s:
            results["checks"]["sanity"] = s
            results["pass"] = False
        else:
            s["pass"] = s["delta_mean"] <= sanity_tol_mean and s["delta_var"] <= sanity_tol_var
            results["checks"]["sanity"] = s
            if not s["pass"]:
                results["pass"] = False

    if "reconstruction" in checks:
        r = run_reconstruction_check(x, y, max_n=recon_max_n, use_mlp=recon_use_mlp)
        if "error" in r:
            results["checks"]["reconstruction"] = r
        else:
            r["pass"] = (np.isfinite(r["R2"]) and r["R2"] < recon_r2_fail_threshold)
            results["checks"]["reconstruction"] = r
            if not r["pass"]:
                results["pass"] = False

    if "multi_run" in checks:
        m = run_multi_run_check(fn, x, cfg, K=multi_run_K)
        if "error" in m:
            results["checks"]["multi_run"] = m
        else:
            results["checks"]["multi_run"] = m

    return results


def validate_skill(
    skill: Any,
    x: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Validate a single Skill when target == 'numeric'; otherwise return a skip report."""
    if getattr(skill, "target", None) != "numeric":
        return {"skipped": True, "reason": "not_numeric", "pass": None}
    cfg = dict(getattr(skill, "default_config", {}))
    if config:
        cfg.update(config)
    return validate_numeric_operator(skill.fn, x, cfg, **kwargs)


def validate_registry_numeric(
    registry: Any,
    x: Optional[np.ndarray] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    n_synthetic: int = 2000,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Validate all skills with target == 'numeric' in the registry.

    When x is None, use a synthetic input vector of length n_synthetic (normal + a few outliers)
    for quick self-checking without real data.
    """
    numeric_skills = registry.list_by_target("numeric") if hasattr(registry, "list_by_target") else []
    if not numeric_skills:
        return {"skills": [], "summary": "no_numeric_skills"}

    if x is None:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n_synthetic)
        x[rng.integers(0, n_synthetic, size=min(50, n_synthetic // 20))] += rng.uniform(2, 4, size=min(50, n_synthetic // 20))

    report: Dict[str, Any] = {"skills": [], "summary": []}
    all_pass = True
    for skill in numeric_skills:
        cfg = config_overrides or {}
        r = validate_skill(skill, x, cfg, **kwargs)
        r["skill_id"] = skill.id
        r["skill_name"] = getattr(skill, "name", skill.id)
        report["skills"].append(r)
        if r.get("pass") is False:
            all_pass = False
            report["summary"].append(f"{skill.id}: FAIL")
        elif r.get("pass") is True:
            report["summary"].append(f"{skill.id}: PASS")
        else:
            report["summary"].append(f"{skill.id}: {r.get('error', 'OK')}")
    report["all_pass"] = all_pass
    return report
