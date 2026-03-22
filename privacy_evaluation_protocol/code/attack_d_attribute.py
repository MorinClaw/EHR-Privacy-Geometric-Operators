# -*- coding: utf-8 -*-
"""
Attack D: attribute inference and a pairwise indistinguishability game.

Attribute inference predicts a sensitive function f(x) (e.g., max(x), whether above a threshold,
or the value at a specific time) from perturbed observations y.

Indistinguishability game samples (x0, x1), draws b ~ Bern(0.5), sets y = T(x_b),
and asks the attacker to guess b. Metrics are accuracy and advantage |acc - 0.5|.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from config import OPERATORS
from data_loader import CohortData, load_cohort_data


def _series_features(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    valid = ~np.isnan(z)
    if valid.sum() < 2:
        return np.zeros(7)
    v = z[valid]
    return np.array([
        np.mean(v), np.std(v), np.min(v), np.max(v),
        np.median(v), np.percentile(v, 25), np.percentile(v, 75),
    ], dtype=float)


# Sensitive attribute functions (defined on raw x).
def attr_max(x: np.ndarray) -> float:
    return float(np.nanmax(x))


def attr_min(x: np.ndarray) -> float:
    return float(np.nanmin(x))


def attr_value_at_t24(x: np.ndarray) -> float:
    if x.size > 24:
        return float(x.flat[24]) if not np.isnan(x.flat[24]) else np.nanmean(x)
    return np.nanmean(x)


def attr_above_threshold(x: np.ndarray, q: float = 0.9) -> float:
    """Whether there exists a point above the q-quantile.

    Returns either a binary value (mean over indicator) or, more generally, the fraction above threshold.
    """
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return 0.0
    th = np.percentile(x[valid], q * 100)
    return float(np.mean(x[valid] > th))


ATTRIBUTE_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "max": attr_max,
    "min": attr_min,
    "value_t24": attr_value_at_t24,
    "above_p90": lambda x: attr_above_threshold(x, 0.9),
}


def run_attribute_one(
    cohort: CohortData,
    operator: str,
    attr_name: str,
    seed: int = 42,
) -> dict:
    """Predict f(x) from y using training labels (y, f(x)).

    The attacker is a simple regressor; evaluation reports R2 and MAE.
    """
    if attr_name not in ATTRIBUTE_FUNCS:
        return {}
    f = ATTRIBUTE_FUNCS[attr_name]
    Y_train = cohort.Y_train[operator]
    Y_test = cohort.Y_test[operator]
    X_train = cohort.X_train
    X_test = cohort.X_test

    attr_train = np.array([f(X_train[i]) for i in range(X_train.shape[0])])
    attr_test = np.array([f(X_test[i]) for i in range(X_test.shape[0])])
    feat_train = np.vstack([_series_features(Y_train[i]) for i in range(Y_train.shape[0])])
    feat_test = np.vstack([_series_features(Y_test[i]) for i in range(Y_test.shape[0])])

    valid_tr = ~np.isnan(attr_train)
    valid_te = ~np.isnan(attr_test)
    if valid_tr.sum() < 10 or valid_te.sum() < 5:
        return {"var": cohort.var, "setting": cohort.setting, "operator": operator, "attr": attr_name,
                "R2": np.nan, "MAE": np.nan}

    X_tr = feat_train[valid_tr]
    y_tr = attr_train[valid_tr]
    X_te = feat_test[valid_te]
    y_te = attr_test[valid_te]

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        reg = LinearRegression().fit(X_tr, y_tr)
        pred = reg.predict(X_te)
        r2 = float(r2_score(y_te, pred))
        mae = float(np.mean(np.abs(pred - y_te)))
    except Exception:
        r2 = np.nan
        mae = np.nan

    return {
        "var": cohort.var,
        "setting": cohort.setting,
        "operator": operator,
        "attr": attr_name,
        "R2": r2,
        "MAE": mae,
    }


def run_distinguish_one(
    cohort: CohortData,
    operator: str,
    seed: int = 42,
    n_trials: int = 500,
) -> dict:
    """
    Pairwise indistinguishability game.

    Sample (x0, x1), draw b ~ Bern(0.5), and set y = T(x_b).
    Train a binary classifier on features (feat(x0), feat(x1), feat(y)) to guess b.
    """
    X_train = cohort.X_train
    Y_train = cohort.Y_train[operator]
    n = X_train.shape[0]
    if n < 20:
        return {"var": cohort.var, "setting": cohort.setting, "operator": operator,
                "acc": np.nan, "advantage": np.nan}

    rng = np.random.default_rng(seed)
    feats_x0 = []
    feats_x1 = []
    feats_y = []
    labels = []
    for _ in range(n_trials):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            j = (j + 1) % n
        b = rng.integers(0, 2)
        x0 = X_train[i]
        x1 = X_train[j]
        y = Y_train[j] if b else Y_train[i]
        feats_x0.append(_series_features(x0))
        feats_x1.append(_series_features(x1))
        feats_y.append(_series_features(y))
        labels.append(b)

    X_att = np.hstack([np.vstack(feats_x0), np.vstack(feats_x1), np.vstack(feats_y)])
    y_att = np.array(labels)
    split = int(0.8 * len(y_att))
    X_tr, X_te = X_att[:split], X_att[split:]
    y_tr, y_te = y_att[:split], y_att[split:]

    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=300, random_state=seed).fit(X_tr, y_tr)
        acc = float(np.mean(clf.predict(X_te) == y_te))
    except ImportError:
        acc = np.nan
    advantage = float(abs(acc - 0.5)) if not np.isnan(acc) else np.nan

    return {
        "var": cohort.var,
        "setting": cohort.setting,
        "operator": operator,
        "acc": acc,
        "advantage": advantage,
        "n_trials": n_trials,
    }


def run_attack_d(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: Optional[list[str]] = None,
    operators: tuple[str, ...] = OPERATORS,
    alpha: float = 0.8,
    seed: int = 42,
    attr_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Attack D: attribute inference table and indistinguishability game table."""
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    setting_path = perturbed_dir / "z"
    if not setting_path.exists():
        setting_path = perturbed_dir / "phys"
    vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()]) if setting_path.exists() else []
    if variables:
        vars_here = [v for v in vars_here if v in variables]
    attrs = attr_names or list(ATTRIBUTE_FUNCS.keys())

    rows_attr = []
    rows_dist = []
    for var in vars_here:
        for setting in ("z", "phys"):
            cohort = load_cohort_data(
                data_dir, perturbed_dir, var, setting,
                operators=operators, alpha=alpha, seed=seed,
            )
            if cohort is None:
                continue
            for op in cohort.Y_train.keys():
                for attr in attrs:
                    r = run_attribute_one(cohort, op, attr, seed=seed)
                    if r:
                        r["alpha"] = alpha
                        rows_attr.append(r)
                r = run_distinguish_one(cohort, op, seed=seed)
                r["alpha"] = alpha
                rows_dist.append(r)

    df_attr = pd.DataFrame(rows_attr)
    df_dist = pd.DataFrame(rows_dist)
    df_attr.to_csv(tables_dir / "table_attack_d_attribute.csv", index=False)
    df_dist.to_csv(tables_dir / "table_attack_d_distinguish.csv", index=False)
    print(f"[Attack D] Attribute inference: {tables_dir / 'table_attack_d_attribute.csv'} ({len(df_attr)} rows)")
    print(f"[Attack D] Distinguishability: {tables_dir / 'table_attack_d_distinguish.csv'} ({len(df_dist)} rows)")
    return df_attr, df_dist
