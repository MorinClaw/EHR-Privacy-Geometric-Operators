# -*- coding: utf-8 -*-
"""
Attack C: membership inference.

Determine whether a candidate record appears in the exported cohort.
Data-level MI trains a classifier on statistical features of y for member vs non-member.

Metrics: ROC-AUC and Advantage = AUC - 0.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import OPERATORS
from data_loader import CohortData, load_cohort_data


def _series_features(z: np.ndarray) -> np.ndarray:
    """Statistical feature vector for a single time series."""
    z = np.asarray(z, dtype=float)
    valid = ~np.isnan(z)
    if valid.sum() < 2:
        return np.zeros(7)
    v = z[valid]
    return np.array([
        np.mean(v), np.std(v), np.min(v), np.max(v),
        np.median(v), np.percentile(v, 25), np.percentile(v, 75),
    ], dtype=float)


def run_membership_one(
    cohort: CohortData,
    operator: str,
    seed: int = 42,
) -> dict:
    """
    Data-level MI:
      - member: y from the training split
      - non-member: y from the test split

    Split both train and test halves to train/evaluate a binary classifier, and report AUC.
    """
    Y_train = cohort.Y_train[operator]   # (n_train, 48)
    Y_test = cohort.Y_test[operator]     # (n_test, 48)
    n_tr = Y_train.shape[0]
    n_te = Y_test.shape[0]
    if n_tr < 20 or n_te < 20:
        return {"var": cohort.var, "setting": cohort.setting, "operator": operator,
                "AUC": np.nan, "Advantage": np.nan}

    rng = np.random.default_rng(seed)
    # Attacker training data: half of train y (label 1) + half of test y (label 0). Remaining halves are used for evaluation.
    idx_tr = rng.permutation(n_tr)
    idx_te = rng.permutation(n_te)
    split_tr = n_tr // 2
    split_te = n_te // 2

    X_att_tr = np.vstack([
        [_series_features(Y_train[i]) for i in idx_tr[:split_tr]],
        [_series_features(Y_test[i]) for i in idx_te[:split_te]],
    ])
    y_att_tr = np.array([1] * split_tr + [0] * split_te)
    X_att_te = np.vstack([
        [_series_features(Y_train[i]) for i in idx_tr[split_tr:]],
        [_series_features(Y_test[i]) for i in idx_te[split_te:]],
    ])
    y_att_te = np.array([1] * (n_tr - split_tr) + [0] * (n_te - split_te))

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_att_tr, y_att_tr)
        proba = clf.predict_proba(X_att_te)[:, 1]
        auc = float(roc_auc_score(y_att_te, proba))
    except ImportError:
        auc = np.nan
    advantage = float(auc - 0.5) if not np.isnan(auc) else np.nan

    return {
        "var": cohort.var,
        "setting": cohort.setting,
        "operator": operator,
        "AUC": auc,
        "Advantage": advantage,
        "n_train_member": n_tr,
        "n_test_nonmember": n_te,
    }


def run_attack_c(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: Optional[list[str]] = None,
    operators: tuple[str, ...] = OPERATORS,
    alpha: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full C-class membership inference evaluation and write the output table."""
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    setting_path = perturbed_dir / "z"
    if not setting_path.exists():
        setting_path = perturbed_dir / "phys"
    vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()]) if setting_path.exists() else []
    if variables:
        vars_here = [v for v in vars_here if v in variables]

    all_rows = []
    for var in vars_here:
        for setting in ("z", "phys"):
            cohort = load_cohort_data(
                data_dir, perturbed_dir, var, setting,
                operators=operators, alpha=alpha, seed=seed,
            )
            if cohort is None:
                continue
            for op in cohort.Y_train.keys():
                row = run_membership_one(cohort, op, seed=seed)
                row["alpha"] = alpha
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = tables_dir / "table_attack_c_membership.csv"
    df.to_csv(out_path, index=False)
    print(f"[Attack C] Wrote {out_path} ({len(df)} rows)")
    return df
