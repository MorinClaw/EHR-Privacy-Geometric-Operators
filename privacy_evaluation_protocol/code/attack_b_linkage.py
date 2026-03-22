# -*- coding: utf-8 -*-
"""
Attack B: record linkage / re-identification.

Given y = T(x*), the candidate set {x^(1),...,x^(m)} contains the true x*.
The attacker predicts an index j^.

Metrics include Re-id@1, Re-id@k, and AUC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import LINKAGE_CANDIDATE_SIZES, OPERATORS
from data_loader import CohortData, load_cohort_data


def _series_features(z: np.ndarray) -> np.ndarray:
    """Feature vector for a single time series (length 48 or n)."""
    z = np.asarray(z, dtype=float)
    valid = ~np.isnan(z)
    if valid.sum() < 2:
        return np.zeros(7)
    v = z[valid]
    return np.array([
        np.mean(v), np.std(v), np.min(v), np.max(v),
        np.median(v), np.percentile(v, 25), np.percentile(v, 75),
    ], dtype=float)


def _build_pair_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Concatenate features of (x, y) for same/diff classification."""
    fx = _series_features(x)
    fy = _series_features(y)
    return np.concatenate([fx, fy])


def run_linkage_one(
    cohort: CohortData,
    operator: str,
    candidate_size: int,
    seed: int = 42,
) -> dict:
    """
    Run record linkage for a single (cohort, operator).

    For each test example y_i, construct a candidate set of size m that includes the true x_i.
    Train a same/diff classifier on the training split, score (x_j, y_i), and select the best match.
    """
    X_train, Y_train = cohort.X_train, cohort.Y_train[operator]
    X_test, Y_test = cohort.X_test, cohort.Y_test[operator]
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    if n_test < 2 or n_train < candidate_size:
        return {"var": cohort.var, "setting": cohort.setting, "operator": operator, "m": candidate_size,
                "Reid_at_1": np.nan, "Reid_at_5": np.nan, "AUC": np.nan, "n_trials": 0}

    # Build training pairs: positive (x, T(x)), negative (x, T(x')) with a randomly sampled different entry.
    rng = np.random.default_rng(seed)
    pos_feats, neg_feats = [], []
    for i in range(X_train.shape[0]):
        x_i = X_train[i]
        y_i = Y_train[i]
        pos_feats.append(_build_pair_features(x_i, y_i))
        j = rng.integers(0, n_train)
        if j == i:
            j = (i + 1) % n_train
        neg_feats.append(_build_pair_features(x_i, Y_train[j]))
    X_tr = np.vstack(pos_feats + neg_feats)
    y_tr = np.array([1] * len(pos_feats) + [0] * len(neg_feats))

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_tr, y_tr)
        scorer = lambda x, y: clf.predict_proba(_build_pair_features(x, y).reshape(1, -1))[0, 1]
    except ImportError:
        # No sklearn: use a heuristic score based on feature distance (more similar => higher score).
        def scorer(x, y):
            f = _build_pair_features(x, y)
            return -float(np.linalg.norm(f[:7] - f[7:]))
        clf = None

    reid_1 = 0
    reid_5 = 0
    all_scores = []
    all_labels = []

    for i in range(n_test):
        y_i = Y_test[i]
        x_true = X_test[i]
        # Candidate set: include the true index and sample m-1 other test indices.
        others = [k for k in range(n_test) if k != i]
        rng.shuffle(others)
        idx_cand = [i] + others[: candidate_size - 1]
        rng.shuffle(idx_cand)  # shuffle order; the true index is no longer guaranteed to be at position 0
        true_idx_in_cand = idx_cand.index(i)

        scores = []
        for k in idx_cand:
            x_k = X_test[k]
            sc = scorer(x_k, y_i)
            scores.append((sc, 1 if k == i else 0))
        scores.sort(key=lambda t: -t[0])
        rank = next(r for r, (_, lab) in enumerate(scores) if lab == 1)
        if rank == 0:
            reid_1 += 1
        if rank < 5:
            reid_5 += 1
        for sc, lab in scores:
            all_scores.append(sc)
            all_labels.append(lab)

    reid_1 /= n_test
    reid_5 /= n_test
    auc = np.nan
    if clf is not None and all_labels:
        try:
            auc = float(roc_auc_score(all_labels, all_scores))
        except Exception:
            pass

    return {
        "var": cohort.var,
        "setting": cohort.setting,
        "operator": operator,
        "m": candidate_size,
        "Reid_at_1": reid_1,
        "Reid_at_5": reid_5,
        "AUC": auc,
        "n_trials": n_test,
    }


def run_attack_b(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: Optional[list[str]] = None,
    candidate_sizes: tuple[int, ...] = LINKAGE_CANDIDATE_SIZES,
    operators: tuple[str, ...] = OPERATORS,
    alpha: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full B-class record linkage evaluation and write the output table."""
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
                for m in candidate_sizes:
                    row = run_linkage_one(cohort, op, m, seed=seed)
                    row["alpha"] = alpha
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = tables_dir / "table_attack_b_linkage.csv"
    df.to_csv(out_path, index=False)
    print(f"[Attack B] Wrote {out_path} ({len(df)} rows)")
    return df
