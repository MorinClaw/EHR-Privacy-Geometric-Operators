#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-mixing pilot runner:
  raw x  ->  qmix x' = Qx  ->  z = T(x')  (T in {T1_uniform, T1_weighted, T2})

Privacy (Attack A):
  attacker observes z, tries to reconstruct *raw x* (not qmix(x)).
  We achieve this by running Attack A with:
    data_dir = raw_ts_dir
    perturbed_dir = qmix_results/perturbed

Utility:
  - Keep delta-AUROC/AUPRC (raw vs op) using existing paired LR utility evaluation.
  - Keep shape metrics (1-KS etc.) on flattened 48h sequences (raw vs op in z space).

Outputs:
  out_root/
    alpha_{a}/<var>/{strong,weak}/
      table_pr.csv
      table_ut.csv
      tradeoff_utility.png
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# For this Q-mixing pilot run, keep the T1/T2 operator set needed for
# privacy/utility evaluation (exclude T3; include both T1 variants).
OPS_KEEP = ["T1_uniform", "T1_weighted", "T2"]


def parse_alpha_dir(alpha: float) -> str:
    a = int(alpha)
    b = int(round((alpha - a) * 10))
    return f"alpha_{a}_{b}"


def run_cmd(cmd: str) -> None:
    import subprocess

    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")


def extract_features(X: np.ndarray) -> np.ndarray:
    n, t = X.shape
    feats = np.full((n, 5), np.nan, dtype=float)
    idx_t = np.arange(t, dtype=float)
    for i in range(n):
        x = X[i]
        m = np.isfinite(x)
        if m.sum() < 2:
            continue
        xv = x[m]
        tv = idx_t[m]
        feats[i, 0] = float(np.mean(xv))
        feats[i, 1] = float(np.std(xv))
        feats[i, 2] = float(np.min(xv))
        feats[i, 3] = float(np.max(xv))
        A = np.vstack([tv, np.ones_like(tv)]).T
        beta, _, _, _ = np.linalg.lstsq(A, xv, rcond=None)
        feats[i, 4] = float(beta[0])
    return feats


def train_test_split_indices(n: int, seed: int = 42, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(n * train_frac)
    return perm[:n_tr], perm[n_tr:]


def make_labels_los_binary(cohort: pd.DataFrame) -> np.ndarray:
    los = pd.to_numeric(cohort["los"], errors="coerce").to_numpy(dtype=float)
    los_med = np.nanmedian(los)
    return (np.nan_to_num(los, nan=los_med) > los_med).astype(int)


def eval_paired_raw_vs_op(X_raw: np.ndarray, X_op: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    good = np.isfinite(X_raw).all(axis=1) & np.isfinite(X_op).all(axis=1)
    Xr = X_raw[good]
    Xo = X_op[good]
    yg = y[good]
    if np.unique(yg).size < 2 or len(yg) < 50:
        return {k: np.nan for k in ["auroc_raw", "auprc_raw", "auroc_op", "auprc_op", "delta_auroc", "delta_auprc", "n_total", "n_train", "n_test"]}

    i_tr, i_te = train_test_split_indices(len(yg), seed=seed)
    Xr_tr, Xr_te = Xr[i_tr], Xr[i_te]
    Xo_tr, Xo_te = Xo[i_tr], Xo[i_te]
    y_tr, y_te = yg[i_tr], yg[i_te]
    if np.unique(y_te).size < 2:
        return {k: np.nan for k in ["auroc_raw", "auprc_raw", "auroc_op", "auprc_op", "delta_auroc", "delta_auprc", "n_total", "n_train", "n_test"]}

    clf_raw = LogisticRegression(max_iter=2000, random_state=seed, C=0.5).fit(Xr_tr, y_tr)
    clf_op = LogisticRegression(max_iter=2000, random_state=seed, C=0.5).fit(Xo_tr, y_tr)
    pr = clf_raw.predict_proba(Xr_te)[:, 1]
    po = clf_op.predict_proba(Xo_te)[:, 1]
    auroc_raw = float(roc_auc_score(y_te, pr))
    auprc_raw = float(average_precision_score(y_te, pr))
    auroc_op = float(roc_auc_score(y_te, po))
    auprc_op = float(average_precision_score(y_te, po))
    return {
        "auroc_raw": auroc_raw,
        "auprc_raw": auprc_raw,
        "auroc_op": auroc_op,
        "auprc_op": auprc_op,
        "delta_auroc": auroc_op - auroc_raw,
        "delta_auprc": auprc_op - auprc_raw,
        "n_total": int(len(yg)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def run_shape_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return {
            "utility_score": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "delta_mean": np.nan,
            "delta_var": np.nan,
            "max_abs_delta": np.nan,
            "unchanged_ratio": np.nan,
            "n_valid": int(mask.sum()),
        }
    xv = x[mask]
    yv = y[mask]
    ks = stats.ks_2samp(xv, yv)
    delta = yv - xv
    abs_delta = np.abs(delta)
    return {
        "utility_score": float(1.0 - ks.statistic),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "delta_mean": float(abs(float(np.mean(xv)) - float(np.mean(yv)))),
        "delta_var": float(abs(float(np.var(xv, ddof=0)) - float(np.var(yv, ddof=0)))),
        "max_abs_delta": float(np.max(abs_delta)),
        "unchanged_ratio": float(np.mean(abs_delta <= 1e-8)),
        "n_valid": int(mask.sum()),
    }


def load_ts48h_matrix(ts_dir: Path, var: str, z: bool) -> tuple[np.ndarray, np.ndarray]:
    fname = f"ts_48h_{var}_zscore.csv" if z else f"ts_48h_{var}.csv"
    df = pd.read_csv(ts_dir / fname)
    stay_ids = df["stay_id"].values.astype(int)
    cols = [c for c in df.columns if c != "stay_id" and str(c).isdigit()]
    cols = sorted(cols, key=int)
    X = df[cols].to_numpy(dtype=float)
    return X, stay_ids


def load_perturbed_matrix(perturbed_base: Path, var: str, op: str, alpha: float) -> np.ndarray:
    stem = f"{op}_n_passes=5_max_diff={alpha}" if op in ("T1_uniform", "T1_weighted") else f"{op}_max_diff={alpha}"
    path = perturbed_base / "perturbed" / "z" / var / f"{stem}.csv"
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    vec = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    n = len(vec)
    n_stays = n // 48
    vec = vec[: n_stays * 48]
    return vec.reshape(n_stays, 48)


def infer_z_to_phys_params(ts_dir: Path, var: str) -> tuple[float, float]:
    """Infer mean/std from paired raw phys and zscore tables (global)."""
    Xz, _ = load_ts48h_matrix(ts_dir, var, z=True)
    Xp, _ = load_ts48h_matrix(ts_dir, var, z=False)
    z = Xz.reshape(-1)
    p = Xp.reshape(-1)
    m = np.isfinite(z) & np.isfinite(p)
    if m.sum() < 100:
        return 0.0, 1.0
    # p ≈ z*std + mean
    std = float(np.nanstd(p[m])) / max(float(np.nanstd(z[m])), 1e-12)
    mean = float(np.nanmean(p[m]) - np.nanmean(z[m]) * std)
    return mean, std


def out_of_range_ratio_from_z(
    z_mat: np.ndarray,
    mean: float,
    std: float,
    lo: float,
    hi: float,
) -> float:
    x = z_mat * std + mean
    m = np.isfinite(x)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((x[m] < lo) | (x[m] > hi)))


DEFAULT_BOUNDS = {
    "HR": (30.0, 220.0),
    "Glucose": (40.0, 500.0),
    "MAP": (30.0, 160.0),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Q-mix pilot (T(Qx) vs x)")
    ap.add_argument("--raw-ts-dir", type=Path, required=True)
    ap.add_argument("--weak-ts-dir", type=Path, required=True)
    ap.add_argument("--cohort-csv", type=Path, required=True)
    ap.add_argument("--variables", nargs="+", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--secret-seed", type=int, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Repository root for locating dependent scripts.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from repo_discovery import find_make_tradeoff_plots

    build_qmix = repo_root / "p7_qmix_pilot" / "code" / "build_qmix_ts_dir.py"
    run_a2 = repo_root / "P3_weak_interpolation" / "code" / "run_a2_on_ts_zonly_np5.py"
    run_attack = repo_root / "privacy_evaluation_protocol" / "code" / "run_privacy_protocol.py"
    tradeoff = find_make_tradeoff_plots(repo_root)

    cohort = pd.read_csv(args.cohort_csv)
    stay_ids = cohort["stay_id"].values.astype(int)
    y_los = make_labels_los_binary(cohort)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for alpha in args.alphas:
        alpha_dir = out_root / parse_alpha_dir(alpha)
        alpha_dir.mkdir(parents=True, exist_ok=True)

        for regime, ts_dir in [("strong", Path(args.raw_ts_dir)), ("weak", Path(args.weak_ts_dir))]:
            # Build qmix ts_dir (zscore only, single-column)
            qmix_ts_dir = alpha_dir / f"qmix_ts_{regime}_seed{args.secret_seed}"
            qmix_ts_dir.mkdir(parents=True, exist_ok=True)
            run_cmd(
                f"PYTHONUNBUFFERED=1 python -u \"{build_qmix}\" "
                f"--raw-ts-dir \"{ts_dir}\" --out-ts-dir \"{qmix_ts_dir}\" "
                f"--variables {' '.join(args.variables)} --secret-seed {int(args.secret_seed)}"
            )

            # Apply operators on qmix input
            a2_out = alpha_dir / f"a2_qmix_{regime}_alpha={alpha}"
            run_cmd(
                f"PYTHONUNBUFFERED=1 python -u \"{run_a2}\" "
                f"--ts-dir \"{qmix_ts_dir}\" --out-dir \"{a2_out}\" "
                f"--variables {' '.join(args.variables)} --alpha {alpha} "
                f"--operators {' '.join(OPS_KEEP)}"
            )

            # Attack A: reconstruct raw x from z=T(Qx)
            run_cmd(
                f"PYTHONUNBUFFERED=1 python -u \"{run_attack}\" --attack A "
                f"--leakage L2 "
                f"--data-dir \"{ts_dir}\" --perturbed-dir \"{a2_out / 'perturbed'}\" "
                f"--variables {' '.join(args.variables)} --alpha {alpha}"
            )
            # snapshot attack table
            src = repo_root / "privacy_evaluation_protocol" / "results" / "tables" / "table_attack_a_reconstruction.csv"
            snap = alpha_dir / f"pr_qmix_{regime}_alpha={alpha}.csv"
            shutil.copyfile(src, snap)

            # Export leaf tables (var-wise)
            pr_df = pd.read_csv(snap)
            pr_df = pr_df[(pr_df["setting"] == "z") & (pr_df["model"] == "linear") & (pr_df["leakage"] == "L2")]
            pr_df = pr_df[pr_df["operator"].isin(OPS_KEEP)]

            for var in args.variables:
                leaf = alpha_dir / var / regime
                leaf.mkdir(parents=True, exist_ok=True)

                pr_v = pr_df[pr_df["var"] == var][["operator", "R2", "MAE_z", "rho", "n_train", "n_test"]].copy()
                pr_v = pr_v.rename(columns={"R2": "attackA_R2", "MAE_z": "attackA_MAEz"})
                pr_v["operator"] = pd.Categorical(pr_v["operator"], categories=OPS_KEEP, ordered=True)
                pr_v = pr_v.sort_values("operator")
                pr_v.to_csv(leaf / "table_pr.csv", index=False)

                # UT: delta-AUC + shape + eps stats + out_of_range
                X_raw_z, sids = load_ts48h_matrix(ts_dir, var, z=True)
                if len(sids) != len(stay_ids) or not np.all(sids == stay_ids):
                    sid2row = {int(s): i for i, s in enumerate(sids)}
                    X_aligned = np.full((len(stay_ids), 48), np.nan)
                    for i, sid in enumerate(stay_ids):
                        if int(sid) in sid2row:
                            X_aligned[i] = X_raw_z[sid2row[int(sid)]]
                    X_raw_z = X_aligned

                feats_raw = extract_features(X_raw_z)
                flat_raw = X_raw_z.reshape(-1)

                mean_phys, std_phys = infer_z_to_phys_params(ts_dir, var)
                lo, hi = DEFAULT_BOUNDS.get(var, (-np.inf, np.inf))

                rows = []
                for op in OPS_KEEP:
                    Z = load_perturbed_matrix(a2_out, var, op, alpha)
                    n = min(Z.shape[0], feats_raw.shape[0])
                    feats_op = extract_features(Z[:n])
                    delta = eval_paired_raw_vs_op(feats_raw[:n], feats_op, y_los[:n], seed=args.seed)

                    flat_op = Z[:n].reshape(-1)
                    nflat = min(len(flat_raw[: n * 48]), len(flat_op))
                    shape = run_shape_metrics(flat_raw[: n * 48][:nflat], flat_op[:nflat])

                    eps = (flat_op[:nflat] - flat_raw[: n * 48][:nflat])
                    eps_abs = np.abs(eps[np.isfinite(eps)])
                    if eps_abs.size:
                        eps_stats = {
                            "eps_std": float(np.nanstd(eps)),
                            "eps_median_abs": float(np.nanmedian(eps_abs)),
                            "eps_p95_abs": float(np.nanpercentile(eps_abs, 95)),
                            "eps_p99_abs": float(np.nanpercentile(eps_abs, 99)),
                        }
                    else:
                        eps_stats = {"eps_std": np.nan, "eps_median_abs": np.nan, "eps_p95_abs": np.nan, "eps_p99_abs": np.nan}

                    oor = out_of_range_ratio_from_z(Z[:n], mean_phys, std_phys, lo=lo, hi=hi)

                    rows.append({
                        "operator": op,
                        **delta,
                        **shape,
                        **eps_stats,
                        "out_of_range_ratio": oor,
                    })

                ut = pd.DataFrame(rows)
                ut["operator"] = pd.Categorical(ut["operator"], categories=OPS_KEEP, ordered=True)
                ut = ut.sort_values("operator")
                cols = [
                    "operator",
                    "delta_auroc", "delta_auprc", "auroc_raw", "auroc_op", "auprc_raw", "auprc_op", "n_total", "n_train", "n_test",
                    "utility_score", "ks_statistic", "ks_pvalue", "delta_mean", "delta_var", "max_abs_delta", "unchanged_ratio", "n_valid",
                    "eps_std", "eps_median_abs", "eps_p95_abs", "eps_p99_abs",
                    "out_of_range_ratio",
                ]
                for c in cols:
                    if c not in ut.columns:
                        ut[c] = np.nan
                ut = ut[cols]
                ut.to_csv(leaf / "table_ut.csv", index=False)

        # tradeoff plots for this alpha_dir (treat as out_v2-like)
        run_cmd(f"PYTHONUNBUFFERED=1 python -u \"{tradeoff}\" --out-v2 \"{alpha_dir}\"")

    print("Done. Results at:", out_root)


if __name__ == "__main__":
    main()

