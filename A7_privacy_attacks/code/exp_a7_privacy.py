#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a7_privacy.py — Sec. A.7: privacy & numeric reconstruction attacks (PDF §1.2.4).

1) Perturbation statistics: mean/std/quantiles of Δ = y − x; tables + boxplots.
2) Regression reconstruction: predict X from Y (perturbed); linear + small MLP; R², RMSE, ρ.
3) Multi-run variability: K runs per operator; pairwise KS and ρ between runs.

Usage (from A7_privacy_attacks/code):
  python3 exp_a7_privacy.py
  python3 exp_a7_privacy.py --variables HR SBP Creatinine --K 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import default_a2_perturbed_dir, default_ts48h_dir, find_a2_operator_code


def _ensure_a2_on_path() -> None:
    p = find_a2_operator_code(ROOT)
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# Fixed threshold in normalized space (aligned with A2 numeric_operators.NORMALIZED_MAX_DIFF)
NORMALIZED_MAX_DIFF = 1.0  # 1.0 for stronger reconstruction resistance (0.5→0.8→1.0)
# Comparable parameters for reconstruction (normalized space)
RECON_PARAMS = {
    "T1_uniform": "n_passes=5,max_diff=1.0",
    "T1_weighted": "n_passes=5,max_diff=1.0",
    "T2": "max_diff=1.0",
    "T3": "max_diff=1.0",
}
# 多次扰动 K 次
DEFAULT_K = 5
# 回归攻击：对比实验固定两档——大量配对 20%、极少配对 0.01%（0.0001）；单档运行时默认用 20%
TRAIN_FRAC = 0.2
RECON_COMPARE_FRAC_LOW = 0.0001  # 0.01%，极少配对
# 数据量低于此值时不再跑 0.01% 档（训练样本过少），仅保留 20% 档
MIN_N_FOR_001PCT = 10000
MLP_HIDDEN = (64, 32)
MLP_MAX_ITER = 200


def _estimate_n_for_recon(data_dir: Path, perturbed_dir: Path, variables: list[str] | None, max_n: int | None) -> int:
    """Estimate typical sample size for reconstruction (whether to run 0.01% regime)."""
    for setting in ["z", "phys"]:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here[:1]:  # first variable only
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            n = int((~np.isnan(x)).sum())
            if max_n and n > max_n:
                n = max_n
            return n
    return 0


def load_single_column(data_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载原始单列。"""
    fname = f"ts_single_column_{var}_zscore.csv" if setting == "z" else f"ts_single_column_{var}.csv"
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def load_perturbed(perturbed_dir: Path, setting: str, var: str, stem: str) -> np.ndarray:
    """加载 A.2 扰动结果。"""
    path = perturbed_dir / setting / var / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def parse_perturbed_filename(stem: str) -> tuple[str, str]:
    """从文件名 stem 解析 operator 与 params。"""
    if stem.startswith("T1_uniform_"):
        op = "T1_uniform"
        rest = stem.replace("T1_uniform_", "")
    elif stem.startswith("T1_weighted_"):
        op = "T1_weighted"
        rest = stem.replace("T1_weighted_", "")
    elif stem.startswith("T2_"):
        op = "T2"
        rest = stem.replace("T2_", "")
    elif stem.startswith("T3_"):
        op = "T3"
        rest = stem.replace("T3_", "")
    else:
        op = stem.split("_")[0] if "_" in stem else stem
        rest = stem
    params = rest.replace("_max_diff=", ",max_diff=").replace("_n_passes=", ",n_passes=")
    return op, params


def stem_from_operator_params(operator: str, params: str) -> str:
    """由 operator 与 params 反推文件名 stem。"""
    if params:
        p = params.replace(",max_diff=", "_max_diff=").replace(",n_passes=", "_n_passes=")
        return f"{operator}_{p}"
    return operator


# ---------- 1) 扰动分布统计 ----------


def run_delta_stats(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: list[str] | None = None,
) -> None:
    """对每个 (var, setting, operator, params) 计算 Δ = y - x 的统计量，写表并画 boxplot。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    box_data: list[dict] = []  # var, setting, operator_label, delta_values

    for setting in ["z", "phys"]:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            var_path = setting_path / var
            for csv_path in sorted(var_path.glob("*.csv")):
                stem = csv_path.stem
                operator, params = parse_perturbed_filename(stem)
                try:
                    y = load_perturbed(perturbed_dir, setting, var, stem)
                except FileNotFoundError:
                    continue
                if len(y) != len(x):
                    continue
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 10:
                    continue
                delta = (y - x)[mask]
                d = delta.astype(float)
                rows.append({
                    "var": var,
                    "setting": setting,
                    "operator": operator,
                    "params": params,
                    "delta_mean": float(np.mean(d)),
                    "delta_std": float(np.std(d)),
                    "delta_p10": float(np.percentile(d, 10)),
                    "delta_p50": float(np.percentile(d, 50)),
                    "delta_p90": float(np.percentile(d, 90)),
                    "delta_max": float(np.max(np.abs(d))),
                    "n": int(mask.sum()),
                })
                # 用于 boxplot：只保留代表参数，避免图过于拥挤
                if params == RECON_PARAMS.get(operator, ""):
                    label = f"{operator}"
                    box_data.append({
                        "var": var,
                        "setting": setting,
                        "operator": label,
                        "delta": d,
                    })

    pd.DataFrame(rows).to_csv(tables_dir / "table_delta_stats.csv", index=False)
    print(f"[A.7.1] 扰动分布统计表: {tables_dir / 'table_delta_stats.csv'} ({len(rows)} 行)")

    # Boxplot：每个变量一张图，按 operator 分组（仅 z、代表参数）
    for var in sorted(set(b["var"] for b in box_data)):
        subset = [b for b in box_data if b["var"] == var and b["setting"] == "z"]
        if not subset:
            continue
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[A.7.1] 跳过 boxplot：未安装 matplotlib")
            break
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [s["operator"] for s in subset]
        data = [s["delta"] for s in subset]
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_ylabel(r"$\Delta = y - x$")
        ax.set_title(f"Perturbation delta ({var}, z)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(figs_dir / f"boxplot_delta_{var}.pdf")
        plt.close()
        print(f"[A.7.1] 图: boxplot_delta_{var}.pdf")


# ---------- 2) 回归重建攻击 ----------


def _linear_regression_fit_predict(Y_train: np.ndarray, X_train: np.ndarray, Y_test: np.ndarray) -> np.ndarray:
    """纯 NumPy OLS：用 Y 预测 X，返回 X_pred。形状 (n,1)。"""
    # X ≈ Y @ coef + intercept；等价于 [Y, 1] @ [coef; intercept]
    n_tr = Y_train.shape[0]
    ones = np.ones((n_tr, 1), dtype=float)
    Y_aug = np.hstack([Y_train.reshape(-1, 1), ones])
    beta, _, _, _ = np.linalg.lstsq(Y_aug, X_train.ravel(), rcond=None)
    n_te = Y_test.shape[0]
    Y_te_aug = np.hstack([Y_test.reshape(-1, 1), np.ones((n_te, 1), dtype=float)])
    return (Y_te_aug @ beta).reshape(-1, 1)


def run_reconstruction_attack(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: list[str] | None = None,
    train_frac: float = TRAIN_FRAC,
    seed: int = 42,
    max_n: int | None = 100000,
    train_n: int | None = None,
    out_table_suffix: str = "",
) -> None:
    """对每个变量、每个算子做线性回归 + 可选 MLP 重建攻击。train_n 非空时固定训练条数（如 100），否则用 train_frac。"""
    use_sklearn = False
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.neural_network import MLPRegressor
        use_sklearn = True
    except ImportError:
        pass

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    run_reconstruction_attack._recon_max_n = max_n
    rows: list[dict] = []

    for setting in ["z", "phys"]:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            mask = ~np.isnan(x)
            if mask.sum() < 500:
                continue
            for operator, params in RECON_PARAMS.items():
                stem = stem_from_operator_params(operator, params)
                try:
                    y = load_perturbed(perturbed_dir, setting, var, stem)
                except FileNotFoundError:
                    continue
                if len(y) != len(x):
                    continue
                valid = mask & ~np.isnan(y)
                if valid.sum() < 500:
                    continue
                X = x[valid].reshape(-1, 1)
                Y = y[valid].reshape(-1, 1)
                n = X.shape[0]
                recon_max_n = getattr(run_reconstruction_attack, "_recon_max_n", 100000)
                if recon_max_n and n > recon_max_n:
                    rng_sub = np.random.default_rng(seed)
                    idx_perm = rng_sub.permutation(n)[:recon_max_n]
                    X, Y = X[idx_perm], Y[idx_perm]
                    n = X.shape[0]
                idx = np.arange(n)
                if train_n is not None:
                    n_tr = min(max(2, train_n), n - 2)
                else:
                    n_tr = max(2, int(n * train_frac))
                    n_tr = min(n_tr, n - 2)
                if use_sklearn:
                    i_train, i_test = train_test_split(idx, train_size=n_tr, random_state=seed, shuffle=True)
                else:
                    perm = rng.permutation(n)
                    i_train, i_test = perm[:n_tr], perm[n_tr:]
                Y_train, Y_test = Y[i_train], Y[i_test]
                X_train, X_test = X[i_train], X[i_test]

                # 线性回归（NumPy OLS）：用 Y 预测 X
                X_pred_lr = _linear_regression_fit_predict(Y_train, X_train, Y_test)
                var_test = float(np.var(X_test))
                r2_lr = float(1 - np.sum((X_test - X_pred_lr) ** 2) / (np.sum((X_test - X_test.mean()) ** 2) + 1e-12)) if var_test > 1e-12 else np.nan
                rmse_lr = float(np.sqrt(np.mean((X_test - X_pred_lr) ** 2)))
                rho_lr = float(np.corrcoef(X_test.ravel(), X_pred_lr.ravel())[0, 1]) if len(X_test) > 1 else np.nan
                # 方案二：按特征自适应阈值（基于该列标准差）计算重构成功率
                # epsilon = alpha * std(x_j)，这里 std 在当前 (var, setting, operator) 的样本上估计
                std_all = float(np.std(X)) if X.size > 1 else 0.0
                succ_rates: dict[str, float] = {}
                for alpha in (0.1, 0.2, 0.5):
                    key = f"succ_std_alpha{alpha}"
                    if std_all > 0:
                        eps = alpha * std_all
                        succ = float(np.mean(np.abs(X_test - X_pred_lr) < eps))
                    else:
                        succ = float("nan")
                    succ_rates[key] = succ
                row_base = {"var": var, "setting": setting, "operator": operator, "params": params, "train_frac": train_frac if train_n is None else (n_tr / n)}
                if train_n is not None:
                    row_base["n_train"] = n_tr
                rows.append({
                    **row_base,
                    "model": "linear", "R2": r2_lr, "RMSE": rmse_lr, "rho": rho_lr,
                    **succ_rates,
                })

                if use_sklearn:
                    mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=seed)
                    mlp.fit(Y_train, X_train.ravel())
                    X_pred_mlp = mlp.predict(Y_test).reshape(-1, 1)
                    r2_mlp = float(1 - np.sum((X_test - X_pred_mlp) ** 2) / (np.sum((X_test - X_test.mean()) ** 2) + 1e-12)) if var_test > 1e-12 else np.nan
                    rmse_mlp = float(np.sqrt(np.mean((X_test - X_pred_mlp) ** 2)))
                    rho_mlp = float(np.corrcoef(X_test.ravel(), X_pred_mlp.ravel())[0, 1]) if len(X_test) > 1 else np.nan
                    rows.append({
                        **row_base,
                        "model": "mlp", "R2": r2_mlp, "RMSE": rmse_mlp, "rho": rho_mlp,
                    })
                    print(f"[A.7.2] {var} / {setting} / {operator} linear R2={r2_lr:.4f} MLP R2={r2_mlp:.4f}")
                else:
                    print(f"[A.7.2] {var} / {setting} / {operator} linear R2={r2_lr:.4f} (无 sklearn，未跑 MLP)")

    out_table = tables_dir / ("table_reconstruction" + (f"_{out_table_suffix}" if out_table_suffix else "") + ".csv")
    pd.DataFrame(rows).to_csv(out_table, index=False)
    print(f"[A.7.2] 回归重建表: {out_table} ({len(rows)} 行, train_frac={train_frac})")
    if not out_table_suffix:
        _plot_reconstruction(out_dir, table_path=None)


def _plot_reconstruction(out_dir: Path, table_path: Path | None = None) -> None:
    """根据指定表或 table_reconstruction_20pct.csv / table_reconstruction.csv 画 R² 按变量、算子分组柱状图。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    path = table_path or tables_dir / "table_reconstruction_20pct.csv"
    if not path.exists():
        path = tables_dir / "table_reconstruction.csv"
    if not path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[A.7.2] 跳过回归攻击图：未安装 matplotlib")
        return
    df = pd.read_csv(path)
    # 仅用 linear 模型，避免图过密
    df = df[df["model"] == "linear"].copy()
    if df.empty:
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub = df[df["setting"] == setting]
        if sub.empty:
            continue
        vars_order = sorted(sub["var"].unique())
        x = np.arange(len(vars_order))
        w = 0.2
        fig, ax = plt.subplots(figsize=(max(6, len(vars_order) * 0.5), 4))
        for i, op in enumerate(ops):
            vals = [sub[(sub["var"] == v) & (sub["operator"] == op)]["R2"].values for v in vars_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            off = (i - 1.5) * w
            ax.bar(x + off, vals, width=w, label=op)
        ax.set_xticks(x)
        ax.set_xticklabels(vars_order, rotation=45, ha="right")
        ax.set_ylabel(r"$R^2$ (linear)")
        ax.set_title(f"Reconstruction attack R² ({title_suffix})")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_reconstruction_R2_{setting}.pdf")
        plt.close()
        print(f"[A.7.2] 图: plot_reconstruction_R2_{setting}.pdf")


def _plot_reconstruction_compare(out_dir: Path) -> None:
    """读取 20% 与 0.01% 两张表，画对比图：同一 setting 下左右两栏（20% vs 0.01%）R² 对比。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    path_20 = tables_dir / "table_reconstruction_20pct.csv"
    path_001 = tables_dir / "table_reconstruction_001pct.csv"
    if not path_20.exists() or not path_001.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df20 = pd.read_csv(path_20)
    df001 = pd.read_csv(path_001)
    df20 = df20[df20["model"] == "linear"].copy()
    df001 = df001[df001["model"] == "linear"].copy()
    if df20.empty or df001.empty:
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub20 = df20[df20["setting"] == setting]
        sub001 = df001[df001["setting"] == setting]
        if sub20.empty or sub001.empty:
            continue
        vars_order = sorted(sub20["var"].unique())
        x = np.arange(len(vars_order))
        w = 0.2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(vars_order) * 1.0), 4), sharey=True)
        for i, op in enumerate(ops):
            v20 = [sub20[(sub20["var"] == v) & (sub20["operator"] == op)]["R2"].values for v in vars_order]
            v20 = [v[0] if len(v) else np.nan for v in v20]
            v001 = [sub001[(sub001["var"] == v) & (sub001["operator"] == op)]["R2"].values for v in vars_order]
            v001 = [v[0] if len(v) else np.nan for v in v001]
            off = (i - 1.5) * w
            ax1.bar(x + off, v20, width=w, label=op, color=f"C{i}", alpha=0.9)
            ax2.bar(x + off, v001, width=w, label=op, color=f"C{i}", alpha=0.9)
        for ax in (ax1, ax2):
            ax.set_xticks(x)
            ax.set_xticklabels(vars_order, rotation=45, ha="right")
            ax.set_ylabel(r"$R^2$ (linear)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        ax1.set_title("20% pairs")
        ax2.set_title("0.01% pairs")
        ax1.legend(loc="lower right", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Reconstruction R² comparison ({title_suffix})", y=1.02)
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_reconstruction_R2_compare_{setting}.pdf")
        plt.close()
        print(f"[A.7.2] 图: plot_reconstruction_R2_compare_{setting}.pdf")


# ---------- 2.5) 无配对攻击（朴素估计 x̂=y）----------

def run_no_pairs_attack(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: list[str] | None = None,
    max_n: int | None = 100000,
) -> None:
    """
    无配对攻击：攻击者只有 y，朴素估计 x̂=y。对每 (变量, setting, 算子) 计算
    MAE(x,y)、RMSE(x,y)、Pearson(x,y)、完全相等比例（atol=1e-8）。
    不训练任何模型，仅衡量扰动幅度。
    """
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for setting in ["z", "phys"]:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            mask = ~np.isnan(x)
            if mask.sum() < 500:
                continue
            for operator, params in RECON_PARAMS.items():
                stem = stem_from_operator_params(operator, params)
                try:
                    y = load_perturbed(perturbed_dir, setting, var, stem)
                except FileNotFoundError:
                    continue
                if len(y) != len(x):
                    continue
                valid = mask & ~np.isnan(y)
                if valid.sum() < 500:
                    continue
                X = x[valid]
                Y = y[valid]
                n = X.shape[0]
                if max_n and n > max_n:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n, size=max_n, replace=False)
                    X, Y = X[idx], Y[idx]
                    n = X.shape[0]
                mae = float(np.mean(np.abs(X - Y)))
                rmse = float(np.sqrt(np.mean((X - Y) ** 2)))
                corr = float(np.corrcoef(X, Y)[0, 1]) if n > 1 and np.std(X) > 1e-12 and np.std(Y) > 1e-12 else np.nan
                exact_ratio = float(np.mean(np.isclose(X, Y, atol=1e-8)))
                rows.append({
                    "var": var,
                    "setting": setting,
                    "operator": operator,
                    "params": params,
                    "n": n,
                    "MAE": mae,
                    "RMSE": rmse,
                    "corr": corr,
                    "exact_ratio": exact_ratio,
                })
                print(f"[A.7.no-pairs] {var} / {setting} / {operator} MAE={mae:.4f} RMSE={rmse:.4f} corr={corr:.4f}")

    out_table = tables_dir / "table_no_pairs_attack.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)
    print(f"[A.7.no-pairs] 无配对攻击表: {out_table} ({len(rows)} 行)")
    _plot_no_pairs_attack(out_dir)


def _plot_no_pairs_attack(out_dir: Path) -> None:
    """根据 table_no_pairs_attack.csv 画 MAE、corr 按变量、算子分组柱状图（z/phys 各一图，两子图）。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    path = tables_dir / "table_no_pairs_attack.csv"
    if not path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub = df[df["setting"] == setting]
        if sub.empty:
            continue
        vars_order = sorted(sub["var"].unique())
        x = np.arange(len(vars_order))
        w = 0.2
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(vars_order) * 0.5), 6), sharex=True)
        for i, op in enumerate(ops):
            mae_vals = [sub[(sub["var"] == v) & (sub["operator"] == op)]["MAE"].values for v in vars_order]
            mae_vals = [v[0] if len(v) else np.nan for v in mae_vals]
            corr_vals = [sub[(sub["var"] == v) & (sub["operator"] == op)]["corr"].values for v in vars_order]
            corr_vals = [v[0] if len(v) else np.nan for v in corr_vals]
            off = (i - 1.5) * w
            ax1.bar(x + off, mae_vals, width=w, label=op, color=f"C{i}", alpha=0.9)
            ax2.bar(x + off, corr_vals, width=w, label=op, color=f"C{i}", alpha=0.9)
        for ax in (ax1, ax2):
            ax.set_xticks(x)
            ax.set_xticklabels(vars_order, rotation=45, ha="right")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        ax1.set_ylabel("MAE (no-pairs: x̂=y)")
        ax1.set_title(f"No-pairs attack ({title_suffix}): MAE (larger = harder to guess x from y)")
        ax1.set_ylim(0, None)
        ax2.set_ylabel("Pearson corr(x, y)")
        ax2.set_title(f"No-pairs attack ({title_suffix}): corr (lower = less linear relation)")
        ax2.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_no_pairs_attack_{setting}.pdf")
        plt.close()
        print(f"[A.7.no-pairs] 图: plot_no_pairs_attack_{setting}.pdf")


# ---------- 3) 多次扰动差异 ----------


def _run_operator_k_times(
    x_filled: np.ndarray,
    nan_mask: np.ndarray,
    operator: str,
    params: str,
    max_diff: float,
    n_passes: int | None,
    seeds: list[int],
) -> list[np.ndarray]:
    """对同一列用同一算子跑 K 次（不同种子）。方案一：先归一化再固定阈值，返回原尺度 [y1, y2, ...]。"""
    _ensure_a2_on_path()
    from numeric_operators import (
        normalize_zscore,
        denormalize_zscore,
        triplet_micro_rotation,
        triplet_micro_rotation_weighted,
        constrained_noise_projection,
        householder_reflection,
    )
    z_work, mu, sigma = normalize_zscore(x_filled)
    outs: list[np.ndarray] = []
    for s in seeds:
        rng = np.random.default_rng(s)
        try:
            if operator == "T1_uniform":
                y_z = triplet_micro_rotation(
                    z_work.copy(), max_diff=max_diff, n_passes=n_passes or 5,
                    x_original=z_work.copy(), rng=rng,
                )
            elif operator == "T1_weighted":
                y_z = triplet_micro_rotation_weighted(
                    z_work.copy(), max_diff=max_diff, n_passes=n_passes or 5,
                    x_original=z_work.copy(), rng=rng,
                )
            elif operator == "T2":
                y_z = constrained_noise_projection(z_work.copy(), max_diff=max_diff, rng=rng)
            elif operator == "T3":
                y_z, _ = householder_reflection(z_work.copy(), max_diff=max_diff, rng=rng, return_n_trials=True)
            else:
                continue
        except (ValueError, RuntimeError):
            continue
        y = denormalize_zscore(y_z, mu, sigma)
        y = np.asarray(y, dtype=float)
        y[nan_mask] = np.nan
        outs.append(y)
    return outs


def apply_strong_pipeline(x: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    numeric 的 privacy-strong 流水线：方案一 先归一化再固定阈值，T1->T2->T3 在归一化空间用 NORMALIZED_MAX_DIFF。
    NaN 用列均值填充，输出中恢复原 NaN 位置。
    """
    _ensure_a2_on_path()
    from numeric_operators import (
        normalize_zscore,
        denormalize_zscore,
        triplet_micro_rotation,
        constrained_noise_projection,
        householder_reflection,
    )
    x = np.asarray(x, dtype=float).ravel()
    nan_mask = np.isnan(x)
    mean_val = np.nanmean(x)
    x_filled = x.copy()
    x_filled[nan_mask] = mean_val
    z_work, mu, sigma = normalize_zscore(x_filled)
    rng = np.random.default_rng(seed)
    # T1
    y1_z = triplet_micro_rotation(
        z_work.copy(), max_diff=NORMALIZED_MAX_DIFF, n_passes=5,
        x_original=z_work.copy(), rng=rng,
    )
    mean1_z = np.nanmean(y1_z)
    y1_filled = np.asarray(y1_z, dtype=float)
    y1_filled[nan_mask] = mean1_z
    # T2
    y2_z = constrained_noise_projection(y1_filled.copy(), max_diff=NORMALIZED_MAX_DIFF, rng=rng)
    mean2_z = np.nanmean(y2_z)
    y2_filled = np.asarray(y2_z, dtype=float)
    y2_filled[nan_mask] = mean2_z
    # T3
    y3_z, _ = householder_reflection(y2_filled.copy(), max_diff=NORMALIZED_MAX_DIFF, rng=rng, return_n_trials=True)
    y3 = denormalize_zscore(y3_z, mu, sigma)
    y3 = np.asarray(y3, dtype=float)
    y3[nan_mask] = np.nan
    return y3


def apply_pipeline_t1_t2(x: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    仅 T1 -> T2（不含 T3）。方案一：先归一化再固定阈值。
    """
    _ensure_a2_on_path()
    from numeric_operators import (
        normalize_zscore,
        denormalize_zscore,
        triplet_micro_rotation,
        constrained_noise_projection,
    )
    x = np.asarray(x, dtype=float).ravel()
    nan_mask = np.isnan(x)
    mean_val = np.nanmean(x)
    x_filled = x.copy()
    x_filled[nan_mask] = mean_val
    z_work, mu, sigma = normalize_zscore(x_filled)
    rng = np.random.default_rng(seed)
    y1_z = triplet_micro_rotation(
        z_work.copy(), max_diff=NORMALIZED_MAX_DIFF, n_passes=5,
        x_original=z_work.copy(), rng=rng,
    )
    y1_filled = np.asarray(y1_z, dtype=float)
    y1_filled[nan_mask] = np.nanmean(y1_z)
    y2_z = constrained_noise_projection(y1_filled.copy(), max_diff=NORMALIZED_MAX_DIFF, rng=rng)
    y2 = denormalize_zscore(y2_z, mu, sigma)
    y2 = np.asarray(y2, dtype=float)
    y2[nan_mask] = np.nan
    return y2


def run_reconstruction_attack_pipeline_strong(
    data_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
    variables: list[str] | None = None,
    train_frac: float = TRAIN_FRAC,
    seed: int = 42,
    max_n: int | None = 100000,
) -> None:
    """
    对 numeric 的两种组合做重建攻击：(1) T1->T2（无 T3，strong 隐私更优）；(2) T1->T2->T3。
    用与 run_reconstruction_attack 相同的 20% 训练线性回归，得到 R²/RMSE/ρ。
    结果写入 table_reconstruction_pipeline_strong.csv，便于与单算子对比。
    """
    use_sklearn = False
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor
        use_sklearn = True
    except ImportError:
        pass

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    pipelines = [
        ("pipeline_T1_T2", "T1->T2", apply_pipeline_t1_t2),
        ("pipeline_strong", "T1->T2->T3", apply_strong_pipeline),
    ]

    for setting in ["z", "phys"]:
        setting_path = perturbed_dir / setting
        if not setting_path.exists():
            continue
        vars_here = sorted([d.name for d in setting_path.iterdir() if d.is_dir()])
        if variables:
            vars_here = [v for v in vars_here if v in variables]
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            mask = ~np.isnan(x)
            if mask.sum() < 500:
                continue
            for op_name, params_str, apply_fn in pipelines:
                y = apply_fn(x, seed=seed)
                valid = mask & ~np.isnan(y)
                if valid.sum() < 500:
                    continue
                X = x[valid].reshape(-1, 1)
                Y = y[valid].reshape(-1, 1)
                n = X.shape[0]
                if max_n and n > max_n:
                    rng_sub = np.random.default_rng(seed)
                    idx_perm = rng_sub.permutation(n)[:max_n]
                    X, Y = X[idx_perm], Y[idx_perm]
                    n = X.shape[0]
                idx = np.arange(n)
                n_tr = max(2, min(int(n * train_frac), n - 2))
                if use_sklearn:
                    i_train, i_test = train_test_split(idx, train_size=n_tr, random_state=seed, shuffle=True)
                else:
                    perm = rng.permutation(n)
                    i_train, i_test = perm[:n_tr], perm[n_tr:]
                Y_train, Y_test = Y[i_train], Y[i_test]
                X_train, X_test = X[i_train], X[i_test]

                X_pred_lr = _linear_regression_fit_predict(Y_train, X_train, Y_test)
                var_test = float(np.var(X_test))
                r2_lr = float(1 - np.sum((X_test - X_pred_lr) ** 2) / (np.sum((X_test - X_test.mean()) ** 2) + 1e-12)) if var_test > 1e-12 else np.nan
                rmse_lr = float(np.sqrt(np.mean((X_test - X_pred_lr) ** 2)))
                rho_lr = float(np.corrcoef(X_test.ravel(), X_pred_lr.ravel())[0, 1]) if len(X_test) > 1 else np.nan
                row_base = {"var": var, "setting": setting, "operator": op_name, "params": params_str, "train_frac": train_frac}
                rows.append({**row_base, "model": "linear", "R2": r2_lr, "RMSE": rmse_lr, "rho": rho_lr})

                if use_sklearn:
                    mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=seed)
                    mlp.fit(Y_train, X_train.ravel())
                    X_pred_mlp = mlp.predict(Y_test).reshape(-1, 1)
                    r2_mlp = float(1 - np.sum((X_test - X_pred_mlp) ** 2) / (np.sum((X_test - X_test.mean()) ** 2) + 1e-12)) if var_test > 1e-12 else np.nan
                    rmse_mlp = float(np.sqrt(np.mean((X_test - X_pred_mlp) ** 2)))
                    rho_mlp = float(np.corrcoef(X_test.ravel(), X_pred_mlp.ravel())[0, 1]) if len(X_test) > 1 else np.nan
                    rows.append({**row_base, "model": "mlp", "R2": r2_mlp, "RMSE": rmse_mlp, "rho": rho_mlp})

                print(f"[A.7.pipeline] {var} / {setting} {op_name} linear R2={r2_lr:.4f}")

    out_table = tables_dir / "table_reconstruction_pipeline_strong.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)
    print(f"[A.7.pipeline] 表: {out_table} ({len(rows)} 行)")
    _plot_reconstruction_with_pipeline(out_dir)


def _plot_reconstruction_with_pipeline(out_dir: Path) -> None:
    """在单算子 R² 图基础上叠加 pipeline_T1_T2 与 pipeline_strong 的 R²（若存在 pipeline 表）。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    path_main = tables_dir / "table_reconstruction_20pct.csv"
    path_pipeline = tables_dir / "table_reconstruction_pipeline_strong.csv"
    if not path_main.exists() or not path_pipeline.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df_main = pd.read_csv(path_main)
    df_main = df_main[df_main["model"] == "linear"].copy()
    df_pipe = pd.read_csv(path_pipeline)
    df_pipe = df_pipe[df_pipe["model"] == "linear"].copy()
    if df_main.empty or df_pipe.empty:
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3", "pipeline_T1_T2", "pipeline_strong"]
    pipeline_ops = ["pipeline_T1_T2", "pipeline_strong"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub = df_main[df_main["setting"] == setting]
        sub_p = df_pipe[df_pipe["setting"] == setting]
        if sub.empty:
            continue
        vars_order = sorted(sub["var"].unique())
        x = np.arange(len(vars_order))
        w = 0.13
        fig, ax = plt.subplots(figsize=(max(6, len(vars_order) * 0.5), 4))
        for i, op in enumerate(ops):
            if op in pipeline_ops:
                vals = [sub_p[(sub_p["var"] == v) & (sub_p["operator"] == op)]["R2"].values for v in vars_order]
            else:
                vals = [sub[(sub["var"] == v) & (sub["operator"] == op)]["R2"].values for v in vars_order]
            vals = [v[0] if len(v) else np.nan for v in vals]
            off = (i - 2.5) * w
            ax.bar(x + off, vals, width=w, label=op)
        ax.set_xticks(x)
        ax.set_xticklabels(vars_order, rotation=45, ha="right")
        ax.set_ylabel(r"$R^2$ (linear)")
        ax.set_title(f"Reconstruction R² ({title_suffix}) — single op vs T1+T2 vs T1+T2+T3")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_reconstruction_R2_with_pipeline_{setting}.pdf")
        plt.close()
        print(f"[A.7.pipeline] 图: plot_reconstruction_R2_with_pipeline_{setting}.pdf")


def run_multi_run_difference(
    data_dir: Path,
    out_dir: Path,
    variables: list[str] | None = None,
    K: int = DEFAULT_K,
    seed_start: int = 42,
    subsample_n: int | None = 30000,
) -> None:
    """同一列上每个算子跑 K 次，对任意两次算 KS 与 ρ。subsample_n 为 None 时用全列，否则用前 subsample_n 点以加速。"""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    seeds = [seed_start + k for k in range(K)]

    # 只跑“代表参数”（方案一：归一化空间固定阈值 0.8）
    operator_param_list: list[tuple[str, float, int | None]] = [
        ("T1_uniform", NORMALIZED_MAX_DIFF, 5),
        ("T1_weighted", NORMALIZED_MAX_DIFF, 5),
        ("T2", NORMALIZED_MAX_DIFF, None),
        ("T3", NORMALIZED_MAX_DIFF, None),
    ]

    data_dir_p = Path(data_dir)
    vars_here = sorted(set(
        f.stem.replace("ts_single_column_", "").replace("_zscore", "")
        for f in data_dir_p.glob("ts_single_column_*.csv")
    ))
    if variables:
        vars_here = [v for v in vars_here if v in variables]

    rows: list[dict] = []
    for setting in ["z", "phys"]:
        for var in vars_here:
            try:
                x = load_single_column(data_dir, var, setting)
            except FileNotFoundError:
                continue
            x = np.asarray(x, dtype=float)
            nan_mask = np.isnan(x)
            if subsample_n is not None and len(x) > subsample_n:
                x = x[:subsample_n]
                nan_mask = nan_mask[:subsample_n]
            x_filled = x.copy()
            if np.any(nan_mask):
                x_filled[nan_mask] = np.nanmean(x[~nan_mask])
            for operator, max_diff, n_passes in operator_param_list:
                ys = _run_operator_k_times(x_filled, nan_mask, operator, "", max_diff, n_passes, seeds)
                if len(ys) < 2:
                    continue
                ks_vals = []
                rho_vals = []
                for i in range(len(ys)):
                    for j in range(i + 1, len(ys)):
                        valid = ~(np.isnan(ys[i]) | np.isnan(ys[j]))
                        if valid.sum() < 100:
                            continue
                        a, b = ys[i][valid], ys[j][valid]
                        ks_vals.append(float(stats.ks_2samp(a, b).statistic))
                        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
                            rho_vals.append(float(np.corrcoef(a, b)[0, 1]))
                if not ks_vals:
                    continue
                rows.append({
                    "var": var,
                    "setting": setting,
                    "operator": operator,
                    "mean_KS": float(np.mean(ks_vals)),
                    "max_KS": float(np.max(ks_vals)),
                    "mean_rho": float(np.mean(rho_vals)) if rho_vals else np.nan,
                    "min_rho": float(np.min(rho_vals)) if rho_vals else np.nan,
                    "n_pairs": len(ks_vals),
                })
                print(f"[A.7.3] {var} / {setting} / {operator} mean_KS={rows[-1]['mean_KS']:.4f} min_rho={rows[-1].get('min_rho', np.nan):.4f}")

    pd.DataFrame(rows).to_csv(tables_dir / "table_multi_run_KS_rho.csv", index=False)
    print(f"[A.7.3] 多次扰动差异表: {tables_dir / 'table_multi_run_KS_rho.csv'} ({len(rows)} 行)")
    _plot_multi_run(out_dir)


def _plot_multi_run(out_dir: Path) -> None:
    """根据 table_multi_run_KS_rho.csv 画 mean_KS、mean_rho 按变量、算子分组柱状图（z / phys 各一图）。"""
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    path = tables_dir / "table_multi_run_KS_rho.csv"
    if not path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[A.7.3] 跳过多次扰动差异图：未安装 matplotlib")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    ops = ["T1_uniform", "T1_weighted", "T2", "T3"]
    for setting, title_suffix in [("z", "z-score"), ("phys", "phys")]:
        sub = df[df["setting"] == setting]
        if sub.empty:
            continue
        vars_order = sorted(sub["var"].unique())
        x = np.arange(len(vars_order))
        w = 0.2
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(vars_order) * 0.5), 6), sharex=True)
        for i, op in enumerate(ops):
            ks = [sub[(sub["var"] == v) & (sub["operator"] == op)]["mean_KS"].values for v in vars_order]
            ks = [v[0] if len(v) else np.nan for v in ks]
            rho = [sub[(sub["var"] == v) & (sub["operator"] == op)]["mean_rho"].values for v in vars_order]
            rho = [v[0] if len(v) else np.nan for v in rho]
            off = (i - 1.5) * w
            ax1.bar(x + off, ks, width=w, label=op)
            ax2.bar(x + off, rho, width=w, label=op)
        for ax in (ax1, ax2):
            ax.set_xticks(x)
            ax.set_xticklabels(vars_order, rotation=45, ha="right")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        ax1.set_ylabel("mean KS")
        ax1.set_title(f"Multi-run difference ({title_suffix}): mean KS (higher = more variability)")
        ax1.set_ylim(0, None)
        ax2.set_ylabel(r"mean $\rho$")
        ax2.set_title(r"Multi-run difference ({0}): mean $\rho$ (lower = more variability)".format(title_suffix))
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(figs_dir / f"plot_multi_run_KS_rho_{setting}.pdf")
        plt.close()
        print(f"[A.7.3] 图: plot_multi_run_KS_rho_{setting}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="A.7 privacy and numeric reconstruction attacks")
    parser.add_argument("--data-dir", type=str, default=None, help="ts_48h directory")
    parser.add_argument("--perturbed-dir", type=str, default=None, help="A.2 results/perturbed")
    parser.add_argument("--out-dir", type=str, default=None, help="A.7 results directory")
    parser.add_argument("--max-diff", type=float, default=None, help="Normalized-space alpha; default: NORMALIZED_MAX_DIFF in this file")
    parser.add_argument("--variables", type=str, nargs="*", default=None)
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Number of multi-run repetitions")
    parser.add_argument("--skip-delta", action="store_true", help="Skip perturbation statistics")
    parser.add_argument("--skip-recon", action="store_true", help="Skip reconstruction attack")
    parser.add_argument("--skip-no-pairs", action="store_true", help="Skip no-pairs attack (naive x_hat=y)")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-run variability")
    parser.add_argument("--figs-only", action="store_true", help="Regenerate figures from existing tables only")
    parser.add_argument("--multi-subsample-n", type=int, default=30000, help="Multi-run: use first N points per column (0=all)")
    parser.add_argument("--recon-max-n", type=int, default=100000, help="Reconstruction: max points per column (0=all)")
    parser.add_argument("--recon-train-n", type=int, default=None, help="Fixed train size (e.g. 100); else train_frac")
    parser.add_argument("--run-pipeline-strong", action="store_true", help="Run T1->T2->T3 pipeline reconstruction")
    args = parser.parse_args()

    # Default data layout is discovered (no hard-coded localized folder names).
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        try:
            data_dir = default_ts48h_dir(ROOT)
        except FileNotFoundError as e:
            print(f"Error: {e} Use --data-dir.")
            sys.exit(1)
    perturbed_dir = (
        Path(args.perturbed_dir)
        if args.perturbed_dir
        else default_a2_perturbed_dir(ROOT)
    )
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR.parent / "results"

    # 可选：用命令行 α 覆盖归一化空间阈值，与扰动数据一致
    if getattr(args, "max_diff", None) is not None:
        alpha = float(args.max_diff)
        globals()["NORMALIZED_MAX_DIFF"] = alpha
        globals()["RECON_PARAMS"] = {
            "T1_uniform": f"n_passes=5,max_diff={alpha}",
            "T1_weighted": f"n_passes=5,max_diff={alpha}",
            "T2": f"max_diff={alpha}",
            "T3": f"max_diff={alpha}",
        }
        print(f"[A.7] Using alpha (max_diff) = {alpha}")

    if not data_dir.exists() and not args.figs_only:
        print(f"Error: data directory does not exist: {data_dir}")
        sys.exit(1)
    if not perturbed_dir.exists() and not args.figs_only:
        print(f"Error: perturbed directory does not exist: {perturbed_dir}")
        sys.exit(1)

    if args.figs_only:
        _plot_reconstruction(out_dir, table_path=out_dir / "tables" / "table_reconstruction_20pct.csv")
        _plot_reconstruction_compare(out_dir)
        _plot_reconstruction_with_pipeline(out_dir)
        _plot_no_pairs_attack(out_dir)
        _plot_multi_run(out_dir)
        print("\nA.7 figures-only pass complete.")
        return

    if not args.skip_delta:
        run_delta_stats(data_dir, perturbed_dir, out_dir, variables=args.variables)
    if not args.skip_recon:
        max_n = None if getattr(args, "recon_max_n", 100000) == 0 else args.recon_max_n
        train_n = getattr(args, "recon_train_n", None)
        # 主档：20% 或固定 train_n 条训练（--recon-train-n 100 时用 100 条训练）
        run_reconstruction_attack(
            data_dir, perturbed_dir, out_dir,
            variables=args.variables,
            train_frac=0.2,
            max_n=max_n,
            train_n=train_n,
            out_table_suffix="20pct",
        )
        if train_n is not None:
            print(f"[A.7.2] Wrote table_reconstruction_20pct.csv with fixed train_n={train_n}")
        est_n = _estimate_n_for_recon(data_dir, perturbed_dir, args.variables, max_n)
        if train_n is None and est_n >= MIN_N_FOR_001PCT:
            run_reconstruction_attack(
                data_dir, perturbed_dir, out_dir,
                variables=args.variables,
                train_frac=RECON_COMPARE_FRAC_LOW,
                max_n=max_n,
                out_table_suffix="001pct",
            )
        elif train_n is None:
            print(f"[A.7.2] n≈{est_n} < {MIN_N_FOR_001PCT}; skipping 0.01% regime, keeping 20% only.")
        _plot_reconstruction(out_dir, table_path=out_dir / "tables" / "table_reconstruction_20pct.csv")
        _plot_reconstruction_compare(out_dir)
    if getattr(args, "run_pipeline_strong", False):
        max_n = None if getattr(args, "recon_max_n", 100000) == 0 else args.recon_max_n
        run_reconstruction_attack_pipeline_strong(
            data_dir, perturbed_dir, out_dir,
            variables=args.variables,
            train_frac=0.2,
            seed=42,
            max_n=max_n,
        )
    if not args.skip_no_pairs:
        run_no_pairs_attack(
            data_dir, perturbed_dir, out_dir,
            variables=args.variables,
            max_n=None if args.recon_max_n == 0 else args.recon_max_n,
        )
    if not args.skip_multi:
        sub_n = None if args.multi_subsample_n == 0 else args.multi_subsample_n
        run_multi_run_difference(data_dir, out_dir, variables=args.variables, K=args.K, subsample_n=sub_n)

    print("\nA.7 done.")


if __name__ == "__main__":
    main()
