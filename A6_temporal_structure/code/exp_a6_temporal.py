#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_a6_temporal.py — 实验 A.6：时间结构检验（PDF 1.2.3）

1) ACF/PACF：对每个变量取 cohort 平均序列（或若干代表 stays），算原始 vs 各算子版本的 ACF/PACF；
   比较主要滞后峰是否保留；输出表 + 曲线图。
2) Ljung–Box Q：在滞后 h=5,10,20 下对原始与扰动序列做检验，统计非白噪声性是否被削弱。
3) 频域：对 HR 等有周期性的变量做 Welch 功率谱，比较主频与能量分布。

输出：
  - results/tables/table_acf_pacf.csv（关键滞后上的 ACF/PACF 值）
  - results/tables/table_ljungbox.csv
  - results/tables/table_spectral.csv（主频、总功率等）
  - results/figs/acf_pacf_{var}.pdf、psd_{var}.pdf、ljungbox_{var}.pdf（Ljung–Box p 值）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from repo_discovery import default_a2_perturbed_dir, default_ts48h_dir

WINDOW_HOURS = 48
REP_VARS = ("HR", "SBP", "Creatinine")  # 代表变量，用于画 ACF/PACF 与 PSD
ACF_LAGS = [1, 5, 10, 15, 20]  # 保存这些滞后的 ACF/PACF
LB_LAGS = [5, 10, 20]


def load_ts48h_matrix(ts_dir: Path, var: str, setting: str) -> np.ndarray:
    """加载 (n_stays, 48) 矩阵。setting z 用 _zscore。"""
    if setting == "z":
        path = ts_dir / f"ts_48h_{var}_zscore.csv"
    else:
        path = ts_dir / f"ts_48h_{var}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    hour_cols = [str(h) for h in range(WINDOW_HOURS)]
    cols = [c for c in hour_cols if c in df.columns]
    return df[cols].astype(float).to_numpy()


def load_perturbed_48h(perturbed_dir: Path, setting: str, var: str, stem: str, n_stays: int) -> np.ndarray:
    """从 A.2 扰动单列读出并 reshape 为 (n_stays, 48)。"""
    path = perturbed_dir / setting / var / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    col = "value" if "value" in df.columns else df.columns[0]
    full = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return full.reshape(n_stays, WINDOW_HOURS)


def stem_from_filename(stem: str) -> tuple[str, str]:
    """从 A.2 文件名 stem 解析 operator, params。"""
    if stem.startswith("T1_uniform_"):
        op, rest = "T1_uniform", stem.replace("T1_uniform_", "")
    elif stem.startswith("T1_weighted_"):
        op, rest = "T1_weighted", stem.replace("T1_weighted_", "")
    elif stem.startswith("T2_"):
        op, rest = "T2", stem.replace("T2_", "")
    elif stem.startswith("T3_"):
        op, rest = "T3", stem.replace("T3_", "")
    else:
        return "", ""
    params = rest.replace("_max_diff=", ",max_diff=").replace("_n_passes=", ",n_passes=")
    return op, params


def cohort_mean_series(mat: np.ndarray) -> np.ndarray:
    """(n_stays, 48) -> (48,) 取列均值，忽略 NaN。"""
    return np.nanmean(mat, axis=0)


def _simple_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """简易 ACF（无 statsmodels 时使用）。"""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)
    c0 = np.dot(x, x) / n
    if c0 <= 0:
        return np.zeros(nlags + 1)
    r = [1.0]
    for k in range(1, min(nlags + 1, n)):
        r.append(np.dot(x[: n - k], x[k:]) / n / c0)
    return np.array(r + [0.0] * (nlags + 1 - len(r)))


def _simple_pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """简易 PACF：用 Durbin-Levinson 递推（仅前几阶）。"""
    acf_vals = _simple_acf(x, nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0
    if nlags < 1:
        return pacf
    pacf[1] = acf_vals[1]
    for k in range(2, nlags + 1):
        num = acf_vals[k] - np.dot(pacf[1:k], acf_vals[k - 1 : 0 : -1])
        den = 1.0 - np.dot(pacf[1:k], acf_vals[1:k])
        pacf[k] = num / den if abs(den) > 1e-12 else 0.0
    return pacf


def _simple_ljungbox(x: np.ndarray, lags: list[int]) -> pd.DataFrame:
    """简易 Ljung-Box：Q = n(n+2) * sum_{k=1}^{h} rho_k^2/(n-k)，近似 chi2(h)。"""
    from scipy import stats as scipy_stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    acf_vals = _simple_acf(x, max(lags))
    rows = []
    for h in lags:
        if h < 1 or h >= n:
            continue
        q = n * (n + 2) * np.sum(acf_vals[1 : h + 1] ** 2 / (n - np.arange(1, h + 1)))
        p = 1.0 - float(scipy_stats.chi2.cdf(q, h))
        rows.append({"lag": h, "lb_value": q, "lb_pvalue": p})
    return pd.DataFrame(rows).set_index("lag")


def run_acf_pacf_and_ljungbox(
    ts_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
) -> None:
    """对 cohort 平均序列算 ACF/PACF（关键滞后）与 Ljung-Box，写表。"""
    try:
        from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
        use_sm = True
    except ImportError:
        use_sm = False
        sm_acf = sm_pacf = acorr_ljungbox = None

    def acf_fn(x, nlags):
        if use_sm:
            return sm_acf(x, nlags=nlags, fft=True)
        return _simple_acf(x, nlags)

    def pacf_fn(x, nlags):
        if use_sm:
            return sm_pacf(x, nlags=nlags)
        return _simple_pacf(x, nlags)

    def lb_fn(x, lags):
        if use_sm:
            return acorr_ljungbox(x, lags=lags, return_df=True)
        return _simple_ljungbox(x, lags)

    ts_dir = Path(ts_dir)
    perturbed_dir = Path(perturbed_dir)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows_acf: list[dict] = []
    rows_lb: list[dict] = []

    for setting in ["z", "phys"]:
        for var in ["HR", "SBP", "DBP", "MAP", "TempC", "RespRate", "SpO2", "Creatinine", "Lactate", "Glucose"]:
            try:
                mat_orig = load_ts48h_matrix(ts_dir, var, setting)
            except FileNotFoundError:
                continue
            n_stays = mat_orig.shape[0]
            mean_orig = cohort_mean_series(mat_orig)
            if np.any(np.isnan(mean_orig)):
                mean_orig = np.nan_to_num(mean_orig, nan=np.nanmean(mean_orig))
            nlags = min(24, len(mean_orig) // 2 - 1)
            if nlags < 2:
                continue

            # 原始 ACF/PACF 与 Ljung-Box
            acf_orig = acf_fn(mean_orig, nlags)
            pacf_orig = pacf_fn(mean_orig, nlags)
            lb_orig = lb_fn(mean_orig, LB_LAGS)
            for lag in ACF_LAGS:
                if lag <= nlags:
                    rows_acf.append({
                        "var": var, "setting": setting, "operator": "original", "params": "",
                        "lag": lag, "acf": float(acf_orig[lag]), "pacf": float(pacf_orig[lag]),
                    })
            for lag in LB_LAGS:
                if lag in lb_orig.index:
                    rows_lb.append({
                        "var": var, "setting": setting, "operator": "original", "params": "",
                        "lag": lag, "lb_stat": float(lb_orig.loc[lag, "lb_value"]), "lb_pvalue": float(lb_orig.loc[lag, "lb_pvalue"]),
                    })

            var_dir = perturbed_dir / setting / var
            if not var_dir.exists():
                continue
            for csv_path in sorted(var_dir.glob("*.csv")):
                stem = csv_path.stem
                op, params = stem_from_filename(stem)
                if not op:
                    continue
                try:
                    mat_pert = load_perturbed_48h(perturbed_dir, setting, var, stem, n_stays)
                except FileNotFoundError:
                    continue
                mean_pert = cohort_mean_series(mat_pert)
                mean_pert = np.nan_to_num(mean_pert, nan=np.nanmean(mean_pert))
                acf_pert = acf_fn(mean_pert, nlags)
                pacf_pert = pacf_fn(mean_pert, nlags)
                lb_pert = lb_fn(mean_pert, LB_LAGS)
                for lag in ACF_LAGS:
                    if lag <= nlags:
                        rows_acf.append({
                            "var": var, "setting": setting, "operator": op, "params": params,
                            "lag": lag, "acf": float(acf_pert[lag]), "pacf": float(pacf_pert[lag]),
                        })
                for lag in LB_LAGS:
                    if lag in lb_pert.index:
                        rows_lb.append({
                            "var": var, "setting": setting, "operator": op, "params": params,
                            "lag": lag, "lb_stat": float(lb_pert.loc[lag, "lb_value"]), "lb_pvalue": float(lb_pert.loc[lag, "lb_pvalue"]),
                        })

    pd.DataFrame(rows_acf).to_csv(tables_dir / "table_acf_pacf.csv", index=False)
    pd.DataFrame(rows_lb).to_csv(tables_dir / "table_ljungbox.csv", index=False)
    print(f"[写出] table_acf_pacf.csv ({len(rows_acf)} 行), table_ljungbox.csv ({len(rows_lb)} 行)")


def run_spectral(ts_dir: Path, perturbed_dir: Path, out_dir: Path) -> None:
    """对 HR 等做 Welch 功率谱，保存主频与总功率到表。"""
    from scipy import signal as scipy_signal

    ts_dir = Path(ts_dir)
    perturbed_dir = Path(perturbed_dir)
    tables_dir = Path(out_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows_spec: list[dict] = []

    for setting in ["z", "phys"]:
        for var in list(REP_VARS) + ["RespRate"]:
            try:
                mat_orig = load_ts48h_matrix(ts_dir, var, setting)
            except FileNotFoundError:
                continue
            n_stays = mat_orig.shape[0]
            mean_orig = cohort_mean_series(mat_orig)
            mean_orig = np.nan_to_num(mean_orig, nan=np.nanmean(mean_orig))
            f_orig, psd_orig = scipy_signal.welch(mean_orig, fs=1.0, nperseg=min(32, len(mean_orig)))
            peak_idx = np.argmax(psd_orig)
            rows_spec.append({
                "var": var, "setting": setting, "operator": "original", "params": "",
                "peak_freq": float(f_orig[peak_idx]), "total_power": float(np.trapz(psd_orig, f_orig)),
            })
            var_dir = perturbed_dir / setting / var
            if not var_dir.exists():
                continue
            for csv_path in sorted(var_dir.glob("*.csv")):
                stem = csv_path.stem
                op, params = stem_from_filename(stem)
                if not op:
                    continue
                try:
                    mat_pert = load_perturbed_48h(perturbed_dir, setting, var, stem, n_stays)
                except FileNotFoundError:
                    continue
                mean_pert = cohort_mean_series(mat_pert)
                mean_pert = np.nan_to_num(mean_pert, nan=np.nanmean(mean_pert))
                f_pert, psd_pert = scipy_signal.welch(mean_pert, fs=1.0, nperseg=min(32, len(mean_pert)))
                peak_idx_p = np.argmax(psd_pert)
                rows_spec.append({
                    "var": var, "setting": setting, "operator": op, "params": params,
                    "peak_freq": float(f_pert[peak_idx_p]), "total_power": float(np.trapz(psd_pert, f_pert)),
                })

    pd.DataFrame(rows_spec).to_csv(tables_dir / "table_spectral.csv", index=False)
    print(f"[写出] table_spectral.csv ({len(rows_spec)} 行)")


def plot_ljungbox(out_dir: Path) -> None:
    """读 table_ljungbox.csv，为代表变量画 Ljung–Box p 值图（p 越大越接近白噪声）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tables_dir = Path(out_dir) / "tables"
    figs_dir = Path(out_dir) / "figs"
    path = tables_dir / "table_ljungbox.csv"
    if not path.exists():
        print("[跳过] Ljung-Box 图：table_ljungbox.csv 不存在")
        return
    figs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    # 只画 setting=z、代表变量；只保留 4 条曲线：original, T1(5,1), T2(1), T3(1)
    df_z = df[df["setting"] == "z"].copy()
    def keep_row(row):
        if row["operator"] == "original" and (row["params"] == "" or pd.isna(row["params"])):
            return True
        if row["operator"] == "T1_uniform" and "n_passes=5" in str(row["params"]) and "max_diff=0.8" in str(row["params"]):
            return True
        if row["operator"] == "T2" and "max_diff=0.8" in str(row["params"]):
            return True
        if row["operator"] == "T3" and "max_diff=0.8" in str(row["params"]):
            return True
        return False
    df_z["keep"] = df_z.apply(keep_row, axis=1)
    df_z = df_z[df_z["keep"]].drop(columns=["keep"])
    df_z["label"] = df_z["operator"].replace({
        "original": "original",
        "T1_uniform": "T1",
        "T1_weighted": "T1_w",
        "T2": "T2",
        "T3": "T3",
    })
    # p=0 时用一小值便于 log 显示
    p = df_z["lb_pvalue"].to_numpy()
    p_safe = np.where(p <= 0, 1e-20, p)
    df_z["lb_pvalue_safe"] = p_safe
    for var in REP_VARS:
        dv = df_z[df_z["var"] == var]
        if dv.empty:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        lags = sorted(dv["lag"].unique())
        labels = dv["label"].unique().tolist()
        # 保证顺序：original, T1, T2, T3
        order = ["original", "T1", "T1_w", "T2", "T3"]
        labels = [x for x in order if x in labels]
        x = np.arange(len(lags))
        w = 0.8 / len(labels)
        for i, lab in enumerate(labels):
            vals = []
            for lag in lags:
                r = dv[(dv["lag"] == lag) & (dv["label"] == lab)]
                v = r["lb_pvalue_safe"].iloc[0] if not r.empty else 1e-20
                vals.append(v)
            off = (i - (len(labels) - 1) / 2) * w
            ax.bar(x + off, vals, width=w, label=lab, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(h)) for h in lags])
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Ljung–Box p-value")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-20, top=1.1)
        ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{var} (z) — Ljung–Box p (larger p = closer to white noise)")
        plt.tight_layout()
        plt.savefig(figs_dir / f"ljungbox_{var}.pdf", bbox_inches="tight")
        plt.close()
        print(f"[写出] ljungbox_{var}.pdf")


def run_plots(
    ts_dir: Path,
    perturbed_dir: Path,
    out_dir: Path,
) -> None:
    """代表变量画 ACF/PACF 曲线与 PSD 图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[跳过] 需要 matplotlib 画图")
        return
    try:
        from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
        def acf_plot(x, nlags): return sm_acf(x, nlags=nlags, fft=True)
        def pacf_plot(x, nlags): return sm_pacf(x, nlags=nlags)
    except ImportError:
        acf_plot = _simple_acf
        pacf_plot = _simple_pacf
    from scipy import signal as scipy_signal

    ts_dir = Path(ts_dir)
    perturbed_dir = Path(perturbed_dir)
    figs_dir = Path(out_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    setting = "z"
    plot_ops = [
        ("original", ""),
        ("T1_uniform", "n_passes=5,max_diff=0.8"),
        ("T2", "max_diff=0.8"),
        ("T3", "max_diff=0.8"),
    ]

    for var in REP_VARS:
        try:
            mat_orig = load_ts48h_matrix(ts_dir, var, setting)
        except FileNotFoundError:
            continue
        n_stays = mat_orig.shape[0]
        mean_orig = cohort_mean_series(mat_orig)
        mean_orig = np.nan_to_num(mean_orig, nan=np.nanmean(mean_orig))
        nlags = min(24, len(mean_orig) // 2 - 1)

        # ACF/PACF 多子图：original + 3 个算子
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        series_list = [("original", mean_orig)]
        for op, params in plot_ops[1:]:
            stem = f"{op}_{params.replace(',', '_')}" if params else ""
            if op == "T1_uniform":
                stem = "T1_uniform_n_passes=5_max_diff=0.8"
            elif op == "T2":
                stem = "T2_max_diff=0.8"
            elif op == "T3":
                stem = "T3_max_diff=0.8"
            try:
                mat_p = load_perturbed_48h(perturbed_dir, setting, var, stem, n_stays)
                mean_p = cohort_mean_series(mat_p)
                mean_p = np.nan_to_num(mean_p, nan=np.nanmean(mean_p))
                series_list.append((op, mean_p))
            except FileNotFoundError:
                pass
        for i, (label, ser) in enumerate(series_list):
            if i >= 4:
                break
            ax_acf = axes[0, i]
            ax_pacf = axes[1, i]
            acf_vals = acf_plot(ser, nlags)
            pacf_vals = pacf_plot(ser, nlags)
            ax_acf.bar(range(len(acf_vals)), acf_vals, width=0.8)
            ax_acf.axhline(0, color="k", linewidth=0.5)
            ax_acf.set_title(f"ACF — {label}")
            ax_acf.set_xlim(-0.5, nlags + 0.5)
            ax_pacf.bar(range(len(pacf_vals)), pacf_vals, width=0.8)
            ax_pacf.axhline(0, color="k", linewidth=0.5)
            ax_pacf.set_title(f"PACF — {label}")
            ax_pacf.set_xlim(-0.5, nlags + 0.5)
        for j in range(len(series_list), 4):
            axes[0, j].set_visible(False)
            axes[1, j].set_visible(False)
        plt.suptitle(f"{var} (z) — ACF / PACF")
        plt.tight_layout()
        plt.savefig(figs_dir / f"acf_pacf_{var}.pdf", bbox_inches="tight")
        plt.close()
        print(f"[写出] acf_pacf_{var}.pdf")

        # PSD：original vs 3 算子
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        for label, ser in series_list:
            f, psd = scipy_signal.welch(ser, fs=1.0, nperseg=min(32, len(ser)))
            ax.semilogy(f, psd, label=label, alpha=0.8)
        ax.set_xlabel("Frequency (1/hour)")
        ax.set_ylabel("PSD")
        ax.set_title(f"{var} (z) — Power spectral density (Welch)")
        ax.legend()
        ax.set_ylim(bottom=1e-6)
        plt.tight_layout()
        plt.savefig(figs_dir / f"psd_{var}.pdf", bbox_inches="tight")
        plt.close()
        print(f"[写出] psd_{var}.pdf")

    # Ljung–Box p 值图（读已生成的 table）
    plot_ljungbox(out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A.6 时间结构检验：ACF/PACF、Ljung-Box、功率谱。")
    parser.add_argument("--ts-dir", type=str, default=None)
    parser.add_argument("--perturbed-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true", help="不画图")
    parser.add_argument("--plots-only", action="store_true", help="只从已有 tables 重新画图（含 Ljung-Box），不重算表")
    args = parser.parse_args()
    root = _REPO_ROOT
    if args.ts_dir:
        args.ts_dir = Path(args.ts_dir)
    else:
        try:
            args.ts_dir = default_ts48h_dir(root)
        except FileNotFoundError as e:
            raise SystemExit(f"错误：{e} 请指定 --ts-dir。") from e
    args.perturbed_dir = Path(
        args.perturbed_dir or default_a2_perturbed_dir(root)
    )
    args.out_dir = Path(args.out_dir or SCRIPT_DIR.parent / "results")
    return args


def main() -> None:
    args = parse_args()
    if args.plots_only:
        # 只画图：仅用已有 table，写出 Ljung-Box 等图
        lb_path = args.out_dir / "tables" / "table_ljungbox.csv"
        if not lb_path.exists():
            print(f"错误：--plots-only 需要已有 {lb_path}")
            sys.exit(1)
        plot_ljungbox(args.out_dir)
        print("\nA.6 仅画图完成（已写 figs/ljungbox_*.pdf）。")
        return
    if not args.ts_dir.exists():
        print(f"错误：ts 目录不存在 {args.ts_dir}")
        sys.exit(1)
    if not args.perturbed_dir.exists():
        print(f"错误：扰动目录不存在 {args.perturbed_dir}")
        sys.exit(1)

    run_acf_pacf_and_ljungbox(args.ts_dir, args.perturbed_dir, args.out_dir)
    run_spectral(args.ts_dir, args.perturbed_dir, args.out_dir)
    if not args.no_plots:
        run_plots(args.ts_dir, args.perturbed_dir, args.out_dir)
    print("\nA.6 完成。")


if __name__ == "__main__":
    main()
