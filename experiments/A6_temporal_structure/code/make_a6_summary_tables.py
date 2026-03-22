#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 A6 三张结果表生成三张 summary 表：
  - table_a6_summary_acf_pacf.csv
  - table_a6_summary_ljungbox.csv
  - table_a6_summary_spectral.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR.parent / "results" / "tables"


def main():
    tables_dir = TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. ACF/PACF summary ----
    acf = pd.read_csv(tables_dir / "table_acf_pacf.csv")
    # 每个 (var, setting) 的 original 的 ACF/PACF（lag=1 最关键）
    orig = acf[acf["operator"] == "original"].copy()
    orig = orig.rename(columns={"acf": "acf_orig", "pacf": "pacf_orig"})[["var", "setting", "lag", "acf_orig", "pacf_orig"]]
    acf = acf.merge(orig, on=["var", "setting", "lag"], how="left")
    acf["acf_dev"] = (acf["acf"] - acf["acf_orig"]).abs()
    acf["pacf_dev"] = (acf["pacf"] - acf["pacf_orig"]).abs()
    # 只保留 lag=1（README 说滞后 1 的峰最重要）
    acf_lag1 = acf[acf["lag"] == 1].copy()
    # 按 (var, setting, operator) 汇总（同一 operator 多组 params 取平均）
    grp = acf_lag1[acf_lag1["operator"] != "original"].groupby(["var", "setting", "operator"], dropna=False)
    summary_acf = grp.agg(
        acf_lag1_mean=("acf", "mean"),
        pacf_lag1_mean=("pacf", "mean"),
        acf_lag1_dev_from_orig_mean=("acf_dev", "mean"),
        pacf_lag1_dev_from_orig_mean=("pacf_dev", "mean"),
        n_configs=("params", "nunique"),
    ).reset_index()
    summary_acf = summary_acf.sort_values(["var", "setting", "operator"])
    summary_acf.to_csv(tables_dir / "table_a6_summary_acf_pacf.csv", index=False)
    print("Wrote table_a6_summary_acf_pacf.csv")

    # ---- 2. Ljung-Box summary ----
    lb = pd.read_csv(tables_dir / "table_ljungbox.csv")
    # original 的 p 值（取 log 避免 0）
    orig_lb = lb[lb["operator"] == "original"][["var", "setting", "lag", "lb_pvalue"]].rename(
        columns={"lb_pvalue": "lb_pvalue_orig"}
    )
    lb = lb.merge(orig_lb, on=["var", "setting", "lag"], how="left")
    # 扰动后 p 越大越接近白噪声；用 log10(p+1e-300) 做平均
    lb["log10_p"] = np.log10(lb["lb_pvalue"].clip(lower=1e-300))
    lb["log10_p_orig"] = np.log10(lb["lb_pvalue_orig"].clip(lower=1e-300))
    lb_pert = lb[lb["operator"] != "original"]
    grp_lb = lb_pert.groupby(["var", "setting", "operator"], dropna=False)
    summary_lb = grp_lb.agg(
        mean_log10_lb_pvalue=("log10_p", "mean"),
        mean_lb_pvalue=("lb_pvalue", "mean"),
        mean_lb_stat=("lb_stat", "mean"),
        n_configs=("params", "nunique"),
    ).reset_index()
    # 与 original 比较：同一 var+setting 下 original 的 mean log10 p
    orig_agg = lb[lb["operator"] == "original"].groupby(["var", "setting"], dropna=False).agg(
        orig_mean_log10_p=("log10_p", "mean")
    ).reset_index()
    summary_lb = summary_lb.merge(orig_agg, on=["var", "setting"], how="left")
    summary_lb["log10_p_diff_from_orig"] = summary_lb["mean_log10_lb_pvalue"] - summary_lb["orig_mean_log10_p"]
    summary_lb = summary_lb.sort_values(["var", "setting", "operator"])
    summary_lb.to_csv(tables_dir / "table_a6_summary_ljungbox.csv", index=False)
    print("Wrote table_a6_summary_ljungbox.csv")

    # ---- 3. Spectral summary ----
    sp = pd.read_csv(tables_dir / "table_spectral.csv")
    orig_sp = sp[sp["operator"] == "original"][["var", "setting", "peak_freq", "total_power"]].rename(
        columns={"peak_freq": "peak_freq_orig", "total_power": "total_power_orig"}
    )
    sp = sp.merge(orig_sp, on=["var", "setting"], how="left")
    sp["peak_freq_unchanged"] = (sp["peak_freq"] == sp["peak_freq_orig"]).astype(int)
    sp["total_power_ratio"] = sp["total_power"] / sp["total_power_orig"].replace(0, np.nan)
    sp_pert = sp[sp["operator"] != "original"]
    grp_sp = sp_pert.groupby(["var", "setting", "operator"], dropna=False)
    summary_sp = grp_sp.agg(
        peak_freq_unchanged_pct=("peak_freq_unchanged", "mean"),
        total_power_ratio_mean=("total_power_ratio", "mean"),
        total_power_ratio_min=("total_power_ratio", "min"),
        total_power_ratio_max=("total_power_ratio", "max"),
        n_configs=("params", "nunique"),
    ).reset_index()
    summary_sp = summary_sp.sort_values(["var", "setting", "operator"])
    summary_sp.to_csv(tables_dir / "table_a6_summary_spectral.csv", index=False)
    print("Wrote table_a6_summary_spectral.csv")


if __name__ == "__main__":
    main()
