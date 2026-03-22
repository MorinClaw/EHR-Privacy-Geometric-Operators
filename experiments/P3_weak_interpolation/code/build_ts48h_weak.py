#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3: Build weak-interp ts_48h:
- hourly alignment
- forward-fill only with a max allowed gap K hours
- no linear interpolation across gaps
Outputs:
- ts_48h_weak/ts_48h_{var}.csv and ts_48h_{var}_zscore.csv
- ts_48h_weak/ts_single_column_{var}.csv and ..._zscore.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config_weak import WINDOW_HOURS, ITEMID_TO_VAR, DEFAULT_VARIABLES


def load_cohort(data_dir: Path) -> pd.DataFrame:
    cpath = data_dir / "cohort_icu_stays.csv"
    if not cpath.exists():
        raise FileNotFoundError(cpath)
    df = pd.read_csv(cpath)
    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    return df[["stay_id", "intime"]]


def load_long_tables(data_dir: Path) -> pd.DataFrame:
    # charts long
    charts = data_dir / "timeseries_long.csv"
    labs = data_dir / "lab_timeseries_long.csv"
    rows = []
    if charts.exists():
        df = pd.read_csv(charts)
        if "charttime" in df.columns:
            df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        df = df[["stay_id", "itemid", "charttime", "valuenum"]]
        df = df.rename(columns={"charttime": "time"})
        rows.append(df)
    if labs.exists():
        df = pd.read_csv(labs)
        # lab 提供 hour；转为相对 intime 小时后仍用 hour_bin 逻辑
        if "hour" in df.columns:
            df = df[["stay_id", "itemid", "hour", "valuenum"]].copy()
            df["time"] = pd.NaT  # 不用真实时间，后面直接使用 provided hour
            df["provided_hour"] = df["hour"].astype(int)
            rows.append(df[["stay_id", "itemid", "time", "valuenum", "provided_hour"]])
    if not rows:
        return pd.DataFrame(columns=["stay_id", "itemid", "time", "valuenum", "provided_hour"])
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "provided_hour" not in out.columns:
        out["provided_hour"] = np.nan
    return out


def build_hourly_matrix_for_var(
    cohort: pd.DataFrame,
    long_df: pd.DataFrame,
    var: str,
    max_ffill_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (values, zscore_values) shape (n_stays, 48)
    Strategy:
      - map itemid -> var; select rows of this var
      - compute hour_bin per event:
          charts: floor((time - intime).hours)
          labs: provided_hour
      - per stay-hour: mean aggregation
      - forward fill within max_ffill_hours; otherwise keep NaN
    """
    n = len(cohort)
    values = np.full((n, WINDOW_HOURS), np.nan, dtype=float)
    sid2idx = {int(s): i for i, s in enumerate(cohort["stay_id"].values.astype(int))}

    var_itemids = [k for k, v in ITEMID_TO_VAR.items() if v == var]
    if not var_itemids:
        return values, values.copy()
    df = long_df[long_df["itemid"].isin(var_itemids)].copy()
    if df.empty:
        return values, values.copy()

    # compute hour_bin
    # charts (with real time)
    charts = df[df["provided_hour"].isna()].copy()
    if not charts.empty:
        charts = charts.merge(cohort, on="stay_id", how="left")
        charts["rel_hours"] = (charts["time"] - charts["intime"]).dt.total_seconds() / 3600.0
        charts["hour_bin"] = charts["rel_hours"].astype(float).apply(
            lambda h: int(h) if (pd.notna(h) and 0 <= h < WINDOW_HOURS) else np.nan
        )
    else:
        charts = pd.DataFrame(columns=["stay_id", "hour_bin", "valuenum"])

    # labs (provided hour)
    labs = df[~df["provided_hour"].isna()].copy()
    if not labs.empty:
        labs["hour_bin"] = labs["provided_hour"].astype(int)
        labs = labs[(labs["hour_bin"] >= 0) & (labs["hour_bin"] < WINDOW_HOURS)]
    else:
        labs = pd.DataFrame(columns=["stay_id", "hour_bin", "valuenum"])

    merged = pd.concat([charts[["stay_id", "hour_bin", "valuenum"]], labs[["stay_id", "hour_bin", "valuenum"]]], ignore_index=True)
    merged = merged.dropna(subset=["hour_bin", "valuenum"])
    if merged.empty:
        return values, values.copy()

    agg = merged.groupby(["stay_id", "hour_bin"], as_index=False)["valuenum"].mean()
    for _, row in agg.iterrows():
        sid = int(row["stay_id"])
        hb = int(row["hour_bin"])
        if sid in sid2idx:
            values[sid2idx[sid], hb] = float(row["valuenum"])

    # forward-fill with max gap
    for i in range(n):
        last_val = np.nan
        last_pos = -10**9
        for t in range(WINDOW_HOURS):
            v = values[i, t]
            if np.isfinite(v):
                last_val = v
                last_pos = t
            else:
                # can we ffill?
                if np.isfinite(last_val) and (t - last_pos) <= max_ffill_hours:
                    values[i, t] = last_val
                else:
                    values[i, t] = np.nan

    # z-score per column using non-missing across all stays (global per-var)
    col_means = np.nanmean(values, axis=0)
    col_stds = np.nanstd(values, axis=0)
    col_stds[col_stds < 1e-8] = 1.0
    zscore = (values - col_means) / col_stds
    return values, zscore


def write_ts48h_tables(
    out_dir: Path,
    var: str,
    values: np.ndarray,
    zvalues: np.ndarray,
    stay_ids: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # wide
    cols = [str(i) for i in range(WINDOW_HOURS)]
    df = pd.DataFrame(values, columns=cols)
    df.insert(0, "stay_id", stay_ids.astype(int))
    df.to_csv(out_dir / f"ts_48h_{var}.csv", index=False)
    dfz = pd.DataFrame(zvalues, columns=cols)
    dfz.insert(0, "stay_id", stay_ids.astype(int))
    dfz.to_csv(out_dir / f"ts_48h_{var}_zscore.csv", index=False)
    # single column (row-major)
    vec = values.reshape(-1)
    vecz = zvalues.reshape(-1)
    pd.DataFrame({"value": vec}).to_csv(out_dir / f"ts_single_column_{var}.csv", index=False)
    pd.DataFrame({"value": vecz}).to_csv(out_dir / f"ts_single_column_{var}_zscore.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Build weak-interpolation ts_48h (ffill<=K, no linear interp)")
    p.add_argument("--data-dir", type=Path, required=True, help="experiment_extracted 根目录（含 long 表与 cohort）")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录（建议 P3_weak_interpolation/ts_48h_weak）")
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--max-ffill-hours", type=int, default=2, help="允许的最大前向填充小时数（含）")
    args = p.parse_args()

    cohort = load_cohort(args.data_dir)
    stay_ids = cohort["stay_id"].values.astype(int)
    long_df = load_long_tables(args.data_dir)

    for var in args.variables:
        vals, zvals = build_hourly_matrix_for_var(cohort, long_df, var, args.max_ffill_hours)
        write_ts48h_tables(args.out_dir, var, vals, zvals, stay_ids)
        print(f"[weak ts] {var}: shape={vals.shape}")

    print("ts_48h_weak written to", args.out_dir)


if __name__ == "__main__":
    main()

