#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.5 实验 B3：复杂度 / 算力（low compute 可落地）

- 对每张表 / 每个 privacy_level：总运行时间（wall-clock）、各算子耗时（Skill.apply 外包计时）。
- 数值型算子：在 N=1e4, 5e4, 1e5, 5e5, 1e6 下的 scaling 行为（O(N)）。
- 高算力 baseline：CTGAN 在 5k–10k 行子集上训练 + 生成，记录训练时间(GPU/CPU)、生成时间、下游任务性能。
- 输出：表（pipeline + baseline 的 time / hardware）；图（N 增大时运行时间曲线，展示接近线性）。

用法（实验根目录，PYTHONPATH 含 agent_demo）：
  python3 B5_complexity_compute/code/exp_b5_compute_metrics.py
  python3 B5_complexity_compute/code/exp_b5_compute_metrics.py --max-rows 5000 --no-baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import (
    default_experiment_extracted_dir,
    find_agent_demo_dir,
    find_b5_default_results_dir,
)

AGENT_DIR = find_agent_demo_dir(ROOT)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _timed_apply(skill, data, config: dict):
    """执行 skill.apply(data, config)，返回 (结果, 耗时秒)。"""
    t0 = time.perf_counter()
    out = skill.apply(data, config)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def run_patient_profile_timed(
    df: pd.DataFrame,
    agent,
    privacy_level: str,
    config: Dict[str, Any],
    verbose: bool = False,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """与 demo 相同的 patient_profile 脱敏逻辑，但记录每步耗时。返回 (脱敏表, 步骤耗时列表)。"""
    from demo_patient_and_timeline import PATIENT_PROFILE_CONFIG

    cfg = config or PATIENT_PROFILE_CONFIG
    out = df.copy()
    registry = agent.registry
    pipeline = agent.plan_pipeline("patient_profile", privacy_level)
    timings: List[Dict[str, Any]] = []

    id_cols = cfg.get("id_columns", [])
    age_col = cfg.get("age_column")
    numeric_quasiid_cols = cfg.get("numeric_quasiid_columns", [])
    cat_cols = cfg.get("cat_columns", [])
    cat_keep_lists = cfg.get("cat_keep_lists") or {}

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(
                    skill, out[col].astype(str).tolist(), {"secret": "demo_salt", "length": 12}
                )
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - id_hash {col}: {el:.3f}s")

        elif sid == "demo_bin":
            if age_col and age_col in out.columns:
                skill = registry.get("demo_bin")
                out[age_col], el = _timed_apply(skill, out[age_col].to_numpy(), {"mode": "age"})
                timings.append({"skill_id": sid, "column": age_col, "time_sec": el})
                if verbose:
                    print(f"  - demo_bin {age_col}: {el:.3f}s")

        elif sid == "microagg":
            skill = registry.get("microagg")
            for col in numeric_quasiid_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(skill, out[col].to_numpy(), {"k": 10})
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - microagg {col}: {el:.3f}s")

        elif sid == "cat_agg":
            skill = registry.get("cat_agg")
            n_rows = len(out)
            min_freq = 1 if n_rows < 10 else 5
            for col in cat_cols:
                if col not in out.columns:
                    continue
                keep_list = cat_keep_lists.get(col)
                out[col], el = _timed_apply(
                    skill,
                    out[col].astype(str).tolist(),
                    {"min_freq": min_freq, "other_label": "OTHER", "keep_list": keep_list},
                )
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - cat_agg {col}: {el:.3f}s")

    return out, timings


def run_timeline_timed(
    df: pd.DataFrame,
    agent,
    privacy_level: str,
    config: Dict[str, Any],
    verbose: bool = False,
) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """与 demo 相同的 timeline 脱敏逻辑，记录每步耗时。返回 (脱敏表, 步骤耗时列表)。"""
    from demo_patient_and_timeline import TIMELINE_CONFIG

    cfg = config or TIMELINE_CONFIG
    out = df.copy()
    registry = agent.registry
    pipeline = agent.plan_pipeline("timeline", privacy_level)
    timings: List[Dict[str, Any]] = []

    id_cols = cfg.get("id_columns", [])
    time_col = cfg.get("time_column")
    num_cols = cfg.get("numeric_columns", [])
    text_cols = cfg.get("text_columns", [])

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(
                    skill, out[col].astype(str).tolist(), {"secret": "demo_salt", "length": 12}
                )
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - id_hash {col}: {el:.3f}s")

        elif sid == "time_rel":
            if time_col and time_col in out.columns:
                skill = registry.get("time_rel")
                times = pd.to_datetime(out[time_col], errors="coerce")
                out[time_col], el = _timed_apply(skill, times.to_numpy(), {"index_time": None, "unit": "D"})
                timings.append({"skill_id": sid, "column": time_col, "time_sec": el})
                if verbose:
                    print(f"  - time_rel: {el:.3f}s")

        elif sid == "time_shift":
            if time_col and time_col in out.columns:
                skill = registry.get("time_shift")
                out[time_col], el = _timed_apply(skill, out[time_col].to_numpy(), {"max_shift_days": 365})
                timings.append({"skill_id": sid, "column": time_col, "time_sec": el})
                if verbose:
                    print(f"  - time_shift: {el:.3f}s")

        elif sid == "text_mask":
            skill = registry.get("text_mask")
            for col in text_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(skill, out[col].astype(str).tolist(), {})
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - text_mask {col}: {el:.3f}s")

        elif sid == "text_phi_surr":
            skill = registry.get("text_phi_surr")
            for col in text_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(skill, out[col].astype(str).tolist(), {"seed": 0})
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - text_phi_surr {col}: {el:.3f}s")

        elif sid == "ds_tab":
            skill = registry.get("ds_tab")
            for col in num_cols:
                if col not in out.columns:
                    continue
                out[col], el = _timed_apply(
                    skill, out[col].to_numpy(), {"synth_prob": 0.3, "noise_scale": 0.5}
                )
                timings.append({"skill_id": sid, "column": col, "time_sec": el})
                if verbose:
                    print(f"  - ds_tab {col}: {el:.3f}s")

    return out, timings


def run_numeric_scaling(
    n_sizes: List[int],
    pipeline_ids: List[str],
    skill_configs: Dict[str, Dict],
) -> pd.DataFrame:
    """数值算子在不同 N 下的耗时。pipeline_ids 如 ['num_triplet','num_noise_proj','num_householder']。"""
    from skills_and_agent import build_default_registry, PrivacyAgent

    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    rng = np.random.default_rng(42)
    rows = []

    for n in n_sizes:
        x = rng.standard_normal(n).astype(float)
        for sid in pipeline_ids:
            skill = registry.get(sid)
            if skill.target != "numeric":
                continue
            cfg = skill_configs.get(sid, {})
            _, el = _timed_apply(skill, x.copy(), cfg)
            rows.append({"N": n, "skill_id": sid, "time_sec": el})
        # 全流水线一次
        full_cfg = {sid: skill_configs.get(sid, {}) for sid in pipeline_ids}
        t0 = time.perf_counter()
        try:
            agent.run_numeric_pipeline(x.copy(), pipeline_ids, full_cfg)
        except Exception:
            pass
        rows.append({"N": n, "skill_id": "pipeline_full", "time_sec": time.perf_counter() - t0})

    return pd.DataFrame(rows)


def run_full_pipeline_timings(
    data_dir: Path,
    patient_rows: int,
    timeline_rows: int,
    privacy_levels: List[str],
) -> pd.DataFrame:
    """对 patient_profile 与 timeline 在各 privacy_level 下跑一遍，记录总时间与各算子耗时。"""
    from demo_patient_and_timeline import (
        PATIENT_PROFILE_CONFIG,
        TIMELINE_CONFIG,
    )
    from skills_and_agent import build_default_registry, PrivacyAgent

    agent = PrivacyAgent(build_default_registry())
    pp_path = data_dir / "patient_profile.csv"
    tl_path = data_dir / "timeline_events.csv"
    rows = []

    if pp_path.exists():
        df_pp = pd.read_csv(pp_path, nrows=patient_rows)
        for pl in privacy_levels:
            t0 = time.perf_counter()
            _, timings = run_patient_profile_timed(df_pp, agent, pl, PATIENT_PROFILE_CONFIG, verbose=False)
            total = time.perf_counter() - t0
            for t in timings:
                rows.append({
                    "table": "patient_profile",
                    "privacy_level": pl,
                    "n_rows": len(df_pp),
                    "skill_id": t["skill_id"],
                    "column": t.get("column", ""),
                    "time_sec": t["time_sec"],
                })
            rows.append({
                "table": "patient_profile",
                "privacy_level": pl,
                "n_rows": len(df_pp),
                "skill_id": "total",
                "column": "",
                "time_sec": total,
            })

    if tl_path.exists():
        df_tl = pd.read_csv(tl_path, nrows=timeline_rows)
        for pl in privacy_levels:
            t0 = time.perf_counter()
            _, timings = run_timeline_timed(df_tl, agent, pl, TIMELINE_CONFIG, verbose=False)
            total = time.perf_counter() - t0
            for t in timings:
                rows.append({
                    "table": "timeline_events",
                    "privacy_level": pl,
                    "n_rows": len(df_tl),
                    "skill_id": t["skill_id"],
                    "column": t.get("column", ""),
                    "time_sec": t["time_sec"],
                })
            rows.append({
                "table": "timeline_events",
                "privacy_level": pl,
                "n_rows": len(df_tl),
                "skill_id": "total",
                "column": "",
                "time_sec": total,
            })

    return pd.DataFrame(rows)


def run_baseline_ctgan(
    data_dir: Path,
    n_train: int = 5000,
    n_generate: int = 5000,
    epochs: int = 50,
) -> Dict[str, Any]:
    """CTGAN baseline：在 patient_profile 子集上训练 + 生成，返回训练时间、生成时间、硬件说明。"""
    try:
        from ctgan import CTGAN
    except ImportError:
        return {"train_time_sec": None, "gen_time_sec": None, "hardware": "ctgan_not_installed", "downstream_auc": None}

    pp_path = data_dir / "patient_profile.csv"
    if not pp_path.exists():
        return {"train_time_sec": None, "gen_time_sec": None, "hardware": "no_data", "downstream_auc": None}

    df = pd.read_csv(pp_path, nrows=n_train)
    # 选列并填缺失
    use_cols = [c for c in ["gender", "anchor_age", "ethnicity", "insurance", "bmi"] if c in df.columns]
    if not use_cols or "anchor_age" not in df.columns:
        return {"train_time_sec": None, "gen_time_sec": None, "hardware": "missing_columns", "downstream_auc": None}
    df = df[use_cols].copy()
    if "anchor_age" in df.columns:
        df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce").fillna(df["anchor_age"].median() if df["anchor_age"].notna().any() else 50)
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(df["bmi"].median() if df["bmi"].notna().any() else 25)
    for c in ["gender", "ethnicity", "insurance"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    discrete = [c for c in ["gender", "ethnicity", "insurance"] if c in df.columns]

    try:
        model = CTGAN(epochs=epochs, verbose=False)
        t0 = time.perf_counter()
        model.fit(df, discrete_columns=discrete)
        train_time = time.perf_counter() - t0
    except Exception as e:
        return {"train_time_sec": None, "gen_time_sec": None, "hardware": f"train_error:{e}", "downstream_auc": None}

    t0 = time.perf_counter()
    synth = model.sample(n_generate)
    gen_time = time.perf_counter() - t0

    # 简单下游：若有 sklearn，用 anchor_age/bmi 预测一个合成二分类
    downstream_auc = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        y = (df["anchor_age"].to_numpy() > np.nanmedian(df["anchor_age"].to_numpy())).astype(int)
        if np.unique(y).size < 2 and "bmi" in df.columns:
            y = (df["bmi"].to_numpy() > np.nanmedian(df["bmi"].to_numpy())).astype(int)
        feats = [c for c in ["anchor_age", "bmi"] if c in df.columns]
        X = df[feats].fillna(0).to_numpy()
        if X.size == 0:
            raise ValueError("no features")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        lr = LogisticRegression(max_iter=500).fit(X_train, y_train)
        downstream_auc = float(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
    except Exception:
        pass

    return {
        "train_time_sec": train_time,
        "gen_time_sec": gen_time,
        "n_train": n_train,
        "n_generate": n_generate,
        "hardware": "CPU",  # CTGAN 默认 CPU；若有 GPU 可在此检测
        "downstream_auc": downstream_auc,
    }


def plot_scaling(df_numeric: pd.DataFrame, out_path: Path) -> None:
    """N vs time 曲线，展示接近线性。"""
    if not HAS_MPL or df_numeric.empty:
        return
    df = df_numeric[df_numeric["skill_id"] != "pipeline_full"]
    if df.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sid in df["skill_id"].unique():
        sub = df[df["skill_id"] == sid].sort_values("N")
        ax.plot(sub["N"], sub["time_sec"], "o-", label=sid)
    df_full = df_numeric[df_numeric["skill_id"] == "pipeline_full"].sort_values("N")
    if not df_full.empty:
        ax.plot(df_full["N"], df_full["time_sec"], "s-", label="pipeline_full")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (column length)")
    ax.set_ylabel("Time (sec)")
    ax.set_title("B.5 Numeric operator scaling (O(N))")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_pipeline_scaling_by_rows(
    data_dir: Path,
    row_sizes: List[int],
    privacy_level: str = "strong",
) -> pd.DataFrame:
    """随表行数增大（如 10k, 50k, 100k）记录 patient_profile / timeline 总运行时间。"""
    from demo_patient_and_timeline import PATIENT_PROFILE_CONFIG, TIMELINE_CONFIG
    from skills_and_agent import build_default_registry, PrivacyAgent

    agent = PrivacyAgent(build_default_registry())
    pp_path = data_dir / "patient_profile.csv"
    tl_path = data_dir / "timeline_events.csv"
    rows = []

    if pp_path.exists():
        for n in row_sizes:
            df = pd.read_csv(pp_path, nrows=n)
            if len(df) < 10:
                continue
            t0 = time.perf_counter()
            run_patient_profile_timed(df, agent, privacy_level, PATIENT_PROFILE_CONFIG, verbose=False)
            rows.append({"table": "patient_profile", "n_rows": len(df), "time_sec": time.perf_counter() - t0})
    if tl_path.exists():
        for n in row_sizes:
            df = pd.read_csv(tl_path, nrows=n)
            if len(df) < 10:
                continue
            t0 = time.perf_counter()
            run_timeline_timed(df, agent, privacy_level, TIMELINE_CONFIG, verbose=False)
            rows.append({"table": "timeline_events", "n_rows": len(df), "time_sec": time.perf_counter() - t0})

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.5 复杂度/算力实验")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=10000, help="patient/timeline 最大行数（pipeline 计时）")
    p.add_argument("--timeline-rows", type=int, default=50000, help="timeline 行数")
    p.add_argument("--numeric-n", type=str, default="10000,100000,500000,1000000", help="数值 scaling 的 N 列表，逗号分隔")
    p.add_argument("--no-baseline", action="store_true", help="不跑 CTGAN baseline")
    p.add_argument("--ctgan-epochs", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        try:
            data_dir = default_experiment_extracted_dir(ROOT)
        except FileNotFoundError as e:
            print(f"错误：{e} 请指定 --data-dir。")
            sys.exit(1)
    out_dir = Path(args.out_dir) if args.out_dir else find_b5_default_results_dir(ROOT)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    n_sizes = [int(x) for x in args.numeric_n.split(",")]

    # 1) 数值算子 scaling
    print("[B.5] 数值算子 scaling (N =", n_sizes, ")...")
    pipeline_ids = ["num_triplet", "num_noise_proj", "num_householder"]
    # 方案一：归一化空间固定阈值 0.8（增强抗重建）
    skill_configs = {sid: {"max_diff": 0.8} for sid in pipeline_ids}
    skill_configs["num_triplet"]["n_passes"] = 3
    df_numeric = run_numeric_scaling(n_sizes, pipeline_ids, skill_configs)
    df_numeric.to_csv(tables_dir / "table_b5_numeric_scaling.csv", index=False)
    print(f"  已保存: {tables_dir / 'table_b5_numeric_scaling.csv'}")
    plot_scaling(df_numeric, figs_dir / "b5_numeric_scaling.png")

    # 2) 各表各 level 总时间 + 各算子耗时
    print("[B.5] Pipeline 计时 (patient_profile + timeline_events)...")
    df_pipeline = run_full_pipeline_timings(
        data_dir,
        patient_rows=min(args.max_rows, 50000),
        timeline_rows=min(args.timeline_rows, 100000),
        privacy_levels=["light", "medium", "strong"],
    )
    df_pipeline.to_csv(tables_dir / "table_b5_pipeline_timings.csv", index=False)
    print(f"  已保存: {tables_dir / 'table_b5_pipeline_timings.csv'}")

    # 3) 汇总表：pipeline + baseline 的 time / hardware
    summary_rows = []
    for _, r in df_pipeline[df_pipeline["skill_id"] == "total"].iterrows():
        summary_rows.append({
            "method": "pipeline",
            "table": r["table"],
            "privacy_level": r["privacy_level"],
            "n_rows": r["n_rows"],
            "time_sec": r["time_sec"],
            "hardware": "CPU (single machine)",
        })
    if not args.no_baseline:
        print("[B.5] CTGAN baseline...")
        ctgan_res = run_baseline_ctgan(data_dir, n_train=min(7000, args.max_rows), n_generate=5000, epochs=args.ctgan_epochs)
        if ctgan_res.get("train_time_sec") is not None:
            summary_rows.append({
                "method": "CTGAN",
                "table": "patient_profile",
                "privacy_level": "-",
                "n_rows": ctgan_res.get("n_train"),
                "time_sec": ctgan_res["train_time_sec"],
                "hardware": ctgan_res.get("hardware", "CPU"),
            })
            summary_rows.append({
                "method": "CTGAN_gen",
                "table": "patient_profile",
                "privacy_level": "-",
                "n_rows": ctgan_res.get("n_generate"),
                "time_sec": ctgan_res["gen_time_sec"],
                "hardware": ctgan_res.get("hardware", "CPU"),
            })
        if ctgan_res.get("downstream_auc") is not None:
            summary_rows.append({
                "method": "CTGAN_downstream_auc",
                "table": "patient_profile",
                "privacy_level": "-",
                "n_rows": None,
                "time_sec": None,
                "hardware": str(ctgan_res["downstream_auc"]),
            })
    pd.DataFrame(summary_rows).to_csv(tables_dir / "table_b5_summary_time_hardware.csv", index=False)
    print(f"  已保存: {tables_dir / 'table_b5_summary_time_hardware.csv'}")

    # 4) 随 N 增大的运行时间曲线（数值列长度）
    if HAS_MPL and not df_numeric.empty:
        plot_scaling(df_numeric, figs_dir / "b5_runtime_vs_N.png")

    # 5) 随表行数增大的 pipeline 运行时间（10k, 50k, 100k）
    row_sizes = [r for r in [10000, 50000, 100000] if r <= max(args.max_rows, args.timeline_rows)]
    if not row_sizes:
        row_sizes = [min(5000, args.max_rows), args.max_rows] if args.max_rows >= 1000 else []
    if row_sizes and data_dir.exists():
        print("[B.5] Pipeline 随行数 scaling...")
        df_row_scale = run_pipeline_scaling_by_rows(data_dir, row_sizes, privacy_level="strong")
        df_row_scale.to_csv(tables_dir / "table_b5_pipeline_runtime_vs_rows.csv", index=False)
        if HAS_MPL and not df_row_scale.empty:
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            for tbl in df_row_scale["table"].unique():
                sub = df_row_scale[df_row_scale["table"] == tbl].sort_values("n_rows")
                ax.plot(sub["n_rows"], sub["time_sec"], "o-", label=tbl)
            ax.set_xlabel("Table rows (N)")
            ax.set_ylabel("Time (sec)")
            ax.set_title("B.5 Pipeline runtime vs table size")
            ax.legend()
            plt.tight_layout()
            plt.savefig(figs_dir / "b5_pipeline_runtime_vs_rows.png", dpi=120)
            plt.close()
    print("B.5 完成。")


if __name__ == "__main__":
    main()
