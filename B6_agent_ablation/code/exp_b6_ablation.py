#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.6 Agent & 算子消融（阶段三）

一、算子消融（operator ablation）
  在 numeric 流水线中比较：only T1；T1+T2；T1+T3；full T1+T2+T3。
  指标：数值重构攻击（MAE、相关系数）、下游 AUROC（同一任务）。

二、Agent 消融（Agent ablation）
  三种模式：手工 fixed pipeline；当前 rule-based Agent；"错误" pipeline（light 当 strong 用）。
  对比：隐私指标（quasi-ID 唯一化、重构误差）、运行时间；说明 Agent 至少不差于 rule-based，
  且错误配置会失败。

用法（实验根目录，PYTHONPATH 含 agent_demo）：
  python3 B6_agent_ablation/code/exp_b6_ablation.py
  python3 B6_agent_ablation/code/exp_b6_ablation.py --data-dir data_preparation/experiment_extracted --out-dir B6_agent_ablation/results
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
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import (
    default_experiment_extracted_dir,
    find_agent_demo_dir,
    find_b6_default_results_dir,
)

AGENT_DIR = find_agent_demo_dir(ROOT)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 一、算子消融：numeric 流水线变体 T1 / T1+T2 / T1+T3 / full
# ---------------------------------------------------------------------------

NUMERIC_ABLATION_VARIANTS: Dict[str, List[str]] = {
    "only_T1": ["num_triplet"],
    "T1_T2": ["num_triplet", "num_noise_proj"],
    "T1_T3": ["num_triplet", "num_householder"],
    "full_T1_T2_T3": ["num_triplet", "num_noise_proj", "num_householder"],
}


def run_numeric_pipeline_variant(
    x: np.ndarray,
    variant: List[str],
    skill_configs: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """对一列数值 x 跑指定算子序列，返回扰动后的列。"""
    from skills_and_agent import build_default_registry, PrivacyAgent

    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    valid = ~np.isnan(x)
    fill_val = np.nanmean(x[valid]) if np.any(valid) else 0.0
    x_fill = np.where(valid, x, fill_val)
    history = agent.run_numeric_pipeline(x_fill, variant, skill_configs)
    out = np.asarray(history[-1]["data"], dtype=float)
    out[~valid] = np.nan
    return out


def eval_reconstruction(orig: np.ndarray, deid: np.ndarray) -> Dict[str, float]:
    """重构攻击评估：MAE、Pearson 相关系数。"""
    mask = np.isfinite(orig) & np.isfinite(deid)
    if mask.sum() < 2:
        return {"recon_mae": np.nan, "recon_corr": np.nan}
    o, d = orig[mask], deid[mask]
    mae = float(np.mean(np.abs(o - d)))
    corr = float(np.corrcoef(o, d)[0, 1])
    return {"recon_mae": mae, "recon_corr": corr}


def run_operator_ablation(
    data_dir: Path,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    在 patient_profile 的 bmi 列上做算子消融：对 bmi 施加 T1 / T1+T2 / T1+T3 / full，
    计算重构 MAE/corr 与下游 AUROC（特征含 bmi 的 LR 预测 y_1y）。
    """
    from demo_patient_profile_tasks import (
        _age_to_bin_labels,
        add_synthetic_outcomes,
    )
    from demo_patient_and_timeline import PATIENT_PROFILE_CONFIG

    pp_path = data_dir / "patient_profile.csv"
    if not pp_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(pp_path, nrows=max_rows)
    for c in ["gender", "ethnicity", "insurance"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "bmi" not in df.columns:
        df["bmi"] = np.nan
    df = add_synthetic_outcomes(df, seed=seed)
    y = df["y_1y"].to_numpy()

    bmi_raw = pd.to_numeric(df["bmi"], errors="coerce").to_numpy()
    valid_bmi = np.isfinite(bmi_raw)
    if valid_bmi.sum() < 50:
        bmi_raw = np.nan_to_num(bmi_raw, nan=np.nanmean(bmi_raw[valid_bmi]) if np.any(valid_bmi) else 25.0)

    # 方案一：归一化空间固定阈值 0.8（增强抗重建）
    skill_configs = {sid: {"max_diff": 0.8} for sid in ["num_triplet", "num_noise_proj", "num_householder"]}
    skill_configs["num_triplet"]["n_passes"] = 3

    rows = []
    for name, variant in NUMERIC_ABLATION_VARIANTS.items():
        try:
            bmi_deid = run_numeric_pipeline_variant(bmi_raw.copy(), variant, skill_configs)
        except RuntimeError as e:
            print(f"  [B.6] 变体 {name} 失败: {e}，跳过")
            continue
        recon = eval_reconstruction(bmi_raw, bmi_deid)
        row = {"pipeline": name, "recon_mae": recon["recon_mae"], "recon_corr": recon["recon_corr"]}

        if HAS_SKLEARN:
            # 下游：age_bin + gender + ethnicity + insurance + bmi_deid -> y_1y
            df_run = df.copy()
            df_run["bmi"] = bmi_deid
            age = df_run["anchor_age"].to_numpy()
            if pd.api.types.is_numeric_dtype(df_run["anchor_age"]):
                age_bin = _age_to_bin_labels(age)
            else:
                age_bin = df_run["anchor_age"].astype(str)
            X = pd.get_dummies(pd.DataFrame({"age": age_bin, "gender": df_run["gender"].astype(str),
                                               "ethnicity": df_run["ethnicity"].astype(str),
                                               "insurance": df_run["insurance"].astype(str),
                                               "bmi": df_run["bmi"]}), drop_first=False).astype(float)
            X = X.fillna(0)
            clf = LogisticRegression(max_iter=2000, random_state=seed, C=0.5)
            clf.fit(X, y)
            prob = clf.predict_proba(X)[:, 1]
            row["downstream_auroc"] = float(roc_auc_score(y, prob))
        else:
            row["downstream_auroc"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 二、Agent 消融：fixed pipeline / rule-based Agent / wrong pipeline
# ---------------------------------------------------------------------------

def run_patient_profile_with_pipeline(
    df: pd.DataFrame,
    pipeline: List[str],
    config: Dict[str, Any],
    registry,
) -> pd.DataFrame:
    """用给定 pipeline（skill_id 列表）对 patient_profile 脱敏，不依赖 Agent.plan_pipeline。"""
    out = df.copy()
    id_cols = config.get("id_columns", [])
    age_col = config.get("age_column")
    numeric_quasiid_cols = config.get("numeric_quasiid_columns", [])
    cat_cols = config.get("cat_columns", [])
    cat_keep_lists = config.get("cat_keep_lists") or {}
    n_rows = len(out)
    min_freq = 1 if n_rows < 10 else 5

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                out[col] = skill.apply(out[col].astype(str).tolist(), {"secret": "demo_salt", "length": 12})
        elif sid == "demo_bin":
            if age_col and age_col in out.columns:
                skill = registry.get("demo_bin")
                out[age_col] = skill.apply(out[age_col].to_numpy(), {"mode": "age"})
        elif sid == "microagg":
            skill = registry.get("microagg")
            for col in numeric_quasiid_cols:
                if col not in out.columns:
                    continue
                out[col] = skill.apply(out[col].to_numpy(), {"k": 10})
        elif sid == "cat_agg":
            skill = registry.get("cat_agg")
            for col in cat_cols:
                if col not in out.columns:
                    continue
                keep_list = cat_keep_lists.get(col)
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"min_freq": min_freq, "other_label": "OTHER", "keep_list": keep_list},
                )
    return out


def quasiid_uniqueness(df: pd.DataFrame, cols: List[str]) -> float:
    """quasi-ID 列组合的唯一比例。"""
    if not cols:
        return 0.0
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return 0.0
    n_unique = df[cols].astype(str).fillna("").drop_duplicates().shape[0]
    return n_unique / len(df) if len(df) else 0.0


def run_agent_ablation(
    data_dir: Path,
    max_rows: int,
) -> pd.DataFrame:
    """
    三种模式：
    - fixed: 手工 strong pipeline [id_hash, demo_bin, microagg, cat_agg]
    - agent_rule: plan_pipeline("patient_profile", "strong")
    - wrong: plan_pipeline("patient_profile", "light") 模拟高隐私需求下误用轻量配置
    对比：quasiid_unique_ratio（脱敏后）、bmi 重构 MAE（若存在）、runtime_sec。
    """
    from demo_patient_and_timeline import PATIENT_PROFILE_CONFIG
    from skills_and_agent import build_default_registry, PrivacyAgent

    pp_path = data_dir / "patient_profile.csv"
    if not pp_path.exists():
        return pd.DataFrame()

    df_raw = pd.read_csv(pp_path, nrows=max_rows)
    for c in ["gender", "ethnicity", "insurance"]:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].fillna("").astype(str)
    if "bmi" not in df_raw.columns:
        df_raw["bmi"] = np.nan
    bmi_raw = pd.to_numeric(df_raw["bmi"], errors="coerce").to_numpy()
    quasi_cols = ["anchor_age", "gender", "ethnicity"]

    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    cfg = PATIENT_PROFILE_CONFIG
    strong_pipeline = ["id_hash", "demo_bin", "microagg", "cat_agg"]
    light_pipeline = agent.plan_pipeline("patient_profile", "light")

    rows = []
    # fixed (strong by hand)
    t0 = time.perf_counter()
    df_fixed = run_patient_profile_with_pipeline(df_raw.copy(), strong_pipeline, cfg, registry)
    t_fixed = time.perf_counter() - t0
    q_fixed = quasiid_uniqueness(df_fixed, quasi_cols)
    bmi_fixed = pd.to_numeric(df_fixed["bmi"], errors="coerce").to_numpy()
    recon_fixed = eval_reconstruction(bmi_raw, bmi_fixed)
    rows.append({
        "mode": "fixed_strong",
        "description": "hand-crafted strong pipeline (no Agent)",
        "quasiid_unique_ratio": q_fixed,
        "recon_mae": recon_fixed["recon_mae"],
        "runtime_sec": t_fixed,
    })

    # agent rule-based (strong)
    t0 = time.perf_counter()
    pipe_strong = agent.plan_pipeline("patient_profile", "strong")
    df_agent = run_patient_profile_with_pipeline(df_raw.copy(), pipe_strong, cfg, registry)
    t_agent = time.perf_counter() - t0
    q_agent = quasiid_uniqueness(df_agent, quasi_cols)
    bmi_agent = pd.to_numeric(df_agent["bmi"], errors="coerce").to_numpy()
    recon_agent = eval_reconstruction(bmi_raw, bmi_agent)
    rows.append({
        "mode": "agent_rule_strong",
        "description": "Agent plan_pipeline(strong)",
        "quasiid_unique_ratio": q_agent,
        "recon_mae": recon_agent["recon_mae"],
        "runtime_sec": t_agent,
    })

    # wrong: light in high-privacy scenario
    t0 = time.perf_counter()
    df_wrong = run_patient_profile_with_pipeline(df_raw.copy(), light_pipeline, cfg, registry)
    t_wrong = time.perf_counter() - t0
    q_wrong = quasiid_uniqueness(df_wrong, quasi_cols)
    bmi_wrong = pd.to_numeric(df_wrong["bmi"], errors="coerce").to_numpy()
    recon_wrong = eval_reconstruction(bmi_raw, bmi_wrong)
    rows.append({
        "mode": "wrong_light_as_strong",
        "description": "wrong: light pipeline in high-privacy scenario",
        "quasiid_unique_ratio": q_wrong,
        "recon_mae": recon_wrong["recon_mae"],
        "runtime_sec": t_wrong,
    })

    return pd.DataFrame(rows)


def plot_operator_ablation(df: pd.DataFrame, out_path: Path) -> None:
    """算子消融：pipeline vs recon_mae / recon_corr / downstream_auroc 柱状图。"""
    if not HAS_MPL or df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    x = np.arange(len(df))
    w = 0.35
    axes[0].bar(x - w/2, df["recon_mae"], width=w, label="recon_mae", color="C0", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["pipeline"], rotation=15, ha="right")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Reconstruction attack (MAE)")

    axes[1].bar(x - w/2, df["recon_corr"], width=w, label="recon_corr", color="C1", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["pipeline"], rotation=15, ha="right")
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Reconstruction (corr, lower=better privacy)")

    if "downstream_auroc" in df.columns:
        axes[2].bar(x - w/2, df["downstream_auroc"], width=w, label="AUROC", color="C2", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["pipeline"], rotation=15, ha="right")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Downstream utility")
    plt.suptitle("B.6 Operator ablation (numeric pipeline variants)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_agent_ablation(df: pd.DataFrame, out_path: Path) -> None:
    """Agent 消融：mode vs quasiid_unique_ratio / recon_mae / runtime。"""
    if not HAS_MPL or df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    x = np.arange(len(df))
    axes[0].bar(x, df["quasiid_unique_ratio"], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["mode"], rotation=20, ha="right")
    axes[0].set_ylabel("Quasi-ID unique ratio")
    axes[0].set_title("Privacy (lower=better)")

    axes[1].bar(x, df["recon_mae"], color="C1", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["mode"], rotation=20, ha="right")
    axes[1].set_ylabel("Recon MAE")
    axes[1].set_title("Reconstruction resistance")

    axes[2].bar(x, df["runtime_sec"], color="C2", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["mode"], rotation=20, ha="right")
    axes[2].set_ylabel("Time (sec)")
    axes[2].set_title("Runtime")
    plt.suptitle("B.6 Agent ablation (fixed vs rule-based vs wrong)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.6 Agent & 算子消融")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
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
    out_dir = Path(args.out_dir) if args.out_dir else find_b6_default_results_dir(ROOT)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 一、算子消融
    print("[B.6] 算子消融 (T1 / T1+T2 / T1+T3 / full)...")
    df_op = run_operator_ablation(data_dir, args.max_rows, args.seed)
    if not df_op.empty:
        df_op.to_csv(tables_dir / "table_b6_operator_ablation.csv", index=False)
        print(f"  已保存: {tables_dir / 'table_b6_operator_ablation.csv'}")
        if HAS_MPL:
            plot_operator_ablation(df_op, figs_dir / "b6_operator_ablation.png")
    else:
        print("  跳过（无 patient_profile 或 bmi）")

    # 二、Agent 消融
    print("[B.6] Agent 消融 (fixed / agent_rule / wrong)...")
    df_agent = run_agent_ablation(data_dir, args.max_rows)
    if not df_agent.empty:
        df_agent.to_csv(tables_dir / "table_b6_agent_ablation.csv", index=False)
        print(f"  已保存: {tables_dir / 'table_b6_agent_ablation.csv'}")
        if HAS_MPL:
            plot_agent_ablation(df_agent, figs_dir / "b6_agent_ablation.png")
    else:
        print("  跳过（无 patient_profile）")

    print("B.6 完成。")


if __name__ == "__main__":
    main()
