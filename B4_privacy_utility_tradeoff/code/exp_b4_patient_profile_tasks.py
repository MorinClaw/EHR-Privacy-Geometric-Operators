#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.4 / B2.1：patient_profile 隐私 vs Utility

使用 data_preparation/experiment_extracted/patient_profile.csv（真实 MIMIC patient_profile），
合成结局 y_1y / y_30d，经 deidentify_patient_profile_df 脱敏后：
- 任务 1：逻辑回归风险预测，比较原始 vs 脱敏的 AUC、Brier，并输出校准曲线；
- 任务 2：按 ethnicity / insurance 分层计算 30d 事件率 + 95% Wilson CI，
  对比原始表与脱敏表分层结果，输出柱状图与森林图。

用法（实验根目录下，PYTHONPATH 含 agent_demo）：
  python3 B4_privacy_utility_tradeoff/code/exp_b4_patient_profile_tasks.py
  python3 B4_privacy_utility_tradeoff/code/exp_b4_patient_profile_tasks.py --data-dir data_preparation/experiment_extracted --privacy-level strong --out-dir B4_privacy_utility_tradeoff/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import (
    default_experiment_extracted_dir,
    find_agent_demo_dir,
    find_b4_default_results_dir,
)

AGENT_DIR = find_agent_demo_dir(ROOT)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from demo_patient_and_timeline import (
    deidentify_patient_profile_df,
    PATIENT_PROFILE_CONFIG,
)
from skills_and_agent import build_default_registry, PrivacyAgent

# 从 demo 复用：年龄分箱、合成结局、特征构造、任务1/2 与绘图
from demo_patient_profile_tasks import (
    _age_to_bin_labels,
    add_synthetic_outcomes,
    build_feature_matrix,
    run_task1,
    print_task1_table,
    plot_calibration,
    wilson_ci,
    run_task2,
    print_task2_tables,
    plot_task2,
)


def load_patient_profile(data_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """从 data_preparation/experiment_extracted 读 patient_profile.csv。"""
    path = data_dir / "patient_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到: {path}")
    df = pd.read_csv(path, nrows=max_rows)
    # 保证任务所需列存在；缺失的用占位
    for col in ["gender", "ethnicity", "insurance"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""
    if "anchor_age" not in df.columns:
        raise ValueError("patient_profile 需包含 anchor_age")
    if "bmi" not in df.columns:
        df["bmi"] = np.nan
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2.1 patient_profile: 风险预测 + 公平性")
    p.add_argument("--data-dir", type=str, default=None, help="数据目录，默认 ROOT/data_preparation/experiment_extracted")
    p.add_argument("--out-dir", type=str, default=None, help="结果目录，默认 B4_privacy_utility_tradeoff/results")
    p.add_argument("--privacy-level", type=str, default="strong", choices=["light", "medium", "strong"])
    p.add_argument("--max-rows", type=int, default=None, help="最多加载行数（默认全量）")
    p.add_argument("--no-plot", action="store_true", help="不生成图片")
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
    out_dir = Path(args.out_dir) if args.out_dir else find_b4_default_results_dir(ROOT)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载真实 patient_profile + 合成结局
    df_raw = load_patient_profile(data_dir, args.max_rows)
    df_raw = add_synthetic_outcomes(df_raw, seed=args.seed)
    y_1y = df_raw["y_1y"].to_numpy()
    y_30d = df_raw["y_30d"].to_numpy()

    # 2) 脱敏（去掉结局列再脱敏，再复制回去）
    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    df_deid = deidentify_patient_profile_df(
        df_raw.drop(columns=["y_1y", "y_30d"]),
        agent,
        args.privacy_level,
        PATIENT_PROFILE_CONFIG,
    )
    df_deid["y_1y"] = y_1y
    df_deid["y_30d"] = y_30d

    # 3) 任务 1：LR 风险预测
    res1 = run_task1(df_raw, df_deid, y_1y, seed=args.seed)
    print_task1_table(res1)
    # 保存 Task1 表
    task1_table = pd.DataFrame([
        {"metric": "AUC", "raw": res1["auc_raw"], "deid": res1["auc_deid"], "diff": res1["auc_deid"] - res1["auc_raw"]},
        {"metric": "Brier", "raw": res1["brier_raw"], "deid": res1["brier_deid"], "diff": res1["brier_deid"] - res1["brier_raw"]},
    ])
    task1_table.to_csv(tables_dir / "table_b4_task1_auc_brier.csv", index=False)
    print(f"  Task1 表已保存: {tables_dir / 'table_b4_task1_auc_brier.csv'}")
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_calibration(res1, str(figs_dir / "task1_calibration_raw_vs_deid.png"))

    # 4) 任务 2：分层公平性（脱敏表 + 原始表对比）
    rates_eth_deid, rates_ins_deid = run_task2(df_deid, y_30d)
    rates_eth_raw, rates_ins_raw = run_task2(df_raw, y_30d)
    print_task2_tables(rates_eth_deid, rates_ins_deid)
    # 保存 Task2 分层表（脱敏 + 原始）
    rates_eth_raw["table"] = "raw"
    rates_eth_deid["table"] = "deid"
    rates_ins_raw["table"] = "raw"
    rates_ins_deid["table"] = "deid"
    rates_eth_raw["stratifier"] = "ethnicity"
    rates_eth_deid["stratifier"] = "ethnicity"
    rates_ins_raw["stratifier"] = "insurance"
    rates_ins_deid["stratifier"] = "insurance"
    task2_eth = pd.concat([rates_eth_raw, rates_eth_deid], ignore_index=True)
    task2_ins = pd.concat([rates_ins_raw, rates_ins_deid], ignore_index=True)
    task2_eth.to_csv(tables_dir / "table_b4_task2_stratified_ethnicity.csv", index=False)
    task2_ins.to_csv(tables_dir / "table_b4_task2_stratified_insurance.csv", index=False)
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_task2(rates_eth_deid, rates_ins_deid, str(figs_dir))
    print("B2.1 patient_profile 完成。")


if __name__ == "__main__":
    main()
