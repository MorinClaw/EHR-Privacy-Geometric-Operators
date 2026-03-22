#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.4 / B2.2：timeline_events 隐私 vs Utility + 隐私攻击

- 任务：ICU 前 24h 的 summary 特征（各 item 的 min/max/mean/std）预测合成结局 y_icu；
  从原始 timeline 与「脱敏 timeline（ID+时间+文本用 Agent，valuenum 用 numeric T1+T2+T3）」用相同规则提取特征；
  LR 或 XGBoost 比较 原始 vs 脱敏 的 AUROC/AUPRC。
- 隐私攻击（真实数据版）：
  1) 数值重构：脱敏 vs 原始的 MAE/RMSE/相关系数（单列 valuenum 或按 item 分列）；
  2) 重识别：quasi-ID（如 age_bin, gender, 若干 lab）在脱敏表上的唯一化比例；
  3) Membership inference：下游模型预测概率在正/负样本上的分布，阈值攻击估计 MI 成功率。

数据：data_preparation/experiment_extracted/timeline_events.csv、cohort_icu_stays.csv。
用法（实验根目录，PYTHONPATH 含 agent_demo）：
  python3 B4_privacy_utility_tradeoff/code/exp_b4_timeline_icu_tasks.py --max-stays 500 --max-rows 50000
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _load_data(data_dir: Path, max_stays: int | None, max_timeline_rows: int | None):
    """加载 cohort_icu_stays 与 timeline_events，返回 (cohort_df, timeline_df)。"""
    cohort_path = data_dir / "cohort_icu_stays.csv"
    timeline_path = data_dir / "timeline_events.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"未找到: {cohort_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"未找到: {timeline_path}")
    cohort = pd.read_csv(cohort_path, nrows=max_stays)
    timeline = pd.read_csv(timeline_path, nrows=max_timeline_rows)
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    timeline["charttime"] = pd.to_datetime(timeline["charttime"], errors="coerce")
    return cohort, timeline


def _add_synthetic_icu_outcome(cohort: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """基于 los、anchor_age 生成合成 y_icu（0/1）。"""
    rng = np.random.default_rng(seed)
    los = np.asarray(cohort["los"], dtype=float)
    age = np.asarray(cohort["anchor_age"], dtype=float)
    np.nan_to_num(los, nan=1.0, copy=False)
    np.nan_to_num(age, nan=50.0, copy=False)
    risk = (los - 2.0) / 5.0 + (age - 60) / 40.0 + rng.normal(0, 0.5, size=len(cohort))
    p = 1.0 / (1.0 + np.exp(-np.clip(risk, -20, 20)))
    return (rng.random(size=len(cohort)) < p).astype(int)


def _extract_24h_features_one_timeline(
    cohort: pd.DataFrame,
    timeline: pd.DataFrame,
    valuenum_col: str = "valuenum",
) -> pd.DataFrame:
    """
    对每个 stay，取 intime 起 24h 内的事件，按 item 聚合 valuenum 的 min/max/mean/std；
    返回 index=stay_id 的 DataFrame，列如 item_220045_min, item_220045_max, ...
    """
    # 只保留在 cohort 中的 stay
    stay_ids = set(cohort["stay_id"].astype(int))
    timeline = timeline[timeline["stay_id"].isin(stay_ids)].copy()
    if timeline.empty:
        return pd.DataFrame(index=cohort["stay_id"])

    intime_by_stay = cohort.set_index("stay_id")["intime"]
    timeline["intime"] = timeline["stay_id"].map(intime_by_stay)
    timeline = timeline.dropna(subset=["intime", "charttime"])
    timeline["hours"] = (timeline["charttime"] - timeline["intime"]).dt.total_seconds() / 3600.0
    first24 = timeline[(timeline["hours"] >= 0) & (timeline["hours"] < 24)]

    rows = []
    for stay_id in cohort["stay_id"]:
        sub = first24[first24["stay_id"] == stay_id]
        if sub.empty:
            rows.append({"stay_id": stay_id})
            continue
        agg = sub.groupby("item")[valuenum_col].agg(["min", "max", "mean", "std"])
        row = {"stay_id": stay_id}
        for item_id, r in agg.iterrows():
            for stat in ["min", "max", "mean", "std"]:
                row[f"item_{item_id}_{stat}"] = r[stat]
        rows.append(row)

    X = pd.DataFrame(rows)
    X = X.set_index("stay_id")
    return X


def _align_feature_matrices(X_raw: pd.DataFrame, X_deid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对齐列，缺的填 0。"""
    all_cols = sorted(set(X_raw.columns) | set(X_deid.columns))
    X_raw_a = X_raw.reindex(columns=all_cols, fill_value=0.0)
    X_deid_a = X_deid.reindex(columns=all_cols, fill_value=0.0)
    return X_raw_a, X_deid_a


def _deidentify_timeline_with_numeric_pipeline(
    timeline: pd.DataFrame,
    privacy_level: str = "strong",
) -> pd.DataFrame:
    """
    仅对 valuenum 做 numeric 流水线（num_triplet + num_noise_proj + num_householder），
    不改 ID/时间/文本，以便与 cohort 按 stay_id 对齐提取 24h 特征。
    用于「原始 vs 脱敏」特征下的 ICU 预测对比；隐私攻击中的重构评估也用此脱敏列。
    """
    from skills_and_agent import build_default_registry, PrivacyAgent

    out = timeline.copy()
    if "valuenum" not in out.columns:
        return out
    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    x = pd.to_numeric(out["valuenum"], errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(x)
    fill_val = np.nanmean(x[valid]) if valid.any() else 0.0
    x_fill = np.where(valid, x, fill_val)
    pipeline_numeric = ["num_triplet", "num_noise_proj", "num_householder"]
    # 方案一：归一化空间固定阈值 0.8（增强抗重建）
    skill_configs = {sid: {"max_diff": 0.8} for sid in pipeline_numeric}
    skill_configs["num_triplet"]["n_passes"] = 3
    try:
        history = agent.run_numeric_pipeline(x_fill, pipeline_numeric, skill_configs)
        y = np.asarray(history[-1]["data"], dtype=float)
        y[~valid] = np.nan
        out["valuenum"] = y
    except Exception as e:
        print(f"  [B2.2] numeric pipeline 失败: {e}，保留原 valuenum")
    return out


def run_icu_prediction(
    cohort: pd.DataFrame,
    timeline_raw: pd.DataFrame,
    timeline_deid: pd.DataFrame,
    y_icu: np.ndarray,
    seed: int = 42,
    use_xgb: bool = False,
) -> dict:
    """提取 24h 特征，训练模型，返回 raw/deid 的 AUROC、AUPRC 及预测概率。"""
    X_raw = _extract_24h_features_one_timeline(cohort, timeline_raw)
    X_deid = _extract_24h_features_one_timeline(cohort, timeline_deid)
    # 对齐 stay_id 顺序
    stay_ids = cohort["stay_id"].values
    X_raw = X_raw.reindex(stay_ids).fillna(0)
    X_deid = X_deid.reindex(stay_ids).fillna(0)
    X_raw, X_deid = _align_feature_matrices(X_raw, X_deid)
    # 可能有 NaN（std 等），再填 0
    X_raw = X_raw.fillna(0)
    X_deid = X_deid.fillna(0)

    n = len(y_icu)
    if n != len(X_raw) or n != len(X_deid):
        raise ValueError("cohort / X 行数与 y_icu 不一致")
    if y_icu.sum() < 5 or (n - y_icu.sum()) < 5:
        return {"auc_raw": np.nan, "auc_deid": np.nan, "auprc_raw": np.nan, "auprc_deid": np.nan}

    if use_xgb and HAS_XGB:
        clf_raw = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed, n_estimators=50)
        clf_deid = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed, n_estimators=50)
    else:
        clf_raw = LogisticRegression(max_iter=2000, random_state=seed, C=0.5)
        clf_deid = LogisticRegression(max_iter=2000, random_state=seed, C=0.5)

    clf_raw.fit(X_raw, y_icu)
    clf_deid.fit(X_deid, y_icu)
    prob_raw = clf_raw.predict_proba(X_raw)[:, 1]
    prob_deid = clf_deid.predict_proba(X_deid)[:, 1]

    return {
        "auc_raw": float(roc_auc_score(y_icu, prob_raw)),
        "auc_deid": float(roc_auc_score(y_icu, prob_deid)),
        "auprc_raw": float(average_precision_score(y_icu, prob_raw)),
        "auprc_deid": float(average_precision_score(y_icu, prob_deid)),
        "y_icu": y_icu,
        "prob_raw": prob_raw,
        "prob_deid": prob_deid,
    }


def run_numeric_reconstruction(timeline_raw: pd.DataFrame, timeline_deid: pd.DataFrame) -> list[dict]:
    """数值重构：valuenum 列 MAE/RMSE/相关系数。"""
    if "valuenum" not in timeline_raw.columns or "valuenum" not in timeline_deid.columns:
        return []
    o = pd.to_numeric(timeline_raw["valuenum"], errors="coerce").to_numpy()
    d = pd.to_numeric(timeline_deid["valuenum"], errors="coerce").to_numpy()
    mask = ~(np.isnan(o) | np.isnan(d))
    if mask.sum() < 10:
        return [{"col": "valuenum", "n": int(mask.sum()), "mae": np.nan, "rmse": np.nan, "corr": np.nan}]
    o, d = o[mask], d[mask]
    mae = float(np.mean(np.abs(o - d)))
    rmse = float(np.sqrt(np.mean((o - d) ** 2)))
    corr = float(np.corrcoef(o, d)[0, 1])
    return [{"col": "valuenum", "n": int(mask.sum()), "mae": mae, "rmse": rmse, "corr": corr}]


def run_quasiid_uniqueness(profile_raw: pd.DataFrame | None, profile_deid: pd.DataFrame | None) -> dict:
    """
    重识别：用 quasi-ID 列（如 anchor_age 分箱、gender、ethnicity）在脱敏表上算唯一组合数/总行数。
    若未提供 profile，返回空 dict。
    """
    if profile_deid is None or profile_deid.empty:
        return {}
    cols = [c for c in ["anchor_age", "gender", "ethnicity"] if c in profile_deid.columns]
    if not cols:
        return {"n_rows": len(profile_deid), "n_unique_quasiid": 0, "unique_ratio": 0.0}
    quasi = profile_deid[cols].astype(str).fillna("")
    n_unique = quasi.drop_duplicates().shape[0]
    n_rows = len(profile_deid)
    return {
        "n_rows": n_rows,
        "n_unique_quasiid": n_unique,
        "unique_ratio": n_unique / n_rows if n_rows else 0.0,
    }


def run_membership_inference(res: dict, threshold: float = 0.5) -> dict:
    """
    Membership inference：正样本 vs 负样本的预测概率分布，用阈值估计“猜中是否在训练集”的成功率。
    简化：用同一模型在相同数据上的预测概率，正样本平均概率 > 负样本 则 MI 攻击可利用该差异。
    """
    y = res["y_icu"]
    prob_raw = res["prob_raw"]
    prob_deid = res["prob_deid"]
    pos_raw = prob_raw[y == 1]
    neg_raw = prob_raw[y == 0]
    pos_deid = prob_deid[y == 1]
    neg_deid = prob_deid[y == 0]
    # 阈值攻击：若 prob > threshold 猜正例，否则负例；准确率即分类准确率（与 MI 的“是否在训练集”不同，这里简化为“预测概率区分度”）
    acc_raw = (np.mean(pos_raw > threshold) + np.mean(neg_raw <= threshold)) / 2
    acc_deid = (np.mean(pos_deid > threshold) + np.mean(neg_deid <= threshold)) / 2
    return {
        "mean_prob_pos_raw": float(np.mean(pos_raw)),
        "mean_prob_neg_raw": float(np.mean(neg_raw)),
        "mean_prob_pos_deid": float(np.mean(pos_deid)),
        "mean_prob_neg_deid": float(np.mean(neg_deid)),
        "threshold": threshold,
        "sep_raw": float(np.mean(pos_raw) - np.mean(neg_raw)),
        "sep_deid": float(np.mean(pos_deid) - np.mean(neg_deid)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B2.2 timeline ICU 预测 + 隐私攻击")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-stays", type=int, default=500, help="cohort 最多 stay 数")
    p.add_argument("--max-rows", type=int, default=100000, help="timeline 最多行数")
    p.add_argument("--privacy-level", type=str, default="strong", choices=["light", "medium", "strong"])
    p.add_argument("--xgb", action="store_true", help="用 XGBoost 替代 LR")
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
    tables_dir.mkdir(parents=True, exist_ok=True)

    cohort, timeline_raw = _load_data(data_dir, args.max_stays, args.max_rows)
    # 只保留 cohort 中 stay 的 timeline 行以节省内存
    stay_ids = set(cohort["stay_id"].astype(int))
    timeline_raw = timeline_raw[timeline_raw["stay_id"].isin(stay_ids)].copy()
    print(f"[B2.2] cohort 行数={len(cohort)}, timeline 行数={len(timeline_raw)}")

    y_icu = _add_synthetic_icu_outcome(cohort, seed=args.seed)
    print(f"[B2.2] y_icu 正例数={y_icu.sum()}, 负例数={len(y_icu)-y_icu.sum()}")

    print("[B2.2] 脱敏 timeline（ID+时间+文本 + numeric pipeline 替代 ds_tab）...")
    timeline_deid = _deidentify_timeline_with_numeric_pipeline(timeline_raw, args.privacy_level)

    if HAS_SKLEARN:
        res = run_icu_prediction(cohort, timeline_raw, timeline_deid, y_icu, seed=args.seed, use_xgb=args.xgb and HAS_XGB)
        print("\n[B2.2] ICU 预测 (24h 特征):")
        print(f"  AUC   raw={res['auc_raw']:.4f}  deid={res['auc_deid']:.4f}")
        print(f"  AUPRC raw={res['auprc_raw']:.4f}  deid={res['auprc_deid']:.4f}")
        pd.DataFrame([
            {"metric": "AUC", "raw": res["auc_raw"], "deid": res["auc_deid"]},
            {"metric": "AUPRC", "raw": res["auprc_raw"], "deid": res["auprc_deid"]},
        ]).to_csv(tables_dir / "table_b4_timeline_auc_auprc.csv", index=False)
        mi = run_membership_inference(res)
        print(f"  MI 分离度 raw={mi['sep_raw']:.4f}  deid={mi['sep_deid']:.4f}")
        pd.DataFrame([mi]).to_csv(tables_dir / "table_b4_timeline_mi.csv", index=False)
    else:
        print("[B2.2] 未安装 sklearn，跳过 ICU 预测与 MI")

    recon = run_numeric_reconstruction(timeline_raw, timeline_deid)
    if recon:
        pd.DataFrame(recon).to_csv(tables_dir / "table_b4_timeline_reconstruction.csv", index=False)
        for r in recon:
            print(f"  数值重构 {r['col']}: n={r['n']} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} corr={r['corr']:.4f}")

    # quasi-ID 需要 patient_profile 脱敏表；若存在则算
    profile_path = data_dir / "patient_profile.csv"
    if profile_path.exists():
        from demo_patient_and_timeline import deidentify_patient_profile_df, PATIENT_PROFILE_CONFIG
        from skills_and_agent import build_default_registry, PrivacyAgent
        profile_raw = pd.read_csv(profile_path, nrows=args.max_stays * 2)  # 可能多 subject
        agent = PrivacyAgent(build_default_registry())
        profile_deid = deidentify_patient_profile_df(profile_raw, agent, args.privacy_level, PATIENT_PROFILE_CONFIG)
        qid = run_quasiid_uniqueness(profile_raw, profile_deid)
        if qid:
            pd.DataFrame([qid]).to_csv(tables_dir / "table_b4_quasiid_uniqueness.csv", index=False)
            print(f"  Quasi-ID 唯一组合数={qid.get('n_unique_quasiid')} 比例={qid.get('unique_ratio', 0):.4f}")

    print("B2.2 timeline_events 完成。")


if __name__ == "__main__":
    main()
