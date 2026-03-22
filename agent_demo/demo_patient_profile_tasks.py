#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_patient_profile_tasks.py

在 patient_profile 脱敏前后做两个示例下游任务，并给出简单可视化：

  任务 1：基于人口学预测 1 年死亡/再入院风险
    - 用年龄段、性别、种族、保险等构造特征，合成 1 年结局标签
    - 在原始 vs 脱敏后的特征上分别训练逻辑回归，比较 AUC / Brier
    - 输出对比表 + 校准曲线示意图

  任务 2：人群分层与公平性分析（按种族/保险的 30d 再入院率）
    - 合成 30d 再入院/急诊复诊标签
    - 按种族、保险分层计算事件率及 Wilson 置信区间
    - 输出柱状图 + 森林图（含 CI）

用法：
    python demo_patient_profile_tasks.py --n 500 --privacy-level strong
    python demo_patient_profile_tasks.py --n 500 --no-plot   # 只打表不画图
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 可选：画图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from demo_patient_profile_synthetic import make_demo_patient_profile
from demo_patient_and_timeline import (
    deidentify_patient_profile_df,
    PATIENT_PROFILE_CONFIG,
)
from skills_and_agent import build_default_registry, PrivacyAgent


# ---------------------------------------------------------------------------
# 年龄分箱（与 non_numeric_operators.demo_binning 一致，便于对齐特征）
# ---------------------------------------------------------------------------

def _age_to_bin_labels(ages: np.ndarray) -> np.ndarray:
    """将年龄数组转为与 T_demo-bin 一致的区间标签。"""
    bins = np.array(list(range(0, 90, 5)) + [200], dtype=float)
    labels = [f"[{int(bins[i])},{int(bins[i+1]-1)}]" for i in range(len(bins) - 2)] + [">=90"]
    idx = np.digitize(np.asarray(ages, dtype=float), bins, right=False) - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array([labels[i] for i in idx], dtype=object)


# ---------------------------------------------------------------------------
# 合成结局标签（与人口学有简单关联，便于展示脱敏前后模型差异）
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def add_synthetic_outcomes(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    在 DataFrame 上增加两列合成结局：
      - y_1y: 1 年死亡/再入院 (0/1)，与年龄、保险、种族有弱相关
      - y_30d: 30 天再入院/急诊复诊 (0/1)，用于分层公平性分析
    不修改原有列，返回带新列的副本。
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)

    age = np.asarray(out["anchor_age"], dtype=float)
    # 若已是分箱字符串，取区间中点近似（仅用于合成逻辑）
    if out["anchor_age"].dtype == object or out["anchor_age"].dtype.name == "string":
        def mid(s: str) -> float:
            s = str(s)
            if s.startswith(">="):
                return 92.0
            if "[" in s and "," in s:
                a, b = s.replace("[", "").replace("]", "").split(",")
                return (float(a) + float(b)) / 2
            return 65.0
        age = np.array([mid(x) for x in out["anchor_age"]])

    insurance = out["insurance"].astype(str)
    ethnicity = out["ethnicity"].astype(str)
    # 风险得分：年龄越大、Medicare/Medicaid、部分种族略高
    risk_1y = (age - 50) / 30.0
    risk_1y += np.where(insurance.isin(["Medicare", "Medicaid"]), 0.4, 0.0)
    risk_1y += np.where(ethnicity == "BLACK", 0.2, 0.0)
    risk_1y += rng.normal(0, 0.5, size=n)
    p_1y = _sigmoid(risk_1y)
    out["y_1y"] = (rng.random(size=n) < p_1y).astype(int)

    # 30d 再入院：与年龄、保险相关，加随机性
    risk_30d = (age - 55) / 25.0 + np.where(insurance == "Self Pay", 0.3, 0.0)
    risk_30d += rng.normal(0, 0.6, size=n)
    p_30d = _sigmoid(risk_30d)
    out["y_30d"] = (rng.random(size=n) < p_30d).astype(int)

    return out


# ---------------------------------------------------------------------------
# 特征构造：用于逻辑回归的 one-hot 等
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    age_col: str = "anchor_age",
    cat_cols: list[str] | None = None,
    use_age_bin: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    从 patient_profile 表构造模型特征。
    - age_col: 若 use_age_bin 且该列为数值，则先转为与 demo_bin 一致的区间再 one-hot；
              若已是字符串（脱敏后），直接当类别用。
    - cat_cols: 类别列，做 one-hot，缺省为 ["gender", "ethnicity", "insurance"]。
    返回 (X_df, feature_names)。
    """
    if cat_cols is None:
        cat_cols = ["gender", "ethnicity", "insurance"]

    out = pd.DataFrame(index=df.index)
    feature_names: list[str] = []

    # 年龄：统一为分箱后 one-hot
    if age_col in df.columns:
        age_vals = df[age_col]
        if use_age_bin and pd.api.types.is_numeric_dtype(age_vals):
            age_bin = _age_to_bin_labels(age_vals.to_numpy())
        else:
            age_bin = age_vals.astype(str)
        age_dummies = pd.get_dummies(age_bin, prefix="age", drop_first=False)
        for c in age_dummies.columns:
            out[c] = age_dummies[c]
            feature_names.append(c)

    for col in cat_cols:
        if col not in df.columns:
            continue
        d = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
        for c in d.columns:
            out[c] = d[c]
            feature_names.append(c)

    return out, feature_names


# ---------------------------------------------------------------------------
# 任务 1：逻辑回归 + AUC/Brier + 校准曲线
# ---------------------------------------------------------------------------

def run_task1(
    df_raw: pd.DataFrame,
    df_deid: pd.DataFrame,
    y_1y: np.ndarray,
    seed: int = 42,
) -> dict:
    """在原始与脱敏特征上分别训练 LR，返回指标与预测概率（用于校准图）。"""
    X_raw, _ = build_feature_matrix(df_raw, use_age_bin=True)
    X_deid, _ = build_feature_matrix(df_deid, use_age_bin=False)

    # 对齐列：取并集并补 0（缺的类别表示未出现）
    all_cols = sorted(set(X_raw.columns) | set(X_deid.columns))
    X_raw_aligned = X_raw.reindex(columns=all_cols, fill_value=0)
    X_deid_aligned = X_deid.reindex(columns=all_cols, fill_value=0)

    clf = LogisticRegression(max_iter=500, random_state=seed, C=0.5)
    clf_raw = clf.fit(X_raw_aligned, y_1y)
    clf_deid = LogisticRegression(max_iter=500, random_state=seed, C=0.5).fit(X_deid_aligned, y_1y)

    prob_raw = clf_raw.predict_proba(X_raw_aligned)[:, 1]
    prob_deid = clf_deid.predict_proba(X_deid_aligned)[:, 1]

    auc_raw = roc_auc_score(y_1y, prob_raw)
    auc_deid = roc_auc_score(y_1y, prob_deid)
    brier_raw = brier_score_loss(y_1y, prob_raw)
    brier_deid = brier_score_loss(y_1y, prob_deid)

    return {
        "auc_raw": auc_raw,
        "auc_deid": auc_deid,
        "brier_raw": brier_raw,
        "brier_deid": brier_deid,
        "y_1y": y_1y,
        "prob_raw": prob_raw,
        "prob_deid": prob_deid,
    }


def print_task1_table(res: dict) -> None:
    print("\n" + "=" * 60)
    print("任务 1：基于人口学预测 1 年死亡/再入院风险")
    print("  逻辑回归在「原始特征」vs「脱敏后特征」上的表现对比")
    print("=" * 60)
    print(f"  {'指标':<12}  {'原始':>10}  {'脱敏后':>10}  {'差异':>10}")
    print("-" * 50)
    for name, key_raw, key_deid in [
        ("AUC", "auc_raw", "auc_deid"),
        ("Brier", "brier_raw", "brier_deid"),
    ]:
        r, d = res[key_raw], res[key_deid]
        delta = d - r
        print(f"  {name:<12}  {r:>10.4f}  {d:>10.4f}  {delta:>+10.4f}")
    print("=" * 60)


def plot_calibration(res: dict, out_path: str | None = None) -> None:
    """简单校准曲线：预测概率分箱 vs 实际正例率。"""
    if not HAS_MATPLOTLIB:
        return
    y = res["y_1y"]
    prob_raw = res["prob_raw"]
    prob_deid = res["prob_deid"]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    def bin_calibration(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_true = []
        mean_pred = []
        for i in range(len(bins) - 1):
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
            if i == len(bins) - 2:
                mask = (prob >= bins[i]) & (prob <= bins[i + 1])
            if mask.sum() == 0:
                mean_true.append(np.nan)
                mean_pred.append(np.nan)
            else:
                mean_true.append(y[mask].mean())
                mean_pred.append(prob[mask].mean())
        return np.array(mean_true), np.array(mean_pred)

    true_raw, pred_raw = bin_calibration(prob_raw)
    true_deid, pred_deid = bin_calibration(prob_deid)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(bin_centers, true_raw, "o-", color="C0", label="Raw")
    ax.plot(bin_centers, true_deid, "s-", color="C1", label="De-id")
    ax.set_xlabel("Mean predicted prob (binned)")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Task 1: Calibration (1-y death/readmission)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"  校准曲线已保存: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 任务 2：分层事件率 + Wilson CI，柱状图与森林图
# ---------------------------------------------------------------------------

def wilson_ci(n: int, p: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间。"""
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    spread = (z / den) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    low = max(0, center - spread)
    high = min(1, center + spread)
    return (low, high)


def run_task2(df: pd.DataFrame, y_30d: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按 ethnicity 和 insurance 分层计算 30d 事件率及 95% Wilson CI。
    返回 (rates_ethnicity, rates_insurance)。
    """
    df = df.copy()
    df["y_30d"] = y_30d

    def stratum_rates(col: str) -> pd.DataFrame:
        rows = []
        for g, grp in df.groupby(col, dropna=False):
            n = len(grp)
            ev = grp["y_30d"].sum()
            r = ev / n if n else 0
            lo, hi = wilson_ci(n, r)
            rows.append({"group": str(g), "n": n, "events": int(ev), "rate": r, "ci_low": lo, "ci_high": hi})
        return pd.DataFrame(rows)

    return stratum_rates("ethnicity"), stratum_rates("insurance")


def print_task2_tables(rates_eth: pd.DataFrame, rates_ins: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("任务 2：人群分层与公平性分析（30d 再入院/急诊复诊率）")
    print("=" * 60)
    print("\n按种族 (ethnicity):")
    print(rates_eth.to_string(index=False))
    print("\n按保险 (insurance):")
    print(rates_ins.to_string(index=False))
    print("=" * 60)


def plot_task2(
    rates_eth: pd.DataFrame,
    rates_ins: pd.DataFrame,
    out_dir: str | None = None,
) -> None:
    """柱状图（按种族、按保险）+ 森林图（含 CI）。"""
    if not HAS_MATPLOTLIB:
        return
    out_dir = out_dir or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 柱状图：种族
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rates, title in [
        (axes[0], rates_eth, "By ethnicity"),
        (axes[1], rates_ins, "By insurance"),
    ]:
        x = np.arange(len(rates))
        ax.bar(x, rates["rate"], color="steelblue", alpha=0.8, edgecolor="gray")
        ax.errorbar(
            x, rates["rate"],
            yerr=[rates["rate"] - rates["ci_low"], rates["ci_high"] - rates["rate"]],
            fmt="none", color="black", capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rates["group"], rotation=45, ha="right")
        ax.set_ylabel("30d event rate")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    plt.suptitle("Task 2: Stratified 30d readmission rate (95% Wilson CI)")
    plt.tight_layout()
    bar_path = str(Path(out_dir) / "task2_stratified_rates.png")
    plt.savefig(bar_path, dpi=120)
    plt.close()
    print(f"  分层柱状图已保存: {bar_path}")

    # 森林图：合并两组，用 group 类型区分
    rates_eth["stratifier"] = "ethnicity"
    rates_ins["stratifier"] = "insurance"
    combined = pd.concat([rates_eth, rates_ins], ignore_index=True)
    combined["label"] = combined["stratifier"] + " / " + combined["group"]

    fig, ax = plt.subplots(1, 1, figsize=(7, max(5, len(combined) * 0.35)))
    y_pos = np.arange(len(combined))[::-1]
    ax.errorbar(
        combined["rate"], y_pos,
        xerr=[combined["rate"] - combined["ci_low"], combined["ci_high"] - combined["rate"]],
        fmt="o", capsize=3, color="steelblue",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined["label"], fontsize=9)
    ax.set_xlabel("30d event rate (95% CI)")
    ax.set_title("Task 2: Forest plot (stratified rates)")
    ax.set_xlim(0, 1)
    ax.axvline(x=combined["rate"].mean(), color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    forest_path = str(Path(out_dir) / "task2_forest_plot.png")
    plt.savefig(forest_path, dpi=120)
    plt.close()
    print(f"  森林图已保存: {forest_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patient profile 任务 1 & 2 示例：风险预测 + 分层公平性")
    p.add_argument("--n", type=int, default=500, help="模拟患者数（建议 ≥300）")
    p.add_argument("--privacy-level", type=str, default="strong", choices=["light", "medium", "strong"])
    p.add_argument("--no-plot", action="store_true", help="不生成图片，只打印表格")
    p.add_argument("--out-dir", type=str, default=".", help="图片输出目录")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = max(200, args.n)
    seed = args.seed

    # 1) 合成 patient_profile + 脱敏
    df_raw = make_demo_patient_profile(n=n, seed=seed)
    df_raw = add_synthetic_outcomes(df_raw, seed=seed)
    y_1y = df_raw["y_1y"].to_numpy()
    y_30d = df_raw["y_30d"].to_numpy()

    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    df_deid = deidentify_patient_profile_df(
        df_raw.drop(columns=["y_1y", "y_30d"]),
        agent,
        args.privacy_level,
        PATIENT_PROFILE_CONFIG,
    )
    # 脱敏表没有结局，用同一批 y（按行对齐）
    df_deid["y_1y"] = y_1y
    df_deid["y_30d"] = y_30d

    # 2) 任务 1
    res1 = run_task1(df_raw, df_deid, y_1y, seed=seed)
    print_task1_table(res1)
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_calibration(res1, str(Path(args.out_dir) / "task1_calibration.png"))

    # 3) 任务 2（用脱敏后的表做分层，体现脱敏后仍可做公平性分析）
    rates_eth, rates_ins = run_task2(df_deid, y_30d)
    print_task2_tables(rates_eth, rates_ins)
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_task2(rates_eth, rates_ins, args.out_dir)
    elif args.no_plot:
        print("  (未安装 matplotlib 或 --no-plot，跳过图片生成)")


if __name__ == "__main__":
    main()
