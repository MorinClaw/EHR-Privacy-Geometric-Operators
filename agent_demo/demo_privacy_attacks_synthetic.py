# filename: demo_privacy_attacks_synthetic.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_privacy_attacks_synthetic.py

在 synthetic 的 patient_profile / timeline_events / notes 上，
跑一遍脱敏流水线，然后模拟几类“攻击”来检验：

  1. 数值列是否很难精确还原原始数据（只能近似、误差较大）；
  2. 自由文本中的典型 PHI（email / 电话 / URL / 长数字 / 日期 / 已知姓名）是否都被去除或替换；
  3. 最后输出一个简单的攻击效果小结。

依赖：
  - skills_and_agent.build_default_registry / PrivacyAgent
  - demo_patient_and_timeline.deidentify_* 及对应 *_CONFIG
  - 三个 synthetic demo 中定义的 make_demo_* 函数

用法示例：
    python demo_privacy_attacks_synthetic.py --privacy-level strong
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from skills_and_agent import build_default_registry, PrivacyAgent
from demo_patient_and_timeline import (
    deidentify_patient_profile_df,
    deidentify_timeline_df,
    deidentify_notes_df,
    PATIENT_PROFILE_CONFIG,
    TIMELINE_CONFIG,
    NOTES_CONFIG,
)

# 复用前面三个 synthetic demo 里定义的生成函数
from demo_patient_profile_synthetic import make_demo_patient_profile
from demo_timeline_synthetic import make_demo_timeline_events
from demo_notes_synthetic import make_demo_notes


# ---------- 数值重构攻击（evaluation）工具函数 ----------

def evaluate_numeric_reconstruction(
    df_orig: pd.DataFrame,
    df_deid: pd.DataFrame,
    numeric_cols: List[str],
    table_name: str,
) -> None:
    """
    模拟“数值重构攻击”的效果评估：
      - 攻击者把脱敏后的数值当作原始值的估计（或者用线性变换等简单模型）
      - 我们在知道 ground truth 的情况下，计算误差指标来衡量“最多可以还原到什么程度”

    这里为了简单，先从最乐观情况出发：
      - 假设攻击者直接用脱敏列作为原始列的估计（best effort baseline）
      - 即 original_hat = deid_value
    """
    print(f"\n=== [{table_name}] 数值重构攻击评估 ===")

    for col in numeric_cols:
        if col not in df_orig.columns or col not in df_deid.columns:
            print(f"  - 跳过列 {col}（在原始或脱敏表中缺失）")
            continue

        orig = pd.to_numeric(df_orig[col], errors="coerce")
        deid = pd.to_numeric(df_deid[col], errors="coerce")

        mask = orig.notna() & deid.notna()
        if mask.sum() == 0:
            print(f"  - 列 {col}: 无可比对样本，跳过。")
            continue

        o = orig[mask].to_numpy()
        d = deid[mask].to_numpy()

        mae = float(np.mean(np.abs(o - d)))
        rmse = float(np.sqrt(np.mean((o - d) ** 2)))
        corr = float(np.corrcoef(o, d)[0, 1]) if len(o) > 1 else float("nan")
        exact_ratio = float(np.mean(np.isclose(o, d, atol=1e-8)))

        print(f"  - 列 {col}:")
        print(f"      样本数           = {len(o)}")
        print(f"      MAE              = {mae:.4f}")
        print(f"      RMSE             = {rmse:.4f}")
        print(f"      Pearson 相关系数 = {corr:.4f}")
        print(f"      完全相等比例     = {exact_ratio * 100:.2f}%")

        print("      说明：")
        print("        * 完全相等比例低 -> 精确还原原始值的可能性极低；")
        print("        * MAE/RMSE 越大 -> 数值被明显扰动或聚合，攻击者只能做粗糙估计；")
        print("        * 相关系数高    -> 整体分布形状还在（利于下游建模），但个体不可精确识别。")


# ---------- 文本 PHI 扫描攻击（regex search）工具函数 ----------

PHI_PATTERNS: Dict[str, str] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b\d{3}[- ]\d{3}[- ]\d{4}\b",
    "date_yyyy_mm_dd": r"\b\d{4}-\d{2}-\d{2}\b",
    "url": r"https?://\S+",
    "long_number(>=8)": r"\b\d{8,}\b",
}

# synthetic 里我们用过的一些具体名字，用来检测是否被替换掉
SYNTHETIC_NAMES: List[str] = [
    "John Smith",
    "Mary Johnson",
    "Alex Lee",
    "Alex Li",
    "Jordan Brown",
    "Chris Brown",
    "Dr. Miller",
    "Dr. Adams",
    "Dr. Wang",
    "Dr. Garcia",
    "MGH",
    "BWH",
    "Stanford Hospital",
    "NYU Langone",
]


def count_pattern(series: pd.Series, pattern: str) -> int:
    text = "\n".join(map(lambda x: "" if pd.isna(x) else str(x), series))
    return len(re.findall(pattern, text))


def count_substring(series: pd.Series, sub: str) -> int:
    text = "\n".join(map(lambda x: "" if pd.isna(x) else str(x), series))
    return text.count(sub)


def evaluate_text_phi_leakage(
    df_orig: pd.DataFrame,
    df_deid: pd.DataFrame,
    text_col: str,
    table_name: str,
) -> None:
    """
    模拟“文本 PHI 扫描攻击”：
      - 攻击者对整列文本做正则搜索，看看还能不能翻到明显的 PHI 片段。
      - 我们比较脱敏前 vs 脱敏后每种模式命中的数量。
    """
    if text_col not in df_orig.columns or text_col not in df_deid.columns:
        print(f"\n=== [{table_name}] 文本 PHI 扫描攻击: 列 {text_col} 不存在，跳过 ===")
        return

    print(f"\n=== [{table_name}] 文本 PHI 扫描攻击 (列: {text_col}) ===")

    s_orig = df_orig[text_col]
    s_deid = df_deid[text_col]

    print("\n  [基于正则的 PHI 模式扫描]")
    for name, pattern in PHI_PATTERNS.items():
        c_orig = count_pattern(s_orig, pattern)
        c_deid = count_pattern(s_deid, pattern)
        print(f"  - 模式 {name}:")
        print(f"      脱敏前命中次数 = {c_orig}")
        print(f"      脱敏后命中次数 = {c_deid}")

    print("\n  [synthetic 中注入的具体姓名 / 机构 是否还存在]")
    for name in SYNTHETIC_NAMES:
        c_orig = count_substring(s_orig, name)
        c_deid = count_substring(s_deid, name)
        if c_orig == 0 and c_deid == 0:
            # 有可能该名字没被采样到
            continue
        print(f"  - 关键字 \"{name}\":")
        print(f"      脱敏前出现次数 = {c_orig}")
        print(f"      脱敏后出现次数 = {c_deid}")
        if c_deid == 0 and c_orig > 0:
            print("      -> 已被成功去标识化（替换为 surrogate 或移除）。")


# ---------- CLI & 主流程 ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic adversarial attacks demo for de-identified EHR tables."
    )
    parser.add_argument(
        "--privacy-level",
        type=str,
        default="strong",
        choices=["light", "medium", "strong"],
        help="隐私强度：light / medium / strong。",
    )
    parser.add_argument(
        "--n-profile",
        type=int,
        default=30,
        help="patient_profile 行数（默认 30）。",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="timeline_events 中 subject 数量（默认 3）。",
    )
    parser.add_argument(
        "--events-per-subject",
        type=int,
        default=4,
        help="timeline_events 每个 subject 的事件数（默认 4）。",
    )
    parser.add_argument(
        "--n-notes",
        type=int,
        default=8,
        help="notes 条数（默认 8）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    privacy_level = args.privacy_level

    # ---------- 构造 synthetic 原始数据 ----------
    df_profile_orig = make_demo_patient_profile(n=max(10, args.n_profile), seed=0)
    df_timeline_orig = make_demo_timeline_events(
        n_subjects=max(1, args.n_subjects),
        events_per_subject=max(1, args.events_per_subject),
        seed=0,
    )
    df_notes_orig = make_demo_notes(n_notes=max(4, args.n_notes), seed=0)

    # ---------- 构建 Agent & 跑脱敏流水线 ----------
    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    df_profile_deid = deidentify_patient_profile_df(
        df_profile_orig.copy(), agent, privacy_level, PATIENT_PROFILE_CONFIG
    )
    df_timeline_deid = deidentify_timeline_df(
        df_timeline_orig.copy(), agent, privacy_level, TIMELINE_CONFIG
    )
    df_notes_deid = deidentify_notes_df(
        df_notes_orig.copy(), agent, privacy_level, NOTES_CONFIG
    )

    print(
        f"\n=== [对抗攻击 demo] privacy_level={privacy_level} ===\n"
        "本脚本会：\n"
        "  1) 在 synthetic 数据上跑一遍脱敏流水线；\n"
        "  2) 模拟数值重构攻击（看还原精度）；\n"
        "  3) 模拟文本 PHI 扫描攻击（看常见 PHI 是否被去掉）。\n"
    )

    # ---------- 数值重构攻击评估 ----------
    evaluate_numeric_reconstruction(
        df_profile_orig,
        df_profile_deid,
        numeric_cols=["anchor_age", "bmi"],
        table_name="patient_profile",
    )

    evaluate_numeric_reconstruction(
        df_timeline_orig,
        df_timeline_deid,
        numeric_cols=["valuenum"],
        table_name="timeline_events",
    )

    # ---------- 文本 PHI 扫描攻击 ----------
    evaluate_text_phi_leakage(
        df_timeline_orig,
        df_timeline_deid,
        text_col="text",
        table_name="timeline_events",
    )

    evaluate_text_phi_leakage(
        df_notes_orig,
        df_notes_deid,
        text_col="text",
        table_name="notes",
    )

    # ---------- 简单结论 ----------
    print("\n=== 总结（示意） ===")
    print("1) 对数值列：")
    print("   - 如果完全相等比例接近 0%，说明单条记录的精确原始值基本无法被还原；")
    print("   - MAE / RMSE 不为 0，说明算子确实对数值做了扰动、聚合或合成；")
    print("   - 若相关系数仍然较高，说明整体分布形状保留，有利于下游建模。")
    print("\n2) 对文本列：")
    print("   - 如果 email / 电话 / URL / 长数字 / 日期 在脱敏后基本检测不到，说明显性 PHI 已被清除；")
    print("   - synthetic 中注入的具体姓名/医院在脱敏后消失，说明替换/去标识化生效。")
    print("\n3) 该脚本只是一个“模拟攻击 + sanity check”：")
    print("   - 证明了在当前流水线下，常见的简单攻击难以直接还原个人层面的原始信息；")
    print("   - 真正严格的理论保证（如差分隐私）需要结合具体算子的数学定义和参数来分析。")


if __name__ == "__main__":
    main()