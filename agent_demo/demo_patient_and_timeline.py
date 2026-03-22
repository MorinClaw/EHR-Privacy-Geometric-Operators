# filename: demo_patient_and_timeline.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_patient_and_timeline.py

基于已有的 SkillRegistry + 各种算子，给出一个针对
  - patient_profile（病人画像表）
  - timeline_events（时间线事件表）
  - notes（自由文本病程记录表，可选）
的 end-to-end 脱敏示例。

输入:  包含 2～3 张表的 JSON 文件，例如 mimic_sample.json
输出:  脱敏后的 JSON 文件，例如 mimic_sample_deid.json

使用方式:
    python demo_patient_and_timeline.py \
        --input mimic_sample.json \
        --output mimic_sample_deid.json \
        --privacy-level strong
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from skills_and_agent import build_default_registry, PrivacyAgent


# ---------------------------------------------------------------------
# 配置：告诉 runner 哪些列是什么含义（你可以按自己的 schema 修改）
# ---------------------------------------------------------------------

# 主干类别：始终保留，不合并为 OTHER，以支持分层统计与公平性分析
# 族裔仅保留主要类型（WHITE/BLACK/ASIAN）；HISPANIC 等少数按频数处理，罕见则合并为 OTHER
CAT_KEEP_MAIN_ETHNICITY = ["WHITE", "BLACK", "ASIAN"]
CAT_KEEP_MAIN_INSURANCE = ["Medicare", "Medicaid", "Private", "Government"]
CAT_KEEP_GENDER = ["M", "F"]

PATIENT_PROFILE_CONFIG: Dict[str, Any] = {
    # 需要做 ID 哈希的列
    "id_columns": ["subject_id"],
    # 年龄列（用于 demo_binning）
    "age_column": "anchor_age",
    # 其它数值型 quasi-identifier（用于 microaggregation_1d）
    "numeric_quasiid_columns": ["bmi"],
    # 类别列及各自的主干保留列表（仅极罕见类别合并为 OTHER）
    "cat_columns": ["gender", "ethnicity", "insurance"],
    "cat_keep_lists": {
        "gender": CAT_KEEP_GENDER,
        "ethnicity": CAT_KEEP_MAIN_ETHNICITY,
        "insurance": CAT_KEEP_MAIN_INSURANCE,
    },
}

TIMELINE_CONFIG: Dict[str, Any] = {
    # 需要做 ID 哈希的列
    "id_columns": ["subject_id", "hadm_id"],
    # 时间列（绝对时间戳）
    "time_column": "charttime",
    # 数值观测列（用于 DS-tab 部分合成）
    "numeric_columns": ["valuenum"],
    # 文本列（病程记录片段 / 事件描述等）
    "text_columns": ["text"],
}

NOTES_CONFIG: Dict[str, Any] = {
    # notes 表里的 ID 列
    "id_columns": ["subject_id", "hadm_id", "note_id"],
    # notes 表的文本列
    "text_columns": ["text"],
}


# ---------------------------------------------------------------------
# patient_profile runner
# ---------------------------------------------------------------------


def deidentify_patient_profile_df(
    df: pd.DataFrame,
    agent: PrivacyAgent,
    privacy_level: str,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    对 patient_profile 表做脱敏。

    将利用 agent.plan_pipeline("patient_profile", privacy_level)
    规划出来的算子序列，并按列含义做映射：
        - id_hash:  对 subject_id 等 ID 列做哈希
        - demo_bin: 对年龄列做粗分箱
        - microagg: 对配置的 numeric_quasiid_columns 做一维微聚合
        - cat_agg:  对类别列合并低频类别
    """
    if config is None:
        config = PATIENT_PROFILE_CONFIG

    out = df.copy()
    registry = agent.registry
    pipeline = agent.plan_pipeline("patient_profile", privacy_level)

    print(f"[patient_profile] pipeline = {pipeline}")

    id_cols: List[str] = config.get("id_columns", [])
    age_col: str | None = config.get("age_column")
    numeric_quasiid_cols: List[str] = config.get("numeric_quasiid_columns", [])
    cat_cols: List[str] = config.get("cat_columns", [])

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                print(f"  - id_hash on column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"secret": "demo_salt", "length": 12},
                )

        elif sid == "demo_bin":
            if age_col and age_col in out.columns:
                print(f"  - demo_bin (age) on column: {age_col}")
                skill = registry.get("demo_bin")
                out[age_col] = skill.apply(
                    out[age_col].to_numpy(),
                    {"mode": "age"},
                )

        elif sid == "microagg":
            skill = registry.get("microagg")
            for col in numeric_quasiid_cols:
                if col not in out.columns:
                    continue
                print(f"  - microagg on numeric quasi-id column: {col}")
                out[col] = skill.apply(
                    out[col].to_numpy(),
                    {"k": 10},
                )

        elif sid == "cat_agg":
            skill = registry.get("cat_agg")
            n_rows = len(out)
            min_freq = 1 if n_rows < 10 else 5
            cat_keep_lists: Dict[str, list] = config.get("cat_keep_lists") or {}
            for col in cat_cols:
                if col not in out.columns:
                    continue
                keep_list = cat_keep_lists.get(col)
                print(f"  - cat_agg on categorical column: {col} (min_freq={min_freq}, keep_list={keep_list})")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"min_freq": min_freq, "other_label": "OTHER", "keep_list": keep_list},
                )

        else:
            # 如果以后增加别的 skill，可以在这里补逻辑
            print(f"  - [patient_profile] 忽略未识别的 skill: {sid}")

    return out


# ---------------------------------------------------------------------
# timeline_events runner
# ---------------------------------------------------------------------


def deidentify_timeline_df(
    df: pd.DataFrame,
    agent: PrivacyAgent,
    privacy_level: str,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    对 timeline_events 表做脱敏。

    对应 plan_pipeline("timeline", privacy_level) 的默认规划：
      - light : ["id_hash", "time_rel"]
      - medium: ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr"]
      - strong: ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr", "ds_tab"]

    映射到列：
      - id_hash:       对 subject_id / hadm_id 做哈希
      - time_rel:      将 charttime 绝对时间转成“距最早时间的日数”
      - time_shift:    在相对时间基础上整体平移一个随机天数
      - text_mask:     规则掩码，兜底显性 PHI（邮箱/电话/长数字/日期/URL）
      - text_phi_surr: 人名等做 surrogate 替换，日期/ID 等用占位符
      - ds_tab:        对数值观测列做 DataSifter 风格部分合成
    """
    if config is None:
        config = TIMELINE_CONFIG

    out = df.copy()
    registry = agent.registry
    pipeline = agent.plan_pipeline("timeline", privacy_level)

    print(f"[timeline_events] pipeline = {pipeline}")

    id_cols: List[str] = config.get("id_columns", [])
    time_col: str | None = config.get("time_column")
    num_cols: List[str] = config.get("numeric_columns", [])
    text_cols: List[str] = config.get("text_columns", [])

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                print(f"  - id_hash on column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"secret": "demo_salt", "length": 12},
                )

        elif sid == "time_rel":
            if time_col and time_col in out.columns:
                print(f"  - time_rel on column: {time_col}")
                skill = registry.get("time_rel")
                # 先转成 pandas datetime，再转 numpy datetime64
                times = pd.to_datetime(out[time_col], errors="coerce")
                out[time_col] = skill.apply(
                    times.to_numpy(),
                    {"index_time": None, "unit": "D"},
                )

        elif sid == "time_shift":
            if time_col and time_col in out.columns:
                print(f"  - time_shift on column: {time_col}")
                skill = registry.get("time_shift")
                out[time_col] = skill.apply(
                    out[time_col].to_numpy(),
                    {"max_shift_days": 365},
                )

        elif sid == "text_mask":
            skill = registry.get("text_mask")
            for col in text_cols:
                if col not in out.columns:
                    continue
                print(f"  - text_mask on text column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {},
                )

        elif sid == "text_phi_surr":
            skill = registry.get("text_phi_surr")
            for col in text_cols:
                if col not in out.columns:
                    continue
                print(f"  - text_phi_surr on text column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"seed": 0},
                )

        elif sid == "ds_tab":
            skill = registry.get("ds_tab")
            for col in num_cols:
                if col not in out.columns:
                    continue
                print(f"  - ds_tab on numeric column: {col}")
                out[col] = skill.apply(
                    out[col].to_numpy(),
                    {"synth_prob": 0.3, "noise_scale": 0.5},
                )

        else:
            print(f"  - [timeline_events] 忽略未识别的 skill: {sid}")

    return out


# ---------------------------------------------------------------------
# notes runner（新增）
# ---------------------------------------------------------------------


def deidentify_notes_df(
    df: pd.DataFrame,
    agent: PrivacyAgent,
    privacy_level: str,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    对 notes（自由文本）表做脱敏。

    对应 plan_pipeline("notes", privacy_level) 的默认规划：
      - light : ["id_hash"]
      - medium: ["id_hash", "text_mask"]
      - strong: ["id_hash", "text_mask", "text_phi_surr"]

    映射到列：
      - id_hash:       对 subject_id / hadm_id / note_id 做哈希
      - text_mask:     掩码邮箱/电话/长数字/URL/日期等显性 PHI
      - text_phi_surr: 人名/机构/地址等做 surrogate 替换
    """
    if config is None:
        config = NOTES_CONFIG

    out = df.copy()
    registry = agent.registry
    pipeline = agent.plan_pipeline("notes", privacy_level)

    print(f"[notes] pipeline = {pipeline}")

    id_cols: List[str] = config.get("id_columns", [])
    text_cols: List[str] = config.get("text_columns", [])

    for sid in pipeline:
        if sid == "id_hash":
            skill = registry.get("id_hash")
            for col in id_cols:
                if col not in out.columns:
                    continue
                print(f"  - id_hash on column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"secret": "demo_salt", "length": 12},
                )

        elif sid == "text_mask":
            skill = registry.get("text_mask")
            for col in text_cols:
                if col not in out.columns:
                    continue
                print(f"  - text_mask on text column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {},
                )

        elif sid == "text_phi_surr":
            skill = registry.get("text_phi_surr")
            for col in text_cols:
                if col not in out.columns:
                    continue
                print(f"  - text_phi_surr on text column: {col}")
                out[col] = skill.apply(
                    out[col].astype(str).tolist(),
                    {"seed": 0},
                )

        else:
            print(f"  - [notes] 忽略未识别的 skill: {sid}")

    return out


# ---------------------------------------------------------------------
# CLI & 主流程
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run privacy pipeline on MIMIC-style patient_profile + timeline_events (+ notes) JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="mimic_sample.json",
        help="输入 JSON 文件路径（包含 patient_profile / timeline_events / 可选 notes）。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mimic_sample_deid.json",
        help="输出脱敏后 JSON 文件路径。",
    )
    parser.add_argument(
        "--privacy-level",
        type=str,
        default="strong",
        choices=["light", "medium", "strong"],
        help="隐私强度：light / medium / strong。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    privacy_level = args.privacy_level

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 构建 SkillRegistry + Agent
    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    # 读取 JSON
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out_data: Dict[str, Any] = {}

    # ---------------- patient_profile ----------------
    if "patient_profile" in data:
        df_profile = pd.DataFrame(data["patient_profile"])
        print("\n[patient_profile] 原始前 3 行:")
        print(df_profile.head(3))

        df_profile_deid = deidentify_patient_profile_df(
            df_profile, agent, privacy_level, PATIENT_PROFILE_CONFIG
        )

        print("\n[patient_profile] 脱敏后前 3 行:")
        print(df_profile_deid.head(3))

        out_data["patient_profile"] = df_profile_deid.to_dict(orient="records")
    else:
        print("\n[patient_profile] 未在 JSON 中找到该表，跳过。")

    # ---------------- timeline_events ----------------
    if "timeline_events" in data:
        df_timeline = pd.DataFrame(data["timeline_events"])
        print("\n[timeline_events] 原始前 3 行:")
        print(df_timeline.head(3))

        df_timeline_deid = deidentify_timeline_df(
            df_timeline, agent, privacy_level, TIMELINE_CONFIG
        )

        print("\n[timeline_events] 脱敏后前 3 行:")
        print(df_timeline_deid.head(3))

        out_data["timeline_events"] = df_timeline_deid.to_dict(orient="records")
    else:
        print("\n[timeline_events] 未在 JSON 中找到该表，跳过。")

    # ---------------- notes（可选） ----------------
    if "notes" in data:
        df_notes = pd.DataFrame(data["notes"])
        print("\n[notes] 原始前 3 行:")
        print(df_notes.head(3))

        df_notes_deid = deidentify_notes_df(
            df_notes, agent, privacy_level, NOTES_CONFIG
        )

        print("\n[notes] 脱敏后前 3 行:")
        print(df_notes_deid.head(3))

        out_data["notes"] = df_notes_deid.to_dict(orient="records")
    else:
        print("\n[notes] 未在 JSON 中找到该表，跳过。")

    # 写回输出 JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n已写出脱敏后的数据到: {output_path.resolve()}")


if __name__ == "__main__":
    main()