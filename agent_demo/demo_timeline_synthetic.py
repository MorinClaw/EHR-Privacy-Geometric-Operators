# filename: demo_timeline_synthetic.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_timeline_synthetic.py

构造一份小的 timeline_events DataFrame，在内存中跑脱敏流水线：
  - 使用 demo_patient_and_timeline 中的 deidentify_timeline_df
  - 使用 skills_and_agent 中的 PrivacyAgent / SkillRegistry

用法示例：
    python demo_timeline_synthetic.py --privacy-level strong
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from skills_and_agent import build_default_registry, PrivacyAgent
from demo_patient_and_timeline import (
    deidentify_timeline_df,
    TIMELINE_CONFIG,
)


def make_demo_timeline_events(
    n_subjects: int = 3,
    events_per_subject: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """
    造一小批 timeline_events 行，字段包括：
      - subject_id
      - hadm_id
      - charttime
      - valuenum
      - text
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    base_time = datetime(2148, 1, 1, 8, 0, 0)

    names = ["John Smith", "Mary Johnson", "Alex Lee", "Chris Brown"]
    hospitals = ["MGH", "BWH", "UCSF", "NYU"]

    for s_idx in range(n_subjects):
        subject_id = 100000 + s_idx
        hadm_id = 200000 + s_idx

        for e_idx in range(events_per_subject):
            # 时间：在入院后 0~72 小时内
            offset_hours = int(rng.integers(0, 72))
            charttime = base_time + timedelta(hours=offset_hours)

            # 模拟一个数值观测，比如血压/实验室指标
            valuenum = float(rng.normal(loc=80.0 + 5 * e_idx, scale=10.0))

            # 文本里放入邮箱/电话/日期/URL/长数字/人名等，方便看脱敏效果
            name = rng.choice(names)
            hosp = rng.choice(hospitals)
            email = f"{name.split()[0].lower()}.{name.split()[-1].lower()}@example.com"
            phone = f"555-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}"
            long_id = f"{rng.integers(10**8, 10**9 - 1)}"
            url = "https://example-hospital.org/visit?id=" + long_id
            date_str = charttime.strftime("%Y-%m-%d")

            text = (
                f"{name} was seen at {hosp} on {date_str}. "
                f"Contact: {email}, phone {phone}. "
                f"Internal ref {long_id}. Notes uploaded to {url}."
            )

            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": charttime.isoformat(),
                    "valuenum": valuenum,
                    "text": text,
                }
            )

    df = pd.DataFrame(rows)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic demo for timeline_events de-identification."
    )
    parser.add_argument(
        "--privacy-level",
        type=str,
        default="strong",
        choices=["light", "medium", "strong"],
        help="隐私强度：light / medium / strong。",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="模拟 subject 数量（默认 3）。",
    )
    parser.add_argument(
        "--events-per-subject",
        type=int,
        default=4,
        help="每个 subject 的事件数（默认 4）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    privacy_level = args.privacy_level
    n_subjects = max(1, args.n_subjects)
    events_per_subject = max(1, args.events_per_subject)

    df_timeline = make_demo_timeline_events(
        n_subjects=n_subjects,
        events_per_subject=events_per_subject,
        seed=0,
    )

    print(
        f"\n=== [Synthetic timeline_events demo] "
        f"subjects={n_subjects}, events/subject={events_per_subject}, "
        f"privacy_level={privacy_level} ==="
    )
    print("\n[timeline_events] 原始前 6 行:")
    print(df_timeline.head(6))

    # 构建 Agent
    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    # 跑脱敏流水线
    df_timeline_deid = deidentify_timeline_df(
        df_timeline, agent, privacy_level, TIMELINE_CONFIG
    )

    print("\n[timeline_events] 脱敏后前 6 行:")
    print(df_timeline_deid.head(6))

    print("\n说明：")
    print("  - subject_id / hadm_id 已通过 T_ID-hash 做不可逆伪 ID；")
    print("  - charttime 先转成相对时间（距最早时间的天数），再整体随机平移；")
    print("  - text 中的邮箱/电话/URL/长数字/日期会被 text_mask 替换为占位符；")
    print("  - 人名等会被 text_phi_surr 替换为 surrogate 名字；")
    print("  - valuenum 在 strong 模式下会通过 ds_tab 做部分合成（加噪声并 shuffle 一部分单元）。")


if __name__ == "__main__":
    main()