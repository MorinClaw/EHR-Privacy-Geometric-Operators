# filename: demo_notes_synthetic.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_notes_synthetic.py

构造一份小的 notes DataFrame，在内存中跑脱敏流水线：
  - 使用 demo_patient_and_timeline 中的 deidentify_notes_df
  - 使用 skills_and_agent 中的 PrivacyAgent / SkillRegistry

用法示例：
    python demo_notes_synthetic.py --privacy-level strong
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from skills_and_agent import build_default_registry, PrivacyAgent
from demo_patient_and_timeline import (
    deidentify_notes_df,
    NOTES_CONFIG,
)


def make_demo_notes(
    n_notes: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """
    构造一小批 notes 行：
      - subject_id
      - hadm_id
      - note_id
      - text（包含人名/医院/地址/电话/邮箱/日期等）
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    names = ["John Smith", "Mary Johnson", "Alex Li", "Jordan Brown"]
    doctors = ["Dr. Miller", "Dr. Adams", "Dr. Wang", "Dr. Garcia"]
    hospitals = ["MGH", "BWH", "Stanford Hospital", "NYU Langone"]

    for i in range(n_notes):
        subject_id = 100000 + rng.integers(0, 5)
        hadm_id = 200000 + rng.integers(0, 10)
        note_id = 300000 + i

        patient_name = rng.choice(names)
        doctor_name = rng.choice(doctors)
        hospital = rng.choice(hospitals)
        email = f"{patient_name.split()[0].lower()}@example.com"
        phone = f"555-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}"
        date = f"2148-{rng.integers(1,12):02d}-{rng.integers(1,28):02d}"
        long_id = str(rng.integers(10**8, 10**9 - 1))

        text = (
            f"{patient_name} visited {hospital} on {date} and was seen by {doctor_name}. "
            f"Address: 123 Main St, Boston. Phone {phone}, email {email}. "
            f"Internal ref {long_id}. "
            f'\"Patient reports headache and fever for 3 days.\"'
        )

        rows.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "note_id": note_id,
                "text": text,
            }
        )

    df = pd.DataFrame(rows)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic demo for notes de-identification."
    )
    parser.add_argument(
        "--privacy-level",
        type=str,
        default="strong",
        choices=["light", "medium", "strong"],
        help="隐私强度：light / medium / strong。",
    )
    parser.add_argument(
        "--n-notes",
        type=int,
        default=6,
        help="模拟 notes 条数（默认 6）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    privacy_level = args.privacy_level
    n_notes = max(2, args.n_notes)

    df_notes = make_demo_notes(n_notes=n_notes, seed=0)

    print(
        f"\n=== [Synthetic notes demo] n_notes={n_notes}, "
        f"privacy_level={privacy_level} ==="
    )
    print("\n[notes] 原始前 5 行:")
    print(df_notes.head(5))

    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    df_notes_deid = deidentify_notes_df(
        df_notes, agent, privacy_level, NOTES_CONFIG
    )

    print("\n[notes] 脱敏后前 5 行:")
    print(df_notes_deid.head(5))

    print("\n说明：")
    print("  - subject_id / hadm_id / note_id 已通过 T_ID-hash 做不可逆伪 ID；")
    print("  - text 中的邮箱/电话/长数字/日期等显性 PHI 会被 text_mask 替换为占位符；")
    print("  - 人名（病人/医生/医院）会被 text_phi_surr 替换为 surrogate 名字（strong 模式）。")


if __name__ == "__main__":
    main()