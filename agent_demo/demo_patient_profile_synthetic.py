# filename: demo_patient_profile_synthetic.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_patient_profile_synthetic.py

构造一份小的 patient_profile DataFrame，在内存中跑脱敏流水线：
  - 使用 demo_patient_and_timeline 中的 deidentify_patient_profile_df
  - 使用 skills_and_agent 中的 PrivacyAgent / SkillRegistry

用法示例：
    python demo_patient_profile_synthetic.py --n 20 --privacy-level strong
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd

from skills_and_agent import build_default_registry, PrivacyAgent
from demo_patient_and_timeline import (
    deidentify_patient_profile_df,
    PATIENT_PROFILE_CONFIG,
)


def make_demo_patient_profile(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    造一小批 patient_profile 行，包含：
      - subject_id
      - anchor_age
      - bmi
      - gender
      - ethnicity
      - insurance
    """
    rng = np.random.default_rng(seed)

    subject_id = np.arange(100001, 100001 + n, dtype=int)
    anchor_age = rng.integers(18, 90, size=n)

    # 模拟 BMI（kg/m^2）：N(27, 4^2)
    bmi = rng.normal(loc=27.0, scale=4.0, size=n)

    genders = np.array(["M", "F"])
    gender = rng.choice(genders, size=n, replace=True)

    # 种族中故意放几个低频类别，方便 cat_agg 合并
    ethnicity_choices = [
        "WHITE",
        "BLACK",
        "ASIAN",
        "HISPANIC",
        "OTHER",
        "NATIVE",
        "UNKNOWN",
    ]
    ethnicity = rng.choice(ethnicity_choices, size=n, replace=True)

    insurance_choices = [
        "Medicare",
        "Medicaid",
        "Private",
        "Self Pay",
        "Government",
        "Unknown",
    ]
    insurance = rng.choice(insurance_choices, size=n, replace=True)

    df = pd.DataFrame(
        {
            "subject_id": subject_id,
            "anchor_age": anchor_age,
            "bmi": bmi,
            "gender": gender,
            "ethnicity": ethnicity,
            "insurance": insurance,
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic demo for patient_profile de-identification."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="模拟 patient_profile 行数（默认 20）。",
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
    n = max(5, args.n)
    privacy_level = args.privacy_level

    # 1) 构造 demo data
    df_profile = make_demo_patient_profile(n=n, seed=0)

    print(f"\n=== [Synthetic patient_profile demo] n={n}, privacy_level={privacy_level} ===")
    print("\n[patient_profile] 原始前 5 行:")
    print(df_profile.head(5))

    # 2) 构建 Agent
    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    # 3) 跑脱敏流水线（复用 demo_patient_and_timeline 里的逻辑）
    df_profile_deid = deidentify_patient_profile_df(
        df_profile, agent, privacy_level, PATIENT_PROFILE_CONFIG
    )

    print("\n[patient_profile] 脱敏后前 5 行:")
    print(df_profile_deid.head(5))

    print("\n说明：")
    print("  - subject_id 已通过 T_ID-hash 做不可逆伪 ID；")
    print("  - anchor_age 若启用 demo_bin 已被粗分箱；")
    print("  - bmi 若启用 microagg 会在一维上做 k=10 的微聚合；")
    print("  - gender / ethnicity / insurance 中低频类别会被 cat_agg 合并为 OTHER（取决于 min_freq 配置）。")


if __name__ == "__main__":
    main()