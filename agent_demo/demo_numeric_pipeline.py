#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_numeric_pipeline.py

命令行 Demo：
  - 使用 build_default_registry 构建 Skill 库
  - 让 PrivacyAgent 在 numeric 数据上根据隐私强度自动规划流水线
  - 在模拟 HR 列上依次运行，并打印每一步的均值 / 方差 / max|Δ|
"""

from __future__ import annotations

import numpy as np

from skills_and_agent import build_default_registry, PrivacyAgent


def main() -> None:
    rng = np.random.default_rng(42)
    n = 300
    # 模拟一列 HR：N(80, 10^2)
    original = rng.normal(loc=80.0, scale=10.0, size=n)

    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    data_type = "numeric"
    privacy_level = "strong"  # 可改为 "light" / "medium" / "strong"

    pipeline = agent.plan_pipeline(data_type=data_type, privacy_level=privacy_level)

    # 为了保证最终相对原始列的 max|Δ| <~ 1，给每一步一个较小约束
    max_diff_per_step = 0.33  # 三步累积约 ≤ 1
    skill_configs = {sid: {"max_diff": max_diff_per_step} for sid in pipeline}

    history = agent.run_numeric_pipeline(original, pipeline, skill_configs=skill_configs)

    print(f"数据类型: {data_type}, 隐私强度: {privacy_level}")
    print("规划得到的流水线:")
    print(
        "  "
        + " -> ".join(
            f"{step['skill_name']}[{step['skill_id']}]" for step in history[1:]
        )
    )

    print("\n每一步统计: (mean, var, max|Δ| 相对于原始列)")
    for h in history:
        sid = h["skill_id"]
        name = h["skill_name"]
        stats = h["stats"]
        if sid == "input":
            print(
                f"[Step {h['step']}] {name:20s}  "
                f"mean={stats['mean']:.4f}, var={stats['var']:.4f}"
            )
        else:
            print(
                f"[Step {h['step']}] {name:20s}  "
                f"mean={stats['mean']:.4f}, var={stats['var']:.4f}, "
                f"max|Δ|={stats['max_abs_delta']:.4f}"
            )


if __name__ == "__main__":
    main()