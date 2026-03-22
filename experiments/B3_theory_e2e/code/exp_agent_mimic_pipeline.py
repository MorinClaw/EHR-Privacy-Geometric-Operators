#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_agent_mimic_pipeline.py — B.3 入口脚本（与需求文档中的命名一致）

本脚本为 B.3「理论约束在端到端 pipeline 中的数值验证」的入口：
- 在 timeline_events 的 numeric 列上调用 Agent 的 numeric pipeline（strong: T1+T2+T3），做 A.3 sanity；
- 验证 ID/时间/文本算子；
- 输出 table_agent_sanity_numeric.csv、table_agent_text_phi_leakage.csv 及 De-identified rows 样例。

实现位于同目录 exp_b3_theory_constraints.py，此处仅转发调用（默认使用data_preparation内数据）。

用法（与 exp_b3_theory_constraints.py 相同）：
  PYTHONPATH=agent_demo python3 exp_agent_mimic_pipeline.py --out-dir ../results
  PYTHONPATH=agent_demo python3 exp_agent_mimic_pipeline.py --max-rows 50000 --out-dir ../results
"""

from __future__ import annotations

import sys
from pathlib import Path

# 同目录
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from exp_b3_theory_constraints import main

if __name__ == "__main__":
    main()
