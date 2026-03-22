#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.2 Agent 自动规划逻辑实验

基于 data_type × privacy_level 的 rule-based 流水线规划，枚举并导出当前（或配置的）规划表，
支持用户自选 data_type / privacy_level 查询，以及从配置文件注入自定义规则。

用法（在实验根目录 repository root 下）：
  python3 B2_privacy_utility/code/exp_b2_plan_pipeline.py --out-dir B2_privacy_utility/results
  python3 B2_privacy_utility/code/exp_b2_plan_pipeline.py --data-type numeric --privacy-level strong --out-dir B2_privacy_utility/results
  PYTHONPATH=agent_demo python3 B2_privacy_utility/code/exp_b2_plan_pipeline.py --out-dir B2_privacy_utility/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import find_agent_demo_dir

AGENT_DIR = find_agent_demo_dir(ROOT)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

try:
    from skills_and_agent import (
        build_default_registry,
        PrivacyAgent,
        DEFAULT_PIPELINE_RULES,
        PRIVACY_LEVEL_SEMANTICS,
    )
except ImportError:
    sys.path.insert(0, str(AGENT_DIR))
    try:
        from skills_and_agent import (
            build_default_registry,
            PrivacyAgent,
            DEFAULT_PIPELINE_RULES,
            PRIVACY_LEVEL_SEMANTICS,
        )
    except ImportError:
        build_default_registry = None
        PrivacyAgent = None
        DEFAULT_PIPELINE_RULES = {
            "numeric": {"light": ["num_triplet"], "medium": ["num_triplet", "num_noise_proj"], "strong": ["num_triplet", "num_noise_proj", "num_householder"]},
            "patient_profile": {"light": ["id_hash", "demo_bin"], "medium": ["id_hash", "demo_bin", "cat_agg"], "strong": ["id_hash", "demo_bin", "microagg", "cat_agg"]},
            "timeline": {"light": ["id_hash", "time_rel"], "medium": ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr"], "strong": ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr", "ds_tab"]},
            "notes": {"light": ["id_hash"], "medium": ["id_hash", "text_mask"], "strong": ["id_hash", "text_mask", "text_phi_surr"]},
            "kg": {"light": ["kg_struct"], "medium": ["id_hash", "kg_struct"], "strong": ["id_hash", "time_rel", "kg_struct", "lap_agg"]},
        }
        PRIVACY_LEVEL_SEMANTICS = {"light": "最小必要脱敏，保留最大效用", "medium": "平衡隐私与效用", "strong": "优先隐私"}


def load_pipeline_rules_from_config(config_path: Path | None):
    """从可选配置文件加载 pipeline_rules（与 DEFAULT_PIPELINE_RULES 同结构的 dict）。"""
    if config_path is None or not config_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_rules_config", config_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "PIPELINE_RULES", None)


def run_b2(
    out_dir: Path,
    data_type: str | None = None,
    privacy_level: str | None = None,
    config_path: Path | None = None,
) -> None:
    """
    枚举或单点查询 plan_pipeline，写出表格与语义说明。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_override = load_pipeline_rules_from_config(config_path)
    rules = rules_override if rules_override else DEFAULT_PIPELINE_RULES
    levels = list(PRIVACY_LEVEL_SEMANTICS.keys())
    data_types = list(rules.keys())

    agent = None
    if build_default_registry is not None and PrivacyAgent is not None:
        registry = build_default_registry()
        agent = PrivacyAgent(registry, pipeline_rules_override=rules_override)

    rows: list[dict] = []
    if data_type and privacy_level:
        if agent is not None:
            pipeline = agent.plan_pipeline(data_type, privacy_level)
        else:
            pipeline = rules.get(data_type, {}).get(privacy_level, [])
        rows.append({
            "data_type": data_type,
            "privacy_level": privacy_level,
            "pipeline": " -> ".join(pipeline),
            "skill_ids": pipeline,
        })
        print(f"[B.2] {data_type} / {privacy_level}: {' -> '.join(pipeline)}")
    else:
        for dt in data_types:
            for pl in levels:
                pipeline = rules.get(dt, {}).get(pl, [])
                rows.append({
                    "data_type": dt,
                    "privacy_level": pl,
                    "pipeline": " -> ".join(pipeline),
                    "skill_ids": pipeline,
                })
                print(f"[B.2] {dt} / {pl}: {' -> '.join(pipeline)}")

    # 写出 CSV（pipeline 为字符串；skill_ids 不写 CSV 可另写 JSON）
    import csv
    table_path = out_dir / "table_plan_pipeline.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["data_type", "privacy_level", "pipeline"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["data_type", "privacy_level", "pipeline"]})
    print(f"[B.2] 表: {table_path}")

    # 写出 JSON（含 skill_ids 列表 + 语义）
    report = {
        "privacy_level_semantics": PRIVACY_LEVEL_SEMANTICS,
        "rows": [
            {k: v for k, v in r.items() if k != "skill_ids" or True}
            for r in rows
        ],
        "config_used": str(config_path) if config_path else "default",
    }
    json_path = out_dir / "plan_pipeline_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[B.2] 报告: {json_path}")

    # 用户自选示例说明
    with open(out_dir / "privacy_level_usage.txt", "w", encoding="utf-8") as f:
        f.write("B.2 规划逻辑与 privacy_level 使用说明\n")
        f.write("=====================================\n\n")
        f.write("privacy_level 语义（非单纯堆叠）：\n")
        for pl, desc in PRIVACY_LEVEL_SEMANTICS.items():
            f.write(f"  {pl}: {desc}\n")
        f.write("\n自选方式：\n")
        f.write("  1) 选择 level：调用 plan_pipeline(data_type, privacy_level) 即可。\n")
        f.write("  2) 自定义流水线：plan_pipeline(data_type, privacy_level, pipeline_override=[...])。\n")
        f.write("  3) 自定义规则表：PrivacyAgent(registry, pipeline_rules_override=<dict>)。\n")
    print(f"[B.2] 说明: {out_dir / 'privacy_level_usage.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B.2 Agent 自动规划逻辑：枚举/查询 plan_pipeline")
    parser.add_argument("--data-type", type=str, default=None, help="仅查询该 data_type")
    parser.add_argument("--privacy-level", type=str, default=None, help="仅查询该 privacy_level")
    parser.add_argument("--config", type=str, default=None, help="pipeline 规则配置模块路径（如 pipeline_rules_config.py）")
    parser.add_argument("--out-dir", type=str, default=None, help="输出目录，默认 B2/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR.parent / "results"
    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path

    if (args.data_type is None) != (args.privacy_level is None):
        print("若指定 --data-type 则须同时指定 --privacy-level，否则忽略单边")
        args.data_type = None
        args.privacy_level = None

    run_b2(
        out_dir=out_dir,
        data_type=args.data_type,
        privacy_level=args.privacy_level,
        config_path=config_path,
    )
    print("\nB.2 规划逻辑实验完成。")


if __name__ == "__main__":
    main()
