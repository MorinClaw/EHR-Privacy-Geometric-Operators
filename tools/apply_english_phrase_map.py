#!/usr/bin/env python3
"""Apply phrase-level Chinese -> English replacements. Run: python tools/apply_english_phrase_map.py"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = {".py", ".md", ".html", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv"}

PHRASES: list[tuple[str, str]] = [
    ("请设置 PYTHONPATH 含 agent_demo", "add agent_demo to PYTHONPATH"),
    ("无法 import skills_and_agent", "cannot import skills_and_agent"),
    ("跳过扰动分布统计", "skip perturbation statistics"),
    ("跳过回归重建", "skip reconstruction attack"),
    ("跳过无配对攻击", "skip no-pairs attack"),
    ("跳过多次扰动差异", "skip multi-run variability"),
    ("多次扰动运行次数", "number of multi-run repetitions"),
    ("数据目录不存在", "data directory does not exist"),
    ("扰动目录不存在", "perturbed directory does not exist"),
    ("请使用 --data-dir。", "Use --data-dir."),
    ("请指定 --input。", "Specify --input."),
    ("输入不存在", "input does not exist"),
    ("错误：", "Error: "),
    ("未找到:", "Not found:"),
    ("仅绘图完成。", "Figures-only pass complete."),
    ("完成。", "Done."),
    ("数据量 n≈", "sample size n≈"),
    ("跳过 0.01% 档", "skipping 0.01% regime"),
    ("仅保留 20% 档", "keeping 20% regime only"),
    ("已用固定", "using fixed"),
    ("归一化空间", "normalized space"),
    ("增强抗重建", "stronger reconstruction resistance"),
]

HAN = re.compile(r"[\u4e00-\u9fff]")


def main() -> None:
    phrases = sorted(PHRASES, key=lambda x: -len(x[0]))
    n = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in EXT:
            continue
        if any(p in path.parts for p in SKIP_DIRS):
            continue
        if path.name == "apply_english_phrase_map.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not HAN.search(text):
            continue
        new = text
        for zh, en in phrases:
            new = new.replace(zh, en)
        if new != text:
            path.write_text(new, encoding="utf-8")
            n += 1
            print(path.relative_to(ROOT))
    print("Files modified:", n)


if __name__ == "__main__":
    main()
