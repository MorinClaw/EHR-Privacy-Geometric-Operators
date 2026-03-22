#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
non_numeric_operators.py

非数值型算子（ID / 时间 / 文本 / KG / DP 等）的简化实现。

说明：
- 这些实现重点是“可运行 + 行为合理 + 接口统一”，并非追求完美的工业级效果；
- 大多数函数既支持 Python list，也支持 numpy 数组。
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Iterable, List, Sequence

import numpy as np


# ============================================================
# ID & 人口学相关算子
# ============================================================


def hash_ids(
    values: Sequence[object],
    secret: str = "demo_salt",
    length: int = 12,
) -> np.ndarray:
    """
    T_ID-hash：对 ID 字段做不可逆哈希，支持跨表关联但不可回推。

    - 使用 HMAC-SHA256，secret 只在院内保存；
    - 返回十六进制哈希前 length 位作为伪 ID。
    """
    key = str(secret).encode("utf-8")
    out: List[str] = []
    for v in values:
        b = str(v).encode("utf-8")
        digest = hmac.new(key, b, hashlib.sha256).hexdigest()
        out.append(digest[:length])
    return np.array(out, dtype=object)


def demo_binning(
    values: Sequence[object],
    mode: str = "age",
) -> np.ndarray:
    """
    T_demo-bin：人口学粗分箱。

    当前实现：
    - mode="age": 假定是年龄，按 [0,4], [5,9], ... [85,89], [90,+) 分箱；
    - 其它 mode 时，直接返回原值（可按需扩展）。
    """
    arr = np.asarray(values, dtype=float)

    if mode != "age":
        return arr  # 简化：其它模式暂时不处理

    # 年龄分箱
    bins = np.array(list(range(0, 90, 5)) + [200], dtype=float)
    labels = []
    for i in range(len(bins) - 2):
        labels.append(f"[{int(bins[i])},{int(bins[i+1]-1)}]")
    labels.append(">=90")

    idx = np.digitize(arr, bins, right=False) - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    out = np.array([labels[i] for i in idx], dtype=object)
    return out


def microaggregation_1d(
    values: Sequence[object],
    k: int = 10,
) -> np.ndarray:
    """
    T_micro：一维微聚合示例实现。

    - 对一列数值排序，按 k 个一组分段；
    - 每组内用组均值替换所有元素；
    - 保证每个输出值至少对应 k 条记录（除最后一组）。
    """
    x = np.asarray(values, dtype=float).copy()
    n = len(x)
    order = np.argsort(x)

    for start in range(0, n, k):
        idx = order[start : start + k]
        mean = float(x[idx].mean())
        x[idx] = mean

    return x


def rare_category_agg(
    values: Sequence[object],
    min_freq: int = 10,
    other_label: str = "OTHER",
    keep_list: Sequence[object] | None = None,
) -> np.ndarray:
    """
    T_cat-agg：将低频类别合并为 OTHER，主干类别可强制保留。

    - values 可以是字符串或整数；
    - 频数 < min_freq 的类别替换为 other_label；
    - 若提供 keep_list，则其中的类别无论频数多少均保留，用于主干族群、主要保险类型等，
      保证分层统计与公平性分析可行，仅极罕见组合被合并以降低重识别风险。
    """
    arr = np.asarray(values, dtype=object)
    keep_set = set(keep_list) if keep_list is not None else None
    uniq, counts = np.unique(arr, return_counts=True)
    rare = set()
    for i in range(len(uniq)):
        if counts[i] < min_freq:
            if keep_set is None or uniq[i] not in keep_set:
                rare.add(uniq[i])
    out = np.array(
        [other_label if v in rare else v for v in arr],
        dtype=object,
    )
    return out


# ============================================================
# 时间相关算子
# ============================================================


def relative_time_1d(
    values: Sequence[object],
    index_time: object | None = None,
    unit: str = "D",
) -> np.ndarray:
    """
    T_time-rel：把绝对时间戳转换为相对 index_time 的偏移量。

    参数:
        values: 可解析为 numpy.datetime64 的字符串或 datetime 对象。
        index_time: 参考时间；若为 None，则取该列最小时间。
        unit: "D"（天）或 "h"（小时）。
    """
    times = np.array(values, dtype="datetime64[ns]")
    if index_time is None:
        ref = times.min()
    else:
        ref = np.datetime64(index_time)

    delta = times - ref
    if unit == "h":
        rel = delta.astype("timedelta64[h]").astype(float)
    else:
        # 默认按天
        rel = delta.astype("timedelta64[D]").astype(float)
    return rel


def time_shift_1d(
    values: Sequence[object],
    max_shift_days: int = 180,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    T_time-shift：在已是“相对时间”的前提下，为整列数据增加一个随机整数偏移。

    - 这里假定 values 已是“天”单位的相对时间；
    - 在真实系统中应该按患者级别分别平移，这里为一维简化版本。
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(values, dtype=float)
    delta = rng.integers(-max_shift_days, max_shift_days + 1)
    return arr + float(delta)


# ============================================================
# 文本相关算子
# ============================================================

# 基本正则
_RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_RE_PHONE = re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b")
_RE_URL = re.compile(r"https?://\S+")
_RE_IDLONG = re.compile(r"\b\d{8,}\b")
_RE_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)

# 粗糙的人名模式: "John Smith", "Mary A. Doe" 等
_RE_NAME = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)


def text_mask_basic(texts: Sequence[str]) -> List[str]:
    """
    T_mask：强规则掩码，用于兜底显性 PHI。
    """
    out: List[str] = []
    for t in texts:
        if t is None:
            out.append("")
            continue
        s = str(t)
        s = _RE_EMAIL.sub("[EMAIL]", s)
        s = _RE_PHONE.sub("[PHONE]", s)
        s = _RE_URL.sub("[URL]", s)
        s = _RE_IDLONG.sub("[ID]", s)
        s = _RE_DATE.sub("[DATE]", s)
        out.append(s)
    return out


def text_phi_surrogate(
    texts: Sequence[str],
    seed: int = 0,
) -> List[str]:
    """
    T_PHI-sur：非常简化版的“PHI 替身替换”。

    - 使用正则大致识别人名、日期、长数字等实体；
    - 把人名替换为伪造名字（保持同一名字在同一 batch 内一致）；
    - 日期 / ID 等替换为占位符 [DATE] / [ID] 等。
    """
    rng = np.random.default_rng(seed)

    first_names = [
        "Alex", "Chris", "Taylor", "Jordan", "Morgan",
        "Jamie", "Casey", "Riley", "Avery", "Sam",
    ]
    last_names = [
        "Brown", "Smith", "Lee", "Garcia", "Kim",
        "Patel", "Nguyen", "Wang", "Lopez", "Khan",
    ]
    name_map: dict[str, str] = {}

    def replace_name(match: re.Match) -> str:
        src = match.group(0)
        if src not in name_map:
            f = rng.choice(first_names)
            l = rng.choice(last_names)
            name_map[src] = f"{f} {l}"
        return name_map[src]

    out: List[str] = []
    for txt in texts:
        if txt is None:
            out.append("")
            continue
        s = str(txt)

        # 人名替身
        s = _RE_NAME.sub(replace_name, s)

        # 其它显性标识符
        s = _RE_EMAIL.sub("[EMAIL]", s)
        s = _RE_PHONE.sub("[PHONE]", s)
        s = _RE_URL.sub("[URL]", s)
        s = _RE_IDLONG.sub("[ID]", s)
        s = _RE_DATE.sub("[DATE]", s)

        out.append(s)
    return out


# ============================================================
# DataSifter 风格的简化版表格 / 文本合成
# ============================================================


def datasifter_tab_1d(
    values: Sequence[object],
    synth_prob: float = 0.2,
    noise_scale: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    T_DS-tab：非常简化的一维“部分合成”：

    - 以概率 synth_prob 选取一部分单元；
    - 对这些单元加 N(0, noise_scale^2) 噪声，并在选中位置内做一次 shuffle；
    - 其余单元保持不变。
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(values, dtype=float).copy()
    n = len(x)
    mask = rng.random(n) < synth_prob
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return x

    x[idx] += rng.normal(0.0, noise_scale, size=len(idx))
    rng.shuffle(x[idx])
    return x


def datasifter_text(
    texts: Sequence[str],
    mask_fraction: float = 0.15,
    seed: int = 0,
) -> List[str]:
    """
    T_DS-text：简化版“文本部分合成”：

    - 对每条文本，随机挑选约 mask_fraction 的 token；
    - 对这些 token 做简单替换（同义词 / [MASK]），尽量不动 [NAME]/[DATE]/[ID] 等占位符；
    - 目标是“语义大致相同但不与原文 1:1 匹配”。
    """
    rng = np.random.default_rng(seed)

    # 很小的同义词词典（只是为了示意）
    synonym_map = {
        "pain": "discomfort",
        "fever": "high temperature",
        "cough": "coughing",
        "headache": "head pain",
        "doctor": "clinician",
        "patient": "subject",
        "hospital": "facility",
        "tablet": "pill",
        "drug": "medication",
    }

    out: List[str] = []
    for txt in texts:
        if txt is None:
            out.append("")
            continue
        s = str(txt)
        tokens = s.split()
        m = len(tokens)
        if m == 0:
            out.append(s)
            continue

        n_mask = max(1, int(m * mask_fraction))
        idx = rng.choice(m, size=n_mask, replace=False)

        for i in idx:
            tok = tokens[i]
            if tok.startswith("[") and tok.endswith("]"):
                # 不动占位符，例如 [NAME]
                continue
            base = tok.strip(",.;:!?()").lower()
            if base in synonym_map:
                # 同义词替换，保留标点
                new_core = synonym_map[base]
                tokens[i] = tok.replace(base, new_core)
            else:
                # 简单降采样
                tokens[i] = "[MASK]"
        out.append(" ".join(tokens))
    return out


# ============================================================
# KG & 差分隐私聚合
# ============================================================


def kg_struct_identity(records: Sequence[object]) -> list[object]:
    """
    T_KG-struct：这里仅做占位，不修改输入。

    在真实系统里，这一步应该：
      - 解析图结构字段（graph/target/matched_items）；
      - 删除其中可能混入的 patient_id / hadm_id 等；
      - 校验只包含 ATC/UMLS 概念与关系标签。
    """
    return list(records)


def laplace_aggregate_counts(
    counts: Sequence[object],
    epsilon: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    T_Lap-agg：对聚合表计数添加 Laplace 噪声，满足 ε-DP（敏感度=1）。

    参数:
        counts: 聚合后的计数数组；
        epsilon: 隐私预算 ε，>0。
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数。")
    if rng is None:
        rng = np.random.default_rng()

    arr = np.asarray(counts, dtype=float)
    scale = 1.0 / float(epsilon)  # Δf=1
    noise = rng.laplace(0.0, scale, size=arr.shape)
    return arr + noise