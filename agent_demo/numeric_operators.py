#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
numeric_operators.py

数值型隐私算子实现：
    1) triplet_micro_rotation        三元组微旋转
    2) constrained_noise_projection  受限噪声投影
    3) householder_reflection        高维 Householder 反射
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ---------- 方案一：先归一化再用固定阈值 ----------
NORMALIZED_MAX_DIFF = 1.0  # 与 A2 一致，调大以增强抗重建


def normalize_zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """z = (x - μ) / σ，返回 (z, μ, σ)。"""
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    if sigma < 1e-10:
        sigma = 1.0
    z = (x - mu) / sigma
    return z, mu, sigma


def denormalize_zscore(z: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """x = z * σ + μ。"""
    return np.asarray(z, dtype=float) * sigma + mu


def _rodrigues_rotate_axis_111(v: np.ndarray, theta: float) -> np.ndarray:
    """绕轴 (1,1,1)/√3 的 Rodrigues 旋转。"""
    axis = np.ones(3, dtype=float) / np.sqrt(3.0)
    v = np.asarray(v, dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return (
        v * cos_t
        + np.cross(axis, v) * sin_t
        + axis * np.dot(axis, v) * (1.0 - cos_t)
    )


def triplet_micro_rotation(
    column: np.ndarray,
    max_diff: float = 1.0,
    n_passes: int = 3,
    theta_init: float = 0.5,
    max_trials: int = 30,
    rng: Optional[np.random.Generator] = None,
    x_original: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    三元组微小旋转算子 (Triplet Micro-Rotation)，均匀分组版本。

    - 每轮对索引做 random permutation，按连续 3 个划三元组；
    - 围绕轴 (1,1,1)/√3 做小角度旋转，保持局部和与范数；
    - 若提供 x_original，则拒绝采样时约束「相对原始列」的 max|Δ| ≤ max_diff（与 PDF 一致）；
      否则约束相对当前轮该三元组的变化（多轮后可能相对原始超界）。

    复杂度: O(n * n_passes)
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(column, dtype=float).copy()
    n = len(x)
    x_orig = np.asarray(x_original, dtype=float).copy() if x_original is not None else x.copy()
    eps = 1e-8

    for _ in range(n_passes):
        perm = rng.permutation(n)
        for start in range(0, n - 2, 3):
            idx = perm[start : start + 3]
            v = x[idx].copy()
            theta_scale = theta_init
            for _ in range(max_trials):
                theta = rng.uniform(-theta_scale, theta_scale)
                v_new = _rodrigues_rotate_axis_111(v, theta)
                # 约束相对原始列（与 PDF 1.1 一致）
                delta_from_orig = v_new - x_orig[idx]
                max_abs = float(np.max(np.abs(delta_from_orig)))
                min_abs = float(np.min(np.abs(delta_from_orig)))
                if max_abs <= max_diff and min_abs > eps:
                    x[idx] = v_new
                    break
                if max_abs > max_diff:
                    theta_scale *= 0.5
                elif min_abs <= eps:
                    theta_scale *= 1.5
    return x


def _weighted_triple_indices(n: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按权重无放回抽样得到排列，每 3 个为一组，返回 shape (n_triples, 3) 的索引数组。"""
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-10)
    w /= w.sum()
    # 加权无放回抽样：得到的一个排列，权重大的更易被抽到
    perm = rng.choice(n, size=n, replace=False, p=w)
    n_triples = n // 3
    out = np.zeros((n_triples, 3), dtype=int)
    for r in range(n_triples):
        out[r] = perm[r * 3 : r * 3 + 3]
    return out


def triplet_micro_rotation_weighted(
    column: np.ndarray,
    max_diff: float = 1.0,
    n_passes: int = 3,
    theta_init: float = 0.5,
    max_trials: int = 30,
    rng: Optional[np.random.Generator] = None,
    x_original: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    三元组微旋转 — 基于局部方差的加权分组（PDF 2.2 / 实验设计 1.1）。

    - 权重 w_i ∝ |x_i - x̄|，每轮按权重无放回抽样构造三元组；
    - 旋转与约束同 triplet_micro_rotation（默认相对原始列约束）。
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(column, dtype=float).copy()
    n = len(x)
    x_bar = float(np.mean(x))
    x_orig = np.asarray(x_original, dtype=float).copy() if x_original is not None else x.copy()
    eps = 1e-8

    for _ in range(n_passes):
        weights = np.abs(x - x_bar)
        if weights.sum() < eps:
            weights = np.ones(n, dtype=float)
        triples = _weighted_triple_indices(n, weights, rng)
        for idx in triples:
            v = x[idx].copy()
            theta_scale = theta_init
            for _ in range(max_trials):
                theta = rng.uniform(-theta_scale, theta_scale)
                v_new = _rodrigues_rotate_axis_111(v, theta)
                delta_from_orig = v_new - x_orig[idx]
                max_abs = float(np.max(np.abs(delta_from_orig)))
                min_abs = float(np.min(np.abs(delta_from_orig)))
                if max_abs <= max_diff and min_abs > eps:
                    x[idx] = v_new
                    break
                if max_abs > max_diff:
                    theta_scale *= 0.5
                elif min_abs <= eps:
                    theta_scale *= 1.5
    return x


def constrained_noise_projection(
    column: np.ndarray,
    max_diff: float = 1.0,
    max_attempts: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    受限噪声投影算子 (Constrained Noise Projection)。

    - 先中心化：z = x - mean(x)；
    - 加高斯噪声得到 z'；
    - 把 z' 投影回 sum=0 的子空间，并把范数缩放回 ||z||，保证均值、方差恢复；
    - 平移回去得到 x_new，拒绝采样确保 |x_new - x| <= max_diff 且每个元素都有变化。

    复杂度: 每次尝试 O(n)，通常几十次内收敛。
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(column, dtype=float)
    n = len(x)

    mu = float(np.mean(x))
    z = x - mu
    original_norm = float(np.linalg.norm(z))

    if original_norm < 1e-9:
        raise ValueError(
            "该列方差为 0，无法在保持方差不变的前提下使所有值都发生变化。"
        )

    noise_scale = max_diff * 0.5

    for _ in range(max_attempts):
        noise = rng.normal(0.0, noise_scale, size=n)
        z_prime = z + noise

        # 投影到 sum=0 子空间
        z_prime -= np.mean(z_prime)

        current_norm = float(np.linalg.norm(z_prime))
        if current_norm < 1e-9:
            continue

        z_final = z_prime * (original_norm / current_norm)
        x_new = z_final + mu

        diff = x_new - x
        max_abs = float(np.max(np.abs(diff)))
        min_abs = float(np.min(np.abs(diff)))

        if max_abs <= max_diff and min_abs > 1e-8:
            return x_new

        if max_abs > max_diff:
            noise_scale *= 0.8
        elif min_abs <= 1e-8:
            noise_scale *= 1.2

    raise RuntimeError(
        "在给定尝试次数内未找到满足约束的解，可以考虑放宽 max_diff 或检查数据分布。"
    )


def householder_reflection(
    column: np.ndarray,
    max_diff: float = 1.0,
    max_trials: int = 200,
    rng: Optional[np.random.Generator] = None,
    return_n_trials: bool = False,
):
    """
    高维 Householder 反射算子 (High-Dimensional Householder Reflection)。

    - 构造 H = I - 2 u u^T，其中 u 为单位向量且 u^T 1 = 0；
      => H 是正交矩阵，保持方差；H1 = 1，保持均值；
    - 随机采样 u，使得反射后的每个分量变化 |Δ| <= max_diff 且非零。

    若 return_n_trials=True，返回 (y, n_trials_used)，便于实验记录采样难度。
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(column, dtype=float)
    n = len(x)
    ones = np.ones(n, dtype=float)
    eps = 1e-8

    for trial in range(max_trials):
        r = rng.normal(size=n)
        r = r - (r @ ones) / float(ones @ ones) * ones
        nr = float(np.linalg.norm(r))
        if nr < eps:
            continue

        u = r / nr
        proj = float(u @ x)
        if abs(proj) < 1e-3:
            continue

        delta = -2.0 * u * proj  # Hx - x
        max_abs = float(np.max(np.abs(delta)))
        min_abs = float(np.min(np.abs(delta)))

        if max_abs <= max_diff and min_abs > 1e-8:
            x_new = x + delta
            if return_n_trials:
                return x_new, trial + 1
            return x_new

    raise RuntimeError(
        "在给定约束下没有采到合适的 Householder 方向，可以增大 max_diff 或放宽条件。"
    )