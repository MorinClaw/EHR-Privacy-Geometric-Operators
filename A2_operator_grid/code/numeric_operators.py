#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
numeric_operators.py（A2 用副本）

数值型隐私算子：triplet_micro_rotation（含 x_original 约束）、
triplet_micro_rotation_weighted、constrained_noise_projection、householder_reflection。
与 agent_demo/numeric_operators.py 保持同步，A2 独立使用。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ---------- 方案一：先归一化再用固定阈值（归一化空间内统一阈值） ----------
NORMALIZED_MAX_DIFF = 1.0  # 归一化空间（z-score）中的固定 ℓ∞ 阈值，调大以增强抗重建


def normalize_zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """z = (x - μ) / σ，返回 (z, μ, σ)。σ 过小时用 1e-10 避免除零。"""
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


# 定义旋转
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

# 算子1
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
    三元组微小旋转算子，均匀分组；可选 x_original 实现「相对原始列」ℓ∞ 约束。
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

#算子1-改良1
def _weighted_triple_indices(n: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """按权重无放回抽样，每 3 个为一组。"""
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-10)
    w /= w.sum()
    perm = rng.choice(n, size=n, replace=False, p=w)
    n_triples = n // 3
    out = np.zeros((n_triples, 3), dtype=int)
    for r in range(n_triples):
        out[r] = perm[r * 3 : r * 3 + 3]
    return out

# 算子1-改良2
def triplet_micro_rotation_weighted(
    column: np.ndarray,
    max_diff: float = 1.0,
    n_passes: int = 3,
    theta_init: float = 0.5,
    max_trials: int = 30,
    rng: Optional[np.random.Generator] = None,
    x_original: Optional[np.ndarray] = None,
) -> np.ndarray:
    """三元组微旋转 — 基于 |x_i - x̄| 的加权分组。"""
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

# 算子2
def constrained_noise_projection(
    column: np.ndarray,
    max_diff: float = 1.0,
    max_attempts: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """受限噪声投影：加噪后投影回均值/方差流形，拒绝采样 ℓ∞。"""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(column, dtype=float)
    n = len(x)
    mu = float(np.mean(x))
    z = x - mu
    original_norm = float(np.linalg.norm(z))
    if original_norm < 1e-9:
        raise ValueError("该列方差为 0。")
    noise_scale = max_diff * 0.5
    for _ in range(max_attempts):
        noise = rng.normal(0.0, noise_scale, size=n)
        z_prime = z + noise
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
    raise RuntimeError("在给定尝试次数内未找到满足约束的解。")

# 算子3
def householder_reflection(
    column: np.ndarray,
    max_diff: float = 1.0,
    max_trials: int = 200,
    rng: Optional[np.random.Generator] = None,
    return_n_trials: bool = False,
):
    """Householder 反射；return_n_trials=True 时返回 (y, n_trials_used)。"""
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
        delta = -2.0 * u * proj
        max_abs = float(np.max(np.abs(delta)))
        min_abs = float(np.min(np.abs(delta)))
        if max_abs <= max_diff and min_abs > 1e-8:
            x_new = x + delta
            if return_n_trials:
                return x_new, trial + 1
            return x_new
    raise RuntimeError("在给定约束下没有采到合适的 Householder 方向。")
