#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
privacy_agent_demo.py

一个最小可用的 “算子工具库 + Agent + Skill” Demo：
- 定义了 3 个数值型隐私算子：
    1) triplet_micro_rotation        三元组微旋转
    2) constrained_noise_projection  受限噪声投影
    3) householder_reflection        高维 Householder 反射
- 用 Skill / PrivacyAgent 封装成可组合的工具库；
- main() 中让 Agent 根据隐私强度自动规划一个流水线，并在模拟数据上运行，
  输出每一步的均值 / 方差 / 最大 |Δ|。

后续你要做实验，可以直接在这个基础上拆分成模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any

import numpy as np


# ============================================================
# 一、三个数值算子
# ============================================================

def triplet_micro_rotation(column: np.ndarray,
                           max_diff: float = 1.0,
                           n_passes: int = 3,
                           theta_init: float = 0.5,
                           max_trials: int = 30,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    三元组微小旋转算子 (Triplet Micro-Rotation)。

    - 把一列数据打乱索引，按三元组分组 (i,j,k)，在 R^3 中视为向量 v=[x_i,x_j,x_k]^T；
    - 围绕轴 n=(1,1,1)/sqrt(3) 做小角度旋转（Rodrigues 公式），保持局部和 & 二范数不变；
    - 通过拒绝采样控制 max |Δ| <= max_diff，且三元组内每个点都发生变化；
    - 多轮 pass 后，整列的均值与方差保持不变，单元格扰动受限，且整体映射依赖随机分组与角度，
      不记录这些随机性时不可逆。

    复杂度: O(n * n_passes)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(column, dtype=float).copy()
    n = len(x)

    # 旋转轴 (1,1,1)/sqrt(3)
    axis = np.ones(3, dtype=float)
    axis /= np.linalg.norm(axis)

    def rodrigues_rotate(v: np.ndarray, theta: float) -> np.ndarray:
        k = axis
        v = np.asarray(v, dtype=float)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # v' = v cosθ + (k×v) sinθ + k (k·v)(1-cosθ)
        return (
            v * cos_t
            + np.cross(k, v) * sin_t
            + k * np.dot(k, v) * (1.0 - cos_t)
        )

    for _ in range(n_passes):
        perm = rng.permutation(n)
        for start in range(0, n - 2, 3):
            idx = perm[start:start + 3]
            v = x[idx]
            theta_scale = theta_init

            for _ in range(max_trials):
                theta = rng.uniform(-theta_scale, theta_scale)
                v_new = rodrigues_rotate(v, theta)
                delta = v_new - v
                max_abs = float(np.max(np.abs(delta)))
                min_abs = float(np.min(np.abs(delta)))

                if max_abs <= max_diff and min_abs > 1e-8:
                    x[idx] = v_new
                    break

                if max_abs > max_diff:
                    theta_scale *= 0.5
                elif min_abs <= 1e-8:
                    theta_scale *= 1.5

    return x


def constrained_noise_projection(column: np.ndarray,
                                 max_diff: float = 1.0,
                                 max_attempts: int = 100,
                                 rng: np.random.Generator | None = None) -> np.ndarray:
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
        raise ValueError("该列方差为 0，无法在保持方差不变的前提下使所有值都发生变化。")

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

    raise RuntimeError("在给定尝试次数内未找到满足约束的解，可以考虑放宽 max_diff 或检查数据分布。")


def householder_reflection(column: np.ndarray,
                           max_diff: float = 1.0,
                           max_trials: int = 200,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    高维 Householder 反射算子 (High-Dimensional Householder Reflection)。

    - 构造 H = I - 2 u u^T，其中 u 为单位向量且 u^T 1 = 0；
      => H 是正交矩阵，保持方差；H1 = 1，保持均值；
    - 随机采样 u，使得反射后的每个分量变化 |Δ| <= max_diff 且非零。

    复杂度: O(n)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(column, dtype=float)
    n = len(x)
    ones = np.ones(n, dtype=float)
    eps = 1e-8

    for _ in range(max_trials):
        r = rng.normal(size=n)
        # 投影到 1^⊥
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
            return x_new

    raise RuntimeError("在给定约束下没有采到合适的 Householder 方向，可以增大 max_diff 或放宽条件。")


# ============================================================
# 二、Skill / Agent 抽象
# ============================================================

@dataclass
class Skill:
    """
    Skill 抽象：把一个算子封装为带元数据的“工具”。
    """
    id: str
    name: str
    target: str               # 作用的数据类型: "numeric" / "time" / "id" / "text" / "kg"
    description: str
    complexity: str           # 例如 "O(N)"
    guarantees: List[str]     # 例如 ["mean", "variance", "l_inf<=1", "non-invertible"]
    fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
    default_config: Dict[str, Any] = field(default_factory=dict)

    def apply(self, x: np.ndarray, config: Dict[str, Any] | None = None) -> np.ndarray:
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        return self.fn(x, cfg)


class SkillRegistry:
    """
    简单的 Skill 注册表，后面 Agent 可以从这里按类型/ID 查技能。
    """
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.id in self._skills:
            raise ValueError(f"Skill id 重复: {skill.id}")
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def list_by_target(self, target: str) -> List[Skill]:
        return [s for s in self._skills.values() if s.target == target]

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())


class PrivacyAgent:
    """
    PrivacyAgent：给定数据类型和隐私强度，在 Skill 库中规划流水线并执行。
    """
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    # ---- 规划阶段 ----

    def plan_pipeline(self,
                      data_type: str,
                      privacy_level: str) -> List[str]:
        """
        返回一串 skill_id，代表流水线。
        """
        if data_type == "numeric":
            if privacy_level == "light":
                return ["num_triplet"]
            elif privacy_level == "medium":
                return ["num_triplet", "num_noise_proj"]
            elif privacy_level == "strong":
                return ["num_triplet", "num_noise_proj", "num_householder"]
            else:
                raise ValueError(f"未知 privacy_level: {privacy_level}")

        # 其他类型先用占位符 skill
        if data_type == "id":
            return ["id_hash", "demo_bin", "microagg"]
        if data_type == "time":
            return ["time_rel", "time_shift"]
        if data_type == "text":
            return ["text_mask", "text_phi_surr"]
        if data_type == "kg":
            return ["kg_struct"]

        raise ValueError(f"未知 data_type: {data_type}")

    # ---- 执行阶段 ----

    def run_pipeline(self,
                     x: np.ndarray,
                     skill_ids: List[str],
                     skill_configs: Dict[str, Dict[str, Any]] | None = None
                     ) -> List[Dict[str, Any]]:
        """
        顺序执行流水线，返回每一步的结果和统计量。
        """
        if skill_configs is None:
            skill_configs = {}

        history: List[Dict[str, Any]] = []

        def stats(arr: np.ndarray, ref: np.ndarray | None = None) -> Dict[str, float]:
            arr = np.asarray(arr, dtype=float)
            mean = float(arr.mean())
            var = float(arr.var())
            res: Dict[str, float] = {"mean": mean, "var": var}
            if ref is not None:
                diff = arr - ref
                res["max_abs_delta"] = float(np.max(np.abs(diff)))
            return res

        current = np.asarray(x, dtype=float)
        original = current.copy()

        history.append({
            "step": 0,
            "skill_id": "input",
            "skill_name": "原始输入",
            "data": current.copy(),
            "stats": stats(current)
        })

        for i, sid in enumerate(skill_ids, start=1):
            skill = self.registry.get(sid)
            cfg = skill_configs.get(sid, {})
            current = skill.apply(current, cfg)

            history.append({
                "step": i,
                "skill_id": sid,
                "skill_name": skill.name,
                "data": current.copy(),
                "stats": stats(current, ref=original)
            })

        return history


# ============================================================
# 三、把 3 个算子注册成 Skill，并添加一些占位符 Skill
# ============================================================

def build_default_registry() -> SkillRegistry:
    reg = SkillRegistry()

    # 数值型 3 个主算子
    reg.register(Skill(
        id="num_triplet",
        name="三元组微旋转 (T_num_triplet)",
        target="numeric",
        description="局部正交旋转，保持列均值和方差，|Δ| 受限。",
        complexity="O(N)",
        guarantees=["mean", "variance", "l_inf<=1", "non-invertible"],
        fn=lambda x, cfg: triplet_micro_rotation(
            x,
            max_diff=cfg.get("max_diff", 1.0),
            n_passes=cfg.get("n_passes", 3),
        ),
        default_config={"max_diff": 1.0, "n_passes": 3}
    ))

    reg.register(Skill(
        id="num_noise_proj",
        name="受限噪声投影 (T_num_noise_proj)",
        target="numeric",
        description="高斯噪声 + 几何投影，均值/方差恢复，扰动受限。",
        complexity="O(N)",
        guarantees=["mean", "variance", "l_inf<=1", "non-invertible"],
        fn=lambda x, cfg: constrained_noise_projection(
            x,
            max_diff=cfg.get("max_diff", 1.0),
        ),
        default_config={"max_diff": 1.0}
    ))

    reg.register(Skill(
        id="num_householder",
        name="Householder 反射 (T_num_householder)",
        target="numeric",
        description="高维镜面反射，保持均值/方差，方向随机。",
        complexity="O(N)",
        guarantees=["mean", "variance", "l_inf<=1"],
        fn=lambda x, cfg: householder_reflection(
            x,
            max_diff=cfg.get("max_diff", 1.0),
        ),
        default_config={"max_diff": 1.0}
    ))

    # 以下是 ID / 时间 / 文本 / KG 的占位符 skill
    def identity(x: np.ndarray, _: Dict[str, Any]) -> np.ndarray:
        return np.asarray(x, dtype=float)

    reg.register(Skill(
        id="id_hash",
        name="ID 哈希伪匿名 (T_ID_hash)",
        target="id",
        description="把 *_id 字段做不可逆哈希映射。",
        complexity="O(N)",
        guarantees=["non-invertible"],
        fn=identity
    ))
    reg.register(Skill(
        id="demo_bin",
        name="人口学分箱 (T_demo_bin)",
        target="id",
        description="年龄/年份粗分箱。",
        complexity="O(N)",
        guarantees=[],
        fn=identity
    ))
    reg.register(Skill(
        id="microagg",
        name="Micro-aggregation (T_micro)",
        target="id",
        description="k-匿名聚类后取均值/众数。",
        complexity="O(N log N)",
        guarantees=["k-anonymity"],
        fn=identity
    ))
    reg.register(Skill(
        id="time_rel",
        name="相对时间编码 (T_time_rel)",
        target="time",
        description="绝对时间戳 -> 相对入院时间。",
        complexity="O(N)",
        guarantees=[],
        fn=identity
    ))
    reg.register(Skill(
        id="time_shift",
        name="时间随机平移 (T_time_shift)",
        target="time",
        description="整条时间线随机平移 + 粒度粗化。",
        complexity="O(N)",
        guarantees=[],
        fn=identity
    ))
    reg.register(Skill(
        id="text_mask",
        name="规则掩码 (T_text_mask)",
        target="text",
        description="正则表达式删掉身份证号、手机号等显性 PHI。",
        complexity="O(N)",
        guarantees=[],
        fn=identity
    ))
    reg.register(Skill(
        id="text_phi_surr",
        name="PHI 替身替换 (T_PHI_surr)",
        target="text",
        description="小模型识别人名机构并用替身替换。",
        complexity="O(N)",
        guarantees=[],
        fn=identity
    ))
    reg.register(Skill(
        id="kg_struct",
        name="KG 结构脱敏 (T_KG_struct)",
        target="kg",
        description="去掉 subject_id，只保留 ATC/UMLS 节点与边结构。",
        complexity="O(|V|+|E|)",
        guarantees=[],
        fn=identity
    ))

    return reg


# ============================================================
# 四、命令行 Demo
# ============================================================

def main() -> None:
    rng = np.random.default_rng(42)
    n = 300
    original = rng.normal(loc=80.0, scale=10.0, size=n)

    registry = build_default_registry()
    agent = PrivacyAgent(registry)

    data_type = "numeric"
    privacy_level = "strong"  # 可改为 "light" / "medium" / "strong"

    pipeline = agent.plan_pipeline(data_type=data_type, privacy_level=privacy_level)
    # 使每步扰动上限 < 1/步数，保证相对原始列的 max|Δ| < 1
    max_diff_per_step = 0.33  # 三步累积约 ≤ 1
    skill_configs = {sid: {"max_diff": max_diff_per_step} for sid in pipeline}
    history = agent.run_pipeline(original, pipeline, skill_configs=skill_configs)

    print(f"数据类型: {data_type}, 隐私强度: {privacy_level}")
    print("规划得到的流水线:")
    print("  " + " -> ".join(f"{step['skill_name']}[{step['skill_id']}]" for step in history[1:]))

    print("\n每一步统计: (mean, var, max|Δ| 相对于原始列)")
    for h in history:
        sid = h["skill_id"]
        name = h["skill_name"]
        stats = h["stats"]
        if sid == "input":
            print(f"[Step {h['step']}] {name:20s}  mean={stats['mean']:.4f}, var={stats['var']:.4f}")
        else:
            print(
                f"[Step {h['step']}] {name:20s}  "
                f"mean={stats['mean']:.4f}, var={stats['var']:.4f}, "
                f"max|Δ|={stats['max_abs_delta']:.4f}"
            )


if __name__ == "__main__":
    main()