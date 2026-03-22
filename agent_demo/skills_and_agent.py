# filename: skills_and_agent.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
skills_and_agent.py

This module defines:
  - a `Skill` abstraction
  - `SkillRegistry`
  - `PrivacyAgent`
  - a default registry with numeric and non-numeric skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

from numeric_operators import (
    normalize_zscore,
    denormalize_zscore,
    NORMALIZED_MAX_DIFF,
    triplet_micro_rotation,
    constrained_noise_projection,
    householder_reflection,
)
from non_numeric_operators import (
    hash_ids,
    demo_binning,
    microaggregation_1d,
    rare_category_agg,
    relative_time_1d,
    time_shift_1d,
    text_mask_basic,
    text_phi_surrogate,
    datasifter_tab_1d,
    datasifter_text,
    kg_struct_identity,
    laplace_aggregate_counts,
)

ArrayLike = Any  # Keep a loose public type; numeric paths convert to float explicitly.


# ============================================================
# Pipeline rules table: (data_type, privacy_level) -> [skill_id].
# Levels correspond to different strategies (minimal / balanced / privacy-first).
# Users can override it via `pipeline_rules_override`.
# ============================================================

PRIVACY_LEVEL_SEMANTICS: Dict[str, str] = {
    "light": "Minimal de-identification with maximal utility (ID pseudonymization + necessary coarse-graining).",
    "medium": "Balanced privacy and utility (category aggregation, rule-based text masking, limited synthesis).",
    "strong": "Privacy-first (micro-aggregation, temporal perturbation, PHI surrogate replacement, numeric operator chains).",
}

# Default (data_type, privacy_level) -> [skill_id]; can be overridden in PrivacyAgent.
DEFAULT_PIPELINE_RULES: Dict[str, Dict[str, List[str]]] = {
    "numeric": {
        "light": ["num_triplet"],
        "medium": ["num_triplet", "num_noise_proj"],
        # strong uses T1 + T2 (negative case includes T3 in the design); alpha is controlled by max_diff.
        "strong": ["num_triplet", "num_noise_proj"],
    },
    "patient_profile": {
        "light": ["id_hash", "demo_bin"],
        "medium": ["id_hash", "demo_bin", "cat_agg"],
        "strong": ["id_hash", "demo_bin", "microagg", "cat_agg"],
    },
    "timeline": {
        "light": ["id_hash", "time_rel"],
        "medium": ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr"],
        "strong": ["id_hash", "time_rel", "time_shift", "text_mask", "text_phi_surr", "ds_tab"],
    },
    "notes": {
        "light": ["id_hash"],
        "medium": ["id_hash", "text_mask"],
        "strong": ["id_hash", "text_mask", "text_phi_surr"],
    },
    "kg": {
        "light": ["kg_struct"],
        "medium": ["id_hash", "kg_struct"],
        "strong": ["id_hash", "time_rel", "kg_struct", "lap_agg"],
    },
}


# ============================================================
# Skill / Registry / Agent
# ============================================================


@dataclass
class Skill:
    """
    `Skill` abstraction: wraps an operator into a metadata-rich callable tool.
    """

    id: str
    name: str
    target: str               # "numeric" / "time" / "id" / "text" / "kg" / "agg" / ...
    description: str
    complexity: str
    guarantees: List[str]
    fn: Callable[[ArrayLike, Dict[str, Any]], ArrayLike]
    default_config: Dict[str, Any] = field(default_factory=dict)

    def apply(self, x: ArrayLike, config: Dict[str, Any] | None = None) -> ArrayLike:
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        return self.fn(x, cfg)


class SkillRegistry:
    """
    A simple registry for looking up skills by target/type and id.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.id in self._skills:
            raise ValueError(f"Duplicate skill id: {skill.id}")
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def list_by_target(self, target: str) -> List[Skill]:
        return [s for s in self._skills.values() if s.target == target]

    def all_skills(self) -> List[Skill]:
        return list(self._skills.values())


class PrivacyAgent:
    """
    PrivacyAgent plans and executes a privacy transformation pipeline given:
      - a data_type
      - a privacy_level

    Planning follows rule-based combinations from DEFAULT_PIPELINE_RULES (levels: minimal/balanced/privacy-first).
    Users can override pipelines via `pipeline_rules_override`.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        pipeline_rules_override: Dict[str, Dict[str, List[str]]] | None = None,
    ):
        self.registry = registry
        self._pipeline_rules = dict(pipeline_rules_override) if pipeline_rules_override else None

    def plan_pipeline(
        self,
        data_type: str,
        privacy_level: str,
        pipeline_override: List[str] | None = None,
    ) -> List[str]:
        """
        Return a list of skill_ids as the planned pipeline.

        - If `pipeline_override` is provided, return it directly.
        - Otherwise, use the rule table (built-in or overridden) for (data_type, privacy_level).
        """
        if pipeline_override is not None:
            return list(pipeline_override)
        rules = self._pipeline_rules if self._pipeline_rules is not None else DEFAULT_PIPELINE_RULES
        if data_type not in rules:
            raise ValueError(f"Unknown data_type: {data_type}")
        level_rules = rules[data_type]
        if privacy_level not in level_rules:
            raise ValueError(f"Unknown privacy_level: {privacy_level}. Allowed: {list(level_rules.keys())}")
        return list(level_rules[privacy_level])

    def list_privacy_levels(self) -> List[str]:
        """Return supported privacy levels."""
        return list(PRIVACY_LEVEL_SEMANTICS.keys())

    def list_data_types(self) -> List[str]:
        """Return supported data types in the pipeline rule table."""
        rules = self._pipeline_rules if self._pipeline_rules is not None else DEFAULT_PIPELINE_RULES
        return list(rules.keys())

    # Operator validation helpers (for integration-time sanity checks).

    def validate_numeric_skills(
        self,
        x: ArrayLike | None = None,
        config_overrides: Dict[str, Any] | None = None,
        n_synthetic: int = 2000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Validate all skills in the registry with target == 'numeric'.

        When `x` is None, use a synthetic input vector (length n_synthetic) to run a quick self-check.
        """
        from operator_validation import validate_registry_numeric
        return validate_registry_numeric(
            self.registry,
            x=np.asarray(x, dtype=float).ravel() if x is not None else None,
            config_overrides=config_overrides,
            n_synthetic=n_synthetic,
            **kwargs,
        )

    # Execution stage (numeric demo runner).

    def run_numeric_pipeline(
        self,
        x: ArrayLike,
        skill_ids: List[str],
        skill_configs: Dict[str, Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a planned pipeline sequentially on a single numeric vector.

        Only skills with target == 'numeric' are executed; other skills are skipped.
        """
        if skill_configs is None:
            skill_configs = {}

        history: List[Dict[str, Any]] = []

        def stats(arr: ArrayLike, ref: ArrayLike | None = None) -> Dict[str, float]:
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

        history.append(
            {
                "step": 0,
                "skill_id": "input",
                "skill_name": "Raw input",
                "data": current.copy(),
                "stats": stats(current),
            }
        )

        for i, sid in enumerate(skill_ids, start=1):
            skill = self.registry.get(sid)
            if skill.target != "numeric":
                # Skip non-numeric skills in numeric runner.
                continue
            cfg = skill_configs.get(sid, {})
            current = np.asarray(skill.apply(current, cfg), dtype=float)

            history.append(
                {
                    "step": i,
                    "skill_id": sid,
                    "skill_name": skill.name,
                    "data": current.copy(),
                    "stats": stats(current, ref=original),
                }
            )

        return history


# ============================================================
# Build a default SkillRegistry containing all built-in skills.
# ============================================================


def build_default_registry() -> SkillRegistry:
    reg = SkillRegistry()

    # Numeric primary operators.
    def _triplet_fn(x, cfg):
        x = np.asarray(x, dtype=float)
        z, mu, sigma = normalize_zscore(x)
        y_z = triplet_micro_rotation(
            z, max_diff=cfg.get("max_diff", NORMALIZED_MAX_DIFF),
            n_passes=cfg.get("n_passes", 3), x_original=z.copy(),
        )
        return denormalize_zscore(y_z, mu, sigma)

    reg.register(
        Skill(
            id="num_triplet",
            name="Triple micro-rotation (T_num_triplet)",
            target="numeric",
            description="Local orthogonal rotation preserving mean/variance; perturbation magnitude is bounded in normalized space.",
            complexity="O(N)",
            guarantees=["mean", "variance", "l_inf<=0.8_std", "non-invertible"],
            fn=_triplet_fn,
            default_config={"max_diff": NORMALIZED_MAX_DIFF, "n_passes": 3},
        )
    )

    reg.register(
        Skill(
            id="num_noise_proj",
            name="Constrained noise projection (T_num_noise_proj)",
            target="numeric",
            description="Gaussian noise plus geometric projection; mean/variance are restored while perturbations stay bounded in normalized space.",
            complexity="O(N)",
            guarantees=["mean", "variance", "l_inf<=0.8_std", "non-invertible"],
            fn=lambda x, cfg: denormalize_zscore(
                constrained_noise_projection(
                    normalize_zscore(np.asarray(x, dtype=float))[0],
                    max_diff=cfg.get("max_diff", NORMALIZED_MAX_DIFF),
                ),
                *normalize_zscore(np.asarray(x, dtype=float))[1:],
            ),
            default_config={"max_diff": NORMALIZED_MAX_DIFF},
        )
    )

    reg.register(
        Skill(
            id="num_householder",
            name="Householder reflection (T_num_householder)",
            target="numeric",
            description="High-dimensional mirror reflection preserving mean/variance; direction is randomized and bounded in normalized space.",
            complexity="O(N)",
            guarantees=["mean", "variance", "l_inf<=0.8_std"],
            fn=lambda x, cfg: denormalize_zscore(
                householder_reflection(
                    normalize_zscore(np.asarray(x, dtype=float))[0],
                    max_diff=cfg.get("max_diff", NORMALIZED_MAX_DIFF),
                ),
                *normalize_zscore(np.asarray(x, dtype=float))[1:],
            ),
            default_config={"max_diff": NORMALIZED_MAX_DIFF},
        )
    )

    # Identity / demographic skills.
    reg.register(
        Skill(
            id="id_hash",
            name="ID hashing pseudonymization (T_ID-hash)",
            target="id",
            description="Apply an irreversible HMAC-SHA256 mapping to *_id fields.",
            complexity="O(N)",
            guarantees=["non-invertible"],
            fn=lambda x, cfg: hash_ids(
                x,
                secret=cfg.get("secret", "demo_salt"),
                length=cfg.get("length", 12),
            ),
        )
    )
    reg.register(
        Skill(
            id="demo_bin",
            name="Demographic binning (T_demo-bin)",
            target="id",
            description="Coarsely bin continuous demographic variables such as age.",
            complexity="O(N)",
            guarantees=[],
            fn=lambda x, cfg: demo_binning(
                x,
                mode=cfg.get("mode", "age"),
            ),
        )
    )
    reg.register(
        Skill(
            id="microagg",
            name="Micro-aggregation (T_micro)",
            target="id",
            description="1D micro-aggregation: group into k and replace with group means.",
            complexity="O(N log N)",
            guarantees=["k-anonymity (simplified 1D version)"],
            fn=lambda x, cfg: microaggregation_1d(
                x,
                k=cfg.get("k", 10),
            ),
        )
    )
    reg.register(
        Skill(
            id="cat_agg",
            name="Rare-category merging / roll-up (T_cat-agg)",
            target="id",
            description="Merge categories with frequency below a threshold into OTHER; keep-list can preserve main categories (e.g., ethnicity, primary insurance).",
            complexity="O(N)",
            guarantees=[],
            fn=lambda x, cfg: rare_category_agg(
                x,
                min_freq=cfg.get("min_freq", 10),
                other_label=cfg.get("other_label", "OTHER"),
                keep_list=cfg.get("keep_list"),
            ),
        )
    )

    # Time skills.
    reg.register(
        Skill(
            id="time_rel",
            name="Relative time encoding (T_time-rel)",
            target="time",
            description="Convert absolute timestamps into relative index time (default uses the earliest time in the column).",
            complexity="O(N)",
            guarantees=[],
            fn=lambda x, cfg: relative_time_1d(
                x,
                index_time=cfg.get("index_time", None),
                unit=cfg.get("unit", "D"),
            ),
        )
    )
    reg.register(
        Skill(
            id="time_shift",
            name="Random time shifting (T_time-shift)",
            target="time",
            description="Add a random integer offset to an already relative time column.",
            complexity="O(N)",
            guarantees=[],
            fn=lambda x, cfg: time_shift_1d(
                x,
                max_shift_days=cfg.get("max_shift_days", 180),
            ),
        )
    )

    # Text skills.
    reg.register(
        Skill(
            id="text_mask",
            name="Rule-based text masking (T_mask)",
            target="text",
            description="Rule-based masking of explicit PHI such as emails, phone numbers, long digit strings, URLs, and dates.",
            complexity="O(N)",
            guarantees=[],
            fn=lambda x, cfg: text_mask_basic(list(x)),
        )
    )
    reg.register(
        Skill(
            id="text_phi_surr",
            name="PHI surrogate replacement (T_PHI-sur)",
            target="text",
            description="Use simple rules to detect entities (e.g., names) and replace with surrogates; dates can be replaced by placeholders.",
            complexity="O(N)",
            guarantees=["non-invertible (batch-level)"],
            fn=lambda x, cfg: text_phi_surrogate(
                list(x),
                seed=cfg.get("seed", 0),
            ),
        )
    )
    reg.register(
        Skill(
            id="ds_text",
            name="Partial text synthesis (T_DS-text)",
            target="text",
            description="Randomly replace a fraction of tokens with synonyms or [MASK] to generate approximate but non-verbatim text.",
            complexity="O(N · iter)",
            guarantees=["non-invertible"],
            fn=lambda x, cfg: datasifter_text(
                list(x),
                mask_fraction=cfg.get("mask_fraction", 0.15),
                seed=cfg.get("seed", 0),
            ),
        )
    )

    # Table synthesis.
    reg.register(
        Skill(
            id="ds_tab",
            name="Partial table synthesis (T_DS-tab)",
            target="clinical",
            description="Partially synthesize sensitive values in a column with noise + shuffle at a given probability.",
            complexity="O(N)",
            guarantees=["non-invertible"],
            fn=lambda x, cfg: datasifter_tab_1d(
                x,
                synth_prob=cfg.get("synth_prob", 0.2),
                noise_scale=cfg.get("noise_scale", 0.5),
            ),
        )
    )

    # KG de-identification and DP aggregation.
    reg.register(
        Skill(
            id="kg_struct",
            name="KG structure de-identification (T_KG-struct)",
            target="kg",
            description="Current implementation is identity; in a real system it should parse and remove potential PHI nodes.",
            complexity="O(|V|+|E|)",
            guarantees=[],
            fn=lambda x, cfg: kg_struct_identity(x),
        )
    )
    reg.register(
        Skill(
            id="lap_agg",
            name="Differentially private aggregation (T_Lap-agg)",
            target="agg",
            description="Add Laplace noise to aggregation counts to achieve epsilon-DP.",
            complexity="O(M)",
            guarantees=["epsilon-DP (sensitivity=1)"],
            fn=lambda x, cfg: laplace_aggregate_counts(
                x,
                epsilon=cfg.get("epsilon", 1.0),
            ),
        )
    )

    return reg


# ============================================================
# Operator validation helpers (used when integrating new operators).
# ============================================================
# Re-export validator helpers for convenience.

def __get_operator_validation():
    try:
        from operator_validation import (
            validate_numeric_operator,
            validate_skill,
            validate_registry_numeric,
        )
        return validate_numeric_operator, validate_skill, validate_registry_numeric
    except ImportError:
        return None, None, None


def validate_numeric_operator(fn: Any, x: Any, config: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Validate a numeric operator fn(x, config) -> y using sanity/reconstruction/multi-run checks."""
    _fn, _, _ = __get_operator_validation()
    if _fn is None:
        return {"error": "operator_validation not found", "pass": False}
    return _fn(fn, x, config, **kwargs)


def validate_skill(skill: Any, x: Any, config: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Validate a single Skill (numeric only)."""
    _, _skill_fn, _ = __get_operator_validation()
    if _skill_fn is None:
        return {"error": "operator_validation not found", "pass": False}
    return _skill_fn(skill, x, config, **kwargs)


def validate_registry_numeric(registry: Any, x: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Validate all numeric skills in the registry and summarize pass/fail per skill_id."""
    _, _, _reg_fn = __get_operator_validation()
    if _reg_fn is None:
        return {"error": "operator_validation not found", "skills": [], "all_pass": False}
    return _reg_fn(registry, x=x, **kwargs)