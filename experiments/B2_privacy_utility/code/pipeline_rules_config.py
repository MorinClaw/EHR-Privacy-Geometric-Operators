#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional B.2 pipeline rule overrides.

Same structure as `skills_and_agent.DEFAULT_PIPELINE_RULES`, injected as:
  PrivacyAgent(registry, pipeline_rules_override=pipeline_rules_config.PIPELINE_RULES)

Edit (data_type, privacy_level) -> [skill_id] here without changing skills_and_agent.py.
"""

from __future__ import annotations

from typing import Dict, List

# Same structure as DEFAULT_PIPELINE_RULES; partial overrides are OK
PIPELINE_RULES: Dict[str, Dict[str, List[str]]] = {
    "numeric": {
        "light": ["num_triplet"],
        "medium": ["num_triplet", "num_noise_proj"],
        # strong: T1+T2 only (T3 reserved as negative case in paper; alpha via max_diff)
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

PRIVACY_LEVEL_SEMANTICS: Dict[str, str] = {
    "light": "Minimal de-identification; maximize utility (IDs + coarse bins only).",
    "medium": "Balanced privacy and utility (aggregation, masking; limited synthesis).",
    "strong": "Privacy-first (micro-aggregation, time shift, PHI surrogates, numeric chains).",
}
