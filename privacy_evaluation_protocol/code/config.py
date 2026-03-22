# -*- coding: utf-8 -*-
"""
Privacy Evaluation Protocol configuration.

This module defines leakage levels, operator names, and evaluation parameters shared across attack implementations.
"""
from __future__ import annotations

# Leakage levels
LEAKAGE_L0 = "L0"  # No-pairs: only observe y, not (x, y)
LEAKAGE_L1 = "L1"  # Few-pairs: ~0.01%
LEAKAGE_L2 = "L2"  # Many-pairs: ~20%
LEAKAGE_L3 = "L3"  # Full-pairs: 100%

LEAKAGE_LEVELS = (LEAKAGE_L0, LEAKAGE_L1, LEAKAGE_L2, LEAKAGE_L3)
LEAKAGE_TRAIN_FRAC = {
    LEAKAGE_L0: 0.0,
    LEAKAGE_L1: 0.0001,
    LEAKAGE_L2: 0.2,
    LEAKAGE_L3: 1.0,
}

# Operators (aligned with A2 naming)
OPERATORS = ("T1_uniform", "T1_weighted", "T2", "T3")
# Alpha (normalization-space max_diff)
ALPHAS = (0.3, 0.5, 0.8, 1.0)
DEFAULT_ALPHA = 0.8

# Attack types
ATTACK_A_RECON = "A"
ATTACK_B_LINKAGE = "B"
ATTACK_C_MEMBERSHIP = "C"
ATTACK_D_ATTRIBUTE = "D"
ATTACK_TYPES = (ATTACK_A_RECON, ATTACK_B_LINKAGE, ATTACK_C_MEMBERSHIP, ATTACK_D_ATTRIBUTE)

# Reconstruction attack: hit-rate deltas in z-score units
HIT_DELTAS = (0.1, 0.5, 1.0)

# Record linkage: candidate set sizes
LINKAGE_CANDIDATE_SIZES = (10, 50, 100)

# Train/Val/Test split ratios (by stay)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2
WINDOW_HOURS = 48
