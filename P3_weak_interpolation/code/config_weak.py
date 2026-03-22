# -*- coding: utf-8 -*-
"""Config for Weak Interpolation (P3)."""

WINDOW_HOURS = 48

# Simplified itemid map (aligned with earlier P2 experiments)
CHART_ITEMIDS = {
    "HR": 220045, "SBP": 220050, "DBP": 220051, "MAP": 220052,
    "TempC": 223762, "RespRate": 220210, "SpO2": 220277,
}
LAB_ITEMIDS = {"Creatinine": 50912, "Lactate": 50813, "Glucose": 50931}
ITEMID_TO_VAR = {v: k for k, v in {**CHART_ITEMIDS, **LAB_ITEMIDS}.items()}

# Default variable list
DEFAULT_VARIABLES = ["HR", "MAP", "SpO2", "RespRate", "TempC", "SBP", "DBP", "Creatinine", "Lactate", "Glucose"]

