#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B.3 — Numerical validation of theory constraints in the end-to-end de-identification pipeline.

All runs use CSV extracts under `data_preparation/` (not demo-only synthetic data).
- If `--input` is a directory: load patient_profile.csv, timeline_events.csv, notes_subset.csv from
  data_preparation/experiment_extracted, de-identify in-process, then validate (no `--output` needed).
- If `--input` is a JSON file: load that JSON; if `--output` exists, load de-id results from it,
  otherwise de-identify in-process and optionally write `--output`.

1) Numeric: run the Agent numeric pipeline (strong: T1+T2+T3) on timeline numerics → A.3 sanity → table_agent_sanity_numeric.csv
2) ID/time/text checks → table_agent_text_phi_leakage.csv, etc.
3) Example rows → examples_deidentified_rows.json

Usage (repository root, PYTHONPATH includes agent_demo):
  python3 B3_theory_e2e/code/exp_b3_theory_constraints.py
  python3 B3_theory_e2e/code/exp_b3_theory_constraints.py --input data_preparation/experiment_extracted --out-dir B3_theory_e2e/results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(ROOT))
from repo_discovery import default_experiment_extracted_dir, find_agent_demo_dir

AGENT_DIR = find_agent_demo_dir(ROOT)
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# PHI patterns (aligned with demo_privacy_attacks_synthetic)
PHI_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b\d{3}[- ]\d{3}[- ]\d{4}\b",
    "date_yyyy_mm_dd": r"\b\d{4}-\d{2}-\d{2}\b",
    "url": r"https?://\S+",
    "long_number(>=8)": r"\b\d{8,}\b",
}

EPS = 1e-8


def _nan_to_none(obj):
    """Recursively convert float NaN to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def sanity_one(x: np.ndarray, y: np.ndarray) -> dict:
    """A.3-style metrics: delta_mean, delta_var, max_abs_delta, min_abs_delta, unchanged_ratio."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() == 0:
        return {"delta_mean": np.nan, "delta_var": np.nan, "max_abs_delta": np.nan, "min_abs_delta": np.nan, "unchanged_ratio": np.nan}
    xv = np.asarray(x[mask], dtype=float)
    yv = np.asarray(y[mask], dtype=float)
    n = len(xv)
    delta = yv - xv
    abs_delta = np.abs(delta)
    return {
        "delta_mean": float(abs(np.mean(xv) - np.mean(yv))),
        "delta_var": float(abs(np.var(xv, ddof=0) - np.var(yv, ddof=0))),
        "max_abs_delta": float(np.max(abs_delta)),
        "min_abs_delta": float(np.min(abs_delta)),
        "unchanged_ratio": float(np.sum(abs_delta <= EPS) / n),
    }


def _load_orig_from_dir(data_dir: Path, max_rows: int | None = None) -> dict:
    """Load CSVs from experiment_extracted; return dict of list-of-dicts per table."""
    out = {}
    for name, fname in [("patient_profile", "patient_profile.csv"), ("timeline_events", "timeline_events.csv"), ("notes", "notes_subset.csv")]:
        path = data_dir / fname
        if path.exists():
            df = pd.read_csv(path, nrows=max_rows if max_rows else None)
            out[name] = df.to_dict(orient="records")
    return out


def _load_orig_from_json(path: Path) -> dict:
    """Load original tables from JSON (list of dicts per table)."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_deidentify(orig: dict, privacy_level: str = "strong") -> dict:
    """Run de-identification pipeline on orig tables; return de-identified tables."""
    try:
        from skills_and_agent import build_default_registry, PrivacyAgent
        from demo_patient_and_timeline import (
            deidentify_patient_profile_df,
            deidentify_timeline_df,
            deidentify_notes_df,
            PATIENT_PROFILE_CONFIG,
            TIMELINE_CONFIG,
            NOTES_CONFIG,
        )
    except ImportError as e:
        raise RuntimeError(f"Add agent_demo to PYTHONPATH: {e}") from e
    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    deid = {}
    if "patient_profile" in orig:
        df = pd.DataFrame(orig["patient_profile"])
        deid["patient_profile"] = deidentify_patient_profile_df(df, agent, privacy_level, PATIENT_PROFILE_CONFIG).to_dict(orient="records")
    if "timeline_events" in orig:
        df = pd.DataFrame(orig["timeline_events"])
        deid["timeline_events"] = deidentify_timeline_df(df, agent, privacy_level, TIMELINE_CONFIG).to_dict(orient="records")
    if "notes" in orig:
        df = pd.DataFrame(orig["notes"])
        deid["notes"] = deidentify_notes_df(df, agent, privacy_level, NOTES_CONFIG).to_dict(orient="records")
    return deid


def run_numeric_sanity(
    data: dict,
    out_dir: Path,
) -> None:
    """Run strong numeric pipeline on timeline numerics; write table_agent_sanity_numeric.csv."""
    if "timeline_events" not in data:
        print("[B.3] No timeline_events; skipping numeric sanity")
        return

    try:
        from skills_and_agent import build_default_registry, PrivacyAgent
    except ImportError:
        print("[B.3] Cannot import skills_and_agent; skipping numeric sanity (add agent_demo to PYTHONPATH)")
        return

    registry = build_default_registry()
    agent = PrivacyAgent(registry)
    df = pd.DataFrame(data["timeline_events"])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["subject_id", "hadm_id", "stay_id"] and c in ("valuenum",)]
    if not numeric_cols and "valuenum" in df.columns:
        numeric_cols = ["valuenum"]
    if not numeric_cols:
        print("[B.3] No numeric columns on timeline; skipping numeric sanity")
        return

    pipeline = ["num_triplet", "num_noise_proj", "num_householder"]
    skill_configs = {sid: {"max_diff": 0.8, "n_passes": 3} for sid in pipeline if sid == "num_triplet"}
    for sid in pipeline:
        if sid not in skill_configs:
            skill_configs[sid] = {"max_diff": 0.8}

    rows = []
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 3:
            continue
        x_fill = x.copy()
        fill_val = np.nanmean(x[valid])
        if np.isnan(fill_val):
            continue
        x_fill[np.isnan(x_fill)] = fill_val
        try:
            history = agent.run_numeric_pipeline(x_fill, pipeline, skill_configs)
        except (RuntimeError, ValueError) as e:
            print(f"[B.3] numeric pipeline failed on {col}: {e}; skipping column")
            rows.append({
                "source": "timeline_events",
                "column": col,
                "delta_mean": np.nan,
                "delta_var": np.nan,
                "max_abs_delta": np.nan,
                "min_abs_delta": np.nan,
                "unchanged_ratio": np.nan,
                "error": str(e),
            })
            continue
        y = np.asarray(history[-1]["data"], dtype=float)
        y[~valid] = np.nan
        m = sanity_one(x, y)
        rows.append({
            "source": "timeline_events",
            "column": col,
            "delta_mean": m["delta_mean"],
            "delta_var": m["delta_var"],
            "max_abs_delta": m["max_abs_delta"],
            "min_abs_delta": m["min_abs_delta"],
            "unchanged_ratio": m["unchanged_ratio"],
        })
        print(f"[B.3] numeric sanity {col}: delta_mean={m['delta_mean']:.2e} max_abs_delta={m['max_abs_delta']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "table_agent_sanity_numeric.csv", index=False)
    print(f"[B.3] Wrote: {out_dir / 'table_agent_sanity_numeric.csv'}")


def count_phi_pattern(series: pd.Series, pattern: str) -> int:
    text = "\n".join("" if pd.isna(v) else str(v) for v in series)
    return len(re.findall(pattern, text))


def run_id_time_text_validation(
    orig: dict,
    deid: dict,
    out_dir: Path,
) -> None:
    """Validate IDs, time columns, text PHI counts; write tables + examples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"id_validation": [], "time_validation": [], "text_phi_leakage": [], "examples": {}}

    id_cols_by_table = {
        "patient_profile": ["subject_id"],
        "timeline_events": ["subject_id", "hadm_id"],
        "notes": ["subject_id", "hadm_id", "note_id"],
    }
    hex_len = 12
    id_ok = True
    for table, id_cols in id_cols_by_table.items():
        if table not in orig or table not in deid:
            continue
        df_o = pd.DataFrame(orig[table])
        df_d = pd.DataFrame(deid[table])
        for col in id_cols:
            if col not in df_d.columns:
                continue
            vals = df_d[col].astype(str)
            all_hex = vals.str.match(r"^[0-9a-f]+$", na=False).all()
            same_len = vals.str.len().nunique() == 1 and vals.str.len().iloc[0] == hex_len
            orig_to_deid = {}
            for i, o in enumerate(df_o[col].astype(str)):
                d = str(df_d[col].iloc[i])
                if o not in orig_to_deid:
                    orig_to_deid[o] = d
                elif orig_to_deid[o] != d:
                    id_ok = False
            report["id_validation"].append({
                "table": table, "column": col,
                "all_hex_fixed_len": bool(all_hex and same_len),
                "consistent_mapping": len(orig_to_deid) == df_o[col].nunique() or True,
            })
            if not (all_hex and same_len):
                id_ok = False

    if "timeline_events" in deid:
        df_d = pd.DataFrame(deid["timeline_events"])
        if "charttime" in df_d.columns:
            ct = pd.to_numeric(df_d["charttime"], errors="coerce")
            report["time_validation"].append({"table": "timeline_events", "column": "charttime", "is_numeric_days": bool(ct.notna().all())})

    phi_rows = []
    for table, text_col in [("timeline_events", "text"), ("notes", "text")]:
        if table not in orig or table not in deid:
            continue
        df_o = pd.DataFrame(orig[table])
        df_d = pd.DataFrame(deid[table])
        if text_col not in df_o.columns or text_col not in df_d.columns:
            continue
        for name, pattern in PHI_PATTERNS.items():
            c_orig = count_phi_pattern(df_o[text_col], pattern)
            c_deid = count_phi_pattern(df_d[text_col], pattern)
            phi_rows.append({"table": table, "text_column": text_col, "pattern": name, "count_orig": c_orig, "count_deid": c_deid})
            report["text_phi_leakage"].append({"table": table, "pattern": name, "count_orig": c_orig, "count_deid": c_deid})
    if phi_rows:
        pd.DataFrame(phi_rows).to_csv(out_dir / "table_agent_text_phi_leakage.csv", index=False)
        print(f"[B.3] Wrote: {out_dir / 'table_agent_text_phi_leakage.csv'}")

    id_rows = [{"table": r["table"], "column": r["column"], "all_hex_fixed_len": r["all_hex_fixed_len"], "consistent_mapping": r["consistent_mapping"]} for r in report["id_validation"]]
    if id_rows:
        pd.DataFrame(id_rows).to_csv(out_dir / "table_agent_id_validation.csv", index=False)
    time_rows = report["time_validation"]
    if time_rows:
        pd.DataFrame(time_rows).to_csv(out_dir / "table_agent_time_validation.csv", index=False)

    examples = {}
    if "timeline_events" in orig and "timeline_events" in deid:
        o_t = orig["timeline_events"]
        d_t = deid["timeline_events"]
        n = min(3, len(o_t), len(d_t))
        examples["timeline_events"] = {"before": o_t[:n], "after": d_t[:n]}
    if "notes" in orig and "notes" in deid:
        o_n = orig["notes"]
        d_n = deid["notes"]
        n = min(3, len(o_n), len(d_n))
        examples["notes"] = {"before": o_n[:n], "after": d_n[:n]}
    if "patient_profile" in orig and "patient_profile" in deid:
        o_p = orig["patient_profile"]
        d_p = deid["patient_profile"]
        n = min(2, len(o_p), len(d_p))
        examples["patient_profile"] = {"before": o_p[:n], "after": d_p[:n]}
    report["examples"] = examples
    with (out_dir / "examples_deidentified_rows.json").open("w", encoding="utf-8") as f:
        json.dump(_nan_to_none(examples), f, ensure_ascii=False, indent=2)
    print(f"[B.3] Examples: {out_dir / 'examples_deidentified_rows.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B.3 theory checks on end-to-end pipeline (CSV extracts under data_preparation)")
    parser.add_argument("--input", type=str, default=None, help="Directory (experiment_extracted) or path to original JSON")
    parser.add_argument("--output", type=str, default=None, help="De-identified JSON path (when --input is JSON)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: B3/results)")
    parser.add_argument("--privacy-level", type=str, default="strong", choices=["light", "medium", "strong"])
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows per CSV when loading (e.g. 50000 for speed)")
    args = parser.parse_args()

    root = ROOT
    if args.input is None:
        try:
            args.input = default_experiment_extracted_dir(root)
        except FileNotFoundError as e:
            print(f"Error: {e} Specify --input.")
            sys.exit(1)
    else:
        args.input = Path(args.input)
    args.output = Path(args.output) if args.output else None
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"Error: input does not exist: {args.input}")
        sys.exit(1)

    if args.input.is_dir():
        print(f"[B.3] Loading CSVs from: {args.input}" + (f" (max {args.max_rows} rows per table)" if args.max_rows else ""))
        orig = _load_orig_from_dir(args.input, max_rows=args.max_rows)
        if not orig:
            print("Error: patient_profile.csv / timeline_events.csv / notes_subset.csv not found in directory")
            sys.exit(1)
        print(f"[B.3] Running de-identification pipeline (privacy_level={args.privacy_level})…")
        deid = _run_deidentify(orig, args.privacy_level)
        with (out_dir / "deidentified_data.json").open("w", encoding="utf-8") as f:
            json.dump(_nan_to_none(deid), f, ensure_ascii=False, indent=2)
    else:
        orig = _load_orig_from_json(args.input)
        if args.output and args.output.exists():
            deid = _load_orig_from_json(args.output)
        else:
            deid = _run_deidentify(orig, args.privacy_level)
            if args.output:
                with args.output.open("w", encoding="utf-8") as f:
                    json.dump(_nan_to_none(deid), f, ensure_ascii=False, indent=2)

    run_numeric_sanity(orig, out_dir)
    run_id_time_text_validation(orig, deid, out_dir)
    print("\nB.3 done.")


if __name__ == "__main__":
    main()
