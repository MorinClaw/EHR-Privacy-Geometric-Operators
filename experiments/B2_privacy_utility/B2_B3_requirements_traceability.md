# B.2 / B.3 requirements traceability

## B.2 Agent planning (`plan_pipeline`)

### Requirements
- `data_type ∈ {numeric, patient_profile, timeline, notes, kg}`
- `privacy_level ∈ {light, medium, strong}`
- Documented as a **rule-based, metadata-aware** agent (not black-box RL)

### Implementation

| Area | Requirement | Code |
|------|-------------|------|
| numeric | light/medium/strong skill chains | `DEFAULT_PIPELINE_RULES` in `skills_and_agent.py` |
| patient_profile | ID + demo + microagg + cat_agg by level | same |
| timeline | ID + time + text + optional `ds_tab` for strong | same |
| notes / kg | as per table | same |

**Scripts:** `exp_b2_plan_pipeline.py` enumerates plans; `pipeline_rules_config.py` allows overrides. B.2 does **not** read raw data — planning only.

---

## B.3 End-to-end theory checks

### Requirements
1. Run **numeric strong** pipeline on real **timeline** numerics; A.3-style sanity (mean, variance, ℓ∞, unchanged ratio).
2. Validate **IDs / times / text** (HMAC-style IDs, relative times, PHI pattern counts, before/after snippets).
3. Emit CSV tables + JSON examples.

### Implementation

| Item | Requirement | Delivered |
|------|-------------|-----------|
| Entry | `exp_agent_mimic_pipeline.py` forwards to B.3 | ✓ thin wrapper → `exp_b3_theory_constraints.py` |
| Timeline numeric | strong T1+T2+T3 on valuenum | ✓ `table_agent_sanity_numeric.csv` |
| ID / time / text | per spec | ✓ `table_agent_id_validation.csv`, `table_agent_time_validation.csv`, `table_agent_text_phi_leakage.csv` |
| Examples | `examples_deidentified_rows.json` | ✓ |
| Data | real CSVs under `data_preparation/experiment_extracted` | ✓ default `--input` |

**Main script:** `B3_theory_e2e/code/exp_b3_theory_constraints.py`.
