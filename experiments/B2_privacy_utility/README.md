# B.2 Agent planning & privacy levels

Rule-based pipeline planning by `(data_type, privacy_level)` and configurable `DEFAULT_PIPELINE_RULES`.

## Contents

- **`skills_and_agent.py`**: `DEFAULT_PIPELINE_RULES`, `PRIVACY_LEVEL_SEMANTICS`, `PrivacyAgent` with overrides.
- **`exp_b2_plan_pipeline.py`**: enumerate / query plans → CSV + JSON reports.
- **`pipeline_rules_config.py`**: optional `--config` injection.
- **`exp_b2_grid_search_privacy.py`**: grid over operator pipelines; ranks pipelines by reconstruction R² to inform `strong` / `medium` / `light` mapping.

### Run planning

```bash
PYTHONPATH=agent_demo python3 B2_privacy_utility/code/exp_b2_plan_pipeline.py --out-dir B2_privacy_utility/results
```

### Run privacy-strength grid (recommended before fixing rules)

```bash
python3 B2_privacy_utility/code/exp_b2_grid_search_privacy.py --out-dir B2_privacy_utility/results
python3 B2_privacy_utility/code/exp_b2_grid_search_privacy.py \
  --data-dir data_preparation/experiment_extracted/ts_48h --out-dir B2_privacy_utility/results
```

Outputs under `results/tables/`: detailed grid, summary, recommendation CSV, and a short usage note.

## Paper positioning

Described as a **rule-based, metadata-aware** agent; richer search/RL is future work.
