# B.6 Operator & Agent ablation

**Operator ablation:** compare numeric pipelines `T1 only`, `T1+T2`, `T1+T3`, `T1+T2+T3` on the same downstream task — reconstruction MAE/corr and AUROC.

**Agent ablation:** compare a **fixed strong** pipeline, **rule-based Agent (strong)**, and a **wrong** setting (e.g. `light` used where `strong` is required) on quasi-ID uniqueness and BMI reconstruction.

## Run

```bash
PYTHONPATH=agent_demo python3 B6_agent_ablation/code/exp_b6_ablation.py
PYTHONPATH=agent_demo python3 B6_agent_ablation/code/exp_b6_ablation.py --data-dir data_preparation/experiment_extracted --out-dir B6_agent_ablation/results --max-rows 5000
```

## Outputs

- `results/tables/table_b6_operator_ablation.csv`, `table_b6_agent_ablation.csv`
- `results/figs/b6_operator_ablation.png`, `b6_agent_ablation.png`

**Dependencies:** `agent_demo`, pandas, numpy, scikit-learn (AUROC), matplotlib.
