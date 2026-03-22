# B.3 End-to-end theory checks on the de-identification pipeline

Numerical checks that **numeric operators** embedded in the Agent still satisfy sanity constraints, and that **ID / time / text** operators behave as designed. Uses **real CSV extracts** under `data_preparation/experiment_extracted/`, not synthetic demo tables only.

## What it does

- **Numeric:** run the strong numeric pipeline on timeline numerics; compare to A.3-style sanity metrics.
- **IDs:** HMAC-style hashed IDs, consistent across rows/tables.
- **Time:** charttime → relative days + global shift.
- **Text:** count PHI-like patterns before/after (aligned with `demo_privacy_attacks_synthetic`).
- **Examples:** a few before/after rows per table for slides.

## Run

```bash
PYTHONPATH=agent_demo python3 B3_theory_e2e/code/exp_b3_theory_constraints.py --out-dir B3_theory_e2e/results
PYTHONPATH=agent_demo python3 B3_theory_e2e/code/exp_b3_theory_constraints.py --max-rows 50000 --out-dir B3_theory_e2e/results
```

Optional JSON mode: `--input orig.json --output deid.json`.

## Outputs

- `table_agent_sanity_numeric.csv`, `table_agent_text_phi_leakage.csv`, ID/time validation tables, `examples_deidentified_rows.json`.
- Figures: `python3 B3_theory_e2e/code/plot_b3_figures.py`.

## Dependencies

`PYTHONPATH` must include `agent_demo`; pandas, numpy.
