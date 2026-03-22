# English localization

The repository **defaults to English** for:

- Top-level `README.md`, `docs/*`, and section **`README.md`** files under each experiment folder.
- `repo_discovery.py` and CLI-oriented scripts under `privacy_evaluation_protocol/` and `p7_qmix_pilot/`.
- `B3_theory_e2e/code/exp_b3_theory_constraints.py` (fully English).

Some legacy modules (notably large files under `agent_demo/` and long experiment drivers such as `A7_privacy_attacks/code/exp_a7_privacy.py`) may still contain **Chinese comments or print strings** from the original research tree. These are being migrated; **pull requests that translate remaining strings to English are welcome**.

### Helper

`tools/apply_english_phrase_map.py` applies a small phrase dictionary across `*.py`, `*.md`, `*.html`. Extend the `PHRASES` list and re-run:

```bash
python tools/apply_english_phrase_map.py
```

Do **not** blindly machine-translate operator math comments without review.
