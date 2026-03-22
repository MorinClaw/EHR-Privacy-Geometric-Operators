"""
Resolve experiment layout without hard-coded non-ASCII directory names.
"""
from __future__ import annotations
from pathlib import Path

_REPO_MARKERS = ("privacy_evaluation_protocol", "experiments")


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__).resolve().parent).resolve()
    if p.is_file():
        p = p.parent
    for cand in [p, *p.parents]:
        if all((cand / name).exists() for name in _REPO_MARKERS):
            return cand
    raise FileNotFoundError(
        "Repository root not found (expected: " + ", ".join(_REPO_MARKERS) + ")."
    )


def find_experiments_dir(repo_root: Path) -> Path:
    return repo_root / "experiments"


def find_prefixed_section_dir(repo_root: Path, prefix: str) -> Path:
    """First directory under experiments/ whose name starts with prefix."""
    exp_dir = find_experiments_dir(repo_root)
    for child in sorted(exp_dir.iterdir()):
        if child.is_dir() and child.name.startswith(prefix):
            return child
    raise FileNotFoundError(
        f"No directory under {exp_dir} starts with {prefix!r}."
    )


def find_a2_operator_code(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if not child.is_dir() or not child.name.startswith("A2_"):
            continue
        code_dir = child / "code"
        if (code_dir / "numeric_operators.py").is_file() and (
            code_dir / "exp_operators_mimic.py"
        ).is_file():
            return code_dir
    raise FileNotFoundError("A2 operator code not found under experiments/.")


def find_first_top_level_with_subpath(repo_root: Path, *parts: str) -> Path:
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue
        nested = child.joinpath(*parts)
        if nested.is_dir():
            return nested
    raise FileNotFoundError(
        f"No top-level directory under {repo_root} contains subpath {parts!r}."
    )


def default_experiment_extracted_dir(repo_root: Path) -> Path:
    return find_first_top_level_with_subpath(repo_root, "experiment_extracted")


def default_ts48h_dir(repo_root: Path) -> Path:
    return default_experiment_extracted_dir(repo_root) / "ts_48h"


def default_a2_perturbed_dir(repo_root: Path) -> Path:
    return find_a2_operator_code(repo_root).parent / "results" / "perturbed"


def find_make_tradeoff_plots(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if not child.is_dir() or not child.name.startswith("P6_"):
            continue
        p = child / "make_tradeoff_plots.py"
        if p.is_file():
            return p
    raise FileNotFoundError("make_tradeoff_plots.py not found under experiments/P6_*/.")


def find_weak_ts48h_dir(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if not child.is_dir() or not child.name.startswith("P3_"):
            continue
        p = child / "ts_48h_weak"
        if p.is_dir():
            return p
    raise FileNotFoundError("Weak ts directory not found under experiments/P3_*/ts_48h_weak.")


def find_b6_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if child.is_dir() and child.name.startswith("B6_"):
            return child / "results"
    return repo_root / "experiments" / "b6_results"


def find_b4_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if child.is_dir() and child.name.startswith("B4_"):
            return child / "results"
    return repo_root / "experiments" / "b4_results"


def find_b5_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(find_experiments_dir(repo_root).iterdir()):
        if child.is_dir() and child.name.startswith("B5_"):
            return child / "results"
    return repo_root / "experiments" / "b5_results"


def find_agent_demo_dir(repo_root: Path) -> Path:
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and (child / "skills_and_agent.py").is_file():
            return child
    raise FileNotFoundError(
        "Agent demo directory not found (expected skills_and_agent.py under a top-level folder)."
    )
