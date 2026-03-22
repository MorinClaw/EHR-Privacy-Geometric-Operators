"""
Resolve experiment layout without hard-coded non-ASCII directory names.

Top-level folder names may be localized; discovery uses structural markers
(privacy_evaluation_protocol, P7_qmix_pilot) and basename prefixes (A2_, P6_, …).
"""

from __future__ import annotations

from pathlib import Path

_REPO_MARKERS = ("privacy_evaluation_protocol", "P7_qmix_pilot")


def find_prefixed_section_dir(repo_root: Path, prefix: str) -> Path:
    """First top-level directory whose name starts with ``prefix`` (e.g. ``'A7_'``)."""
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and child.name.startswith(prefix):
            return child
    raise FileNotFoundError(
        f"No top-level directory under {repo_root} starts with {prefix!r}."
    )


def find_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` until both marker sibling directories exist."""
    p = (start or Path(__file__).resolve().parent).resolve()
    if p.is_file():
        p = p.parent
    for cand in [p, *p.parents]:
        if all((cand / name).is_dir() for name in _REPO_MARKERS):
            return cand
    raise FileNotFoundError(
        "Repository root not found (expected sibling dirs: "
        + ", ".join(_REPO_MARKERS)
        + ")."
    )


def find_a2_operator_code(repo_root: Path) -> Path:
    """Return ``A2_*/code`` containing numeric operator experiments."""
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("A2_"):
            continue
        code_dir = child / "code"
        if (code_dir / "numeric_operators.py").is_file() and (
            code_dir / "exp_operators_mimic.py"
        ).is_file():
            return code_dir
    raise FileNotFoundError(
        "A2 operator code not found (expected A2_*/code with "
        "numeric_operators.py and exp_operators_mimic.py)."
    )


def find_first_top_level_with_subpath(repo_root: Path, *parts: str) -> Path:
    """e.g. ``('experiment_extracted',)`` -> ``<any>/<experiment_extracted>``."""
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
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("P6_"):
            continue
        p = child / "make_tradeoff_plots.py"
        if p.is_file():
            return p
    raise FileNotFoundError(
        "make_tradeoff_plots.py not found (expected P6_*/make_tradeoff_plots.py)."
    )


def find_weak_ts48h_dir(repo_root: Path) -> Path:
    for child in sorted(repo_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("P3_"):
            continue
        p = child / "ts_48h_weak"
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Weak ts directory not found (expected P3_*/ts_48h_weak)."
    )


def find_b6_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and child.name.startswith("B6_"):
            return child / "results"
    return repo_root / "b6_results"


def find_b4_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and child.name.startswith("B4_"):
            return child / "results"
    return repo_root / "b4_results"


def find_b5_default_results_dir(repo_root: Path) -> Path:
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and child.name.startswith("B5_"):
            return child / "results"
    return repo_root / "b5_results"


def find_agent_demo_dir(repo_root: Path) -> Path:
    """Directory containing ``skills_and_agent.py`` (Agent demo / skills)."""
    for child in sorted(repo_root.iterdir()):
        if child.is_dir() and (child / "skills_and_agent.py").is_file():
            return child
    raise FileNotFoundError(
        "Agent demo directory not found (expected skills_and_agent.py under a top-level folder)."
    )
