"""Where the shipped data lives (M8_DESIGN.md section 8.1).

Catalogs, benchmark rows, model fixtures, the trace manifest and the bundled trace samples
are package data under `llmplan/data/`, located with `importlib.resources` so the same
paths work from a checkout, an editable install and an installed wheel. There is no
repository-relative fallback. Users who want other catalogs pass their own files.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from llmplan.errors import CatalogError


def data_dir() -> Path:
    """The package data directory as a filesystem path. Raises `CatalogError` when the
    package is imported from somewhere that is not a directory (e.g. a zip file)."""
    root = files("llmplan") / "data"
    if not isinstance(root, Path) or not root.is_dir():
        raise CatalogError(f"llmplan package data not found as a directory at {root}")
    return root


DATA_DIR = data_dir()
GPUS_YAML = DATA_DIR / "gpus.yaml"
PRICES_YAML = DATA_DIR / "prices.yaml"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
MODEL_FIXTURES_DIR = DATA_DIR / "fixtures" / "model_configs"
TRACE_MANIFEST = DATA_DIR / "traces" / "manifest.yaml"
TRACE_SAMPLES_DIR = DATA_DIR / "traces" / "samples"
