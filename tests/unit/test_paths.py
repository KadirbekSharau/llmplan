from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from llmplan import paths
from llmplan.errors import CatalogError


def test_data_comes_from_the_package() -> None:
    assert Path(paths.__file__).parent / "data" == paths.DATA_DIR
    for path in (paths.GPUS_YAML, paths.PRICES_YAML, paths.TRACE_MANIFEST):
        assert path.is_file()
    for path in (paths.BENCHMARKS_DIR, paths.MODEL_FIXTURES_DIR, paths.TRACE_SAMPLES_DIR):
        assert path.is_dir()


def test_package_data_must_be_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "llmplan.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("llmplan/data/gpus.yaml", "[]")
    monkeypatch.setattr(paths, "files", lambda _: zipfile.Path(archive, "llmplan/"))
    with pytest.raises(CatalogError, match="not found as a directory"):
        paths.data_dir()
