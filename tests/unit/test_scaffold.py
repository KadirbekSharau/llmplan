from __future__ import annotations

import llmplan
from llmplan import errors


def test_version_is_set() -> None:
    assert llmplan.__version__ == "0.2.0"


def test_error_hierarchy() -> None:
    assert issubclass(errors.UnsupportedArchitecture, errors.CatalogError)
    assert issubclass(errors.FetchError, errors.CatalogError)
    assert issubclass(errors.CatalogError, errors.LLMPlanError)
    assert issubclass(errors.ValidationError, errors.LLMPlanError)
    exc = errors.UnsupportedArchitecture("bad", field="architectures")
    assert exc.field == "architectures"
