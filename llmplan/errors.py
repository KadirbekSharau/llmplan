"""Exception hierarchy for llmplan (ARCHITECTURE.md section 7).

Every error raised by the package derives from `LLMPlanError`. The CLI maps subclasses to
exit codes: 2 usage/validation, 3 catalog/fetch, 4 infeasible, 5 solver.
"""

from __future__ import annotations


class LLMPlanError(Exception):
    """Base class for every error raised by llmplan."""


class CatalogError(LLMPlanError):
    """A catalog entry is missing or malformed (unknown GPU id, bad YAML row)."""


class UnsupportedArchitecture(CatalogError):
    """A model config cannot be mapped; `field` names the key that could not be derived."""

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field = field


class FetchError(CatalogError):
    """A Hugging Face config fetch failed or the requested id was disallowed."""


class ValidationError(LLMPlanError):
    """Input failed validation at a package boundary."""


class WorkloadFormatError(ValidationError):
    """M2: a trace file does not match its format (bad header, ambiguous, mostly invalid)."""


class UnknownRegistryKey(LLMPlanError):
    """A registry lookup used a key that was never registered."""


class InfeasiblePlan(LLMPlanError):
    """M4: no fleet satisfies the constraints; `reason` explains which one."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class SolverError(LLMPlanError):
    """M4: backend failure or time limit reached without an incumbent."""


class PerfError(LLMPlanError):
    """M3: no performance backend could produce an estimate; the message says why."""


class BenchmarkError(CatalogError):
    """M3: a benchmark row is malformed or below the physical floor; names file and row."""
