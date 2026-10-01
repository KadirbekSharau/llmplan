"""Web UI presets (M6_DESIGN.md sections 3, 5, 6): traffic presets, defaults, and limits.

Traffic presets are the bundled public trace samples (`llmplan/data/traces/samples/`, made by
`scripts/make_samples.py`) plus one synthetic preset. The UI never parses a full public
trace. No Streamlit import here, so the presets are testable on their own.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llmplan.catalog.models import DEFAULT_FIXTURE_DIR, FIXTURE_PREFIX
from llmplan.paths import TRACE_SAMPLES_DIR
from llmplan.planner import SLO
from llmplan.types import DType
from llmplan.workload import Distribution, Workload, generate, load_workload

SAMPLES_DIR = TRACE_SAMPLES_DIR

# Limits (section 6). The upload cap is checked before any parsing.
MAX_UPLOAD_BYTES = 50_000_000
MAX_TIME_LIMIT_S = 30.0
MAX_SIM_REQUESTS = 200_000
MAX_PLANS_PER_HOUR = 30

# Defaults (section 3).
DEFAULT_SLO = SLO(ttft_ms_p95=500.0, tpot_ms_p95=50.0, utilization_target=0.8)
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_TIME_LIMIT_S = 10.0
TP_CHOICES = (1, 2, 4, 8)
DTYPE_CHOICES: tuple[DType, ...] = ("bf16", "fp16", "fp8", "int8", "int4")
DEFAULT_DTYPES: tuple[DType, ...] = ("bf16", "fp8")
MAX_NUM_SEQS_CHOICES = (16, 32, 64, 128, 256, 512)
DEFAULT_MAX_NUM_SEQS = (32, 64, 128, 256)
PERF_BACKENDS = ("auto", "roofline", "table")
CLASS_CHOICES = ("2x2", "1", "3x3")  # M7: request-size classes; the first is the default
SOLVERS = ("highs", "cp_sat", "scip", "gurobi")

# Model picker: shipped fixtures (offline) and popular Hugging Face ids, dense and (M8)
# mixture of experts (fetched only when a plan runs), listed by family (M9). The gpt2 and
# deepseek-v3 fixtures are the unsupported-architecture test cases, not offered.
UNSUPPORTED_FIXTURES = frozenset({"gpt2", "deepseek-v3"})
FIXTURE_MODELS = tuple(
    f"{FIXTURE_PREFIX}{path.stem}"
    for path in sorted(DEFAULT_FIXTURE_DIR.glob("*.json"))
    if path.stem not in UNSUPPORTED_FIXTURES
)
CUSTOM_MODEL = "Other Hugging Face id"
POPULAR_MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-30B-A3B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
)
DEFAULT_MODEL = f"{FIXTURE_PREFIX}llama3-8b"


MODEL_GROUPS = ("Llama", "Qwen", "Mistral", "MoE")


def model_group(model_id: str) -> str:
    """The family a model id is listed under (one of `MODEL_GROUPS`)."""
    lowered = model_id.lower()
    if "mixtral" in lowered or "-a3b" in lowered:
        return "MoE"
    return next(g for g in MODEL_GROUPS if g.lower() in lowered)


MODEL_CHOICES = tuple(
    sorted((*FIXTURE_MODELS, *POPULAR_MODELS), key=lambda m: MODEL_GROUPS.index(model_group(m)))
)

# Token lengths of the synthetic preset and the synthetic form (`workload synth` defaults).
DEFAULT_IN_TOKENS = "lognormal:6.2:0.8"
DEFAULT_OUT_TOKENS = "lognormal:5.5:0.9"

TRAFFIC_MODES = ("Preset sample", "Upload CSV", "Synthetic")
TRAFFIC_HELP = ("Bundled public traces", "Your trace, parsed in memory", "Seeded Poisson arrivals")
# Latency-target presets (M9 section 3): TTFT and TPOT p95 in ms; None is no target.
TARGET_PRESETS: dict[str, tuple[float | None, float | None]] = {
    "Chat": (500.0, 50.0),
    "Batch": (None, None),
    "Strict": (200.0, 30.0),
}
# Every input lives in session state under its widget key; these are the defaults (gpu_ids
# and providers come from the shipped catalog). Share links (`share.py`) and example
# scenarios write the same keys. Number inputs take their bounds from BOUNDS.
DEFAULTS: dict[str, object] = {
    "model_choice": DEFAULT_MODEL,
    "model_custom": "",
    "traffic_mode": TRAFFIC_MODES[0],
    "preset": "azure2024-conv",
    "syn_rate": 2.0,
    "syn_duration": 3600.0,
    "syn_in": DEFAULT_IN_TOKENS,
    "syn_out": DEFAULT_OUT_TOKENS,
    "syn_seed": 0,
    "ttft": DEFAULT_SLO.ttft_ms_p95,
    "tpot": DEFAULT_SLO.tpot_ms_p95,
    "utilization": DEFAULT_SLO.utilization_target,
    "gpu_ids": [],
    "providers": [],
    "tp": list(TP_CHOICES),
    "dtypes": list(DEFAULT_DTYPES),
    "seqs": list(DEFAULT_MAX_NUM_SEQS),
    "max_model_len": DEFAULT_MAX_MODEL_LEN,
    "classes": CLASS_CHOICES[0],
    "perf_backend": PERF_BACKENDS[0],
    "solver": SOLVERS[0],
    "time_limit": DEFAULT_TIME_LIMIT_S,
}
BOUNDS: dict[str, tuple[float, float] | tuple[int, int]] = {
    "syn_rate": (0.01, 1000.0),
    "syn_duration": (1.0, 86_400.0),
    "syn_seed": (0, 2**31 - 1),
    "ttft": (0.1, 600_000.0),
    "tpot": (0.1, 60_000.0),
    "utilization": (0.05, 1.0),
    "max_model_len": (256, 1_048_576),
    "time_limit": (1.0, MAX_TIME_LIMIT_S),
}
CHOICES: dict[str, tuple[object, ...]] = {
    "tp": TP_CHOICES,
    "dtypes": DTYPE_CHOICES,
    "seqs": MAX_NUM_SEQS_CHOICES,
    "classes": CLASS_CHOICES,
    "perf_backend": PERF_BACKENDS,
    "solver": SOLVERS,
}


class SamplePreset(BaseModel):
    """A bundled sample: `filename` under `SAMPLES_DIR` in the generic `csv` format, with
    its row count and the public dataset it was cut from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    filename: str = Field(pattern=r"^[a-z0-9_-]+\.csv$")
    rows: int = Field(gt=0)
    source_url: str = Field(pattern=r"^https://\S+$")


class SyntheticPreset(BaseModel):
    """A seeded synthetic Poisson workload (`llmplan.workload.generate`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    rate_rps: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    input_tokens: Distribution
    output_tokens: Distribution
    seed: int = Field(default=0, ge=0)


TracePreset = SamplePreset | SyntheticPreset

SYNTHETIC_PRESET = SyntheticPreset(
    key="synthetic-2rps-1h",
    label="Synthetic: 2 req/s for 1 hour (Poisson, chat-like token lengths)",
    rate_rps=2.0,
    duration_s=3600.0,
    input_tokens=Distribution(kind="lognormal", mean=6.2, sigma=0.8),
    output_tokens=Distribution(kind="lognormal", mean=5.5, sigma=0.9),
)

_AZURE = "https://github.com/Azure/AzurePublicDataset"
# Rows and windows: llmplan/data/traces/samples/README.md (scripts/make_samples.py output);
# each file is the key with underscores, e.g. azure2024_conv.csv.
_SAMPLES = (
    ("azure2024-conv", "Azure 2024 conversation: busiest + median hour (4.6% of rows)", 19_999),
    ("azure2024-code", "Azure 2024 code: busiest + median hour (5.5% of rows)", 19_999),
    ("azure2023-conv", "Azure 2023 conversation: whole trace (58 min)", 19_366),
    ("azure2023-code", "Azure 2023 code: whole trace (57 min)", 8_819),
    ("burstgpt-1", "BurstGPT: busiest + median hour (60% of rows)", 19_999),
)
SAMPLE_PRESETS: tuple[SamplePreset, ...] = tuple(
    SamplePreset(
        key=key,
        label=label,
        filename=f"{key.replace('-', '_')}.csv",
        rows=rows,
        source_url="https://github.com/HPMLL/BurstGPT" if key.startswith("burst") else _AZURE,
    )
    for key, label, rows in _SAMPLES
)

PRESETS: tuple[TracePreset, ...] = (*SAMPLE_PRESETS, SYNTHETIC_PRESET)


def preset(key: str) -> TracePreset:
    """The preset with `key`; raises `KeyError` naming the known keys otherwise."""
    for candidate in PRESETS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"unknown preset {key!r}; known: {', '.join(p.key for p in PRESETS)}")


def load_preset(chosen: TracePreset) -> Workload:
    """The workload of a preset: a bundled sample parsed as generic `csv` (from the package
    data directory, never a user path), or the synthetic workload generated from its seed."""
    if isinstance(chosen, SamplePreset):
        return load_workload(SAMPLES_DIR / chosen.filename, format="csv")
    return generate(
        rate_rps=chosen.rate_rps,
        duration_s=chosen.duration_s,
        input_tokens=chosen.input_tokens,
        output_tokens=chosen.output_tokens,
        seed=chosen.seed,
    )
