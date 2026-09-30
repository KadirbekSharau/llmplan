# ARCHITECTURE.md — LLM Capacity Planner

This document is the contract between milestones. Implementing agents build against the
interfaces here. If an interface must change, update this document in the same PR and say why.

---

## 1. Engineering standards (non-negotiable)

These exist because the code will be written by several agents over time and must stay
small, readable, and safe.

**Small and clean**
- Pure functions for all arithmetic and modeling. Inputs in, results out, no hidden state.
- One responsibility per module. If a module needs a second paragraph to describe, split it.
- No abstraction until there are two real implementations. Registries (below) exist only
  where a second implementation is already planned in MILESTONES.md.
- Prefer deleting code to adding it. A PR that adds a feature and grows the package by more
  than ~300 lines needs a one-paragraph justification in its description.
- No utility grab-bags (`utils.py`, `helpers.py`, `common.py`). Name the thing.

**Typed and validated**
- Every public data structure is a frozen `pydantic` v2 model with field constraints
  (`gt=0`, `Literal[...]`, etc.). Validation happens at the boundary, once. Internals trust
  their inputs.
- `mypy --strict` passes on the package. `from __future__ import annotations` everywhere.
- Units are in the field name: `vram_bytes`, `price_usd_per_hour`, `ttft_p95_ms`. Never a
  bare `memory` or `cost`.

**Reliable**
- Deterministic: same inputs, same outputs, including solver runs (fixed seeds, single
  thread by default, time limits set explicitly).
- Errors are typed exceptions from one hierarchy (section 7). No bare `except:`. No
  silent fallbacks: if a value is unknown, the result carries `confidence: "unknown"` and the
  caller decides.
- Every module has unit tests; every milestone has acceptance tests with expected values.
- No network in tests. External fetches sit behind a protocol with a fixture-backed fake.

**Secure**
- The only external inputs are: Hugging Face config.json, user-uploaded CSV traces, YAML
  catalogs, and CLI/UI arguments. Each is validated by schema before use.
- YAML is loaded with `yaml.safe_load` only. No `pickle`, no `eval`, no `exec`, no
  `subprocess` in the package.
- Hugging Face fetches go only to `https://huggingface.co/<org>/<name>/resolve/<rev>/config.json`
  with `org`/`name`/`rev` validated against `^[A-Za-z0-9._-]+$`. Optional token is read from
  the environment (`HF_TOKEN`), never from a file we write, never logged.
- Uploaded traces are size-capped (default 200 MB), parsed with an explicit column schema,
  and never executed or templated.
- No secrets in the repo. `data/` contains only public catalog data with source URLs.
- Dependencies pinned in `uv.lock`; `pip-audit` (or `uv audit`) runs in CI.
- Logging never includes tokens, file contents, or full user traces; log shapes and counts.

**Scalable and extensible**
- Extension via small registries keyed by string (section 6): model architectures, trace
  formats, performance backends, solver backends, output renderers. Adding one is a new
  file plus a registration line; nothing else changes.
- The planner core is a library. The CLI and the web UI are thin adapters over it. Nothing
  in the core imports Streamlit, argparse, or matplotlib.
- Data flows through immutable models, so stages can run in parallel or be cached by hash
  of their inputs later without redesign.

---

## 2. System overview

```
                    +-------------------+      +----------------------+
  HF config.json -->|  catalog.models   |      |  catalog.hardware    |<-- data/gpus.yaml
  (or fixture)      |  ModelSpec        |      |  GPUSpec, PriceRow   |<-- data/prices.yaml
                    +---------+---------+      +----------+-----------+
                              |                           |
                              v                           v
                    +---------------------------------------------+
  EngineProfile --->|  memory        (exact VRAM arithmetic)      |  M1
                    |  fit(model, gpu, engine, tp, dtype) -> Fit  |
                    +----------------------+----------------------+
                                           |
  trace CSV / public trace --> workload ---+---> perf (throughput/latency model) M3
        (M2)                  Workload     |          PerfEstimate
                                           v
                    +---------------------------------------------+
                    |  planner  (MILP over fleet + replica config)|  M4
                    |  plan(workload, models, gpus, prices, slo)  |
                    +----------------------+----------------------+
                                           |
                                           v
                    +---------------------------------------------+
                    |  simulate (trace replay on chosen fleet)    |  M5
                    |  timeline, SLO violations, binding constraint|
                    +----------------------+----------------------+
                                           |
                        +------------------+------------------+
                        v                                     v
                  cli (typer)                          ui (Streamlit)     M6
```

Each box is a package under `llmplan/`. Arrows are function calls passing frozen models.
There is no shared mutable state and no database in v1.

---

## 3. Package layout

```
llmplan/
  __init__.py            # version only
  errors.py              # exception hierarchy (section 7)
  types.py               # shared Literal aliases: DType, KVDType, Provider
  catalog/
    models.py            # ModelSpec + loaders (HF fetch behind protocol, fixtures)
    architectures/       # registry: per-architecture parameter/KV formulas
      __init__.py        # register(), get()
      llama_like.py      # llama, mistral, qwen2, qwen3 (dense)
      ...                # moe.py etc. added in later milestones
    hardware.py          # GPUSpec, PriceRow, catalog loaders (YAML)
  memory/
    dtypes.py            # bytes-per-element table
    weights.py           # parameter counting -> weight bytes
    kv_cache.py          # bytes per token, with TP and GQA handling
    engine.py            # EngineProfile (vLLM defaults), overhead model
    fit.py               # fit() -> FitResult  (M1 public API)
  workload/              # M2
    schema.py            # Request rows, Workload
    formats/             # registry: azure2023, azure2024, burstgpt, csv
    stats.py             # peak windows, token distributions
  perf/                  # M3
    backend.py           # PerfBackend protocol
    table.py             # benchmark-table interpolation backend
    vidur.py             # optional Vidur backend (lazy import)
  planner/               # M4
    model.py             # MathOpt formulation
    solve.py             # backend selection, time limits, determinism
    result.py            # PlanResult, explanation of binding constraints
  simulate/              # M5
    replay.py
    timeline.py
  render/                # output adapters: text table, JSON, vLLM command line, plots
    vllm_cmd.py
    ...
  cli.py                 # typer app; thin
  ui/                    # Streamlit app; thin (M6)
data/
  gpus.yaml
  prices.yaml
  benchmarks/            # M3
  fixtures/model_configs/*.json
tests/
  unit/<package>/
  acceptance/test_m1.py ...
```

---

## 4. Core data models (frozen pydantic v2)

Field lists are normative. Add fields only with an ARCHITECTURE.md update.

```python
# llmplan/types.py
DType   = Literal["fp32", "bf16", "fp16", "fp8", "int8", "int4"]
KVDType = Literal["bf16", "fp16", "fp8"]
Attention = Literal["mha", "gqa", "mqa"]   # derived from head counts

# llmplan/catalog/models.py
class ModelSpec(BaseModel, frozen=True):
    id: str                     # "meta-llama/Llama-3.1-70B-Instruct" or "fixture:llama3-70b"
    architecture: str           # registry key, e.g. "llama_like"
    hidden_size: int            # gt=0
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int           # == num_attention_heads for MHA
    head_dim: int               # explicit; default hidden_size // num_attention_heads
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool        # qwen2 has q/k/v bias
    mlp_bias: bool
    qk_norm: bool = False       # Qwen3: per-layer q_norm/k_norm of size head_dim (M1)
    max_position_embeddings: int
    sliding_window: int | None  # informational in M1
    param_count_override: int | None   # user-supplied when architecture unsupported
    source: Literal["huggingface", "fixture", "manual"]
    # strict=True: config values must already be ints/bools. Property `attention` derives
    # mha/gqa/mqa. Validator: num_attention_heads % num_kv_heads == 0.

class DerivedModelInfo(BaseModel, frozen=True):   # computed by memory.weights.model_info
    param_count: int
    attention: Attention
    weight_bytes_by_dtype: dict[DType, int]

# llmplan/catalog/hardware.py
class GPUSpec(BaseModel, frozen=True):
    id: str                     # "h100-sxm-80gb"
    vendor: Literal["nvidia"]
    name: str
    vram_bytes: int             # as reported by nvidia-smi total, in bytes
    memory_bandwidth_gbps: float | None
    fp16_dense_tflops: float | None
    fp8_dense_tflops: float | None
    nvlink: bool
    source_url: str
    as_of: date

class PriceRow(BaseModel, frozen=True):
    provider: str               # "aws", "gcp", "azure", "lambda", "runpod", "onprem"
    instance: str               # "p5.48xlarge"
    gpu_id: str                 # FK -> GPUSpec.id
    gpu_count: int
    price_usd_per_hour: float   # for the whole instance
    commitment: Literal["on_demand", "reserved_1y", "reserved_3y", "spot"]
    region: str | None
    source_url: str
    as_of: date

# llmplan/memory/engine.py
class EngineProfile(BaseModel, frozen=True):
    engine: Literal["vllm"]     # extend later
    gpu_memory_utilization: float = 0.9     # 0 < x <= 1
    max_num_batched_tokens: int = 8192
    fixed_overhead_bytes: int = 1 * 2**30   # CUDA context, graphs; documented assumption
    activation_multiplier: float = 16.0     # see M1 design, calibrated in M3
    kv_dtype: KVDType = "bf16"

# llmplan/memory/fit.py
class FitRequest(BaseModel, frozen=True):
    model: ModelSpec
    gpu: GPUSpec
    engine: EngineProfile
    tensor_parallel: int = 1    # ge=1
    dtype: DType = "bf16"
    context_len: int            # tokens per sequence used for concurrency estimate

class FitResult(BaseModel, frozen=True):
    fits: bool
    per_gpu_usable_bytes: int
    per_gpu_weight_bytes: int
    per_gpu_overhead_bytes: int
    per_gpu_kv_budget_bytes: int        # may be <= 0 when fits is False
    kv_bytes_per_token_per_gpu: int
    kv_token_capacity: int              # total tokens in flight across the replica
    max_concurrent_seqs_at_context: int
    binding: Literal["weights", "overhead", "ok"]
    notes: tuple[str, ...]              # human-readable assumptions applied
    confidence: Literal["exact", "estimated"]   # "estimated" if any override/assumption used
```

Later milestones add `Workload`, `PerfEstimate`, `SLO`, `PlanRequest`, `PlanResult`,
`Timeline` following the same conventions. Their fields are specified in their design docs
and copied here when merged.

---

## 5. Public API per milestone

Downstream code calls only these.

| Milestone | Function | Signature |
|---|---|---|
| M1 | `llmplan.memory.fit.fit` | `(FitRequest) -> FitResult` |
| M1 | `llmplan.catalog.models.load_model` | `(id: str, *, fetcher: ConfigFetcher \| None) -> ModelSpec` |
| M1 | `llmplan.catalog.hardware.load_gpus / load_prices` | `(path: Path \| None) -> Mapping[str, GPUSpec]` / `(path: Path \| None, *, gpus: Mapping[str, GPUSpec] \| None) -> tuple[PriceRow, ...]` (`gpus` is the FK target; default: shipped catalog) |
| M2 | `llmplan.workload.load_workload` | `(source: str \| Path, *, format: str \| None) -> Workload` |
| M3 | `llmplan.perf.estimate` | `(model, gpu, tp, config, workload_stats, *, backend="table") -> PerfEstimate` |
| M4 | `llmplan.planner.plan` | `(PlanRequest) -> PlanResult` |
| M5 | `llmplan.simulate.replay` | `(PlanResult, Workload) -> Timeline` |

---

## 6. Extension registries

Each registry is a module-level `dict[str, Callable | type]` with `register(key)` decorator
and `get(key)` that raises `UnknownRegistryKey`. Keep them this simple; no plugin discovery,
no entry points, until an external contributor needs one.

| Registry | Location | Interface | Initial members |
|---|---|---|---|
| Architectures | `catalog/architectures` | `hf_classes: Mapping[str, HFClassDefaults]`, `count_params(ModelSpec) -> int`, `embedding_params(ModelSpec) -> int`, `kv_heads_per_gpu(ModelSpec, tp) -> int`; `resolve_hf_class(name)` maps HF class -> key | `llama_like` |
| Trace formats | `workload/formats` | `parse(path) -> Workload` | `csv`, `azure2023`, `azure2024`, `burstgpt` (M2) |
| Perf backends | `perf` | `PerfBackend` protocol | `table` (M3), `vidur` (optional) |
| Solver backends | `planner/solve.py` | MathOpt `SolverType` map | `highs` default, `scip`, `cp_sat`, `gurobi` |
| Renderers | `render` | `render(result) -> str \| bytes` | `text`, `json`, `vllm_cmd` |

---

## 7. Error hierarchy

```python
class LLMPlanError(Exception): ...
class CatalogError(LLMPlanError): ...          # missing GPU id, bad YAML row
class UnsupportedArchitecture(CatalogError):   # carries `field` that could not be derived
class FetchError(CatalogError): ...            # HF fetch failed / disallowed id
class ValidationError(LLMPlanError): ...       # wraps pydantic errors at boundaries
class UnknownRegistryKey(LLMPlanError): ...    # registry get() with an unregistered key (section 6)
class InfeasiblePlan(LLMPlanError): ...        # M4: no fleet satisfies constraints; carries reason
class SolverError(LLMPlanError): ...           # M4: backend failure / time limit without incumbent
```

CLI maps these to exit codes 2 (usage/validation, unknown registry key), 3 (catalog/fetch),
4 (infeasible), 5 (solver). Messages are one line, actionable, and name the offending field or id.

---

## 8. Configuration and data files

- `data/gpus.yaml`: list of `GPUSpec`. `data/prices.yaml`: list of `PriceRow`. Both validated
  on load; a bad row fails the whole load with the row index and field.
- Users may pass `--gpus`/`--prices` to override with their own files (same schema).
- Fixtures in `data/fixtures/model_configs/` are hand-written JSON containing only the
  architectural integers needed by `ModelSpec`, not copies of upstream config files.

---

## 9. Observability

- `structlog` or stdlib `logging` with JSON formatter; one logger per package.
- Log at INFO: stage start/end with input hashes and durations. At DEBUG: solver stats.
- Never log raw traces, tokens, or catalog contents.

---

## 10. Testing strategy

- `tests/unit/`: per module, fast, no I/O beyond fixtures.
- `tests/acceptance/test_m<N>.py`: the numbers from each milestone design doc. These are the
  definition of done.
- Property tests (`hypothesis`) for arithmetic invariants, e.g. `weight_bytes` is monotone in
  dtype width, `kv_token_capacity` is non-increasing in `tensor_parallel` per-GPU share.
- CI: `ruff check`, `ruff format --check`, `mypy --strict llmplan`, `pytest`, `uv audit`.
