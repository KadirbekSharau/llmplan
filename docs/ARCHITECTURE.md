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
- Uploaded traces are size-capped (50 MB in the web UI, checked before parsing; never written to disk), parsed with an explicit column schema,
  and never executed or templated.
- No secrets in the repo. `data/` contains only public catalog data with source URLs.
- Dependencies pinned in `uv.lock`; `pip-audit` (or `uv audit`) runs in CI.
- Logging never includes tokens, file contents, or full user traces; log shapes and counts.

**Scalable and extensible**
- Extension via small registries keyed by string (section 6): model architectures, trace
  formats, performance backends, solver backends, output renderers. Adding one is a new
  file plus a registration line; nothing else changes.
- The planner core is a library. The CLI and the web UI are thin adapters over it. Nothing
  in the core imports Streamlit, argparse, or matplotlib (only `render/plots.py` does, and
  it is imported lazily by the CLI).
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
  types.py               # shared Literal aliases: DType, KVDType, Attention, Commitment
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
  workload/              # M2 (implemented)
    __init__.py          # load_workload() (M2 public API)
    schema.py            # Workload, Distribution, WorkloadStats
    formats/             # registry: csv (generic_csv.py), azure2023/azure2024 (azure.py),
                         #   burstgpt; reader.py = chunked parsing + row validation,
                         #   InMemoryTrace (M6 uploads, never written to disk)
    stats.py             # compute_stats(): peak windows, token percentiles, diurnal
    synth.py             # deterministic synthetic generator
    fetch.py             # consented, checksum-verified public trace download
    classes.py           # M7: DemandClass, classify() (request-size classes), classify_spec,
                         #   assign_classes (request -> class for the simulator)
  perf/                  # M3
    config.py            # ReplicaConfig, fit_for() (M1 fit for one replica)
    estimate.py          # StatsLike, PerfEstimate, PerfBackend protocol + registry, estimate()
    roofline.py          # roofline backend (first-principles bounds)
    benchmarks.py        # BenchmarkRow/BenchmarkTable, YAML loader, physical-bound check
    table.py             # benchmark-table interpolation backend
    vidur.py             # optional Vidur backend (lazy import; not built in M3)
  planner/               # M4
    __init__.py          # plan() (M4 public API): validate, evaluate, prune, solve, explain
    request.py           # SLO, PlanOptions, PlanRequest
    candidates.py        # candidate enumeration and pre-solve checks, Column, dominance pruning
    classes.py           # M7: per-class candidate estimates and verdicts, class demands
    model.py             # MathOpt formulation (pure: columns in, model out); M7 class model
                         #   (allocations x_{r,k}) and the routing LP over a fixed fleet
    solve.py             # backend selection, parameters, determinism, LP relaxation and
                         #   routing LP (GLOP)
    explain.py           # M7 (moved from __init__): capacity, binding LP, assumptions, baseline
    baseline.py          # best homogeneous fleet by enumeration (no solver)
    result.py            # CandidateEval, ReplicaPlan, FleetItem, SolverInfo, PlanResult
  simulate/              # M5
    __init__.py          # replay(), replay_requests() (M5 public API)
    replica.py           # ReplicaSpec from a planned candidate, ReplicaState (slots, KV, queue)
    events.py            # heap-based event loop, RequestLog, per-replica step logs
    routing.py           # routing policy registry: least_outstanding, round_robin
    timeline.py          # SimOptions, Timeline, WindowRecord, ReplicaWindowRecord,
                         #   SimulationSummary, window aggregation (numpy)
  render/                # output adapters: text table, JSON, vLLM command line, plots
    __init__.py          # Renderer protocol, register(), get()
    text.py              # M1
    json_render.py       # M1 (named to avoid shadowing stdlib json)
    workload_text.py     # M2 `workload stats` text (moved from cli_workload.py in M4)
    plan_text.py         # M4 `plan` text
    vllm_cmd.py          # M4 `vllm serve` lines (functions, not a registry member)
    timeline_text.py     # M5 `simulate` text
    timeline_json.py     # M5 `simulate` JSON
    plots.py             # M5 timeline PNG (matplotlib Agg; imported only for --png and the UI)
    ...
  cli.py                 # typer app; thin
  cli_perf.py            # `llmplan perf` typer sub-app (M3), registered in cli.py
  cli_workload.py        # M2: `workload` and `traces` sub-apps, registered in cli.py
  cli_plan.py            # M4: `llmplan plan` command, registered in cli.py
  cli_simulate.py        # M5: `llmplan simulate` command, registered in cli.py
  cli_ui.py              # M6: `llmplan ui` (Streamlit bootstrap in-process; lazy import)
  ui/                    # M6: Streamlit app; thin. Only app.py and views.py import streamlit
    app.py               # the page: sidebar inputs, one Plan button, last outcome (< 400 lines)
    views.py             # result sections rendered from library results
    state.py             # pure helpers: PlanRequest from inputs, PriceRow validation of the
                         #   edited table, cache key, run_plan (plan + replay), window, brake
    presets.py           # SamplePreset/SyntheticPreset, defaults, section 6 limits
    usage_log.py         # opt-in JSON-lines usage log (LLMPLAN_USAGE_LOG)
data/
  gpus.yaml
  prices.yaml
  benchmarks/            # M3: <gpu-id>.yaml rows, aliases.yaml
  fixtures/model_configs/*.json
  traces/manifest.yaml   # M2: public trace URLs + SHA-256 (full data files never committed)
  traces/samples/        # M6: CC-BY-4.0 samples (<= 20,000 rows each) + README; UI presets
scripts/                 # M6: make_samples.py (cuts the samples), usage_summary.py
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
    engine: Literal["vllm"] = "vllm"   # extend later
    gpu_memory_utilization: float = 0.9     # 0 < x <= 1
    max_num_batched_tokens: int = 8192
    fixed_overhead_bytes: int = 1 * 2**30   # CUDA context, graphs; documented assumption
    activation_multiplier: float = 16.0     # see M1 design, calibrated in M3
    kv_dtype: KVDType = "bf16"

# llmplan/memory/fit.py
class FitRequest(BaseModel, frozen=True):
    model: ModelSpec
    gpu: GPUSpec
    engine: EngineProfile = EngineProfile()
    tensor_parallel: int = 1    # ge=1
    dtype: DType = "bf16"
    quantize_embeddings: bool = False   # M1 design 4.2: int8/int4 keep 16-bit embeddings
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

```python
# llmplan/perf/estimate.py (M3)
@runtime_checkable
class StatsLike(Protocol):             # read-only properties, all float, tokens per request
    input_tokens_mean; input_tokens_p50; input_tokens_p95
    output_tokens_mean; output_tokens_p50; output_tokens_p95
    # The WorkloadStats fields M3 reads (M2_DESIGN.md section 3). M3 was built in parallel
    # with M2, so it types `stats` with this protocol instead of importing llmplan.workload;
    # M2's WorkloadStats satisfies it structurally.

class PerfEstimate(BaseModel, frozen=True):
    backend: Literal["roofline", "table"]
    confidence: Literal["roofline", "interpolated", "measured"]
    effective_batch: int                # ge=1; concurrency the estimate assumes
    decode_tokens_per_s: float          # aggregate output tokens/s for the replica
    prefill_tokens_per_s: float         # aggregate input tokens/s
    requests_per_s_capacity: float      # effective_batch / service time per request
    ttft_ms_p50: float                  # service time only, no queueing
    ttft_ms_p95: float
    tpot_ms_p50: float
    tpot_ms_p95: float
    assumptions: tuple[str, ...]
    source_urls: tuple[str, ...]        # empty for roofline

class PerfBackend(Protocol):
    name: str
    def estimate(self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig,
                 stats: StatsLike) -> PerfEstimate | None: ...   # None = cannot answer
    def explain(self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig,
                stats: StatsLike) -> str: ...   # why estimate() returned None (one line)

# llmplan/perf/config.py (M3)
class ReplicaConfig(BaseModel, frozen=True):
    tensor_parallel: int = 1            # ge=1
    dtype: DType = "bf16"
    kv_dtype: KVDType = "bf16"
    max_num_seqs: int = 256             # gt=0
    max_model_len: int                  # <= model.max_position_embeddings (checked by estimate)
    gpu_memory_utilization: float = 0.9 # 0 < x <= 1
    max_num_batched_tokens: int = 8192  # gt=0

# llmplan/perf/benchmarks.py (M3)
class BenchmarkRow(BaseModel, frozen=True):
    model_id: str                       # canonical id (see aliases.yaml) or fixture id
    gpu_id: str                         # FK -> GPUSpec.id; must equal the file stem
    engine: Literal["vllm", "trtllm", "sglang", "nim"]   # nim: container, engine unnamed
    engine_version: str
    tensor_parallel: int
    dtype: DType
    concurrency: int                    # concurrent requests during the measurement
    input_len: int                      # tokens
    output_len: int
    output_tokens_per_s: float          # aggregate output throughput
    ttft_ms_p50: float | None
    ttft_ms_p95: float | None
    tpot_ms_p50: float | None
    tpot_ms_p95: float | None
    source_url: str
    as_of: date

class BenchmarkTable(BaseModel, frozen=True):
    rows: tuple[BenchmarkRow, ...]
    aliases: dict[str, str]             # model id -> canonical id; canonical(id) method
```

```python
# llmplan/workload/schema.py (M2)
class Workload(BaseModel, frozen=True, arbitrary_types_allowed=True):
    source: str                 # path or "synthetic:<seed>"
    format: str                 # registry key, or "synthetic"
    frame: pd.DataFrame         # exactly these columns, default RangeIndex, >= 1 row:
                                #   arrival_s float64 >= 0, non-decreasing, first == 0.0
                                #   input_tokens int64 >= 1; output_tokens int64 >= 0
                                #   model, tenant: pandas "string" (pd.NA), may be all <NA>
    dropped_rows: int           # rows removed during parsing
    notes: tuple[str, ...] = ()

class Distribution(BaseModel, frozen=True):       # synthetic token lengths
    kind: Literal["fixed", "lognormal", "uniform"]
    value: int | None = None    # fixed only
    mean: float | None = None   # lognormal only: mean of the underlying normal
    sigma: float | None = None  # lognormal only: sigma of the underlying normal
    lo: int = 1                 # clip / uniform bounds, inclusive
    hi: int = 131_072

class WorkloadStats(BaseModel, frozen=True):      # compute_stats(workload, window_s=60.0)
    n_requests: int
    duration_s: float           # max(arrival_s) - min(arrival_s)
    window_s: float
    n_windows: int
    mean_rps: float             # n_requests / duration_s (0.0 if duration_s == 0)
    peak_window_rps: float      # max over windows of count / window_s
    peak_window_index: int
    input_tokens_p50: float; input_tokens_p95: float; input_tokens_p99: float
    input_tokens_mean: float; input_tokens_max: int
    output_tokens_p50: float; output_tokens_p95: float; output_tokens_p99: float
    output_tokens_mean: float; output_tokens_max: int
    peak_input_tokens_per_s: float    # max over windows of sum(input_tokens) / window_s
    peak_output_tokens_per_s: float
    hourly_rps: tuple[float, ...] | None   # 24 entries when >= 24 hour windows, else None
```

```python
# llmplan/types.py (M4)
Commitment = Literal["on_demand", "reserved_1y", "reserved_3y", "spot"]   # PriceRow.commitment

# llmplan/planner/request.py (M4)
class SLO(BaseModel, frozen=True):
    ttft_ms_p95: float | None = None          # gt=0; service-time bound (queueing is M5)
    tpot_ms_p95: float | None = None          # gt=0
    utilization_target: float = 0.8           # 0 < x <= 1; capacity is derated by this

class PlanOptions(BaseModel, frozen=True):    # choice tuples non-empty, no duplicates
    gpu_ids: tuple[str, ...] | None = None    # None = every GPU with at least one price row
    providers: tuple[str, ...] | None = None
    commitments: tuple[Commitment, ...] = ("on_demand",)
    tensor_parallel_choices: tuple[int, ...] = (1, 2, 4, 8)
    dtype_choices: tuple[DType, ...] = ("bf16", "fp8")
    max_num_seqs_choices: tuple[int, ...] = (32, 64, 128, 256)
    max_model_len: int                        # required; <= model.max_position_embeddings
    homogeneous: bool = False                 # force a single price row
    perf_backend: str = "auto"                # "auto" or a perf registry key
    solver: Literal["highs", "cp_sat", "scip", "gurobi"] = "highs"
    time_limit_s: float = 60.0
    seed: int = 0                             # 0 <= seed <= 2**31 - 1
    max_instances_per_row: int = 1000         # big-M bound

class PlanRequest(BaseModel, frozen=True):
    model: ModelSpec
    stats: WorkloadStats
    slo: SLO
    engine: EngineProfile
    options: PlanOptions
    gpus: Mapping[str, GPUSpec]               # catalogs passed explicitly (tests inject rows)
    prices: tuple[PriceRow, ...]
    classes: tuple[DemandClass, ...] = ()     # M7: from classify() on the same trace; () = one
                                              #   class (the whole workload, M4 behaviour)

# llmplan/planner/result.py (M4)
class CandidateEval(BaseModel, frozen=True):
    price_row: PriceRow
    config: ReplicaConfig
    replicas_per_instance: int                # gpu_count // tensor_parallel
    fit: FitResult | None                     # None if rejected before the memory check
    perf: PerfEstimate | None
    status: Literal["eligible", "no_fit", "slo_ttft", "slo_tpot", "no_perf", "tp_gt_gpus"]
    reason: str
    usd_per_hour_per_rps: float | None        # price / (replicas_per_instance * derated rps)

class ReplicaPlan(BaseModel, frozen=True):
    candidate: CandidateEval
    count: int                                # replicas of this candidate
    instances: int                            # ceil(count * tp / gpu_count)

class FleetItem(BaseModel, frozen=True):
    price_row: PriceRow
    instances: int
    usd_per_day: float

class SolverInfo(BaseModel, frozen=True):
    backend: str                              # solver key, or "enumeration" for a baseline
    status: Literal["optimal", "feasible_time_limit"]
    objective_usd_per_day: float              # recomputed from the integers with float prices
    best_bound_usd_per_day: float | None
    solve_time_s: float                       # wall clock; excluded from serialization
    n_variables: int
    n_constraints: int

class PlanResult(BaseModel, frozen=True):
    fleet: tuple[FleetItem, ...]
    replicas: tuple[ReplicaPlan, ...]
    cost_usd_per_day: float
    baseline: PlanResult | None               # best homogeneous fleet; None if homogeneous
                                              #   requested or no single row meets demand
    baseline_saving_pct: float | None
    demand_rps: float                         # stats.peak_window_rps
    demand_output_tokens_per_s: float         # stats.peak_output_tokens_per_s
    capacity_rps: float                       # derated, summed over replicas
    capacity_output_tokens_per_s: float
    binding: Literal["requests", "tokens", "both", "none"]   # tight in the LP relaxation over
                                              #   the chosen candidates (tol 1e-6 relative)
    candidates: tuple[CandidateEval, ...]     # all; eligible by usd_per_hour_per_rps, then
                                              #   rejected (empty on a baseline)
    solver: SolverInfo
    assumptions: tuple[str, ...]
```

```python
# llmplan/planner/result.py (M5): load_plan_json(path, *, max_bytes=50 MB)
#   -> tuple[PlanResult, SLO | None]   reads `llmplan plan --format json` output; the SLO is
#   request.slo (None for a bare PlanResult dump); solve_time_s (not serialized) loads as 0.0

# llmplan/simulate/timeline.py (M5)
class SimOptions(BaseModel, frozen=True):
    window_s: float = 60.0               # gt=0, finite
    routing: Literal["least_outstanding", "round_robin"] = "least_outstanding"
    max_requests: int = 500_000          # trace is truncated (with a note) beyond this
    seed: int = 0                        # reserved for tie-breaking (unused: ties are by index)
    ttft_budget_ms: float | None = None  # defaults to slo.ttft_ms_p95 when given
    tpot_budget_ms: float | None = None

class ReplicaWindowRecord(BaseModel, frozen=True):
    replica_index: int
    utilization: float                   # busy slot-seconds / (slots * window_s), 0..1
    kv_tokens_in_use_mean: int           # time-weighted over the window, rounded
    kv_tokens_in_use_max: int
    kv_bytes_in_use_max: int             # summed over the replica's GPUs
    weight_bytes: int                    # summed over the replica's GPUs
    queue_depth_mean: float
    queue_depth_max: int
    requests_started: int
    requests_completed: int
    vram_bytes_total: int | None         # M6: GPU vram_bytes x tensor parallel; None when
                                         #   replay() was not given the GPU spec

class WindowRecord(BaseModel, frozen=True):
    index: int
    start_s: float
    arrivals: int
    completions: int
    demand_rps: float                    # arrivals / window_s
    capacity_rps: float                  # from the plan (derated), constant
    ttft_ms_p95: float | None            # over requests completed in this window
    e2e_ms_p95: float | None
    ttft_violations: int
    tpot_violations: int
    replicas: tuple[ReplicaWindowRecord, ...]

class SimulationSummary(BaseModel, frozen=True):
    n_requests: int                      # simulated (after truncation)
    n_truncated: int
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_p95: float
    e2e_ms_p95: float
    ttft_violation_pct: float
    tpot_violation_pct: float
    mean_utilization: float              # across replicas and windows
    max_queue_depth: int

class Timeline(BaseModel, frozen=True):  # no wall-clock fields: JSON is byte-identical
    plan_cost_usd_per_day: float
    options: SimOptions                  # budgets resolved against the SLO
    windows: tuple[WindowRecord, ...]    # from the first arrival until the last completion
    summary: SimulationSummary
    assumptions: tuple[str, ...]

# llmplan/simulate/events.py (M5)
class RequestLog(BaseModel, frozen=True, arbitrary_types_allowed=True):
    frame: pd.DataFrame                  # one row per simulated request, trace order:
                                         #   arrival_s, start_s, complete_s (float64 s),
                                         #   ttft_ms, tpot_ms, e2e_ms (float64 ms),
                                         #   replica_index, kv_tokens (int64)
```

```python
# llmplan/workload/classes.py (M7)
class DemandClass(BaseModel, frozen=True):   # also a perf StatsLike (token means/p50/p95)
    index: int
    input_lo: int; input_hi: int          # inclusive token bounds; classes tile
    output_lo: int; output_hi: int        #   [1, max input] x [0, max output]
    share: float                          # fraction of requests
    peak_rps: float                       # class requests in the fleet-wide peak request window
    peak_output_tokens_per_s: float       # class output tokens in the fleet-wide peak token window
    input_tokens_mean: float; output_tokens_mean: float
    input_tokens_p50: float; output_tokens_p50: float   # M7: added so a class is a StatsLike
    input_tokens_p95: float; output_tokens_p95: float
    notes: tuple[str, ...] = ()           # cells merged in (< 1% of requests), edges dropped

# classify(workload, *, input_bins=2, output_bins=2, method="quantile"|"fixed",
#          edges=None, window_s=60.0) -> tuple[DemandClass, ...]
# classify_spec(workload, "1"|"AxB"|"fixed:<in edges>/<out edges>") ("1" -> () = no classes)
# assign_classes(classes, input_tokens, output_tokens) -> int64 class index per request
```

```python
# llmplan/workload/formats/reader.py (M6)
@dataclass(frozen=True)
class InMemoryTrace:                     # an upload parsed without touching the disk
    name: str                            # labels messages and Workload.source
    data: bytes                          # never in repr/str

# llmplan/ui/state.py (M6)
class PlanRun(BaseModel, frozen=True):   # the outcome of one Plan click
    request: PlanRequest
    result: PlanResult
    timeline: Timeline

# llmplan/ui/presets.py (M6); TracePreset = SamplePreset | SyntheticPreset
class SamplePreset(BaseModel, frozen=True):
    key: str; label: str
    filename: str                        # under data/traces/samples/, generic csv
    rows: int                            # gt=0
    source_url: str
class SyntheticPreset(BaseModel, frozen=True):
    key: str; label: str
    rate_rps: float; duration_s: float   # gt=0
    input_tokens: Distribution; output_tokens: Distribution
    seed: int = 0
```

---

## 5. Public API per milestone

Downstream code calls only these.

| Milestone | Function | Signature |
|---|---|---|
| M1 (implemented) | `llmplan.memory.fit.fit` | `(FitRequest) -> FitResult` |
| M1 (implemented) | `llmplan.catalog.models.load_model` | `(id: str, *, fetcher: ConfigFetcher \| None) -> ModelSpec` |
| M1 (implemented) | `llmplan.catalog.hardware.load_gpus / load_prices` | `(path: Path \| None) -> Mapping[str, GPUSpec]` / `(path: Path \| None, *, gpus: Mapping[str, GPUSpec] \| None) -> tuple[PriceRow, ...]` (`gpus` is the FK target; default: shipped catalog) |
| M2 (implemented) | `llmplan.workload.load_workload` | `(source: str \| Path \| InMemoryTrace, *, format: str \| None = None, max_bytes: int = 2 GiB) -> Workload` (`format=None` detects from the header; `max_bytes` lets the M6 upload path pass its 50 MB cap; `InMemoryTrace(name, data)`, added in M6, parses an upload from memory so it is never written to disk) |
| M3 (implemented) | `llmplan.perf.estimate` | `(model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig, stats: StatsLike, *, backend: str = "auto", backends: Mapping[str, PerfBackend] \| None = None) -> PerfEstimate` (`tp` lives in `config`; `"auto"` tries table then roofline; `backends` overrides registry entries for one call) |
| M3 (implemented) | `llmplan.perf.benchmarks.load_benchmarks` | `(directory: Path \| None, *, gpus: Mapping[str, GPUSpec] \| None) -> BenchmarkTable` |
| M4 (implemented) | `llmplan.planner.plan` | `(PlanRequest) -> PlanResult` (raises `InfeasiblePlan` with a reason from the candidate statuses, `SolverError` for an unavailable backend or a time limit without a fleet) |
| M5 (implemented) | `llmplan.simulate.replay` | `(plan: PlanResult, workload: Workload, *, slo: SLO \| None = None, options: SimOptions \| None = None, gpus: Mapping[str, GPUSpec] \| None = None) -> Timeline` (window length lives in `options`; `gpus`, added in M6, is the catalog the plan used and supplies `vram_bytes_total`) |
| M5 (implemented) | `llmplan.simulate.replay_requests` | `(plan: PlanResult, workload: Workload, *, options: SimOptions \| None = None) -> RequestLog` (the per-request records of the same replay) |
| M7 | `llmplan.workload.classify` | `(workload: Workload, *, input_bins: int = 2, output_bins: int = 2, method: Literal["quantile", "fixed"] = "quantile", edges: tuple[tuple[int, ...], tuple[int, ...]] \| None = None, window_s: float = 60.0) -> tuple[DemandClass, ...]` (`window_s`, added in M7, is the peak-window length and must match the `WorkloadStats` the plan uses) |
| M6 (implemented) | `llmplan.ui.state.run_plan` | `(request: PlanRequest, workload: Workload, options: SimOptions, gpus: Mapping[str, GPUSpec]) -> PlanRun` (plan, then replay with the request's SLO budgets; the web UI's only entry into the planner, cached under `cache_key(request, options, workload)`) |

---

## 6. Extension registries

Each registry is a module-level `dict[str, Callable | type]` with `register(key)` decorator
and `get(key)` that raises `UnknownRegistryKey`. Keep them this simple; no plugin discovery,
no entry points, until an external contributor needs one.

| Registry | Location | Interface | Initial members |
|---|---|---|---|
| Architectures | `catalog/architectures` | `hf_classes: Mapping[str, HFClassDefaults]`, `count_params(ModelSpec) -> int`, `embedding_params(ModelSpec) -> int`, `kv_heads_per_gpu(ModelSpec, tp) -> int`; `resolve_hf_class(name)` maps HF class -> key | `llama_like` |
| Trace formats | `workload/formats` | `TraceFormat` protocol: `matches(header, first_row) -> bool`, `parse(source, *, max_bytes) -> Workload`; `detect(source) -> str` (`source` is a `Path` or, since M6, an `InMemoryTrace`) | `csv`, `azure2023`, `azure2024`, `burstgpt` (M2) |
| Perf backends | `perf/estimate.py` | `PerfBackend` protocol: `name`, `estimate(model, gpu, config, stats) -> PerfEstimate \| None`, `explain(...) -> str` | `roofline`, `table` (M3), `vidur` (optional, not built) |
| Solver backends | `planner/solve.py` | MathOpt `SolverType` map | `highs` default, `scip`, `cp_sat`, `gurobi` |
| Routing policies | `simulate/routing.py` | `(outstanding: Sequence[int], index: int) -> int` (replica index) | `least_outstanding`, `round_robin` (M5) |
| Renderers | `render` | `Renderer` protocol, one method per result type returning `str`: `fit(FitRequest, FitResult)`, `model_info(ModelSpec)`, `gpus(Mapping[str, GPUSpec])` (M1); `perf_estimate(ModelSpec, GPUSpec, ReplicaConfig, StatsLike, PerfEstimate)`, `benchmarks(Sequence[BenchmarkRow])` (M3); `workload_stats(Workload, WorkloadStats)`, `plan(PlanRequest, PlanResult)` (M4); `timeline(Timeline)` (M5); later milestones add a method per new result | `text`, `json` (M1). `render/vllm_cmd.py` (M4) holds plain functions (`serve_command`, `plan_commands`) used by the text renderer and `plan --format vllm`; it renders only replica configs, so it is not a registry member. `render/plots.py` (M5) holds `save_png(Timeline, Path)` and (M6) `render_png(Timeline) -> bytes`, plain functions for the binary PNG output (`llmplan simulate --png`, the web UI) |

---

## 7. Error hierarchy

```python
class LLMPlanError(Exception): ...
class CatalogError(LLMPlanError): ...          # missing GPU id, bad YAML row
class UnsupportedArchitecture(CatalogError):   # carries `field` that could not be derived
class FetchError(CatalogError): ...            # HF fetch failed / disallowed id
class ValidationError(LLMPlanError): ...       # wraps pydantic errors at boundaries
class WorkloadFormatError(ValidationError): ... # M2: trace header/format mismatch, >50% bad rows
class UnknownRegistryKey(LLMPlanError): ...    # registry get() with an unregistered key (section 6)
class InfeasiblePlan(LLMPlanError): ...        # M4: no fleet satisfies constraints; carries reason
class SolverError(LLMPlanError): ...           # M4: backend failure / time limit without incumbent
class PerfError(LLMPlanError): ...             # M3: no perf backend could answer; names each tried
class BenchmarkError(CatalogError): ...        # M3: bad benchmark row; names file, row index, model_id
```

CLI maps these to exit codes 2 (usage/validation, unknown registry key), 3 (catalog/fetch,
including `BenchmarkError`), 4 (infeasible), 5 (solver); `PerfError` has no mapping and
exits 1. Messages are one line, actionable, and name the offending field or id.

---

## 8. Configuration and data files

- `data/gpus.yaml`: list of `GPUSpec`. `data/prices.yaml`: list of `PriceRow`. Both validated
  on load; a bad row fails the whole load with the row index and field.
- Users may pass `--gpus`/`--prices` to override with their own files (same schema).
  `llmplan plan` uses `--gpus` for a list of GPU ids (M4_DESIGN.md section 9), so its catalog
  overrides are `--gpu-catalog PATH` and `--prices PATH`.
- `data/benchmarks/<gpu-id>.yaml`: lists of `BenchmarkRow`; `data/benchmarks/aliases.yaml`:
  mapping of model id -> canonical id. Every row must resolve to a model fixture (directly or
  through an alias) and pass the physical floor (roofline at 100% bandwidth and MFU), or the
  whole load fails with `BenchmarkError` naming the file, row index, and model id.
- Fixtures in `data/fixtures/model_configs/` are hand-written JSON containing only the
  architectural integers needed by `ModelSpec`, not copies of upstream config files.
- `data/traces/manifest.yaml` (M2): list of `TraceSource` rows
  (`llmplan.workload.fetch`: `name`, `url | None`, `sha256 | None`, `size_bytes | None`,
  `as_of`, `license_url`). `llmplan traces fetch` downloads only with `--yes`, caps at
  2 GiB, verifies the SHA-256, and never writes into the repo unless `--dest` points there.

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
