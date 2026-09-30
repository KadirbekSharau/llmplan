# M3 Design — Performance Model

Status: ready for implementation once M1 is merged (independent of M2 except for
`WorkloadStats`, whose fields are fixed in M2_DESIGN.md section 3 and may be constructed
directly in tests). Assignee: developer agent. Branch: `m3-perf`.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9.

---

## 1. Goal

Estimate, for one replica (model, GPU type, replica config) under a workload's token
distributions: decode and prefill throughput, request capacity, and service-time latency
(TTFT, TPOT). This is the only empirical part of the planner. It must be honest about
confidence, physically bounded, and replaceable.

Two backends ship:
- `roofline`: first-principles bounds from the M1 memory arithmetic and the GPU catalog
  (bandwidth, TFLOPS). Always available. Confidence `"roofline"`.
- `table`: interpolation over published benchmark rows. Confidence `"measured"` when an
  exact row matches, `"interpolated"` otherwise. Falls back to `roofline` when no rows match.

## 2. Deliverables

1. `llmplan/perf/config.py`: `ReplicaConfig`.
2. `llmplan/perf/estimate.py`: `PerfEstimate`, `PerfBackend` protocol, `estimate()` facade.
3. `llmplan/perf/roofline.py`: roofline backend.
4. `llmplan/perf/table.py`: benchmark table loader, validator, interpolation backend.
5. `data/benchmarks/*.yaml`: seed rows with sources (section 6).
6. CLI: `llmplan perf estimate`, `llmplan perf benchmarks`.
7. Tests, README, CHANGELOG, `M3_NOTES.md`, and ARCHITECTURE.md model additions.

## 3. Data models

```python
class ReplicaConfig(BaseModel, frozen=True):
    tensor_parallel: int = 1
    dtype: DType = "bf16"
    kv_dtype: KVDType = "bf16"
    max_num_seqs: int = 256
    max_model_len: int                  # must be <= model.max_position_embeddings
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 8192

class PerfEstimate(BaseModel, frozen=True):
    backend: Literal["roofline", "table"]
    confidence: Literal["roofline", "interpolated", "measured"]
    effective_batch: int                # concurrency the estimate assumes (see 4.3)
    decode_tokens_per_s: float          # aggregate output tokens/s for the replica
    prefill_tokens_per_s: float         # aggregate input tokens/s
    requests_per_s_capacity: float      # sustainable request rate at effective_batch (4.4)
    ttft_ms_p50: float                  # service time only, no queueing
    ttft_ms_p95: float
    tpot_ms_p50: float
    tpot_ms_p95: float
    assumptions: tuple[str, ...]
    source_urls: tuple[str, ...]        # empty for roofline

class PerfBackend(Protocol):
    name: str
    def estimate(self, model: ModelSpec, gpu: GPUSpec, config: ReplicaConfig,
                 stats: WorkloadStats) -> PerfEstimate | None: ...   # None = cannot answer
```

`estimate(model, gpu, config, stats, *, backend="auto")`: `"auto"` tries `table` then
`roofline`; a named backend is used alone and raises `PerfError` if it returns `None`.

## 4. Roofline backend

Uses M1's `fit()` to get `per_gpu_weight_bytes`, `kv_bytes_per_token_per_gpu`, and
`kv_token_capacity`. Requires `gpu.memory_bandwidth_gbps` and `gpu.fp16_dense_tflops`
(or `fp8_dense_tflops` when `dtype == "fp8"`); if either is `null`, returns `None`.

Constants (module-level, documented as assumptions, appended to `assumptions`):
`BANDWIDTH_EFFICIENCY = 0.70`, `PREFILL_MFU = 0.50`, `DECODE_MFU = 0.30`,
`P95_FACTOR = 1.5` (p95 service time = p50 x factor; replaced by data in the table backend).

### 4.1 Effective batch
```
avg_ctx_tokens   = stats.input_tokens_mean + stats.output_tokens_mean / 2
kv_seq_capacity  = kv_token_capacity // max(avg_ctx_tokens, 1)
effective_batch  = max(1, min(config.max_num_seqs, kv_seq_capacity))
```

### 4.2 Decode step time (per token per sequence)
```
bytes_per_step   = per_gpu_weight_bytes + effective_batch * avg_ctx_tokens * kv_bytes_per_token_per_gpu
t_mem            = bytes_per_step / (bandwidth_bytes_per_s * BANDWIDTH_EFFICIENCY)
flops_per_step   = 2 * param_count * effective_batch / tensor_parallel
t_compute        = flops_per_step / (tflops * 1e12 * DECODE_MFU)
tpot_s           = max(t_mem, t_compute)
decode_tokens_per_s = effective_batch / tpot_s
```
`bandwidth_bytes_per_s = memory_bandwidth_gbps * 1e9`. `tflops` is the dense figure for the
weight dtype (fp16/bf16 -> `fp16_dense_tflops`; fp8 -> `fp8_dense_tflops`; int8/int4 ->
`fp16_dense_tflops`, with an assumption note that dequant kernels are not modeled).

### 4.3 Prefill
```
prefill_tokens_per_s = tflops * 1e12 * PREFILL_MFU * tensor_parallel / (2 * param_count)
ttft_s(n_input)      = n_input / prefill_tokens_per_s
ttft_ms_p50          = ttft_s(stats.input_tokens_p50) * 1000
ttft_ms_p95          = ttft_s(stats.input_tokens_p95) * 1000
```
Note the `tensor_parallel` factor assumes ideal scaling; add an assumption note.

### 4.4 Request capacity
```
service_s_per_request = ttft_s(stats.input_tokens_mean) + stats.output_tokens_mean * tpot_s
requests_per_s_capacity = effective_batch / service_s_per_request
```
This is the rate at which `effective_batch` concurrent slots turn over. Queueing is M5's
job; M4 applies a utilization target on top of this number.

## 5. Table backend

### 5.1 Row schema (`data/benchmarks/<gpu-id>.yaml`, list of rows)
```
model_id: str            # HF id or fixture id the row was measured on
gpu_id: str              # FK -> GPUSpec.id
engine: "vllm" | "trtllm" | "sglang"
engine_version: str
tensor_parallel: int
dtype: DType
concurrency: int         # concurrent requests during the measurement
input_len: int           # tokens
output_len: int
output_tokens_per_s: float
ttft_ms_p50: float | null
ttft_ms_p95: float | null
tpot_ms_p50: float | null
tpot_ms_p95: float | null
source_url: str
as_of: date
```

### 5.2 Validation at load
- FK checks against the GPU catalog and, when the model id is a fixture, the fixture set.
- Physical bound: for each row, compute the roofline `tpot_s` at the row's concurrency and
  context (`input_len + output_len/2`) with `BANDWIDTH_EFFICIENCY = 1.0` and `DECODE_MFU =
  1.0` (the absolute floor). If the row's implied per-token time
  (`concurrency / output_tokens_per_s`) is below that floor, raise `BenchmarkError` naming
  the row. This catches transcription mistakes and marketing numbers.

### 5.3 Matching and interpolation
1. Filter rows by `(model_id, gpu_id, tensor_parallel, dtype)`. Model matching is by exact
   id; add an alias map `data/benchmarks/aliases.yaml` for `fixture:llama3-70b ->
   meta-llama/Llama-3.1-70B-Instruct` and similar. No match -> return `None`.
2. Choose the rows with the nearest `(input_len, output_len)` in log space; if the nearest
   pair is more than 2x away in either dimension, return `None` (do not extrapolate shape).
3. Interpolate `output_tokens_per_s` and latencies linearly in `log(concurrency)` between the
   two bracketing rows at `effective_batch` (computed as in 4.1 but capped by the max
   concurrency in the table; note the cap). Exact bracket hit -> `"measured"`, else
   `"interpolated"`. Below the smallest or above the largest concurrency -> clamp and note.
4. `prefill_tokens_per_s` comes from the roofline unless the rows carry TTFT, in which case
   `input_len / ttft_ms_p50` is used.

### 5.4 Seed data
Seed at least these row groups, each from a public page with `source_url` and `as_of`:
- Llama-3.1-70B-Instruct on H100 (tp=4 and/or tp=8), bf16 or fp8, at two or more
  concurrencies. Candidate sources: NVIDIA developer blog "LLM Inference Benchmarking: How
  Much Does Your LLM Inference Cost?", vLLM's published performance dashboards, InferenceMAX.
- Llama-3.1-8B-Instruct on H100 tp=1 and on an L4 or A10G if a public row exists.
If a source only gives a chart without numbers, do not read values off the chart; skip it
and note it. Every number must be printed in the source. Fewer verified rows beat more
guessed rows.

## 6. CLI

```
llmplan perf estimate --model <id> --gpu <gpu-id> [--tp 1] [--dtype bf16]
                      [--max-num-seqs 256] [--max-model-len 8192]
                      (--trace PATH | --in-mean 512 --in-p50 400 --in-p95 1500 --out-mean 256 --out-p50 200 --out-p95 800)
                      [--backend auto|roofline|table] [--format text|json]
llmplan perf benchmarks [--gpu <gpu-id>] [--model <id>] [--format text|json]
```
`--trace` requires M2; if M2 is not merged when M3 is implemented, implement only the
explicit stats flags and leave `--trace` with a `TODO(M2)` and a clear error.

## 7. ARCHITECTURE.md updates
Add `ReplicaConfig`, `PerfEstimate`, `PerfBackend` to section 4; add `PerfError` and
`BenchmarkError(CatalogError)` to section 7; mark the M3 API row implemented.

## 8. Allowed dependencies
None new at runtime (numpy already present via M2 or add it here).

## 9. Acceptance tests (`tests/acceptance/test_m3.py`)

Construct `WorkloadStats` directly with: `input_tokens_mean 512, p50 400, p95 1500`,
`output_tokens_mean 256, p50 200, p95 800` (other fields arbitrary but valid).

**9.1 Roofline, llama3-70b bf16, h100-sxm-80gb, tp=4, max_num_seqs=256, max_model_len=8192.**
Using M1 numbers (`per_gpu_weight_bytes 35_276_853_248`, `kv_bytes_per_token_per_gpu 81_920`,
`kv_token_capacity 469_612`, `param_count 70_553_706_496`) and catalog bandwidth 3350 GB/s,
fp16 dense 989 TFLOPS:
- `avg_ctx_tokens = 640`, `kv_seq_capacity = 733`, `effective_batch = 256`.
- `bytes_per_step = 35_276_853_248 + 256*640*81_920 = 48_698_626_048`.
- `t_mem = 48_698_626_048 / (3.35e12 * 0.7) = 0.020767 s` (approx).
- `flops_per_step = 2*70_553_706_496*256/4 = 9.0309e12`; `t_compute = 9.0309e12/(989e12*0.3) = 0.030437 s`.
- `tpot_s = 0.030437` (compute-bound at this batch); `decode_tokens_per_s ≈ 8411`.
- `prefill_tokens_per_s = 989e12*0.5*4/(2*70_553_706_496) ≈ 14_018`.
- `ttft_ms_p50 ≈ 28.5`, `ttft_ms_p95 ≈ 107.0`, `tpot_ms_p50 ≈ 30.44`, `tpot_ms_p95 ≈ 45.66`.
- `service_s_per_request = 512/14_018 + 256*0.030437 ≈ 7.828`; `requests_per_s_capacity ≈ 32.7`.
Assert each with `rel=2e-2` (the implementer's exact constants must reproduce these to
within 2%; the CTO recomputes independently).

**9.2 Roofline, tp=1 same model on h100:** `fit()` says no fit -> backend returns `None`
and `estimate(backend="roofline")` raises `PerfError` mentioning "does not fit".

**9.3 Roofline requires catalog fields:** a `GPUSpec` with `memory_bandwidth_gbps=None`
makes the roofline backend return `None`; `estimate(backend="auto")` then raises `PerfError`
listing both backends tried.

**9.4 Table validation:** a test-only YAML row with `output_tokens_per_s` set 10x above the
physical floor raises `BenchmarkError` naming `row index` and `model_id`.

**9.5 Table interpolation:** a test-only table with rows at concurrency 8 and 64 for the same
(model, gpu, tp, dtype, input_len 512, output_len 256); with `max_num_seqs=32` the estimate
is `"interpolated"`, `effective_batch == 32`, and `output_tokens_per_s` lies strictly between
the two rows' values and equals the log-linear interpolation to `rel=1e-6`. With
`max_num_seqs=64` it is `"measured"`. A request for `input_len 8192` (more than 2x from 512)
returns `None` from the table backend and `"auto"` falls back to roofline.

**9.6 Seed data sanity:** every shipped benchmark row loads, passes the physical bound, and
has a non-empty `source_url` and `as_of`.

**9.7 Determinism:** two calls with identical inputs produce identical `model_dump_json()`.

## 10. Implementation order
1. `ReplicaConfig`, `PerfEstimate`, protocol, facade, errors.
2. Roofline backend + 9.1 to 9.3.
3. Table loader + validation + 9.4, 9.6 (with an empty seed table first).
4. Interpolation + 9.5, 9.7.
5. Seed data from sources; CLI; README; CHANGELOG; notes; ARCHITECTURE update.

## 11. Questions for founder
- Whether to accept vendor-published (NVIDIA) numbers at face value when they use TensorRT-LLM
  rather than vLLM. Current answer: yes, with `engine` recorded, and the planner (M4) will
  prefer rows whose engine matches the user's choice.
