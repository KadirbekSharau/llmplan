# llmplan — LLM Capacity Planner

CPU-only planner for self-hosted LLM inference: given a model, traffic, a latency target and
GPU prices, find the cheapest fleet and replica configuration, and show the utilization
timeline that proves it. No GPU required to run it.

Status: M1 to M5 implemented (model and GPU/price catalogs with exact VRAM fit, workload
ingestion, performance model, MILP fleet planner, trace replay with a utilization
timeline). Documents drive the work:

- [docs/PLAN.md](docs/PLAN.md) — what, why, goals, non-goals, principles
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — engineering standards, package layout, data models, interfaces
- [docs/MILESTONES.md](docs/MILESTONES.md) — M0 to M6, acceptance criteria, review checklist
- [docs/milestones/M1_DESIGN.md](docs/milestones/M1_DESIGN.md) — detailed design for the first milestone
- [docs/DEFINITION_OF_DONE.md](docs/DEFINITION_OF_DONE.md) — rules every milestone must satisfy
- [docs/FOUNDER_QUESTIONS.md](docs/FOUNDER_QUESTIONS.md) — decisions waiting on the founder
- [docs/CTO_ASSESSMENT.md](docs/CTO_ASSESSMENT.md) — background reasoning (optional reading)

Implementing agents: start with PLAN.md, then ARCHITECTURE.md, then your milestone's design doc.

## Usage

From a checkout (catalogs and fixtures are read from `data/`):

```
uv sync
```

**`llmplan fit`** — does a model fit on a GPU at a tensor-parallel degree, and how many
full-context sequences fit in the remaining KV cache? A non-fit is an answer (exit 0).

```
uv run llmplan fit --model fixture:llama3-70b --gpu h100-sxm-80gb --tp 2
```

Options: `--dtype {fp32,bf16,fp16,fp8,int8,int4}`, `--kv-dtype {bf16,fp16,fp8}`,
`--context 8192`, `--gpu-mem-util 0.9`, `--max-num-batched-tokens 8192`,
`--param-count-override N`, `--gpus PATH`, `--format {text,json}`. `--model` also accepts a
Hugging Face repo id (`org/name`); gated repos need `HF_TOKEN` in the environment.

**`llmplan model-info`** — architecture integers, exact parameter count, KV bytes per token,
and weight size in every dtype.

```
uv run llmplan model-info --model fixture:qwen2.5-7b
```

**`llmplan gpus`** — the GPU catalog (`data/gpus.yaml`, or `--gpus PATH`).

```
uv run llmplan gpus --format json
```

**`llmplan workload stats`** — summarize a request trace: mean and peak request rate over
`--window` seconds, token percentiles, peak token rates, and an hour-of-day profile when the
trace covers a day. The format (`csv`, `azure2023`, `azure2024`, `burstgpt`) is detected from
the header unless `--format` is given.

```
uv run llmplan workload stats --trace tests/fixtures/workload_10.csv --window 60
```

Options: `--format {auto,csv,azure2023,azure2024,burstgpt}`, `--window 60`,
`--format-out {text,json}`. The generic `csv` format has columns `arrival_s` (seconds) or
`timestamp` (ISO-8601), `input_tokens`, `output_tokens`, and optionally `model`, `tenant`.

**`llmplan workload synth`** — write a deterministic synthetic trace (Poisson arrivals) in the
generic `csv` format, for users without a trace.

```
uv run llmplan workload synth --rps 5 --duration 3600 --in-tokens lognormal:6.2:0.8 --out-tokens lognormal:5.5:0.9 --seed 1 --out synth.csv
```

Token distributions: `fixed:N`, `lognormal:MEAN:SIGMA[:LO:HI]` (parameters of the underlying
normal), `uniform:LO:HI`.

**`llmplan traces fetch`** — download a public trace listed in `data/traces/manifest.yaml`
(`azure2023-code`, `azure2023-conv`, `azure2024-code`, `azure2024-conv`, `burstgpt-1`). It
refuses to run without `--yes`, verifies the recorded SHA-256, and caps downloads at 2 GiB.

```
uv run llmplan traces fetch azure2023-conv --dest ~/traces --yes
```

**`llmplan perf estimate`** — throughput, TTFT, and TPOT of one replica under a workload's
token distribution. `--backend auto` (default) interpolates published benchmark rows
(`data/benchmarks/`) when they match the model, GPU, tensor parallelism, dtype, and request
shape, and otherwise falls back to a roofline bound; the output states the backend,
confidence (`measured`, `interpolated`, or `roofline`), assumptions, and sources.

```
uv run llmplan perf estimate --model fixture:llama3-70b --gpu h100-sxm-80gb --tp 4 \
  --in-mean 512 --in-p50 400 --in-p95 1500 --out-mean 256 --out-p50 200 --out-p95 800
```

Options: `--dtype`, `--max-num-seqs 256`, `--max-model-len 8192`,
`--backend {auto,roofline,table}`, `--format {text,json}`. Pass either all six token
statistics or `--trace PATH`, which computes them from a trace as `llmplan workload stats`
does (`uv run llmplan perf estimate --model fixture:llama3-8b --gpu l40s-48gb --trace
tests/fixtures/workload_10.csv`). Latencies are service times without queueing.

**`llmplan perf benchmarks`** — the shipped benchmark rows, optionally filtered by GPU and
model (fixture ids match through `data/benchmarks/aliases.yaml`).

```
uv run llmplan perf benchmarks --gpu h100-sxm-80gb --model fixture:llama3-8b
```

**`llmplan plan`** — the cheapest fleet (instances of which price rows) and replica
configurations (tensor parallelism, dtype, `max_num_seqs`) that meet the workload's peak
request and output-token rates within the latency SLO. It solves a mixed-integer program
(OR-Tools MathOpt, HiGHS by default), compares the answer with the best single-GPU-type
fleet, says which demand constraint was binding, lists every candidate with the reason it
was used or rejected, and prints a `vllm serve` command per replica.

```
uv run llmplan plan --model fixture:llama3-8b --trace tests/fixtures/workload_10.csv --max-model-len 8192
```

Options: `--stats-json PATH` instead of `--trace` (the output of `workload stats
--format-out json`, or a bare stats object), `--ttft-p95-ms`, `--tpot-p95-ms` (service-time
bounds; queueing is not modeled), `--utilization 0.8` (capacity derating), `--gpus
h100-sxm-80gb,l40s-48gb`, `--providers aws,lambda`, `--tp 1,2,4,8`, `--dtypes bf16,fp8`,
`--max-num-seqs 32,64,128,256`, `--homogeneous`, `--perf-backend {auto,roofline,table}`,
`--solver {highs,cp_sat,scip,gurobi}` (SCIP and Gurobi only when OR-Tools can load them),
`--time-limit 60`, `--gpu-catalog PATH`, `--prices PATH`, `--format {text,json,vllm}`.
Solves are single-threaded and seeded, so the JSON output is byte-identical across runs.
The fleet is sized for the peak window only (no autoscaling yet).

**`llmplan simulate`** — replay a trace on a planned fleet and show what it does over time:
per-window demand against capacity, utilization, KV cache in use and queue depth per
replica, latency p95s, and SLO violations. `--plan` takes the output of `llmplan plan
--format json`; each replica gets `effective_batch` slots, the fit's KV-token capacity and
the perf estimate's service times, and a request waits in its replica's FIFO queue until a
slot and enough KV tokens are free. TTFT here includes queueing.

```
uv run llmplan plan --model fixture:llama3-8b --trace tests/fixtures/workload_10.csv --max-model-len 8192 --ttft-p95-ms 500 --format json > plan.json
uv run llmplan simulate --plan plan.json --trace tests/fixtures/workload_csv_50.csv --png timeline.png
```

Options: `--window 60` (seconds), `--routing {least_outstanding,round_robin}`,
`--ttft-p95-ms`, `--tpot-p95-ms` (budgets; default: the SLO recorded in the plan file, and
without either no violations are counted), `--png PATH` (four-panel figure, rendered
headless), `--format {text,json}`. The plan can also be piped:
`uv run llmplan plan ... --format json | uv run llmplan simulate --plan /dev/stdin --trace
TRACE`. The JSON output is byte-identical across runs.

Exit codes: 0 success, 1 no performance estimate possible (`perf estimate`), 2
usage/validation, 3 catalog/fetch error (including invalid benchmark rows), 4 no feasible
plan, 5 solver error. Shipped fixtures:
`fixture:llama3-70b`, `fixture:llama3-8b`, `fixture:mistral-7b-v0.1`, `fixture:qwen2.5-7b`.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict llmplan
uv run pytest            # includes coverage (fails under 90%)
uv run pytest -m slow --no-cov   # performance test (200k-request replay), excluded by default
uv audit
uv build
```
