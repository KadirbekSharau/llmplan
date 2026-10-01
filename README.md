# llmplan — LLM Capacity Planner

[![ci](https://github.com/KadirbekSharau/llmplan/actions/workflows/ci.yml/badge.svg)](https://github.com/KadirbekSharau/llmplan/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

CPU-only planner for self-hosted LLM inference: given a model, traffic, a latency target and
GPU prices, find the cheapest fleet and replica configuration, and show the utilization
timeline that proves it. No GPU required to run it.

Status: M1 to M7 implemented (model and GPU/price catalogs with exact VRAM fit, workload
ingestion, performance model, MILP fleet planner with request-size demand classes, trace
replay with a utilization timeline, and a Streamlit web UI). Documents drive the work:

- [docs/PLAN.md](docs/PLAN.md) — what, why, goals, non-goals, principles
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — engineering standards, package layout, data models, interfaces
- [docs/MILESTONES.md](docs/MILESTONES.md) — M0 to M7, acceptance criteria, review checklist
- [docs/milestones/M1_DESIGN.md](docs/milestones/M1_DESIGN.md) — detailed design for the first milestone
- [docs/DEFINITION_OF_DONE.md](docs/DEFINITION_OF_DONE.md) — rules every milestone must satisfy
- [docs/FOUNDER_QUESTIONS.md](docs/FOUNDER_QUESTIONS.md) — decisions waiting on the founder
- [docs/CTO_ASSESSMENT.md](docs/CTO_ASSESSMENT.md) — background reasoning (optional reading)
- [docs/DEPLOY.md](docs/DEPLOY.md) — hosting the web UI (Docker, Streamlit Community Cloud)
- [docs/LAUNCH.md](docs/LAUNCH.md) — launch checklist and draft posts

Implementing agents: start with PLAN.md, then ARCHITECTURE.md, then your milestone's design doc.

## Usage

From a checkout (catalogs, fixtures and trace samples ship inside the package, under
`llmplan/data/`):

```
uv sync
```

**`llmplan ui`** — the web UI: pick a model (a shipped fixture or a Hugging Face id), traffic
(a bundled public trace sample, an uploaded CSV up to 50 MB, or synthetic), a p95
TTFT/TPOT target and GPU prices (editable), click Plan, and get the cheapest fleet, a
`vllm serve` line per replica, the utilization timeline of a replay, every candidate with
its reason, the assumptions, and JSON downloads. Nothing recomputes until Plan is clicked.
Under Advanced, "Request-size classes" (default 2x2) plans each request size on the GPUs
that meet the target for it; the results then show the routing table, each class's
replayed latency, and the saving against sizing every replica for the mean request.

```
uv run llmplan ui
```

Options: `--port 8501`, `--address localhost` (`0.0.0.0` in a container), `--headless`
(do not open a browser). It needs no secrets; `HF_TOKEN` (optional) is read server-side for
gated models, and `LLMPLAN_USAGE_LOG=PATH` enables the anonymous usage log (fields listed in
the page footer and docs/DEPLOY.md). Limits: 50 MB uploads (refused before parsing), a 30 s
solver time limit, 200,000 simulated requests, 30 plans per hour per session. The presets
are samples of the Azure 2023/2024 and BurstGPT traces in `llmplan/data/traces/samples/` (CC-BY-4.0;
see its README). Docker: `docker build -t llmplan . && docker run --rm -p 8501:8501
llmplan`; hosting steps in docs/DEPLOY.md.

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

**`llmplan gpus`** — the GPU catalog (`llmplan/data/gpus.yaml`, or `--gpus PATH`).

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

**`llmplan traces fetch`** — download a public trace listed in `llmplan/data/traces/manifest.yaml`
(`azure2023-code`, `azure2023-conv`, `azure2024-code`, `azure2024-conv`, `burstgpt-1`). It
refuses to run without `--yes`, verifies the recorded SHA-256, and caps downloads at 2 GiB.

```
uv run llmplan traces fetch azure2023-conv --dest ~/traces --yes
```

**`llmplan perf estimate`** — throughput, TTFT, and TPOT of one replica under a workload's
token distribution. `--backend auto` (default) interpolates published benchmark rows
(`llmplan/data/benchmarks/`) when they match the model, GPU, tensor parallelism, dtype, and request
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
model (fixture ids match through `llmplan/data/benchmarks/aliases.yaml`).

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
`--time-limit 60`, `--gpu-catalog PATH`, `--prices PATH`, `--format {text,json,vllm}`,
`--classes {1,2x2,3x3,...,fixed:<input edges>/<output edges>}` (request-size classes,
needs `--trace`; default `1`, one class sized for the mean request). Solves are
single-threaded and seeded, so the JSON output is byte-identical across runs. The fleet is
sized for the peak window only (no autoscaling yet).

With classes, the trace is cut into input x output token bins (`2x2`: median splits;
`fixed:1024,4096/256`: your edges, inclusive upper bounds), each class is estimated at its
own request shape and served only by candidates that meet the SLO for it, and the output
adds a class table and a routing table (the share of each class's requests per replica
type):

```
uv run llmplan plan --model fixture:llama3-8b --trace llmplan/data/traces/samples/azure2024_conv.csv --max-model-len 8192 --ttft-p95-ms 500 --tpot-p95-ms 50 --classes 2x2
```

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

Options: `--window 60` (seconds), `--routing
{auto,least_outstanding,round_robin,class_weighted}` (`auto`, the default, follows a
class plan's routing weights and is `least_outstanding` otherwise), `--kv-accounting
{incremental,full}` (`incremental`, the default, reserves a request's mean KV occupancy
and grows its KV as tokens are generated; `full` reserves input + output at admission),
`--ttft-p95-ms`, `--tpot-p95-ms` (budgets; default: the SLO recorded in the plan file, and
without either no violations are counted), `--png PATH` (four-panel figure, rendered
headless; its VRAM panel splits the fleet's memory into weights, KV in use and free),
`--gpu-catalog PATH` (the GPU catalog the plan used, for total VRAM; default: shipped),
`--requests-csv PATH` (one row per simulated request: arrival, start, completion, TTFT,
TPOT, E2E, replica, KV tokens, class), `--format {text,json}`. A plan with classes adds
one summary line (and JSON entry) per class. The plan can also be piped:
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
uv run pytest -m slow --no-cov   # 200k-request replay and every UI preset, excluded by default
uv run pytest -m docker --no-cov # docker build (M6 9.7), excluded by default; skipped without Docker
uv audit
uv build
```

The web UI tests run headless and offline with `streamlit.testing.v1.AppTest`. The M7
Mélange cross-check is reproduced with `uv run --with pulp==2.8.0 python
scripts/melange_crosscheck.py --melange-dir DIR` on a checkout of melange-release
(docs/milestones/M7_NOTES.md). The bundled
trace samples are regenerated with `uv run python scripts/make_samples.py TRACES_DIR` from
the full traces (`llmplan traces fetch NAME --dest TRACES_DIR --yes`, outside the repo).

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The bundled trace samples
in `llmplan/data/traces/samples/` are excerpts of the Azure LLM inference traces and
BurstGPT, redistributed under CC-BY-4.0 with attribution in NOTICE. Security reports:
[SECURITY.md](SECURITY.md). Contributions, including your own benchmark rows:
[CONTRIBUTING.md](CONTRIBUTING.md).
