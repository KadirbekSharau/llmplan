# llmplan — LLM Capacity Planner

CPU-only planner for self-hosted LLM inference: given a model, traffic, a latency target and
GPU prices, find the cheapest fleet and replica configuration, and show the utilization
timeline that proves it. No GPU required to run it.

Status: M1 implemented (model catalog, GPU/price catalogs, exact VRAM fit, CLI). Documents
drive the work:

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

Exit codes: 0 success, 2 usage/validation, 3 catalog/fetch error. Shipped fixtures:
`fixture:llama3-70b`, `fixture:llama3-8b`, `fixture:mistral-7b-v0.1`, `fixture:qwen2.5-7b`.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict llmplan
uv run pytest            # includes coverage (fails under 90%)
uv audit
uv build
```
