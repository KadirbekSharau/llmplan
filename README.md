# llmplan — LLM Capacity Planner

CPU-only planner for self-hosted LLM inference: given a model, traffic, a latency target and
GPU prices, find the cheapest fleet and replica configuration, and show the utilization
timeline that proves it. No GPU required to run it.

Status: pre-implementation. Documents drive the work:

- [docs/PLAN.md](docs/PLAN.md) — what, why, goals, non-goals, principles
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — engineering standards, package layout, data models, interfaces
- [docs/MILESTONES.md](docs/MILESTONES.md) — M0 to M6, acceptance criteria, review checklist
- [docs/milestones/M1_DESIGN.md](docs/milestones/M1_DESIGN.md) — detailed design for the first milestone
- [docs/CTO_ASSESSMENT.md](docs/CTO_ASSESSMENT.md) — background reasoning (optional reading)

Implementing agents: start with PLAN.md, then ARCHITECTURE.md, then your milestone's design doc.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
uv sync --locked
uv run ruff check && uv run ruff format --check
uv run mypy --strict llmplan
uv run pytest
uv audit
```
