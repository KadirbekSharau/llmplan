# MILESTONES.md — LLM Capacity Planner

Each milestone is independently assignable to a developer agent. A milestone is done when
its acceptance tests pass, CI is green, and every rule in docs/DEFINITION_OF_DONE.md is
satisfied (the CTO review checklist at the bottom is a summary of those rules). Design docs live in `docs/milestones/M<N>_DESIGN.md`. Status is tracked in the table below.

| Milestone | Status | Branch / merge |
|---|---|---|
| M0 | done | main (0824153) |
| M1 | in progress | main |
| M2 | designed | m2-workload |
| M3 | designed | m3-perf |
| M4 | not designed | |
| M5 | not designed | |
| M6 | not designed | |

Effort estimates assume one capable agent working continuously and include tests and docs.

---

## M0 — Repo scaffold (part of M1's first PR)

Deliverables: `pyproject.toml`, `uv.lock`, `ruff`/`mypy`/`pytest` config, CI workflow,
`llmplan/__init__.py`, `errors.py`, `types.py`, empty package dirs from ARCHITECTURE.md
section 3, `README.md` pointing to docs.
Done when: `uv run pytest` passes with one trivial test; `mypy --strict llmplan` passes.

## M1 — Catalog and exact VRAM fit  (effort: ~2 days)

Answers question shape 1: "Does model M fit on GPU G with TP T, and how many concurrent
sequences at context L?"

Scope:
- `ModelSpec` from HF config.json (fetch behind protocol) or fixtures.
- `llama_like` architecture: exact parameter count and KV bytes per token, GQA/MQA aware,
  tensor-parallel aware.
- `GPUSpec`/`PriceRow` catalogs from YAML with source URLs.
- `EngineProfile` for vLLM with a documented overhead model.
- `fit()` and CLI `llmplan fit`, `llmplan model-info`, `llmplan gpus`.
Out of scope: MoE, MLA, throughput, traces, solver.
Acceptance: docs/milestones/M1_DESIGN.md section 9.

## M2 — Workload ingestion and characterization  (effort: ~2 days)

Scope:
- `Workload` schema: rows of (arrival_s, input_tokens, output_tokens, optional model/tenant).
- Format registry: generic CSV, Azure LLM Inference 2023 and 2024, BurstGPT. Loaders
  download nothing; they parse a local path the user provides. A `llmplan traces fetch`
  helper may download the public files with explicit user consent and checksum verification.
- Stats: requests/s over sliding windows, p50/p95/p99 input and output tokens, peak window
  detection, diurnal profile (hour-of-day buckets), synthetic trace generator (Poisson
  arrivals with configurable token distributions) for users with no trace.
Acceptance: parses each public trace sample fixture; stats match hand-computed values on a
50-row fixture; synthetic generator is deterministic under a seed.

## M3 — Performance model  (effort: ~3 days; highest uncertainty)

Scope:
- `PerfBackend` protocol: given (model, gpu, tp, replica config, workload stats) return
  throughput (tokens/s, requests/s) and latency estimates (TTFT p50/p95, TPOT p50/p95) with a
  confidence label.
- `table` backend: benchmark rows in `data/benchmarks/*.yaml` with `source_url`, `as_of`,
  engine version, and interpolation on batch size and context length. Seed from NVIDIA
  inference benchmarking posts, vLLM published benchmarks, InferenceMAX.
- Roofline sanity bound: decode is memory-bandwidth bound; TPOT lower bound =
  weight_bytes / bandwidth per token per replica. Table values below the bound are rejected
  at load time.
- Optional `vidur` backend behind a lazy import; not required for acceptance.
Acceptance: interpolated values reproduce held-out benchmark rows within 15%; roofline
bound rejects a deliberately wrong fixture row.

## M4 — MILP planner  (effort: ~4 days)

Scope:
- OR-Tools MathOpt formulation. Decision variables: integer count of each (gpu type,
  provider/price row, tp degree, replica config) tuple; optional time-of-day scaling
  buckets. Objective: minimize $/day. Constraints: per-window demand <= sum of replica
  throughput, VRAM fit (from M1), SLO estimate (from M3) holds.
- Backends: HiGHS default; SCIP, CP-SAT (with integer-cent costs), Gurobi optional.
- Deterministic solves: explicit time limit, single thread default, seed.
- `PlanResult` with the fleet, per-replica config, cost vs. naive baseline (all
  single-cheapest-GPU-that-fits), and a binding-constraint explanation derived from slack
  and duals of the LP relaxation.
- Renderer: vLLM command line per replica.
Acceptance: reproduce the Mélange conversational scenario within 10% of reported cost using
its public GPU set and prices; infeasible inputs raise `InfeasiblePlan` with a reason.

## M5 — Simulation and timeline  (effort: ~3 days)

Scope:
- Discrete-event replay of the workload on the planned fleet using the M3 model for
  service times; queueing per replica; simple least-loaded routing.
- `Timeline`: per-replica utilization, VRAM split (weights/KV/free), queue depth, SLO
  violations per window.
- Plot renderer (matplotlib, headless) and JSON export.
Acceptance: replay of a plan that M4 marks feasible produces SLO violations under 5% of
requests; a deliberately under-provisioned fleet produces visible violations.

## M6 — Streamlit UI and launch  (effort: ~3 days)

Scope:
- Single-page app: model picker (HF id or fixture), trace picker (public preset, upload,
  synthetic), SLO inputs, price catalog editor, run, results with timeline and command
  lines, download JSON.
- Upload limits and validation per ARCHITECTURE.md security section.
- Anonymous usage log: input shapes only (model id, GPU ids, trace stats), no uploads
  retained.
- Launch posts drafted for Show HN, r/LocalLLaMA, discuss.vllm.ai (founder posts).
Acceptance: end-to-end run on the Azure 2024 preset completes under 60 s on a laptop CPU.

---

## Dependencies

```
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6
             \__________/   (M2 and M3 can proceed in parallel after M1)
```

## CTO review checklist (every milestone)

- [ ] Acceptance tests from the design doc exist and pass unchanged (no relaxed tolerances).
- [ ] No new dependency without a one-line reason in the PR.
- [ ] Package growth is proportional to scope; no `utils.py`.
- [ ] Public API matches ARCHITECTURE.md section 5, or ARCHITECTURE.md was updated in the PR.
- [ ] Every catalog/benchmark row added has `source_url` and `as_of`.
- [ ] No network in tests; no secrets; `yaml.safe_load` only; `uv audit` clean.
- [ ] Error messages name the offending field or id.
- [ ] README usage section updated for any new CLI command.
