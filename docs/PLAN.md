# PLAN.md — LLM Capacity Planner

Owner: Founder (product) / CTO agent (specs). Implementation is delegated to developer agents.
Read order for a new agent: this file, then ARCHITECTURE.md, then MILESTONES.md, then the
design doc for the milestone you were assigned (docs/milestones/M<N>_DESIGN.md).
Background and the reasoning behind these choices: docs/CTO_ASSESSMENT.md (optional).

---

## 1. What we are building

A CPU-only planning tool that answers, for self-hosted LLM inference:

> "Given this model, this traffic, this latency target, and these GPU prices, what is the
> cheapest fleet and per-replica configuration that meets the target, and what does its
> utilization look like over time?"

It is an offline planner, not a runtime system. It never runs on a customer's cluster, never
touches a GPU, and never sits in the request path.

Working package name: `llmplan`. CLI: `llmplan`. (Repo directory name is a legacy name.)

## 2. Why this and not something bigger

- The founder has no GPUs and no budget to rent any. The planner needs neither.
- The core of the planner is a mixed-integer program. That is the technical interest driving
  the project.
- Academic prior art proves both the formulation (Mélange, UC Berkeley: ILP over GPU
  heterogeneity, up to 77% cost reduction) and the CPU-only feasibility (Vidur, Microsoft:
  simulator with <9% latency error). Neither is a usable product.
- Existing online calculators answer "will the model load?" They do not answer "what is the
  cheapest fleet that meets my SLO under my real traffic?"
- Runtime products (telemetry agents, routers, schedulers) are deferred. See CTO_ASSESSMENT.md
  sections 3 and Revision 2 for why.

## 3. Users and the questions they ask

Primary user: an engineer at a company self-hosting open-weight models on vLLM (or similar),
sizing a deployment. Evidence: sizing questions on discuss.vllm.ai, vLLM GitHub issues,
r/LocalLLaMA, Hacker News.

The three question shapes we must answer, in priority order:
1. **Fit.** "Does model M in dtype D fit on GPU G with tensor parallel T, and how many
   concurrent requests at context length L can it hold?" (Milestone 1)
2. **Size.** "For traffic that looks like this and a p95 TTFT / TPOT target of X, how many
   replicas of which GPU do I need, and what do I set max_num_seqs / gpu_memory_utilization /
   max_model_len to?" (Milestones 2 to 4)
3. **Cheapest.** "Across GPU types and providers, what mix minimizes $/day for that traffic
   and SLO?" (Milestone 4, with the timeline proof in Milestone 5)

## 4. Deliverables and interfaces

Three layers, built in this order. Each is a thin adapter over the one below it.

| Layer | What it is | Who uses it | Milestones |
|---|---|---|---|
| Library `llmplan` | pip-installable Python package; every capability is a pure function returning a frozen model | the CLI, the UI, future API, partner integrations, tests | M1 to M5 |
| CLI `llmplan` | one subcommand per capability (`fit`, `workload`, `perf`, `plan`, `simulate`); text tables, JSON, PNG plots | engineers scripting sizing; CTO verification of each milestone; forum power users | each milestone ships its command |
| Web UI | Streamlit single-page app: model picker, trace preset/upload/synthetic, SLO inputs, editable price catalog, results with vLLM command lines and the utilization timeline | the public launch audience (vLLM forum, r/LocalLLaMA, Hacker News); free, hosted, no login | M6 |

The web UI is the launch deliverable. The CLI and library exist so the UI contains no logic
and so every milestone is verifiable end to end without a UI. A hosted HTTP API is a likely
follow-on and is one more adapter, not a rewrite.

## 5. Goals

- G1: Exact, testable VRAM arithmetic from a model's Hugging Face config.json. No estimates
  where an exact formula exists.
- G2: Reproduce a published Mélange-style result on a public trace to validate the MILP.
- G3: Every answer comes with an explanation: which constraint was binding (VRAM, throughput,
  or SLO) and by how much.
- G4: A ready-to-paste vLLM command line for every recommended replica.
- G5: A public, free web UI within roughly six weeks of implementation effort, launched on
  the forums where the questions are asked.

## 6. Non-goals (do not build these, even if asked nicely by a test or a comment)

- Anything that connects to a Kubernetes cluster, a GPU, or a live inference server.
- Online request routing or scheduling.
- Reimplementing an inference engine or its batching logic.
- Training workloads.
- Non-NVIDIA accelerators in v1 (the catalog is extensible; do not spend time on it).
- MIG partitioning as an optimization variable in v1.

## 7. Principles for implementing agents

1. **Exact before empirical.** VRAM is arithmetic; throughput is empirical. Keep them in
   separate modules with separate confidence labels.
2. **Every number has a source.** Hardware specs, prices, and benchmark rows carry a
   `source_url` and `as_of` date. Unknown fields are `null`, never guessed.
3. **Solver-agnostic modeling.** The MILP is written against OR-Tools MathOpt so the backend
   (HiGHS default; SCIP, CP-SAT, Gurobi optional) is a one-line switch.
4. **Python only, CPU only.** No Go, no C++, no CUDA, no network calls in tests.
5. **Tests are the spec.** Each milestone design doc lists acceptance tests with expected
   values. A milestone is done when they pass, not when the code "looks right."
6. **Small public API per module.** Downstream milestones depend on the interfaces in
   ARCHITECTURE.md, not on internals.
7. **Do not widen scope.** If a milestone needs something from a later milestone, stub it
   behind the documented interface and leave a TODO referencing the milestone number.

## 8. Success metrics

- M1 to M5: acceptance tests in each design doc pass; the Mélange reproduction is within
  10% of the paper's reported cost for the conversational scenario.
- Launch (M6): at least 200 distinct plan runs in the first 30 days; at least 20 inbound
  comments or issues describing a real deployment.
- Longer term: users asking for "connect my Prometheus" is the signal to revisit the
  deferred runtime products.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Throughput model is wrong (the only empirical part) | Isolate in one module; label confidence; support Vidur as an alternative backend; validate against published benchmark points |
| Model architectures we can't parse (MoE, MLA, hybrid attention) | Explicit `UnsupportedArchitecture` error with the field that failed; user-supplied override for parameter count |
| Price catalogs go stale | `as_of` on every row; UI shows the date; catalog is editable |
| Scope creep toward runtime systems | Non-goals list above; CTO review at each milestone boundary |

## 10. Tooling and repo conventions

- Python 3.11+, `uv` for environments, `pyproject.toml` with `hatchling` build.
- `ruff` (lint + format), `pytest`, `mypy --strict` on `llmplan/` (allow `# type: ignore`
  only with a comment).
- Layout: `llmplan/` package, `tests/`, `data/` (catalogs, fixtures), `docs/`.
- No network access in unit tests. Hugging Face config fetches are behind an interface with
  a fixture-backed fake.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).
