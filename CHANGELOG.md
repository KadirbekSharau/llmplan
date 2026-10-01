# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- M0: package scaffold, tooling (uv, ruff, mypy --strict, pytest with coverage), CI workflow,
  error hierarchy.
- M1: `ModelSpec` from Hugging Face config.json or fixtures, with a validated https-only
  fetcher; `llama_like` architecture (Llama, Mistral, Qwen2, Qwen3) with exact parameter
  counts; GPU and price catalogs (`data/gpus.yaml`, `data/prices.yaml`) with sources;
  exact weight and KV-cache arithmetic and the vLLM overhead model behind `fit()`; text and
  JSON renderers; CLI commands `llmplan fit`, `llmplan model-info`, `llmplan gpus`.
- M2: `Workload`, `WorkloadStats`, and `Distribution` models; trace parsers for generic CSV,
  Azure LLM inference 2023 and 2024, and BurstGPT with header detection and row validation;
  `compute_stats()` (peak windows, token percentiles, diurnal profile); a seeded synthetic
  trace generator; a checksum-verified public trace manifest and fetcher; CLI commands
  `llmplan workload stats`, `llmplan workload synth`, `llmplan traces fetch`.
- M3: performance model. `ReplicaConfig`, `PerfEstimate`, the `PerfBackend` protocol, and
  `llmplan.perf.estimate()` with `auto`/`roofline`/`table` backends; roofline bounds from
  M1 memory arithmetic and catalog bandwidth/TFLOPS; benchmark table (`data/benchmarks/`)
  with a load-time physical-bound check and log-concurrency interpolation; 105 seed rows
  from the NVIDIA NIM performance page (Llama-3.1-70B on H100, Llama-3.1-8B on H100 and
  L40S); CLI commands `llmplan perf estimate` and `llmplan perf benchmarks`.
- M4: MILP fleet planner. `SLO`, `PlanOptions`, `PlanRequest`, `PlanResult` and
  `llmplan.planner.plan()`: candidate enumeration over price rows x tensor parallelism x
  dtype x `max_num_seqs` with M1 fit, M3 estimates, SLO checks and dominance pruning; an
  OR-Tools MathOpt formulation (GPU packing, request and token demand, optional
  homogeneous fleet) solved deterministically with HiGHS, CP-SAT (integer-scaled), SCIP or
  Gurobi; the best homogeneous fleet as a baseline; the binding constraint from an LP
  relaxation; `vllm serve` command lines per replica; CLI command `llmplan plan`.
  `llmplan perf estimate --trace` now computes statistics from a trace, and the workload
  stats renderers moved into `llmplan.render` (output unchanged).
- M5: trace replay and timeline. `llmplan.simulate.replay()` runs a heap-based
  discrete-event simulation of a workload on a `PlanResult`'s fleet (slots from the perf
  estimate, KV-token admission from the M1 fit, per-replica FIFO queues, `least_outstanding`
  or `round_robin` routing) and returns a `Timeline` of per-window demand, utilization, KV
  in use, queue depth, latency p95s and SLO violations with a summary;
  `replay_requests()` gives the per-request records. `load_plan_json()` reads `llmplan plan
  --format json` output. Four-panel PNG (matplotlib, Agg), text and JSON renderers; CLI
  command `llmplan simulate`. The M3 acceptance tests import the real `WorkloadStats`.
- M6: Streamlit web UI (`llmplan ui`, `llmplan/ui/`): model picker (fixtures and Hugging
  Face ids), traffic from bundled public trace samples, a CSV upload (50 MB, parsed in
  memory via the new `InMemoryTrace`) or synthetic traffic, SLO inputs, an editable price
  table validated through `PriceRow`, and results with the answer card, fleet and replica
  tables, `vllm serve` lines, the replay timeline, candidates, assumptions and JSON
  downloads; per-session plan brake and an opt-in anonymous usage log
  (`LLMPLAN_USAGE_LOG`). Five CC-BY-4.0 trace samples in `data/traces/samples/` cut by
  `scripts/make_samples.py`; `scripts/usage_summary.py`; Dockerfile, docs/DEPLOY.md and
  docs/LAUNCH.md. Timeline windows record `vram_bytes_total` (`replay(..., gpus=...)`,
  `llmplan simulate --gpu-catalog`) and the plot's VRAM panel shows weights, KV in use and
  free.
- M7: request-size demand classes. `llmplan.workload.classify()` (quantile or fixed
  input x output token bins, classes under 1% merged, demand in the fleet-wide peak
  windows) and `DemandClass`; `PlanRequest.classes`; the MILP gains per-class allocation
  variables (milli-replica integers for CP-SAT) with per-class SLO eligibility and
  estimates at each class's shape, a routing LP over the chosen fleet, and
  `PlanResult.classes`, `routing` (`RoutingRule`) and `class_binding`; one class
  reproduces M4 exactly. `llmplan plan --classes`; class and routing tables in the text
  output. Simulation: `class_weighted` routing (deterministic deficit rule), per-class
  summaries in the `Timeline`, `class_index` in the request log, incremental KV accounting
  (default; `kv_accounting="full"` keeps M5's), `llmplan simulate --routing auto
  --kv-accounting --requests-csv`. Web UI: request-size classes selector (2x2 default),
  routing table and "saving from request-size routing". Validation: a brute-force
  exactness test over 50 random instances and `scripts/melange_crosscheck.py` (results in
  docs/milestones/M7_NOTES.md).
