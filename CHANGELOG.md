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
