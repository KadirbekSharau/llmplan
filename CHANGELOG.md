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
- M3: performance model. `ReplicaConfig`, `PerfEstimate`, the `PerfBackend` protocol, and
  `llmplan.perf.estimate()` with `auto`/`roofline`/`table` backends; roofline bounds from
  M1 memory arithmetic and catalog bandwidth/TFLOPS; benchmark table (`data/benchmarks/`)
  with a load-time physical-bound check and log-concurrency interpolation; 105 seed rows
  from the NVIDIA NIM performance page (Llama-3.1-70B on H100, Llama-3.1-8B on H100 and
  L40S); CLI commands `llmplan perf estimate` and `llmplan perf benchmarks`.
