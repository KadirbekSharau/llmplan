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
