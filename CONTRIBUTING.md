# Contributing to llmplan

Thank you for helping. The two most useful contributions are **benchmark rows** from your
own deployments and **issues describing a real sizing question** the tool gets wrong.

## Contributing benchmark rows

Every throughput number in llmplan has a source; most models and GPUs today only have the
uncalibrated roofline model ("expect ±30%"). Your measurements fix that.

1. Run `vllm bench serve ... --save-result` (add `--metric-percentiles 50,95,99` to record
   p95 latencies), or write the rows as an llmplan CSV with the columns
   `model_id,gpu_id,engine,engine_version,tensor_parallel,dtype,concurrency,input_len,`
   `output_len,output_tokens_per_s,ttft_ms_p50,ttft_ms_p95,tpot_ms_p50,tpot_ms_p95`
   (latency columns may be empty).
2. Check them against your model: in the web UI under "Calibrate with your own benchmarks",
   or with `llmplan plan ... --benchmarks results.json --benchmarks-gpu h100-sxm-80gb`
   (`llmplan perf estimate` takes the same flags). Rows faster than the physical floor of
   the GPU, or for another model, are rejected with the reason.
3. Click "Contribute these rows" in the UI (it opens a prefilled GitHub issue; nothing is
   sent until you submit it) or open an issue yourself with the CSV, and fill in the
   engine version, driver and CUDA version, the date and the benchmark command.

Maintainers add accepted rows to `llmplan/data/benchmarks/<gpu-id>.yaml` with a
`source_url` and `as_of` date; the shipped table is validated on every load.

## Contributing code

Read `docs/PLAN.md` and `docs/ARCHITECTURE.md` first. In short (the full checklist is
`docs/DEFINITION_OF_DONE.md`):

- Python 3.11+, managed with `uv`. Frozen pydantic v2 models at boundaries, pure functions
  inside, units in field names, typed errors from `llmplan/errors.py`, no `utils.py`.
- No new runtime dependency without a one-line reason. `yaml.safe_load` only; no pickle,
  eval, exec or subprocess in the package; no network in tests; no secrets anywhere.
- Every change comes with tests; acceptance values in `tests/acceptance/` are not relaxed.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`, `ci:`), one logical
  change each. Update README (usage), `docs/ARCHITECTURE.md` (interfaces) and
  `CHANGELOG.md` with the change.

Before opening a pull request, run:

```
uv sync --locked --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict llmplan
uv run pytest                       # unit and acceptance tests (CI job `check`)
uv run pytest -m ui                 # the Streamlit AppTest suite (CI job `ui`)
uv run pytest -m slow --no-cov      # multi-second tests, including the wheel install test
uv audit
```

By contributing you agree that your contribution is licensed under the Apache License,
Version 2.0 (see `LICENSE`).
