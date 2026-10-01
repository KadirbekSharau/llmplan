# M8 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the deliverables
order of M8_DESIGN.md section 2; "9b" is the carry-over item.

## Implementation notes

- **9b — Carry-over done first.** CI is two jobs: `check` (lint, format, mypy, the default
  pytest selection, audit, build, CLI entry point) and `ui` (the Streamlit AppTest suite
  with its own coverage gate on `llmplan/ui/*`, then the `slow` tests). The AppTest modules
  (`tests/unit/ui/test_app.py`, `tests/acceptance/test_m6.py`) carry
  `pytestmark = pytest.mark.ui`; `addopts` deselects `ui` (and the new `network` marker) by
  default like `slow` and `docker`. Tests over two seconds in the baseline run moved to
  `slow`: `test_vram_split_with_and_without_the_gpu_spec` (3.5 s),
  `test_8_5_determinism` (2.6 s) and `test_simulate_json_flags_override_and_png` (2.5 s).
  Locally: `uv run pytest` is the `check` selection, `uv run pytest -m ui` the AppTest
  suite, `uv run pytest -m slow --no-cov` the slow tests.
- **Step 1 — Confidence sentences (`llmplan/perf/confidence.py`).** One function builds the
  sentence for the plan text (a `Confidence  ...` line directly under `Cost`, two spaces
  after the label as in the design's example, since `Confidence` fills the 10-character
  label column), `llmplan perf estimate` text (under `Backend`) and the UI banner:
  `roofline (uncalibrated first-principles model; expect ±30% on throughput)`,
  `measured (NVIDIA NIM performance page, as of 2026-09-30)`, `interpolated between
  measured rows (...)`, and `mixed (some replicas use measured rows, others the roofline
  model, which is uncalibrated ...; measured rows: ...)`. A source is named from a URL
  prefix table (only the NIM page today; any other URL is printed as is) and dated with
  the newest `as_of` of the shipped rows carrying that URL. "±30%" is written with the
  plus-minus sign, so no `-` precedes a percentage (section 4).
- **Step 1 — UI banner.** `views.confidence_banner` renders `st.warning` for roofline and
  mixed, `st.info` for measured and interpolated, above the cost cards, followed by a
  "What the confidence means" expander listing `BANDWIDTH_EFFICIENCY`, `PREFILL_MFU`,
  `DECODE_MFU` and `P95_FACTOR` with their values. On the bundled presets the default
  plans are roofline (no shipped row is near their request shapes).
- **Step 1 — Test double.** `FakePerf` (tests/fake_planner.py) gains `confidence`
  (default `"measured"`, the previous hard-coded value), so test 9.1 can mix confidences.

## Deviations from the design doc

- **Step 1 — `perf_confidence` and `perf_sources` are properties of `PlanResult`, not
  serialized fields.** Both are derived from the chosen replicas' `PerfEstimate`s, which the
  plan already carries; as fields they would add two keys to every plan JSON and break the
  byte-for-byte pre-M7 golden comparison of M7 acceptance test 8.1, which must pass
  unchanged. `ModelSpec.attention` and `PlanRun.class_saving_pct` set the precedent.
  ARCHITECTURE.md section 4 documents them.

## Questions for founder

None beyond docs/FOUNDER_QUESTIONS.md item 7 (PyPI account and trusted publishing).
