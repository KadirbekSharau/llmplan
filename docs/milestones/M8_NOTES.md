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
- **Step 2 — Where the comparison lives.** `llmplan/simulate/compare.py` plans the request
  without classes and, given the trace, replays that fleet (least-outstanding routing, the
  request's SLO budgets); it sits in `simulate` because it needs both `plan` and `replay`.
  The UI (`state.run_plan`) and `llmplan plan --classes` both use it, so the CLI text gains
  a "Request-size routing" section before the class table, and the CLI JSON a
  `class_comparison` object (only for class plans; K=1 output is byte-identical, which the
  M7 golden test checks). `load_plan_json` ignores that object, so `llmplan simulate
  --plan` still reads class plans. `PlanRun` gains `single_class_ttft_violation_pct` and a
  `comparison` property.
- **Step 2 — Sentence choice.** Costs are compared to the cent. Class plan dearer: when the
  single-class replay has TTFT violations, "Sized for the mean request this fleet would
  cost $A/day, but the replay shows it would miss the latency target for V% of requests.
  The class-sized plan costs $B/day."; when it was not replayed, or replays with 0.0%
  violations, the clause is "but it would under-provision the long-request class" (the
  conservative reading: claiming a miss the replay does not show would be false). Cheaper:
  "Request-size routing saves $X/day (Y%) versus sizing every replica for the mean
  request." Equal: "Request-size routing does not change the fleet for this traffic." No
  single-class fleet: "No fleet sized for the mean request meets the target; the
  class-sized plan costs $B/day."
- **Step 2 — Measured on the M7 Azure sample** (`azure2024_conv.csv`, UI default SLO 500 ms /
  50 ms, all GPUs): class plan $83.76/day (1 x H100), single-class plan $44.66/day (1 x
  L40S), and the single-class fleet's replay at the sample's own rate (peak 4.18 req/s)
  shows 0.0% TTFT violations, so the text reads "... but it would under-provision the
  long-request class". (M7_NOTES.md's 20.2% violations were measured on the same trace with
  arrivals compressed 21.9-fold; the replay's per-replica TPOT is the whole-workload
  estimate, so it cannot show the long classes' TPOT misses that make the class plan
  dearer.)
- **Step 2 — No negative percentage.** The baseline saving is printed as "saving Y%", or
  "baseline Y% cheaper" when a time-limited solve returns a fleet dearer than its baseline
  (the only way it can be negative); the UI metric follows the same rule. The fleet headroom
  line already clamps at +0.0%. Test 9.2 checks the plan text and every text and metric of
  the UI page against `-\d[\d,.]*%`.
- **Step 2 — Tests changed for the wording.** `tests/unit/ui/test_app.py::
  test_request_size_classes_default_to_2x2_and_show_the_routing` asserted the old
  "Saving from request-size routing" headline; it now asserts the M8 sentence (on H100
  alone both plans buy one H100: "does not change the fleet for this traffic"), and
  `tests/unit/ui/test_state.py::test_run_plan_with_classes_routes_by_weight_and_compares`
  called the removed `views._saving`; it now checks the M8 sentences of the same two runs.
  No M1 to M7 acceptance test asserted the old wording.

## Deviations from the design doc

- **Step 1 — `perf_confidence` and `perf_sources` are properties of `PlanResult`, not
  serialized fields.** Both are derived from the chosen replicas' `PerfEstimate`s, which the
  plan already carries; as fields they would add two keys to every plan JSON and break the
  byte-for-byte pre-M7 golden comparison of M7 acceptance test 8.1, which must pass
  unchanged. `ModelSpec.attention` and `PlanRun.class_saving_pct` set the precedent.
  ARCHITECTURE.md section 4 documents them.

## Questions for founder

None beyond docs/FOUNDER_QUESTIONS.md item 7 (PyPI account and trusted publishing).
