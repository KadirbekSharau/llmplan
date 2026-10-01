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
- **9b — Two coverage gates.** The Streamlit page modules (`ui/app.py`, `ui/views.py`,
  `ui/calibrate.py`) run only under AppTest, so `[tool.coverage.report] omit` leaves them
  out of the default run's 90% gate, and the `ui` job gates all of `llmplan/ui/*` at 90%
  with `coverage report --include='llmplan/ui/*' --omit='llmplan/ui/__init__.py'` (an
  explicit `--omit` replaces the configured list; `--omit=''` does not).
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
- **Step 3 — vLLM benchmark JSON, verified.** Read on 2026-10-01 from vLLM **v0.30.0**
  (tag `v0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, the latest release
  tag), `vllm/benchmarks/serve.py` (the `vllm bench serve` implementation; the older
  `benchmarks/benchmark_serving.py` wrote the same keys). The saved object is the run's
  setup (`date`, `endpoint_type`, `backend`, `label`, `model_id`, `tokenizer_id`,
  `num_prompts`, `request_rate`, `burstiness`, `max_concurrency`, lines 2273-2297) merged
  with the result (`duration`, `completed`, `failed`, `total_input_tokens`,
  `total_output_tokens`, `request_throughput`, `request_goodput`, `output_throughput`,
  `total_token_throughput`, `max_output_tokens_per_s`, `max_concurrent_requests`, `rtfx`,
  lines 1286-1306) and, per metric in `--percentile-metrics` (default `ttft,tpot,itl`),
  `mean_<m>_ms`, `median_<m>_ms`, `std_<m>_ms` and `p<P>_<m>_ms` for each `P` in
  `--metric-percentiles` (default `99` only; lines 1392-1397). Per-request lists
  (`input_lens`, `ttfts`, ...) are dropped unless `--save-detailed` (lines 2395-2407);
  `--append-result` appends one JSON object per line (line 2420), which the parser also
  accepts. `output_throughput` is `sum(output lens) / duration` (line 745), the aggregate
  output tokens/s `BenchmarkRow.output_tokens_per_s` means. Mapping implemented:

  | vLLM key | BenchmarkRow field |
  |---|---|
  | `model_id` | `model_id` (must name the model being planned, directly or by alias) |
  | `max_concurrency` (null: `num_prompts`, with a note) | `concurrency` |
  | `round(total_input_tokens / completed)` | `input_len` |
  | `round(total_output_tokens / completed)` | `output_len` |
  | `output_throughput` | `output_tokens_per_s` |
  | `median_ttft_ms`, `median_tpot_ms` | `ttft_ms_p50`, `tpot_ms_p50` |
  | `p95_ttft_ms`, `p95_tpot_ms` (only with `--metric-percentiles` including 95) | `ttft_ms_p95`, `tpot_ms_p95` |
  | `p99_*`, `mean_*`, `std_*`, `*_itl_ms`, everything else | ignored |
  | (user) GPU, tensor parallel, dtype, vLLM version | `gpu_id`, `tensor_parallel`, `dtype`, `engine_version`; `engine = "vllm"` |

  `max_concurrency` is null when the run had no client-side limit (only a request rate);
  `num_prompts` is then an upper bound on the requests in flight, and the upload notes say
  so, as the design asks.
- **Step 3 — Run details the file lacks.** CLI: `--benchmarks-gpu` (required for a vLLM
  JSON in `plan`; `perf estimate` defaults to its own `--gpu`), `--benchmarks-tp` (default
  1, or `--tp` in `perf estimate`), `--benchmarks-dtype` (default bf16, or `--dtype`),
  `--benchmarks-engine-version` (default `unknown`, since `engine_version` is a required
  string). UI: a "vLLM JSON details" expander under the uploader with the same four fields.
  For an llmplan CSV they are ignored (the CSV has the columns).
- **Step 3 — Validation.** Exactly the shipped checks where they apply: schema
  (`BenchmarkRow`), `gpu_id` in the GPU catalog, tensor parallelism dividing the heads, and
  the physical floor (`benchmarks.row_problem`, factored out of the shipped loader so the
  messages are identical). The fixture-alias FK of shipped rows becomes "the row's model
  must be the model being planned" (resolved through the shipped `aliases.yaml`), because
  the physical floor needs that model's architecture and the rows exist to calibrate it.
  File-level problems (over 5 MB or 500 rows, not UTF-8, missing or unknown CSV columns,
  invalid JSON, a vLLM JSON without a GPU id) raise `ValidationError` (CLI exit 2, one UI
  error box); row problems are listed with their 0-based index. The CLI prints the counts,
  each rejection and each note to stderr, so stdout (text or JSON) is unchanged.
- **Step 3 — Session-only.** The CLI reads the file once; the UI keeps the bytes in the
  uploader and the validated report in `st.session_state` (`BenchmarkFile.data` is left out
  of `repr`). Rows reach the planner as `upload_backends(rows)` through the new
  `plan(request, *, backends=...)` keyword, threaded to every `perf.estimate` call
  (candidates and request-size classes) and to `run_plan`. The UI's plan cache key adds the
  rows' JSON, so a cached plan made with uploaded rows is only reused for the same rows;
  the cached `PlanRun` holds estimates, not the rows. Nothing is logged: the usage log's
  fields are unchanged (M6 test 9.6).
- **Step 3 — Which rows answer.** Uploaded rows are tried first for a (model, GPU, tensor
  parallel, dtype) match; the shipped rows of the same match are used only when the
  uploaded ones cannot answer (no shape within 2x, or no TTFT and no TFLOPS). Merging them
  into one pool would let the table's "lowest throughput per concurrency" rule mix a
  visitor's own measurement with a NIM row, and test 9.3 requires
  `source_urls == ("user-upload",)`.
- **Step 3 — Contribute.** `contribute_url(rows)` builds
  `https://github.com/KadirbekSharau/llmplan/issues/new?title=...&body=...` (query encoded
  with `/`, `:` and `,` left unescaped, which RFC 3986 allows in a query, to keep CSV rows
  short): a model / GPU / tensor-parallel / row-count header, the rows as an llmplan CSV in
  a code block, and a checklist for engine version, driver and CUDA, date and benchmark
  command. Over 6,000 characters it returns the rows-free URL and `False`; the UI then
  offers the CSV download next to the issue link. The UI shows it after a plan that used an
  upload (`st.link_button`, a new tab); nothing is sent automatically.
- **Step 3 — UI modules.** The sidebar section and the post-plan report live in
  `llmplan/ui/calibrate.py` (a third module allowed to import Streamlit; the static test's
  allow-list in `tests/unit/ui/test_presets.py` gained it). To keep `app.py` under the M6
  design's 400 lines (now 398), the page title and intro moved to `views.header()`. The
  banner's expander links to the section with `[How to calibrate](#calibrate)`.
- **Step 4 — MoE registry members (`llmplan/catalog/architectures/moe.py`).** `mixtral`
  (`MixtralForCausalLM`) and `qwen_moe` (`Qwen3MoeForCausalLM` with `qk_norm`,
  `Qwen2MoeForCausalLM` with q/k/v bias) share one formula class; the per-part formulas
  (attention, norms, dense MLP, embeddings) moved to functions in `llama_like.py` so both
  families count attention identically. Expected values reproduce exactly:
  Mixtral-8x7B 46,702,792,704 / 12,879,925,248, Qwen3-30B-A3B 30,532,122,624 /
  3,353,032,704, bf16 weights 61,064,245,248, KV 131,072 and 98,304 bytes per token.
  `active_params` counts embeddings and lm_head as active (as the official "active"
  figures do), so a dense model's active count equals its total.
- **Step 4 — Loading.** Each family reads its own keys through the new
  `Architecture.config_fields(model_id, raw)` (dense: none); `moe_layer_indices` is derived
  there (Mixtral: every layer; Qwen: layer `i` when `i` is not in `mlp_only_layers` and
  `(i + 1) % decoder_sparse_step == 0`, as transformers does). A missing required key is
  `UnsupportedArchitecture(field=<key>)`, a wrong type `CatalogError` naming the field, as
  for the common keys. `shared_expert_intermediate_size` is read for Qwen2-MoE only (the
  design's "Qwen2-MoE only, 0 if absent"). `ModelSpec` validates that `1 <= k <= E`,
  that `moe_intermediate_size` is set, and that the MoE layers are distinct, sorted and in
  range, and that a dense spec has none of them.
- **Step 4 — Memory and roofline.** Weights count every expert (`count_params`), KV is
  unchanged. `perf.roofline.decode_weight_bytes(model, dtype, tp, batch, per_gpu)` is the
  bytes one GPU reads per decode step: dense models return `per_gpu` unchanged (so every
  dense estimate is bit-identical to M7), MoE models `(non_expert + expert * min(1, B * k /
  E)) / tp` with `expert = expert_weight_bytes(model, dtype)` (routed experts only; the
  router and Qwen2's shared expert are non-expert). Decode and prefill FLOPs use
  `active_param_count`, in the roofline, the table backend's roofline prefill and the
  benchmark physical floor (which also reads only the touched experts). Every MoE
  estimate states the assumption (uniform routing, even tensor-parallel split of experts),
  and `fit()` notes that all experts are resident. A MoE model with
  `param_count_override` has no known expert split and is treated as dense (active =
  total, all weights read), the conservative reading.
- **Step 4 — MLA.** `DeepseekV2ForCausalLM` and `DeepseekV3ForCausalLM` raise
  `UnsupportedArchitecture` ("multi-head latent attention (MLA) is not modeled",
  `field="architectures"`) before any key is read; `fixture:deepseek-v3` (architectural
  integers of the public config) is the test case, excluded from the UI picker like gpt2.
- **Step 4 — Fixtures and UI.** `mixtral-8x7b.json` and `qwen3-30b-a3b.json` carry only the
  integers the loader reads (values from the public configs; Qwen3's explicit `head_dim`
  128 differs from 2048 / 32, which `fit()` notes as trusted). The UI picker offers both
  fixtures and the two Hugging Face ids. `llmplan model-info` text gains an `Experts` line
  for MoE models only (dense output unchanged); its JSON `derived` object and every
  `ModelSpec` dump gain the new fields.
- **Step 4 — Test changed for the new scope.** `tests/unit/catalog/test_architectures.py::
  test_resolve_hf_class` used `MixtralForCausalLM` as its example of an unsupported class;
  it now asserts that Mixtral resolves to `mixtral` and uses `GPT2LMHeadModel` as the
  unsupported example.
- **Step 5 — Live Hugging Face checks (9.11).** `tests/live/test_hf_configs.py`
  (`pytestmark = pytest.mark.network`, deselected by default and in both CI jobs) fetches
  each config through the M1 `HttpConfigFetcher`, asserts the parameter count within 1.5%
  of the official figure recorded in the test, and runs `fit()` (bf16, one H100, 4,096
  tokens; "fit runs" is the requirement, not "fits"). Run once on 2026-10-01 12:09 UTC
  with `uv run pytest -m network tests/live --no-cov -v -s` (no `HF_TOKEN` set):

  ```
  Qwen/Qwen2.5-7B-Instruct: llama_like, params 7,615,616,512 (active 7,615,616,512), official 7.61B (model card: Number of Parameters 7.61B), off by 0.07%; fit bf16 on 1 x H100: fits=True binding=ok
  PASSED
  mistralai/Mistral-7B-v0.3: llama_like, params 7,248,023,552 (active 7,248,023,552), official 7.25B (Hugging Face model page: 7.25B params), off by 0.03%; fit bf16 on 1 x H100: fits=True binding=ok
  PASSED
  mistralai/Mixtral-8x7B-Instruct-v0.1: mixtral, params 46,702,792,704 (active 12,879,925,248), official 46.7B (Mistral AI: 46.7B total parameters), off by 0.01%; fit bf16 on 1 x H100: fits=False binding=weights
  PASSED
  Qwen/Qwen3-30B-A3B: qwen_moe, params 30,532,122,624 (active 3,353,032,704), official 30.5B (model card: 30.5B total, 3.3B activated), off by 0.11%; fit bf16 on 1 x H100: fits=True binding=ok
  PASSED
  meta-llama/Llama-3.1-8B-Instruct SKIPPED (gated; set HF_TOKEN to check it)
  ========================= 4 passed, 1 skipped in 2.15s =========================
  ```

  The live Mixtral and Qwen3-MoE counts equal the fixture values exactly, so the fixtures
  carry the real integers. Llama-3.1-8B is gated and no token is available to this agent;
  its integers equal `fixture:llama3-8b` (8,030,261,248, M1), 0.00% from 8.03B.

## Deviations from the design doc

- **Step 1 — `perf_confidence` and `perf_sources` are properties of `PlanResult`, not
  serialized fields.** Both are derived from the chosen replicas' `PerfEstimate`s, which the
  plan already carries; as fields they would add two keys to every plan JSON and break the
  byte-for-byte pre-M7 golden comparison of M7 acceptance test 8.1, which must pass
  unchanged. `ModelSpec.attention` and `PlanRun.class_saving_pct` set the precedent.
  ARCHITECTURE.md section 4 documents them.

- **Step 3 — `BenchmarkRow.source_url` accepts the literal `"user-upload"`.** The design
  fills uploaded rows' `source_url` with `user-upload`, which the `^https://\S+$` pattern
  rejected. The pattern is now `^(https://\S+|user-upload)$`, and `load_benchmarks` rejects
  `user-upload` in a shipped file, so every shipped row still carries a real URL.
  ARCHITECTURE.md section 4 updated.
- **Step 3 — `plan()` and `run_plan()` gain `backends=`.** The design routes uploads through
  "the existing `backends=` override mechanism (M3)", which existed only on
  `perf.estimate`; the planner calls `estimate` internally, so it needs the same keyword.
  `compare_single_class` passes it on too. ARCHITECTURE.md section 5 updated.

- **Step 4 — `Architecture` gains `config_fields`, `active_params` and `expert_params`.**
  The design lists the new `ModelSpec` and `DerivedModelInfo` fields but not where the
  family-specific config keys are read or how decode bytes split experts from the rest;
  keeping both in the registry member keeps "a new family is one module plus one import".
  ARCHITECTURE.md sections 3, 4 and 6 updated.

## Questions for founder

None beyond docs/FOUNDER_QUESTIONS.md item 7 (PyPI account and trusted publishing).
