# M3 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the implementation
order in M3_DESIGN.md section 10.

## Implementation notes

- **Coordination with M2: `StatsLike` protocol.** M2 (`llmplan/workload/**`) was built in
  parallel on `m2-workload`, so M3 never imports it. `llmplan/perf/estimate.py` defines a
  runtime-checkable `StatsLike` protocol with the six `WorkloadStats` fields M3 reads
  (`input_tokens_mean/p50/p95`, `output_tokens_mean/p50/p95`, all `float`), declared as
  read-only properties so frozen pydantic models satisfy it under `mypy --strict`. M2's
  `WorkloadStats` satisfies it structurally. `tests/acceptance/test_m3.py` carries a
  field-for-field copy of M2's `WorkloadStats` (M2_DESIGN.md section 3) with a `TODO(M2)` to
  import the real one after the merge; unit tests use a six-field `FakeStats`.
- **Step 1 — Input validation in `estimate()`.** `max_model_len <= max_position_embeddings`
  needs the model, so `estimate()` checks it (not `ReplicaConfig`), raising
  `ValidationError` naming `max_model_len`. It also checks `stats` is `StatsLike` and that
  every field is finite, inputs `>= 1` and outputs `>= 0` (M2's row constraints).
- **Step 1 — `fit_for(model, gpu, config)`** maps a `ReplicaConfig` onto M1 `fit()` with
  `max_model_len` as `context_len`; `fixed_overhead_bytes` and `activation_multiplier` keep
  their M1 defaults (the design does not put them in `ReplicaConfig`).
- **Step 1 — Registry location.** The design puts the protocol and facade in `estimate.py`,
  so the `register`/`get` registry lives there too. `llmplan/perf/__init__.py` imports
  `roofline` and `table` to register them. `llmplan.perf.estimate` (the attribute) is the
  function, as ARCHITECTURE.md section 5 names it; `from llmplan.perf.estimate import ...`
  still resolves the module.
- **Step 2 — Roofline check order:** model fits (M1) first, then `memory_bandwidth_gbps`,
  then the dense TFLOPS field for the weight dtype. `P95_FACTOR` applies to TPOT only; TTFT
  p95 uses the p95 input length (section 4.3). `kv_seq_capacity` uses floor division by the
  float mean context, as written in section 4.1. Test 9.1 reproduces every stated value to
  4 significant figures: TPOT 30.4377 ms, decode 8410.6 tokens/s, prefill 14017.7 tokens/s,
  TTFT 28.54 / 107.01 ms, capacity 32.70 requests/s.
- **Step 3 — `benchmarks.py` split from `table.py`.** Loading and validation (schema, FKs,
  aliases, physical floor) and matching/interpolation are separate responsibilities; each
  module stays under 230 lines.
- **Step 3 — Physical floor details (section 5.2).** Rows do not record their KV-cache dtype
  or embedding precision, so the floor uses the smallest storage the row could have used:
  fp8 KV cache and embeddings at the weight dtype. It is the larger of the memory and
  compute terms that the GPU catalog can supply (either alone is a valid lower bound); a
  row whose GPU has neither bandwidth nor TFLOPS is rejected because it cannot be checked.
- **Step 3 — Every row must resolve to a model fixture** (its `model_id` is a fixture, or an
  `aliases.yaml` fixture key maps to its canonical id); otherwise the load fails. Section
  5.2 only FK-checks fixture ids, but the physical bound needs the architecture and tests
  cannot fetch from Hugging Face. Conservative reading: an uncheckable row is rejected.
- **Step 3 — Other load checks:** a row's `gpu_id` must equal its file stem
  (`<gpu-id>.yaml`); `tensor_parallel` must divide the model's attention heads; aliases may
  not chain. Error messages read `<file> row index <i> (model_id '<id>'): ...`.
- **Step 4 — Matching choices (section 5.3).** Rows of one engine are used: `vllm` if
  present (the engine `EngineProfile` models), else the alphabetically first. The workload
  shape is `(input_tokens_mean, max(output_tokens_mean, 1))`; the nearest benchmark shape
  minimizes Euclidean distance in log space (ties: smaller input, then output). "More than
  2x" means a ratio above 2 in either dimension; exactly 2x is accepted. When several rows
  share a concurrency, the lowest throughput is kept (conservative for capacity planning).
- **Step 4 — Clamping below the smallest concurrency** keeps that row's per-sequence rate
  (`output_tokens_per_s * batch / concurrency`) and latencies, rather than its aggregate
  throughput, which would overstate a smaller batch. Confidence is `"interpolated"`.
- **Step 4 — Latencies.** A latency is interpolated only when both bracketing rows carry it.
  Missing TPOT p50 is derived as `effective_batch / output_tokens_per_s`; missing p95 is
  `p50 x P95_FACTOR`. The prefill rate is `input_len / TTFT p50` when rows carry TTFT, else
  the roofline rate; TTFT p50/p95 are then the workload's p50/p95 input lengths over that
  rate, as in the roofline backend. Request capacity uses the section 4.4 formula with the
  table's TPOT. KV-cache dtype is not part of the match (rows do not record it); every
  table estimate says so in `assumptions`.
- **Step 4 — The table backend also returns `None`** when the model does not fit the
  replica (rows cannot make it fit) and when rows lack TTFT and the GPU lacks TFLOPS.
- **Step 5 — Seed source.** All 105 rows come from one page, the NVIDIA NIM LLMs
  Benchmarking "Performance" page
  (https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html, footer "Last
  updated on Apr 01, 2026", read 2026-09-30). The numbers were extracted by a script from
  the page's HTML table text and spot-checked against an independent manual reading of the
  same tables. `as_of` is the read date, as M1 did for its catalogs.
  - Throughput column = aggregate output tokens/s, per the companion Metrics page
    ("total output token throughput across all simultaneous requests").
  - GPU: tables say "H100 80G"; the page's hardware table lists NVIDIA DGX H100 with
    "H100 80GB HBM3(GH100)", the SXM part, hence `h100-sxm-80gb`. L40S tables map to
    `l40s-48gb`.
  - Seeded: Llama-3.1-70B fp8 tp4 on 4x H100 (NIM 1.3.0), Llama-3.1-8B fp8 and bf16 tp1 on
    1x H100 and 1x L40S (NIM 1.8.0), shapes 200/200, 500/2000, 1000/1000, 5000/500,
    concurrency 1 to 250.
  - Not seeded: 20000/2000 shapes (beyond the 8192-token fixtures); Llama-3.3-70B tables
    (a different checkpoint id; no fixture alias; would need its own canonical id); the
    NIM 1.3.0 Llama-3.1-8B tables (same match key as the 1.8.0 rows, would mix container
    versions); H200 and A100 tables (not required by section 5.4; can be added later).
  - Checked and unusable, per a research pass: the NVIDIA blog "LLM Inference Benchmarking:
    How Much Does Your LLM Inference Cost?" (results only in charts), the vLLM v0.6.0 blog
    (charts, request rate not concurrency), InferenceMAX (JavaScript dashboard, no text
    numbers), the vLLM performance dashboard (JavaScript), and TensorRT-LLM perf-overview
    tables (max-throughput runs with no concurrency stated). No public text row was found
    for Llama-3.1-8B on L4 or A10G.
- **CLI.** `llmplan/cli_perf.py` is a typer sub-app registered in `llmplan/cli.py` with one
  import and one `app.add_typer(perf_app, name="perf")` line, to keep the diff small for the
  M2 merge. It reuses `cli._run` through a call-time import (a module-level import would be
  circular). Only the design's flags exist (no `--kv-dtype`, `--gpus`). The six token
  statistics are required when `--trace` is absent: the design's numbers read as examples,
  and a hidden default workload would be silently wrong. `--trace` raises
  `ValidationError` (exit 2) with a `TODO(M2)`.
- **`PerfError` exits 1.** ARCHITECTURE.md section 7 gives it no code; `cli.exit_code` was
  left untouched (its fallback is 1) to keep `cli.py` unchanged apart from the registration.
- **Renderers.** `perf_estimate` and `benchmarks` methods were appended to the `Renderer`
  protocol and to both renderers, as ARCHITECTURE.md section 6 prescribes. M2 will likely
  append its own methods at the same places; a textual conflict there (and at the end of
  `CHANGELOG.md`) resolves by keeping both sides.
- **M1's `TODO(M3)` comments in `data/gpus.yaml`** (nvidia-smi totals, A10G bandwidth and
  TFLOPS) are outside M3_DESIGN.md's scope and were not changed. A10G therefore has no
  roofline estimate (both figures are null) and no benchmark rows.
- **Package growth (~870 lines).** ARCHITECTURE.md section 1 asks for a justification above
  ~300 lines: M3 ships two backends, a validated data format, and two CLI commands
  (`perf/`: ~650 lines across five modules, `cli_perf.py`: ~150, renderer methods: ~90).
  Nothing is speculative: `vidur` was not built, and every module has one job.
- **`docs/MILESTONES.md`** is set to "ready for review"; the CTO records the merge hash.

## Deviations from the design doc

- **`estimate()` signature.** ARCHITECTURE.md section 5 listed
  `(model, gpu, tp, config, workload_stats, *, backend="table")`; M3_DESIGN.md section 3
  has `estimate(model, gpu, config, stats, *, backend="auto")` with `tp` inside
  `ReplicaConfig`. The design doc's form is implemented, plus a keyword
  `backends: Mapping[str, PerfBackend] | None` that overrides registry entries for one call
  (test 9.5 needs `"auto"` over a test-only table). ARCHITECTURE.md section 5 updated.
- **`PerfBackend.explain()` added.** Tests 9.2 and 9.3 need `PerfError` messages that say
  why each backend could not answer ("does not fit", "memory_bandwidth_gbps null"), but
  `estimate()` returns only `None`. `explain()` returns that one-line reason and is called
  only after `None`; both methods share one private evaluation, so no state is kept.
  ARCHITECTURE.md sections 4 and 6 updated.
- **`stats` is typed `StatsLike`, not `WorkloadStats`** (see the coordination note above).
  ARCHITECTURE.md section 4 updated.
- **`BenchmarkRow.engine` adds `"nim"`.** The only source that met the "numbers printed on
  the page" rule benchmarks NVIDIA NIM containers and does not name the engine inside them
  (TensorRT-LLM or vLLM). Recording `trtllm` would be a guess; `"nim"` with the container
  version as `engine_version` is exact. ARCHITECTURE.md section 4 updated.
- **Seed rows have null TTFT/TPOT.** The source prints TTFT and ITL without naming the
  statistic (mean or median); storing them as `*_p50` would be a guess. The printed values
  are kept in each row's YAML comment with a `TODO(M3): verify` in the file header. Table
  estimates therefore derive TPOT from throughput and take the prefill rate from the
  roofline.
- **Module layout.** ARCHITECTURE.md section 3 listed `perf/backend.py`; the design doc's
  deliverables are `config.py`, `estimate.py`, `roofline.py`, `table.py`, and loading and
  validation moved to `benchmarks.py` (see above). ARCHITECTURE.md section 3 updated.
- **`TableBackend(table: BenchmarkTable | None = None)`** takes an explicit table (tests)
  and otherwise loads the shipped table once per process (`default_table()`, cached).

## Questions for founder

None. Per the founder's rule in docs/FOUNDER_QUESTIONS.md, these CTO-decidable items were
decided here and are open to CTO review:

- **TTFT/ITL statistic on the NIM page.** If the CTO can confirm the statistic (NIM's
  AIPerf reports averages by default), the `*_p50` fields could be filled from the row
  comments, or a `*_mean` field added to `BenchmarkRow`. Left null until then.
- **`engine: nim`.** If the CTO prefers mapping NIM rows to `trtllm` (NIM's optimized
  fp8 profiles on H100 are TensorRT-LLM based), it is a one-word change per file.
