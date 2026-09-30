# M5 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the implementation
order in M5_DESIGN.md section 10; "9b" is the carry-over item.

## Implementation notes

- **9b — Carry-over done first.** `tests/acceptance/test_m3.py` imports
  `llmplan.workload.schema.WorkloadStats`; the local copy and its `TODO(M2)` are gone. No
  expected value changed.
- **Step 1 — Replica parameters.** `slots = perf.effective_batch` (not `max_num_seqs`),
  `kv_token_capacity = fit.kv_token_capacity`, `kv_bytes_per_token =
  fit.kv_bytes_per_token_per_gpu * tensor_parallel`, and `ReplicaWindowRecord.weight_bytes =
  fit.per_gpu_weight_bytes * tensor_parallel`, so all byte figures are per replica (summed
  over its GPUs). A planned candidate without a fit or perf estimate, or with a KV capacity
  below one token, raises `ValidationError` (it cannot serve).
- **Step 1 — Event loop.** The heap holds only completions `(time_s, seq, replica,
  request)`; arrivals are read in trace order from the sorted arrays and compared with the
  heap top. At equal times the completion is processed first, so a slot freed at `t` serves
  a request arriving at `t`. Test 9.3 needs this: request 2 arrives at exactly 0.5 s, when
  request 0 completes on replica 0; processing the arrival first would queue it for zero
  time and report `max_queue_depth == 1`. Pushing all arrivals onto the heap would also
  cost one heap operation per request for nothing. `seq` breaks ties between completions.
- **Step 1 — Exact service times.** `service = input / prefill_tokens_per_s + output *
  tpot_s` is computed once per request; `ttft = wait + prefill` and `e2e = wait + service`
  (not `complete - arrival`), so 9.1's values are exact floats (100.0 and 500.0 ms).
- **Step 1 — Oversized requests.** A request needing more KV tokens than its replica holds
  would block its FIFO queue forever. Conservative reading: it is not dropped; its need is
  capped at the replica's capacity (it runs alone) and the timeline notes how many were
  capped.
- **Step 1 — State logs.** Each replica logs `(time, busy slots, KV tokens, queue depth)`
  once per event that touched it, after the event's changes. Several events at one instant
  (9.4's ten arrivals at t = 0) each log a state, and zero-duration states count toward
  window maxima (so 9.4's queue reaches 7 at t = 0) but not toward time-weighted means.
- **Step 1 — TPOT.** Every request's `tpot_ms` is its replica's constant
  `perf.tpot_ms_p50`, and TPOT violations are counted over all requests, including those
  with zero output tokens (conservative; it only matters when a budget is set below the
  estimate, which then flags every request).
- **Step 2 — Routing.** Policies are pure functions of the per-replica outstanding counts
  (admitted plus queued) and the request's trace index. `least_outstanding` breaks ties by
  lowest index (`min` returns the first minimum). Both policies are deterministic, so
  `SimOptions.seed` currently has no effect; it is kept because section 3 lists it.
- **Step 3 — Windows.** Half-open windows of `window_s` from the first arrival, as in M2,
  but they extend to the last completion (M2's windows end at the last arrival): otherwise
  the completions, busy time and latencies of requests still running after the last arrival
  would be dropped. Windows after the last arrival have zero demand. Requests are assigned
  to windows by `floor((t - start) / window_s)` as in M2; step functions are integrated
  against the same window bounds.
- **Step 3 — Aggregation details.** Utilization is clipped at 1.0 against float noise;
  `kv_tokens_in_use_mean` is the time-weighted mean rounded to the nearest integer;
  percentiles use `numpy.percentile` ("linear", as M2); window percentiles and violations
  are over requests completed in the window; summary percentages are over simulated
  requests; `mean_utilization` is the unweighted mean over (replica, window) records;
  `demand_rps = arrivals / window_s` and `capacity_rps = plan.capacity_rps` (derated).
- **Step 3 — Budgets.** `replay(..., slo=...)` fills unset budgets from
  `slo.ttft_ms_p95` / `slo.tpot_ms_p95`, and `Timeline.options` stores the resolved
  values, so a timeline says which budgets its violation counts used. The assumptions say
  when a budget is unset.
- **Step 3 — Acceptance tests 9.1 and 9.2 landed with step 3**, not step 1, because they
  assert window utilization and summary fields that exist from step 3. The engine-level
  parts of 9.1, 9.2 and 9.4 are unit-tested in step 1's commit
  (`tests/unit/simulate/test_events.py`).
- **Step 4 — Plot.** `render/plots.py` builds a `matplotlib.figure.Figure` on an explicit
  `FigureCanvasAgg` (no pyplot, no global backend state, no display) and saves with
  `metadata={"Software": None}`. Colors come from the dataviz reference categorical palette
  in fixed order; with more than eight replicas the utilization and KV panels show the mean
  and maximum across replicas instead of one line each. The fourth panel plots queue depth
  and violations on one axis because both are counts of requests (no dual axis). Only this
  module imports matplotlib, and the CLI imports it only for `--png`, so `llmplan --help`
  and every other command do not pay the import.
- **Step 4 — Dependency.** Runtime adds `matplotlib>=3.11` (section 11; tested with
  3.11.2). It pulls in pillow, kiwisolver, fonttools, cycler, contourpy, pyparsing and
  python-dateutil; `uv audit` reports no known vulnerabilities. matplotlib ships inline
  types, so no mypy override is needed.
- **Step 5 — CLI.** `llmplan/cli_simulate.py` holds the command, registered in `cli.py` with
  one import and one `app.command("simulate")` line, like `cli_plan.py`. Budgets come from
  `--ttft-p95-ms` / `--tpot-p95-ms`, else from the SLO recorded in the plan file
  (`request.slo`); the design does not say, and a plan made with an SLO should be checked
  against it rather than silently reporting zero violations. The trace format is detected
  from the header. `--plan /dev/stdin` reads a piped plan; no `-` alias was added.
- **Step 5 — Test fakes.** `FakePerf` gains `prefill_tokens_per_s` (default 10,000, the
  previously hard-coded value, so M4 tests are unchanged). `tests/fake_planner.sim_plan`
  plans one replica on fake row A with the fake backend (`effective_batch = max_num_seqs`,
  `tpot_ms_p50 = tpot_ms_p95 / 2`), then sets the replica count and optionally the KV
  capacity with `model_copy`, as section 9 asks ("fixture GPU with large VRAM"; the fake
  GPU has 10 TB, so the KV cache would otherwise never bind in 9.4).
- **Step 5 — Performance.** Test 9.10 (201,865 requests, 4 replicas, lognormal tokens,
  92% mean utilization, maximum queue depth 48) replays in **1.34 s** on this laptop (Apple Silicon, Python 3.11),
  against the 30 s target. The section 5 target, 500,000 requests (a 504,208-request trace
  truncated by `max_requests`), took 2.93 s. 9.10 is marked `slow`; pytest's `addopts`
  deselect it by default (`-m 'not slow'`) and `uv run pytest -m slow --no-cov` runs it.
- **Package growth (~990 lines).** ARCHITECTURE.md section 1 asks for a justification above
  ~300 lines: the event engine (174 lines incl. docstrings), replica model (86), routing
  registry (46), timeline models and aggregation (269), `replay` with its assumption lines
  (135), plot (115), text and JSON renderers (75 + 18), the CLI (69), and the plan loader
  (~55 in `planner/result.py`). Each module has one job; none exceeds 300 lines.
- **`docs/MILESTONES.md`** is set to "ready for review"; the CTO records the merge hash.

## Deviations from the design doc

- **`replay` signature.** Section 2 writes `replay(plan, workload, *, window_s, slo,
  options)`; `window_s` is also a `SimOptions` field, and two sources for one value invite
  disagreement. Implemented `replay(plan, workload, *, slo=None, options=None)` with the
  window in `options` (default `SimOptions()`). ARCHITECTURE.md section 5 updated (it listed
  `(PlanResult, Workload) -> Timeline`, which this still accepts).
- **Per-request records: `replay_requests` and `RequestLog`.** Tests 9.1, 9.2, 9.3 and 9.4
  assert per-request values (every `ttft_ms`, the last request's TTFT, start times), and
  section 12 says per-request data is available, but `Timeline`'s field list (normative)
  has none, and a pydantic model per request would be slow and huge at 500k requests.
  `replay_requests(plan, workload, *, options)` returns a frozen `RequestLog` wrapping a
  pandas frame (the `Workload` precedent). ARCHITECTURE.md sections 4 and 5 updated.
- **Plan loader returns the SLO too.** `load_plan_json(path) -> (PlanResult, SLO | None)`
  instead of a bare `PlanResult` loader, so the CLI can default its budgets to the plan's
  SLO without reading the file twice. `SolverInfo.solve_time_s` is excluded from
  serialization (M4), so a loaded plan reports 0.0 (also its baseline). The file is capped
  at 50 MB. Round-trip tested (`tests/unit/planner/test_plan_loader.py`): the loaded plan's
  `model_dump_json()` equals the original's byte for byte.
- **9.2 is read with 10 s windows.** The design gives 9.2 no window; with 9.1's 60 s the
  expected "utilization of the busy window == 1.0" is false. Arithmetic: 100 arrivals
  every 0.25 s (0 to 24.75 s), one slot, 0.5 s service, so request `i` starts at `0.5 i`
  and the replica is busy from 0 to 50 s; a 60 s window [0, 60) holds 50 busy
  slot-seconds, utilization 50 / 60 = 0.833. With 10 s windows, windows 0 to 4 are fully
  busy (1.0 each). The expected values are unchanged; the other 9.2 checks are
  window-independent (max queue 50 >= 45, last TTFT (49.5 - 24.75) s + 0.1 s = 24,850 ms >
  20,000, violations 99% > 50).
- **9.7 replays at 3x the remaining replica's capacity, not 3x the plan's demand.** The
  10.9 plan is one g6.xlarge (L4) replica, fp8, `effective_batch` 152, prefill 7,534
  tokens/s, `tpot_ms_p50` 93.03 ms, `requests_per_s_capacity` 29.29 (23.43 derated), KV
  capacity 88,194 tokens; its demand is `workload_10.csv`'s peak, 0.1 req/s. At 3x demand
  (0.3 req/s) with the workload's token range (inputs 100 to 1,000, outputs 10 to 100) the
  longest service is 1000 / 7534 + 100 x 0.09303 = 9.44 s, so on average 0.3 x 9.44 = 2.8
  requests arrive per longest service time, while queueing needs 152 concurrent requests
  (or, KV-bound, at least 88,194 / 1,100 = 80). Measured: at 0.3 req/s over 3,600 s, 0.0%
  violations and a maximum queue depth of 0, so `ttft_violation_pct > 50` and
  `max_queue_depth > 10` cannot hold. Truncating to one replica is a no-op for this plan.
  The test replays at `3 * perf.requests_per_s_capacity` = 87.9 req/s for 300 s instead
  (26,604 requests): measured 99.4% violations and a maximum queue depth of 18,213. The
  assertions (`> 50`, `> 10`) are unchanged.
- **9.6 and 9.7 inputs the design leaves open.** Token lengths are uniform over
  `workload_10.csv`'s range (inputs 100 to 1,000, outputs 10 to 100), because the plan's
  TTFT estimate was made for that workload; duration 3,600 s for 9.6 (160 requests at
  0.05 req/s), 300 s for 9.7; seed 0. The TTFT budget is twice the planned replica's
  `perf.ttft_ms_p95` (126.76 ms, so 253.5 ms).
- **9.4's `queue_depth_max == 7`** is asserted on the replica's record in window 0 (the
  field of that name); `summary.max_queue_depth` agrees.
- **9.10 inputs.** 252 req/s for 800 s (201,865 requests with seed 0), lognormal tokens
  (the `workload synth` defaults' shape), 4 replicas of 64 slots, prefill 10,000 tokens/s,
  tpot 5 ms: loaded enough to queue, so the loop exercises the FIFO path.
- **Windows extend to the last completion** (see implementation notes); section 7 says
  "windows as in M2", whose windows end at the last arrival.
- **Heap layout.** Section 5 describes one heap of `(time, seq, kind, payload)` for both
  event kinds; arrivals are merged from the sorted trace instead (see implementation
  notes). Order of processing is the same except that completions precede arrivals at
  equal times, which 9.3 requires.
- **New modules and interfaces.** `render/timeline_text.py` (the text output, keeping
  `render/text.py` small, as M4 did), `cli_simulate.py` (the M4 CLI pattern), a
  `Renderer.timeline(Timeline)` method (section 6's one-method-per-result rule), and the
  routing registry row. The planned `simulate/replay.py` does not exist; `replay` lives in
  `simulate/__init__.py` like `plan()` in `planner/__init__.py`, with the design's
  `events.py`, `replica.py`, `routing.py` and `timeline.py`. ARCHITECTURE.md sections 1, 3,
  4, 5 and 6 updated.
- **The plot shows KV in use, not free VRAM.** Section 1 describes a "VRAM split (weights /
  KV in use / free)", but the normative `ReplicaWindowRecord` has `weight_bytes` and
  `kv_bytes_in_use_max` and no total or free VRAM, so the KV panel plots KV bytes in use
  (max per window). Adding the replica's KV budget to the record would let M6 draw the
  full split; left for the CTO to decide.

## Questions for founder

None new. Section 12's question (per-request scatter plots in M6 or window aggregates)
keeps its interim decision: `Timeline` and the CLI carry aggregates only; per-request
records are available from the library (`replay_requests`). Per the founder's rule in
docs/FOUNDER_QUESTIONS.md, these CTO-decidable items were decided here and are open to CTO
review:

- **Per-request data from the CLI.** Section 12 says per-request data is "available in
  JSON"; the CLI's JSON is the `Timeline` only. A `--requests-csv PATH` flag writing the
  `RequestLog` frame would be a few lines if wanted.
- **Free VRAM in the timeline.** See the last deviation: one more field
  (`kv_budget_bytes` or `vram_bytes`) on `ReplicaWindowRecord` would complete the split.
