# M7 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. Section numbers refer to
docs/milestones/M7_DESIGN.md; "8b" is the carry-over list.

## Implementation notes

- **Acceptance tests first.** Section 8 is prose; `tests/acceptance/test_m7.py` turns each
  item into concrete tests with the expected values derived below ("Acceptance arithmetic").
  They were committed before any implementation, marked `xfail(strict=True)`, and each
  marker was removed in the commit that made its test pass; no expected value changed
  afterwards. M7 names are imported inside the tests so the file was collectable before
  M7 existed. Test doubles added for them: the `fake_shape` backend (every figure derived
  from the token statistics it is given, so per-class estimates follow from the class
  shapes), the `fake_class` backend (fixed capacities per GPU, tensor parallel and class),
  the `fake_class_perf` fixture, and two pre-M7 golden outputs under `tests/fixtures/m7/`
  (captured on main at f69ec54 with `llmplan plan ... --gpus l4-24gb,l40s-48gb --tp 1
  --dtypes fp8 --max-num-seqs 64,128 --format json` on `workload_10.csv` and `llmplan
  simulate` of that plan on `workload_csv_50.csv`; they must be regenerated if the shipped
  catalogs change).

- **Step 2 — Classes (`llmplan/workload/classes.py`).** Quantile cuts are
  `floor(numpy.percentile(..., 100 i / bins))` (linear, as M2), applied as inclusive upper
  bounds (`tokens <= cut` is the lower bin); duplicate cuts and cuts outside `[min, max)`
  are dropped (a note only for user-given fixed edges). The lowest bins start at 1 input
  and 0 output tokens and the highest end at the observed maxima, so the classes tile the
  plane and any request (also from another trace) can be assigned: `assign_classes` picks
  the class containing it, else the smallest total token distance to a class's bounds,
  ties to the lowest index. "Merged into their nearest neighbour": input bins under 1% of
  all requests merge first (into the adjacent bin whose mean input is nearest), then the
  cells of each input bin along the output axis (nearest mean output), smallest first,
  until none is under 1%. Merging only along one axis keeps every class a rectangle, so
  classes never overlap. Empty bins merge into their lower neighbour silently; non-empty
  merges are noted on the receiving class. Classes are numbered by input bin, then output
  bin. Bins per axis are limited to 1..6 (the design mentions up to 3x3).
- **Step 2 — Peak windows.** "Class demand is measured in the fleet-wide peak window": a
  class's `peak_rps` is its count in the window `compute_stats` reports as the request
  peak, and its `peak_output_tokens_per_s` its output tokens in the window it reports as
  the token peak (M4 sizes requests and tokens against those two windows, which may
  differ). So class demands sum to the workload's, and a single 1x1 class reproduces
  `peak_window_rps`, `peak_output_tokens_per_s` and the token statistics bit for bit (same
  numpy calls on the same arrays; test 8.1 relies on it).
- **Step 2 — `classify_spec`.** The CLI and UI share one parser for `--classes`: `1` (no
  classes, the M4 path), `AxB` (quantile), `fixed:<input edges>/<output edges>` with
  comma-separated edges, either side possibly empty (e.g. `fixed:1024,4096/256`).

- **Step 3 — Per-class candidates (`planner/classes.py`).** A candidate that passed the
  tensor-parallel and memory checks is estimated once per class by the M3 `estimate()`
  with the `DemandClass` as its `StatsLike`, cached by (GPU id, replica config, class) so
  price rows sharing a GPU reuse the estimate. Class verdicts reconcile with the
  whole-workload verdict as follows: eligible for some class -> `eligible` (a candidate
  that failed the whole workload's SLO says "eligible for class 0 only (whole workload:
  ...)"; one that passed it but misses classes appends "not eligible for class 1 (...)");
  eligible for none -> the first class's rejection. `CandidateEval.perf` and
  `usd_per_hour_per_rps` stay the whole-workload estimate (the class estimate only when the
  whole workload had none), because `perf` also drives the simulator's service times.
  With classes, a class that has demand but no eligible candidate raises `InfeasiblePlan`
  naming the class and its bounds before any solve.
- **Step 3 — One class is M4.** With one class `x_r = m_r` is optimal (capacities are
  non-negative), so the formulation with one class (or none) is built exactly as M4's,
  with the class's demand; only two or more classes add `x_{r,k}`. Test 8.1 checks the
  results serialize identically, and the pre-M7 CLI golden still matches byte for byte.
- **Step 3 — Class model.** `x_{r,k}` exists only where candidate r is eligible for class k
  (capacity > 0). HiGHS sees them as continuous. CP-SAT needs integers: allocations are
  milli-replicas `X = 1000 x` (`sum_k X <= 1000 m`, exact), capacities are floored to
  milli-units as in M4, and the class demand is multiplied by 10^6 and rounded up, so
  every scaled solution is feasible unscaled. On razor-thin margins CP-SAT's whole
  milli-replicas may need one more replica than HiGHS; the reported cost is always
  recomputed from the integers.
- **Step 3 — Routing LP.** The MILP's `x` is not unique (any split with enough capacity
  is optimal), so after the solve a GLOP LP over the chosen fleet fixes it: stage 1
  maximizes the uniform headroom `theta` (every class's capacity >= theta x its demand,
  rows written as capacity / demand so coefficients are O(1)); stage 2 keeps theta >=
  theta* (1 - 1e-9) and maximizes the allocated replica time, so time no class needs is
  still given to classes the replica can serve. Allocations under 1e-7 are reported as
  0. theta is capped at 10,000: with a 10^9 cap GLOP returned IMPRECISE on the 8.2
  baseline, and a smaller cap also tightened its answers (1.0666667223 against 16/15 with
  10^9, exact to 1e-13 with 10^4). With classes, `capacity_rps` and
  `capacity_output_tokens_per_s` are the capacity this LP gives the classes, summed, and
  the below-demand guard checks theta* >= 1 - 1e-6.
- **Step 3 — Binding with classes.** The LP relaxation over the chosen columns is the class
  model relaxed; `binding` says whether any class's request (token) constraint is tight.
  Per-class labels are exposed in step 4 (`class_binding`).
- **Step 3 — Baseline with classes.** One column must serve every class with demand; its
  replicas split their time, so it needs `ceil(sum_k max(D_k / cap_k, T_k / tok_k))`
  replicas (the same guard against float noise as M4). With one class this is M4's
  `max(ceil(D / cap), ceil(T / tok), 1)`. M4's `replicas_needed` became
  `replica_equivalents`.
- **Step 3 — `planner/explain.py`.** `plan()` in `planner/__init__.py` was 283 lines; the
  M4 helpers that explain a fleet (binding LP, assumption lines, headroom, the baseline
  `PlanResult`) moved to `explain.py` unchanged, with capacity added, so no module exceeds
  300 lines.

- **Step 4 — Routing weights.** A weight is the replica type's share of the class's
  allocated request capacity, `x_{r,k} cap_{r,k} / sum_r x_{r,k} cap_{r,k}`, from the
  routing LP (with one class, `x_r = m_r`); see Deviations for why not `/ D_k`. A class
  the LP gives nothing (no peak demand) is spread over the chosen replicas eligible for it
  in proportion to `m_r cap_{r,k}`; a class no chosen replica can serve gets no rules (the
  simulator then routes it class-blind). Each rule also carries its replica-equivalents
  and allocated capacities. The baseline carries the same fields.
- **Step 4 — Assumptions with two or more classes** add the class split, the routing LP's
  headroom factor, any merge notes, and one binding line per class instead of M4's single
  line. With one class (or none) the assumptions are M4's, word for word.
- **Step 4 — CLI default `--classes 1`.** The prompt allows a 2x2 default only once K=1
  compatibility is proven by tests; it is (8.1), but the CLI still defaults to `1`: the
  `--stats-json` input has no rows to classify, M4/M5 unit tests compare `--trace` and
  `--stats-json` output line by line, and README examples would silently change cost. The
  UI, which always has the rows, defaults to 2x2 as the design says (step 8).
  `--classes 2x2` with `--stats-json` exits 2 ("--classes needs --trace").
- **Step 4 — Text output** gains a class table (bounds, share, peak demand, binding) and a
  routing table (weight, replica-equivalents, allocated req/s, replica type) after the
  replicas, only when the plan has classes; K=1 text is unchanged. JSON gains `classes`,
  `routing` and `class_binding` (empty lists for K=1) after every pre-M7 field.

- **Step 5 — `class_weighted` routing.** Requests are classified with the plan's class
  bounds (`assign_classes`; a request outside every class goes to the nearest one).
  "Lowest cumulative-deficit rule" is read as: send the request to the replica type whose
  count so far is furthest below its target (`weight x requests of the class so far -
  requests sent there`, i.e. the most negative surplus), ties to the lowest type, which is
  a deterministic weighted round robin that keeps each type within one request of its
  share; then the least-outstanding replica of that type (M5's rule). The target type of
  every request depends only on the class sequence, so it is precomputed and the policy
  keeps M5's `(outstanding, index) -> replica` signature. A replica type is a
  `ReplicaPlan`; rules are matched to it by candidate equality (a rule naming no planned
  replica raises `ValidationError`). Requests of a class without rules, and every request
  on a plan without classes, are routed least-outstanding over all replicas (noted).
- **Step 5 — Per-class results.** `Timeline.classes` holds one `ClassSummary` per plan
  class (request count, TTFT/E2E p95, TTFT/TPOT violation percentages), for any routing
  policy, so class-blind routing can be compared (test 8.4). Per-window per-class figures
  were not added (the design asks for per-class p95 and violations; the window table stays
  M5's). The text output adds one line per class.
- **Step 5 — Incremental KV (8b).** A request reserves `input + ceil(output / 2)` tokens
  (its mean occupancy, as the planner's `effective_batch` assumes; capped at the KV
  capacity) for admission, and its KV in use grows linearly from `input` at admission to
  `input + output` at completion, as the prompt allows instead of per-step events. Each
  replica keeps the in-use line (value, slope, time) and logs value and slope after every
  event; window means integrate the piecewise-linear function (trapezoids) and window
  maxima take segment ends. An idle replica's line is reset to exactly zero so float noise
  cannot accumulate. Admission reserves the mean, so in-use KV can briefly exceed the
  cache when many long requests finish together (vLLM would preempt; not modeled, noted in
  the assumptions; the VRAM plot already clips free memory at zero). `kv_tokens` in the
  request log is the peak held (`input + output`, capped), as before.
- **Step 5 — `SimOptions.kv_accounting`** defaults to `"incremental"`; `"full"` keeps M5's
  behaviour exactly (the step-function path is untouched, so values are bit-identical).
  `llmplan simulate --kv-accounting full` reproduces the pre-M7 simulate JSON byte for
  byte apart from the new `options.kv_accounting` and `classes` fields (test 8.1). M5's
  acceptance test 9.4 asserts that exactly 3 of 10 requests fit 1,000 KV tokens at 300
  tokens each, which holds only under full reservation (incremental reserves 250, so 4
  fit), so it now passes `SimOptions(kv_accounting="full")`, with a comment; its expected
  values are unchanged. Every other M5 acceptance test passes unchanged under the
  incremental default.
- **Step 5 — CLI.** `llmplan simulate --routing auto` (the new default) picks
  `class_weighted` for a plan with routing weights and `least_outstanding` otherwise, so
  K=1 plans replay exactly as before; `--kv-accounting incremental|full`; `--requests-csv
  PATH` writes the `RequestLog` frame (pandas `to_csv`, no index; an unwritable path exits
  2). The CSV is produced by `replay_requests`, which repeats the replay; acceptable for a
  CLI export and keeps both public functions unchanged.
- **Step 5 — Modules.** The step-function window helpers moved from `timeline.py` to
  `simulate/stepfn.py` with the new piecewise-linear variant; `timeline.py` is 307 lines
  (models plus aggregation, one job) and `workload/classes.py` 316 (model, classifier,
  spec parser, assigner of one concept), both just over the ~300-line guideline.
- **Step 5 — Performance.** M5's 9.10 (201,865 requests, 4 replicas) now replays in 2.03 s
  with incremental accounting (1.34 s in M5 on an idle machine; this run shared the CPU
  with other jobs), far inside the 30 s target.

- **Step 6 — Exactness test (section 6.1, acceptance 8.3).** `tests/brute_force.py` draws
  seeded instances (1 to 3 rows of 1 or 2 GPUs at $0.50 to $6.00/h, at most 4 candidates:
  tp 1 and tp 2 on two-GPU rows, 1 or 2 classes, each candidate ineligible for a class
  with probability 1/4 through a 10 s TTFT against a 500 ms SLO, capacities 0.5 to 4 req/s
  and 50 to 500 tokens/s per GPU, demands 0.5 to 10 req/s and 0 to 1,500 tokens/s,
  at most 6 instances per row) and plans them through the full `plan()` pipeline with the
  `fake_class` backend, so candidate evaluation, per-class verdicts, pruning, the MILP and
  the cost recomputation are all under test. The brute force enumerates the 7^P instance
  vectors by cost; for each, every maximal replica vector (replicas are free, so a
  non-maximal vector is never better); a vector is feasible when per-class capacity checks
  pass and (two classes) the allocation LP is feasible in GLOP. Result over the 50 seeds:
  every MILP cost equals the brute-force optimum within 1e-6 (37 feasible, of them 15 with
  two classes and 10 with a mixed fleet; 13 infeasible, all raising `InfeasiblePlan`), in
  **1.27 s** (target 10 s). `tests/unit/test_brute_force.py` checks the brute force itself
  on hand-solved instances.

- **Step 7 — Validation records.** Sections 6.2 and 6.3 are recorded below ("Mélange
  cross-check", "Real-trace saving"). `scripts/melange_crosscheck.py` reproduces 6.2: it
  builds the scenarios, solves them with `plan()`, and runs Mélange's solver from a
  checkout passed as `--melange-dir` (PuLP is not a dependency: `uv run --with
  pulp==2.8.0`); `tests/unit/test_melange_crosscheck.py` covers the llmplan side with a
  stub in place of Mélange.

- **Step 8 — UI.** Advanced gains "Request-size classes" (2x2 default, 1, 3x3; the
  selection is classified with `classify_spec` on the traffic when Plan is clicked).
  `state.run_plan` replays a plan with classes with `class_weighted` routing and plans the
  same request without classes; `PlanRun.single_class_cost_usd_per_day` and the
  `class_saving_pct` property feed "Saving from request-size routing" in a new
  "Request-size routing" section (after the replicas): the saving with the single-class
  cost, a caption that it can be negative, a class table (bounds, share, peak demand,
  binding, replayed TTFT p95 and violations per class) and the routing table (weight,
  replica type, replica-equivalents, req/s). The two plans run under the same lock and
  cache entry. The usage log is unchanged (its fields are fixed by M6's test 9.6).
  `app.py` is 396 lines (design limit 400).
- **Step 8 — Timing.** With 2x2 classes a UI plan runs two plans (with and without
  classes), the class evaluation and a class-weighted replay: every bundled preset still
  plans and replays in under 3 s through AppTest (`-m slow -k every_preset`: Azure 2024
  conversation 2.76 s, the slowest), on this laptop while other jobs kept its load average
  near 17 on 8 cores.
- **Package growth (~1,360 net lines under `llmplan/`).** ARCHITECTURE.md section 1 asks
  for a justification above ~300 lines: classes (`workload/classes.py`, 316), the planner's
  class evaluation and routing (`planner/classes.py`, 186), the class formulation and
  routing LP (~120 in `model.py`, ~75 in `solve.py`), `planner/explain.py` (221, of which
  ~150 moved out of `planner/__init__.py`, which shrank by ~100), the simulator's
  class-weighted routing, incremental KV and per-class summaries (~300 across
  `simulate/`), and the CLI, renderers and UI (~210). Each module keeps one job; only
  `workload/classes.py` (316) and `simulate/timeline.py` (307) pass 300 lines, slightly.
- **Test suite time.** `uv run pytest` (coverage on) took 51 to 61 s here across runs, with
  other jobs holding the load average between 12 and 22 on 8 cores; without coverage it
  takes about 25 s. M7 adds about 6 s under that load (the exactness test is 1.3 s). On an
  idle laptop the suite is well inside the 60 s budget; under heavy contention it can brush
  it.

## Deviations from the design doc

- **Routing weights are normalized per class.** Section 4 defines the weight as
  `x_{r,k} cap_{r,k} / D_k`, which sums to more than 1 whenever the fleet has headroom
  (every solution with spare capacity), while section 8 requires weights that sum to 1 per
  class. The weight is therefore the share of the class's allocated capacity (the design's
  formula divided by its sum), and the `x` come from the routing LP rather than the MILP
  (whose `x` are not unique).
- **`PlanResult.class_binding: tuple[Binding, ...]`** carries the "per-class binding" of
  section 4; `PlanResult.classes` holds the request's `DemandClass`es (no separate result
  model), and `RoutingRule` gains `replicas`, `capacity_rps` and
  `capacity_output_tokens_per_s` next to the design's `(class_index, candidate, weight)`.
  ARCHITECTURE.md section 4 updated.

- **`DemandClass` gains `input_tokens_p50`, `output_tokens_p50` and `notes`.** The perf
  model's `StatsLike` needs p50s (the roofline's TTFT p50 is input p50 over the prefill
  rate), and the design wants merges "noted" while `classify` returns only classes. With
  the p50s a class is passed to the M3 `estimate()` as-is. ARCHITECTURE.md section 4
  updated.
- **`classify(..., window_s=60.0)`.** The design's signature has no window; class demand
  is measured in peak windows, so their length is a parameter (default: `compute_stats`'s
  60 s). ARCHITECTURE.md section 5 updated.
- **New interface fields** (ARCHITECTURE.md sections 4 and 5 updated in the commits that
  added them): `PlanRequest.classes` (where the plan gets its classes; the design does not
  say), `SimOptions.kv_accounting` (8b), `Timeline.classes` / `ClassSummary` ("timeline
  gains per-class p95 latency and violations"; summaries over the whole replay, not per
  window), `RequestLog`'s `class_index` column, `PlanRun.single_class_cost_usd_per_day`
  (the UI's saving), `ReplicaState`'s KV-in-use line, and the `class_weighted` builder
  (`routing.class_weighted(...)`, not a registry entry, because the policy needs the plan
  and each request's class).
- **The CLI defaults to `--classes 1`; the UI to 2x2.** Section 7 sets the UI default only.
  See implementation notes, step 4.
- **`llmplan simulate --routing` defaults to `auto`** (class-weighted for class plans,
  least-outstanding otherwise) instead of `least_outstanding`, so a class plan is replayed
  as planned without extra flags; K=1 plans behave as before.
- **M5 acceptance test 9.4 now passes `SimOptions(kv_accounting="full")`.** Its values hold
  only under M5's full reservation, and 8b makes incremental accounting the default. The
  prompt asks for exactly this; the expected values are unchanged.
- **6.2's synthetic scenarios use TPOT p95 100 ms**, not the UI's 50 ms, at which the L4 is
  eligible for no bucket and the scenarios would compare only L40S against H100.
- **6.3 found no saving.** Section 6.3 expects a saving from request-size routing; on the
  bundled Azure 2024 conversation sample the classes plan costs the same or more (see
  "Real-trace saving"), which is recorded as measured. It is surfaced in the UI as the
  design asks, with a caption that the value can be negative.

## Questions for founder

None. These CTO-decidable items were decided here and are open to review:

- **UI default 2x2.** The design makes 2x2 the UI default, and it is. On the bundled
  presets it raises the planned cost against M6 for the same inputs (Azure 2024
  conversation at the default SLO: $44.66 to $83.76/day), because per-class sizing drops
  the optimism of sizing every replica for the mean request (the replay at full rate shows
  the single-class fleet missing the TTFT target for 20% of requests). If the launch should
  show the M6 numbers by default, set `CLASS_CHOICES` to start with `"1"` in
  `llmplan/ui/presets.py`; the saving line then appears when a visitor picks 2x2.
- **Per-class SLO strictness.** A class is checked against the SLO at its own token shape
  (p95 TTFT from its p95 input; TPOT at its mean context), so long-input classes can fail
  TPOT on GPUs the whole-workload check accepts. This follows the design ("candidates whose
  SLO fails for a class are ineligible for that class") and Mélange's per-bucket profiles;
  a mixed batch in vLLM would see a context between the classes'.

## Acceptance arithmetic

### 8.1 K=1 reproduces M4

No new arithmetic: each M4 acceptance request (10.1 to 10.8, 10.11, and 10.9 on the real
catalogs) is planned twice, without classes and with one explicit class equal to the whole
workload, and the two results must serialize identically once the three new fields are
left out. The costs are M4's: 10.1 504.0, 10.2 576.0, 10.3 480.0, 10.4 144.0, 10.5 576.0,
10.7 456.0, 10.8 504.0, and 10.11 504.0 (10.1's capacities with fp8 weights; the fake
backend ignores the dtype). 10.6 must raise the same `InfeasiblePlan` message. With one
class, its routing weights sum to 1 and `class_binding == (binding,)`.

The CLI part compares `llmplan plan --format json` (default classes, and `--classes 1`)
byte for byte with the pre-M7 golden after deleting `classes`, `routing` and
`class_binding` (top level and baseline), and `llmplan simulate --kv-accounting full`
with the pre-M7 simulate golden after deleting `classes` and `options.kv_accounting`.

### 8.2 Two classes; the cheap GPU meets the short-class SLO only

Workload: 600 requests at `0.1 i` s (`i = 0..599`, all in one 60 s window); even `i`
short (100 input, 10 output tokens), odd `i` long (2,000 input, 65 output).

Classes, `classify(input_bins=2, output_bins=1)`: the input median (linear) of 300 x 100
and 300 x 2,000 is (100 + 2,000) / 2 = 1,050, so class 0 is inputs 1..1,050 and class 1 is
1,051..2,000; one output bin. Shares 0.5 each. Peak window demand:
`D0 = D1 = 300 / 60 = 5.0` req/s; `T0 = 300 x 10 / 60 = 50.0` and `T1 = 300 x 65 / 60 =
325.0` output tokens/s.

Fake shape backend, `max_num_seqs = 4`, tpot 10 ms on both GPUs, utilization 1.0, TTFT
SLO 500 ms. Row SA: 1 GPU, $1.00/h ($24/day), prefill 2,000 tokens/s. Row SB: 1 GPU,
$3.00/h ($72/day), prefill 20,000 tokens/s. Decode capacity is 4 / 0.01 = 400 tokens/s on
both.

| Candidate, class | TTFT p95 (ms) | service s | req/s cap | eligible |
|---|---|---|---|---|
| SA, short | 100 / 2,000 = 50 | 0.05 + 10 x 0.01 = 0.15 | 4 / 0.15 = 26.667 | yes |
| SA, long | 2,000 / 2,000 = 1,000 | | | no (> 500) |
| SB, short | 100 / 20,000 = 5 | 0.005 + 0.1 = 0.105 | 4 / 0.105 = 38.095 | yes |
| SB, long | 2,000 / 20,000 = 100 | 0.1 + 0.65 = 0.75 | 4 / 0.75 = 5.333 | yes |

Replica-equivalents needed (`max(D / cap, T / tok)`): long on SB `max(5 / 5.333, 325 /
400) = max(0.9375, 0.8125) = 0.9375`; short on SB `max(5 / 38.095, 50 / 400) =
max(0.13125, 0.125) = 0.13125`; short on SA `max(5 / 26.667, 0.125) = 0.1875`.

- One SB alone: 0.9375 + 0.13125 = 1.06875 > 1 replica, infeasible. SA alone cannot serve
  long. So every fleet has at least one SB, and one SB needs help with the short class.
- 1 SA + 1 SB: long 0.9375 on SB, short 0.1875 on SA: feasible, $24 + $72 = **$96/day**.
- 2 SB: $144/day. Anything else costs more. Optimum: `{sa-1x: 1, sb-1x: 1}`, cost 96.0.
- Baseline (best single column, same class model): SB needs `ceil(1.06875) = 2` replicas,
  $144/day (SA cannot serve long). Saving `(144 - 96) / 144 = 33.333%`.
- Binding per class (LP relaxation over SA and SB): long needs x = 0.9375 on SB, where the
  request constraint is tight (5.333 x 0.9375 = 5) and tokens slack (375 > 325); short
  goes to SA ($24 / 26.667 = $0.90/day per req/s against SB's $72 / 38.095 = $1.89), x =
  0.1875 makes requests tight (26.667 x 0.1875 = 5) and tokens slack (75 > 50). So
  `class_binding == ("requests", "requests")` and `binding == "requests"`.
- Routing on the chosen fleet: SA can serve only the short class and SB must give the long
  class 0.9375 (and at least that x headroom): the headroom-balancing LP gives theta* =
  min over SB of `1 / 0.9375 = 1.0667` (long) and `26.667 / 5` (short on SA) = 1.0667 with
  long on SB at x = 1 and short on SA, so weights are short -> SA 1.0, long -> SB 1.0.
- Without classes the same request uses the whole-workload shape: input mean 1,050, input
  p95 2,000 (`numpy.percentile`, linear: position 0.95 x 599 = 569.05, in the 2,000 half),
  output mean (10 + 65) / 2 = 37.5. SA: TTFT p95 1,000 ms > 500, rejected. SB: service
  1,050 / 20,000 + 37.5 x 0.01 = 0.4275 s, cap 4 / 0.4275 = 9.357 req/s; demand 10 req/s
  and (3,000 + 19,500) / 60 = 375 tokens/s, so `max(10 / 9.357, 375 / 400) = 1.06875` ->
  2 SB, **$144/day**. Saving from request-size routing on this instance: 33.3%.

### 8.3 Exactness

Fifty seeded random instances (at most 3 price rows of 1 or 2 GPUs, at most 4 candidates,
1 or 2 classes, at most 6 instances per row). The brute force enumerates every instance
vector in order of cost and, for each, every maximal replica vector the GPUs allow, and
solves the allocation LP with GLOP; the first feasible fleet is the optimum. The MILP's
cost must equal it to 1e-6, and an instance the brute force finds infeasible must raise
`InfeasiblePlan`. All 50 seeds within 10 s.

### 8.4 Simulation of the 8.2 plan

`class_weighted`: short requests (every 0.2 s from 0.0) go to SA, service 0.15 s, so at
most one runs at a time (4 slots): TTFT = 100 / 2,000 = 0.05 s = **50.0 ms**. Long requests
(every 0.2 s from 0.1) go to SB, service 0.75 s: at an arrival at t the ones from t - 0.2,
t - 0.4 and t - 0.6 still run and the one from t - 0.8 finished at t - 0.05, so at most 4
run (4 slots), no queueing: TTFT = 2,000 / 20,000 = **100.0 ms**. No violations in either
class (budget 500 ms), maximum queue depth 0.

`least_outstanding` on the same two replicas (SA is replica 0): t = 0 short -> SA; t = 0.1
long -> SB (SA busy until 0.15); t = 0.2 short -> SA (SA idle again, SB busy); t = 0.3 long:
SA holds the 0.2 request (until 0.35) and SB the 0.1 request (until 0.85), a tie broken
to the lowest index, so the long request runs on SA with TTFT >= 2,000 / 2,000 = 1,000 ms
> 500: the long class has violations.

### 8.5 Determinism

Two plans of the 8.2 request render identical JSON; two `class_weighted` replays give
identical timeline JSON; `llmplan plan --classes 2x2` on `workload_csv_50.csv` twice gives
identical stdout with four classes.

### 8b.1 Incremental KV accounting

`fixture:llama3-8b`, bf16, tp 1, max_num_seqs 256, max_model_len 8,192, on
`h100-sxm-80gb` (lambda, one GPU). Requests are 2,000 input + 1,000 output tokens, 20 per
second for 300 s (6,000). M1 fit gives a KV capacity of 448,308 tokens (131,072 bytes per
token: 32 layers x 8 KV heads x 128 x 2 (K and V) x 2 bytes). The roofline's effective
batch is `min(256, 448,308 // (2,000 + 1,000 / 2)) = min(256, 179) = 179` (179.3), so the
replica has 179 slots. Its capacity is about 5.6 req/s, so at 20 req/s one replica is
overloaded and its queue never empties after the first seconds.

- Incremental: a request reserves its projected mean occupancy, 2,000 + 1,000 / 2 = 2,500
  tokens; 179 x 2,500 = 447,500 <= 448,308 < 180 x 2,500, and 179 slots, so 179 run at
  once: steady-state concurrency = 179 = 100% of the effective batch (within 15%).
- Full reservation (M5): 3,000 tokens each; 448,308 // 3,000 = 149 (447,000), so 149 run:
  149 / 179 = 0.832, more than 15% below. Both are measured as mean utilization x 179 over
  windows 1 to 4 (60 to 300 s).

### 8b.2 Requests CSV

`llmplan simulate --requests-csv PATH` on the pre-M7 golden plan and `workload_csv_50.csv`
writes a header equal to `RequestLog`'s columns and 50 rows whose `arrival_s` equal the
trace's.

## Mélange cross-check (section 6.2)

Recorded, not asserted. Source: https://github.com/tyler-griggs/melange-release, `main` at
`d46ab43855bcdbfed4740a42058d2e269374ea55`, cloned into a scratch directory outside the
repository on 2026-10-01; solver `melange/solver.py` (PuLP 2.8.0 with its bundled CBC, run
on Apple Silicon without the README's Homebrew workaround). Mapping: a Mélange bucket is
an llmplan class (demand = share x rate, no token demand), a GPU is a one-GPU price row at
Mélange's hourly cost, a bucket throughput is the class capacity (already derated). A
bucket a GPU cannot serve within the SLO is ineligible in llmplan and gets 1e-9 req/s in
Mélange. Mélange ran at slice factors 1, 4 and 16; the table compares llmplan with the
best of the three (USD per hour).

| Scenario | Mélange (best of 1/4/16) | llmplan | Gap |
|---|---|---|---|
| toy (`melange/config/example.json`) | $5.69 (2 A10G + 1 A100; $6.70 at slice 1) | $4.68 (1 A10G + 1 A100) | -17.75% |
| short-chat heavy | $5.0996 (2 L4 + 1 H100) | $5.0996 (2 L4 + 1 H100) | 0.00% |
| long-document heavy | $13.96 (4 H100) | $13.96 (4 H100) | 0.00% |
| mixed | $10.47 (3 H100) | $10.47 (3 H100) | 0.00% |

llmplan is at or below Mélange on every scenario, as the slice-factor-to-infinity argument
predicts. The one gap above 5% (the toy) was investigated: alone, the A100 would carry
`6 x 0.05 + 3 x 0.05 + 15 x 0.025 + 6 x 0.05 = 1.125` GPUs of load; moving 5 of bucket
(1, 0)'s 15 req/s to the A10G (load 1/5 per req/s) frees 0.125 and fills the A10G exactly
(5 x 0.2 = 1.0). That split is a third of the bucket, which slices of 1/4 or 1/16 of it
cannot express, so Mélange needs a second A10G. Rerun at slice factors 3 and 48 (multiples
of 3), Mélange returns $4.68 (1 A10G + 1 A100), equal to llmplan.

Synthetic scenarios: `fixture:llama3-8b`, fp8, tp 1, the shipped one-GPU rows g6.xlarge (L4,
$0.8048/h), g6e.xlarge (L40S, $1.861/h) and runpod h100-sxm (H100, $3.49/h), 20 req/s, SLO
TTFT p95 500 ms and TPOT p95 100 ms (at the UI's 50 ms no L4 bucket is eligible, which
leaves nothing heterogeneous to compare), capacity derated by 0.8 and taken as the best of
max_num_seqs 32/64/128/256 that meets the SLO, from the perf model (`auto`: the table
backend on the NIM rows for H100 and L40S where a row shape is near, roofline otherwise).
Buckets, on a 2 x 2 grid with input and output edges at 500 tokens, sit at the NIM row
shapes: chat 200/200, generation 500/2,000, document 5,000/500, balanced 1,000/1,000
(input/output tokens). Mixes (`[[chat, generation], [document, balanced]]`): short-chat
heavy `[[0.70, 0.10], [0.05, 0.15]]`, long-document heavy `[[0.10, 0.05], [0.70, 0.15]]`,
mixed `[[0.25, 0.25], [0.25, 0.25]]`. Capacities (req/s, derated): L4 chat 8.21, other
buckets 0 (TPOT or TTFT); L40S 24.95 / 0.54 / 1.27 / 2.74; H100 51.82 / 4.91 / 5.03 / 9.22
(chat / generation / document / balanced).

Reproduce:

```
git clone https://github.com/tyler-griggs/melange-release /tmp/melange-release
uv run --with pulp==2.8.0 python scripts/melange_crosscheck.py --melange-dir /tmp/melange-release
```

## Real-trace saving (section 6.3)

The bundled Azure 2024 conversation sample (`data/traces/samples/azure2024_conv.csv`,
19,999 requests, peak 4.18 req/s), `fixture:llama3-8b`, the shipped catalogs, default
options (`max_model_len 8192`, every GPU and provider, tp 1/2/4/8, bf16/fp8,
max_num_seqs 32 to 256, utilization 0.8), K=1 (`--classes 1`) against K=4 (`--classes
2x2`). Classes (2x2 quantile): inputs 1..956 / 957..7,999 x outputs 0..39 / 40..1,200,
shares 28.3% / 21.7% / 22.1% / 27.9%.

| SLO | K=1 | K=4 | Saving from request-size routing |
|---|---|---|---|
| TTFT 500 ms, TPOT 50 ms (UI default) | $44.66/day (1 x g6e.xlarge, L40S) | $83.76/day (1 x runpod H100) | -87.5% |
| TTFT 500 ms | $44.66/day (1 x L40S) | $44.66/day (1 x L40S) | 0.0% |
| none | $19.32/day (1 x g6.xlarge, L4) | $38.63/day (2 x L4) | -100.0% |

**No saving on this trace; the classes cost the same or more.** The design expected a
saving; the measurement says otherwise, for two reasons. (1) The sample is small: its peak
fits on one GPU, so there is nothing to split between GPU types (the cheapest feasible
fleet is one instance either way). (2) The single-class plan sizes every replica for the
mean request (1,650 input, 102 output tokens), and the M3 model's service time and KV
limit are not linear in the request size, so the mean shape is optimistic: on an L4, the
long/long class (mean input about 3,500 tokens) holds about 24 sequences of KV and serves
about 0.8 req/s derated, so that class alone needs 1.35 L4s, while the mean shape claims
4.35 req/s per L4 for the whole mix. Per class, the TPOT check is also stricter: the
long-input classes miss the 50 ms TPOT on the L40S (roofline TPOT grows with the KV read
per step), which the whole-workload check at the mean context passes. The classes plan is
the honest one, not the K=1 plan with a discount.

The same trace with its arrival times compressed 21.9-fold (x 0.0457, back to the source
hours' request rate, peak 74.75 req/s; built in memory, not committed) shows the same: at
TTFT 500 ms (with or without TPOT 50 ms) K=1 buys L40S + H100 for $128.42/day and K=4
2 x H100 for $167.52/day (-30.4%), and replaying each plan on that traffic with its SLO
gives the K=1 fleet **20.2% TTFT violations** (20.6% without the TPOT target) and the K=4
fleet **0.0%** in every class (`class_weighted` routing).

Reproduce (the two sample rows of the table):

```
uv run llmplan plan --model fixture:llama3-8b --trace data/traces/samples/azure2024_conv.csv --max-model-len 8192 --ttft-p95-ms 500 --tpot-p95-ms 50 --classes 1
uv run llmplan plan --model fixture:llama3-8b --trace data/traces/samples/azure2024_conv.csv --max-model-len 8192 --ttft-p95-ms 500 --tpot-p95-ms 50 --classes 2x2
```
