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

## Questions for founder

None yet.

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
