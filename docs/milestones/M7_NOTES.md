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

## Deviations from the design doc

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
