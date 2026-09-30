# M5 Design — Simulation and Timeline

Status: ready for implementation once M4 is merged. Assignee: developer agent.
Branch: `m5-simulate`. Prerequisites: docs/PLAN.md, docs/ARCHITECTURE.md,
docs/DEFINITION_OF_DONE.md, and the M1 to M4 public APIs.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9.

---

## 1. Goal

Replay a real or synthetic request trace against a planned fleet and show what the plan
would actually do over time: per-replica utilization, VRAM split (weights / KV in use /
free), queue depth, and SLO violations per window. This is the proof behind the planner's
answer and the "scheduling visualizer" the product promises. It also catches what a
peak-window sizing misses: queueing under bursts.

## 2. Deliverables

1. `llmplan/simulate/events.py`: discrete-event engine (heap-based, pure Python, numpy for
   aggregation).
2. `llmplan/simulate/replica.py`: replica state (slots, queue, KV tokens in use).
3. `llmplan/simulate/routing.py`: routing policy registry with `least_outstanding`.
4. `llmplan/simulate/timeline.py`: `Timeline`, `WindowRecord`, `ReplicaWindowRecord`,
   `SimulationSummary`, aggregation.
5. `llmplan/simulate/__init__.py`: `replay(plan, workload, *, window_s, slo, options) -> Timeline`.
6. `llmplan/render/plots.py`: matplotlib (Agg backend) figure with four stacked panels
   (rps demand vs capacity, utilization per replica, KV usage per replica, queue depth
   and violations), saved as PNG; `llmplan/render/timeline_json.py`.
7. CLI `llmplan simulate`.
8. Tests, README, CHANGELOG, `M5_NOTES.md`, ARCHITECTURE.md updates.

## 3. Data models

```python
class SimOptions(BaseModel, frozen=True):
    window_s: float = 60.0
    routing: Literal["least_outstanding", "round_robin"] = "least_outstanding"
    max_requests: int = 500_000          # trace is truncated (with a note) beyond this
    seed: int = 0                        # only used by tie-breaking
    ttft_budget_ms: float | None = None  # defaults to slo.ttft_ms_p95 when given
    tpot_budget_ms: float | None = None

class ReplicaWindowRecord(BaseModel, frozen=True):
    replica_index: int
    utilization: float                   # busy slot-seconds / (slots * window_s), 0..1
    kv_tokens_in_use_mean: int
    kv_tokens_in_use_max: int
    kv_bytes_in_use_max: int
    weight_bytes: int
    queue_depth_mean: float
    queue_depth_max: int
    requests_started: int
    requests_completed: int

class WindowRecord(BaseModel, frozen=True):
    index: int
    start_s: float
    arrivals: int
    completions: int
    demand_rps: float
    capacity_rps: float                  # from the plan, constant
    ttft_ms_p95: float | None            # over requests completed in this window
    e2e_ms_p95: float | None
    ttft_violations: int
    tpot_violations: int
    replicas: tuple[ReplicaWindowRecord, ...]

class SimulationSummary(BaseModel, frozen=True):
    n_requests: int
    n_truncated: int
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_p95: float
    e2e_ms_p95: float
    ttft_violation_pct: float
    tpot_violation_pct: float
    mean_utilization: float              # across replicas and windows
    max_queue_depth: int

class Timeline(BaseModel, frozen=True):
    plan_cost_usd_per_day: float
    options: SimOptions
    windows: tuple[WindowRecord, ...]
    summary: SimulationSummary
    assumptions: tuple[str, ...]
```

## 4. Replica model

For each `ReplicaPlan` in the `PlanResult`, instantiate `count` replicas. Each has:
- `slots = candidate.perf.effective_batch`.
- `kv_token_capacity = candidate.fit.kv_token_capacity` and
  `kv_bytes_per_token = candidate.fit.kv_bytes_per_token_per_gpu * tensor_parallel`.
- Service times from `candidate.perf`: `prefill_s(n) = n / prefill_tokens_per_s`,
  `tpot_s = tpot_ms_p50 / 1000` (constant; assumption noted: batch-size dependence of
  tpot is not modeled in M5, the estimate is the full-batch value, which is conservative).
- A request occupies one slot and `input_tokens + output_tokens` KV tokens (assumption:
  full allocation at admission, conservative) for `prefill_s(input) + output * tpot_s`.
- Admission requires a free slot **and** enough KV tokens; otherwise the request waits in
  the replica's FIFO queue. (This is where VRAM, not slots, can be the real limit.)

## 5. Event loop

Heap of `(time, seq, kind, payload)`; `seq` breaks ties deterministically. Events: `arrive`,
`complete`. On `arrive`: route (section 6), try to admit, else enqueue. On `complete`:
free slot and KV, admit from the queue in FIFO order while resources allow.

Per request record: `arrival_s`, `start_s`, `ttft_s = (start_s - arrival_s) + prefill_s`,
`e2e_s = (complete_s - arrival_s)`, `tpot_s`, `replica_index`.

Utilization accounting: integrate busy slots over time per replica (update on every state
change), then split into windows.

Complexity target: 500k requests in under 30 s on a laptop CPU. Use `heapq`, plain tuples
in the loop, and numpy only for the per-window aggregation afterwards.

## 6. Routing

Registry `simulate/routing.py`: `least_outstanding` (fewest admitted + queued requests,
ties by lowest replica index), `round_robin`. Both pure functions over replica states.

## 7. Windows and summary

Windows as in M2 (half-open, from first arrival). Percentiles over requests *completed* in
the window; `None` if none completed. Violations: `ttft_ms > ttft_budget_ms` and
`tpot_ms > tpot_budget_ms` when budgets are set; otherwise 0 and the assumption says so.

## 8. CLI

```
llmplan simulate --plan PLAN.json --trace PATH [--window 60] [--routing least_outstanding]
                 [--ttft-p95-ms 500] [--tpot-p95-ms 50] [--png OUT.png] [--format text|json]
```
`PLAN.json` is the `--format json` output of `llmplan plan` (M4); add a `PlanResult`
loader in `planner/result.py` if missing. Text output: summary block plus a compact table of
windows (index, demand, completions, utilization mean, queue max, violations).

## 9. Acceptance tests (`tests/acceptance/test_m5.py`)

Tests build a `PlanResult` directly with the M4 fake perf backend values (fixed
`prefill_tokens_per_s`, `tpot_ms_p50`, `effective_batch`) and a fixture GPU with large VRAM,
so all service times are exact.

**9.1 No queueing.** One replica, `effective_batch=1`, prefill 1000 tok/s, tpot 10 ms.
Workload: 100 requests, one every 1.0 s, input 100 tokens, output 40 tokens (service =
0.1 + 0.4 = 0.5 s). Expected: every `ttft_ms == 100.0`, every `e2e_ms == 500.0`,
`max_queue_depth == 0`, window utilization (window 60 s, first window) `== 0.5` (30 busy
slot-seconds over 60), summary `ttft_violation_pct == 0.0` with budget 200 ms.

**9.2 Overload queues.** Same replica, one request every 0.25 s for 100 requests. Service
0.5 s so the queue grows by one every 0.5 s. Expected: `max_queue_depth >= 45`, the last
request's `ttft_ms` > 20_000, `ttft_violation_pct > 50` with budget 200 ms, utilization of
the busy window `== 1.0` (to 1e-9).

**9.3 Two replicas halve the wait.** Same as 9.2 but two replicas with `least_outstanding`
routing: no queueing at all (arrival every 0.25 s, two slots, service 0.5 s), so
`max_queue_depth == 0` and all `ttft_ms == 100.0`.

**9.4 KV limits admission before slots.** One replica, `effective_batch=8`,
`kv_token_capacity=1000`; requests need 300 tokens each (input 200, output 100), so only 3
fit at once. Ten simultaneous arrivals at t=0: exactly 3 start at t=0, the 4th starts when
the first completes; `queue_depth_max == 7`.

**9.5 Routing tie-break determinism.** With `round_robin` and equal replicas, request `i`
lands on replica `i % n`. Two runs give byte-identical `model_dump_json()`.

**9.6 Real plan, well provisioned.** Take the M4 acceptance plan 10.9 (llama3-8b, roofline)
and replay the synthetic workload `generate(rate_rps=stats.peak_window_rps * 0.5, ...)`
(so demand is half the plan's peak): `ttft_violation_pct < 5` with budget
`2 * perf.ttft_ms_p95`, `mean_utilization < 0.8`.

**9.7 Real plan, under-provisioned.** Same plan with `replicas` truncated to one replica
and rate at 3x the plan's demand: `ttft_violation_pct > 50`, `max_queue_depth > 10`.

**9.8 Truncation.** `max_requests=50` on a 100-request workload: `n_truncated == 50` and an
assumption note mentions truncation.

**9.9 Plot.** `render.plots.save_png(timeline, path)` writes a PNG whose first 8 bytes are
the PNG signature; runs headless (no display).

**9.10 Performance.** A synthetic 200k-request workload on 4 replicas replays in under 30 s
(mark `@pytest.mark.slow`; excluded from the default run but present).

## 9b. Carry-over item (in scope for M5)
- Replace the local `WorkloadStats` copy in `tests/acceptance/test_m3.py` with an import of `llmplan.workload.schema.WorkloadStats` and delete the `TODO(M2)`.

## 10. Implementation order
1. Replica state, event loop, per-request records; 9.1, 9.2, 9.4.
2. Routing registry; 9.3, 9.5.
3. Windows, summary, `Timeline`; 9.8.
4. Plots and JSON; 9.9.
5. CLI, `PlanResult` loader, real-plan tests 9.6, 9.7, perf test 9.10, docs.

## 11. Allowed dependencies
Runtime adds: `matplotlib`. Nothing else.

## 12. Questions for founder
- Whether the UI (M6) should show per-request scatter plots (heavier) or window aggregates
  only. Interim: aggregates only; per-request data is available in JSON.
