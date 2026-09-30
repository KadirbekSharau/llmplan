# M4 Design — MILP Fleet Planner

Status: ready for implementation once M2 and M3 are merged. Assignee: developer agent.
Branch: `m4-planner`. Prerequisites: docs/PLAN.md, docs/ARCHITECTURE.md,
docs/DEFINITION_OF_DONE.md, and the M1 to M3 public APIs.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 10.

---

## 1. Goal

Given one model, a workload's statistics, an SLO, and a price catalog, choose the fleet
(instances of which price rows) and the replica configurations (tensor parallel, dtype,
max_num_seqs) that minimize $/day while meeting demand and the SLO. Explain the answer:
what was binding, what each candidate costs per unit of capacity, and why rejected
candidates were rejected. Compare against the best homogeneous fleet (the "naive" answer).

This is the mixed-integer program the project exists for. It is an offline planner; solve
times of seconds to a minute are acceptable.

## 2. Deliverables

1. `llmplan/planner/request.py`: `SLO`, `PlanOptions`, `PlanRequest`.
2. `llmplan/planner/candidates.py`: candidate enumeration and pre-solve evaluation.
3. `llmplan/planner/model.py`: MathOpt formulation (pure function: candidates -> model).
4. `llmplan/planner/solve.py`: backend selection, parameters, determinism, `SolverInfo`.
5. `llmplan/planner/result.py`: `PlanResult`, `FleetItem`, `ReplicaPlan`, `CandidateEval`,
   explanation.
6. `llmplan/planner/baseline.py`: best homogeneous fleet by enumeration (no solver).
7. `llmplan/planner/__init__.py`: `plan(PlanRequest) -> PlanResult`.
8. `llmplan/render/vllm_cmd.py`: vLLM command line per replica plan.
9. CLI `llmplan plan`.
10. Tests, README, CHANGELOG, `M4_NOTES.md`, ARCHITECTURE.md updates.

## 3. Data models

```python
class SLO(BaseModel, frozen=True):
    ttft_ms_p95: float | None = None          # service-time TTFT bound (queueing is M5)
    tpot_ms_p95: float | None = None
    utilization_target: float = 0.8           # 0 < x <= 1; capacity is derated by this

class PlanOptions(BaseModel, frozen=True):
    gpu_ids: tuple[str, ...] | None = None    # None = every GPU with at least one price row
    providers: tuple[str, ...] | None = None
    commitments: tuple[Commitment, ...] = ("on_demand",)
    tensor_parallel_choices: tuple[int, ...] = (1, 2, 4, 8)
    dtype_choices: tuple[DType, ...] = ("bf16", "fp8")
    max_num_seqs_choices: tuple[int, ...] = (32, 64, 128, 256)
    max_model_len: int                        # required; <= model.max_position_embeddings
    homogeneous: bool = False                 # force a single price row
    perf_backend: Literal["auto", "roofline", "table"] = "auto"
    solver: Literal["highs", "cp_sat", "scip", "gurobi"] = "highs"
    time_limit_s: float = 60.0
    seed: int = 0
    max_instances_per_row: int = 1000         # big-M bound

class PlanRequest(BaseModel, frozen=True):
    model: ModelSpec
    stats: WorkloadStats
    slo: SLO
    engine: EngineProfile
    options: PlanOptions
    # catalogs are passed explicitly so tests can inject rows:
    gpus: Mapping[str, GPUSpec]
    prices: tuple[PriceRow, ...]

class CandidateEval(BaseModel, frozen=True):
    price_row: PriceRow
    config: ReplicaConfig
    replicas_per_instance: int                # gpu_count // tensor_parallel
    fit: FitResult
    perf: PerfEstimate | None
    status: Literal["eligible", "no_fit", "slo_ttft", "slo_tpot", "no_perf", "tp_gt_gpus"]
    reason: str
    usd_per_hour_per_rps: float | None        # price / (replicas_per_instance * derated capacity)

class ReplicaPlan(BaseModel, frozen=True):
    candidate: CandidateEval
    count: int                                # replicas of this candidate
    instances: int                            # instances of its price row used

class FleetItem(BaseModel, frozen=True):
    price_row: PriceRow
    instances: int
    usd_per_day: float

class SolverInfo(BaseModel, frozen=True):
    backend: str
    status: Literal["optimal", "feasible_time_limit"]
    objective_usd_per_day: float
    best_bound_usd_per_day: float | None
    solve_time_s: float
    n_variables: int
    n_constraints: int

class PlanResult(BaseModel, frozen=True):
    fleet: tuple[FleetItem, ...]
    replicas: tuple[ReplicaPlan, ...]
    cost_usd_per_day: float
    baseline: PlanResult | None               # best homogeneous fleet, None if homogeneous already
    baseline_saving_pct: float | None
    demand_rps: float                         # stats.peak_window_rps
    demand_output_tokens_per_s: float         # stats.peak_output_tokens_per_s
    capacity_rps: float                       # derated, summed over replicas
    capacity_output_tokens_per_s: float
    binding: Literal["requests", "tokens", "both", "none"]   # constraint with zero slack (tol 1e-6)
    candidates: tuple[CandidateEval, ...]     # all, including rejected, sorted by usd_per_hour_per_rps
    solver: SolverInfo
    assumptions: tuple[str, ...]
```

## 4. Candidate enumeration (`candidates.py`)

For each price row in scope x tp in choices x dtype in choices x max_num_seqs in choices:
1. `tp > price_row.gpu_count` -> status `tp_gt_gpus`.
2. `fit()` with the engine profile and `context_len = max_model_len`. `fits == False` ->
   `no_fit` with the fit's `binding` in the reason.
3. `perf.estimate(...)` with `options.perf_backend`. `PerfError` -> `no_perf`.
4. SLO: `perf.ttft_ms_p95 > slo.ttft_ms_p95` -> `slo_ttft`; same for tpot.
5. Eligible: `replicas_per_instance = gpu_count // tp`,
   `derated_rps = perf.requests_per_s_capacity * utilization_target`,
   `derated_tps = perf.decode_tokens_per_s * utilization_target`,
   `usd_per_hour_per_rps = price / (replicas_per_instance * derated_rps)`.

Dominance pruning (keeps the MILP small and is exact): among eligible candidates sharing
the same price row and tp, keep only those not dominated on both `derated_rps` and
`derated_tps` by another with the same or lower VRAM use. Record the pruned count.

## 5. Formulation (`model.py`)

Sets: eligible candidates `r ∈ R`, price rows `p ∈ P`, `R_p` = candidates using row `p`.
Parameters: `price_p` ($/hour), `g_p` (GPUs per instance), `tp_r`, `cap_r` (derated rps),
`tok_r` (derated output tokens/s), `D_req = demand_rps`, `D_tok = demand_output_tokens_per_s`.

Variables: `n_p ∈ Z≥0` instances of row `p` (upper bound `max_instances_per_row`);
`m_r ∈ Z≥0` replicas of candidate `r`; if `homogeneous`: `y_p ∈ {0,1}`.

```
minimize    Σ_p 24 * price_p * n_p
subject to  Σ_{r∈R_p} tp_r * m_r  <=  g_p * n_p            for each p     (GPU packing)
            Σ_r cap_r * m_r        >=  D_req                                (request demand)
            Σ_r tok_r * m_r        >=  D_tok                                (token demand)
            [homogeneous] n_p <= max_instances_per_row * y_p ;  Σ_p y_p <= 1
```
Leftover GPUs on an instance are paid for and unused (documented assumption).

For CP-SAT all coefficients are integers: prices in cents/day (`round(24*price*100)`),
capacities in milli-units (`round(cap*1000)`); the `demand` side is rounded up. The
objective reported to the user is recomputed from the chosen integers with float prices so
both backends print the same cost.

## 6. Solving (`solve.py`)

OR-Tools MathOpt. Parameters: `time_limit=options.time_limit_s`, `threads=1`,
`random_seed=options.seed`, `enable_output=False`. Status handling:
- `OPTIMAL` -> `status="optimal"`.
- `FEASIBLE` at time limit -> `"feasible_time_limit"` with `best_bound`.
- `INFEASIBLE` -> `InfeasiblePlan` with a reason built from the candidate statuses
  (e.g. "0 of 96 candidates eligible: 40 no_fit, 56 slo_ttft").
- Anything else -> `SolverError` with the raw termination.
Backends: `highs` (default, always installed), `cp_sat` (always installed), `scip` and
`gurobi` only if MathOpt reports them available; otherwise `SolverError("backend not
available")`.

## 7. Baseline and explanation

`baseline.py`: for each price row alone, the minimum instances such that the best single
eligible candidate on that row meets both demands; the cheapest over rows is the baseline.
Pure enumeration, no solver. `baseline_saving_pct = (baseline - cost) / baseline * 100`.

`binding`: after solving, compute slack of the request and token constraints; zero slack
(tolerance `1e-6` relative) marks it binding.

Assumptions appended to every result: utilization derating, leftover-GPU waste, single
peak window sizing (no autoscaling in M4), perf confidence of the chosen candidates.

## 8. vLLM command renderer

```
vllm serve <model.id> --tensor-parallel-size <tp> --max-num-seqs <n> --max-model-len <L>
  --gpu-memory-utilization <u> --dtype <bf16|float16> [--quantization fp8] [--kv-cache-dtype fp8]
```
`dtype="fp8"` renders `--dtype bfloat16 --quantization fp8`. `int4`/`int8` render a comment
line stating a pre-quantized checkpoint is required. `model.id` starting with `fixture:`
renders `<MODEL_ID>` as a placeholder.

## 9. CLI

```
llmplan plan --model <id> (--trace PATH | --stats-json PATH) --max-model-len 8192
             [--ttft-p95-ms 500] [--tpot-p95-ms 50] [--utilization 0.8]
             [--gpus h100-sxm-80gb,l40s-48gb] [--providers aws,lambda]
             [--tp 1,2,4,8] [--dtypes bf16,fp8] [--max-num-seqs 32,64,128,256]
             [--homogeneous] [--perf-backend auto] [--solver highs] [--time-limit 60]
             [--format text|json|vllm]
```
Text output: fleet table, replica table with vLLM command lines, cost vs baseline, binding
constraint, and the top 10 candidates by `usd_per_hour_per_rps` with status and reason.

## 10. Acceptance tests (`tests/acceptance/test_m4.py`)

Tests inject a fake `PerfBackend` (registered under the name `"fake"` in the test) that
returns fixed capacities per `(gpu_id, tp)`, and test-only price rows. Fit is real (M1) but
the fake GPUs have huge VRAM so every candidate fits. `utilization_target=1.0`,
`tensor_parallel_choices=(1,)`, `dtype_choices=("bf16",)`, `max_num_seqs_choices=(64,)`,
token demand set to 0 unless stated.

**10.1 Heterogeneous beats homogeneous.**
Row A: 1 GPU, $2.00/h, capacity 3 rps per replica. Row B: 8 GPUs, $19.00/h, 4 rps per replica
(32 rps per instance). Demand 34 rps.
Expected: fleet = {B: 1 instance, A: 1 instance}, `cost_usd_per_day == 21*24 == 504.0`,
baseline = 12 x A = `576.0` (2 x B = 912 is worse), `baseline_saving_pct == pytest.approx(12.5)`,
`binding == "requests"`, `capacity_rps == 35.0`, solver status `optimal`.

**10.2 Homogeneous option.** Same rows, demand 34, `homogeneous=True`: fleet = 12 x A,
cost 576.0, `baseline is None`.

**10.3 Tie stays deterministic.** Row B at $20.00/h, demand 30: A x 10 and B x 1 both cost
$480/day. Two solves with the same seed give identical `model_dump_json()`; the chosen
fleet must be one of the two and `cost_usd_per_day == 480.0`.

**10.4 Token demand binds.** Row A capacity 3 rps and 300 tok/s per replica; demand 6 rps
and 900 tok/s -> 3 x A (tokens bind), `binding == "tokens"`, cost 144.0.

**10.5 SLO rejection.** Fake backend returns `ttft_ms_p95 = 800` for row B and 100 for A;
`SLO(ttft_ms_p95=500)`; demand 34 -> fleet 12 x A, and the B candidate appears in
`candidates` with `status == "slo_ttft"` and a reason containing "800" and "500".

**10.6 Infeasible.** Every candidate rejected (SLO 10 ms) -> `InfeasiblePlan` whose message
contains "0 of" and "slo_ttft".

**10.7 GPU packing.** Row B only (8 GPUs), tp choices (1, 2, 4): fake capacities per
replica tp1: 4 rps, tp2: 9 rps, tp4: 20 rps. Demand 40 rps. Best packing on one instance:
2 x tp4 = 40 rps (exact). Expected fleet 1 x B, replicas = {tp4: 2}, cost 456.0 (B at $19).

**10.8 CP-SAT agrees with HiGHS.** Run 10.1 with `solver="cp_sat"`: same fleet and cost.

**10.9 Real catalogs, roofline backend.** `fixture:llama3-8b`, shipped `data/gpus.yaml` and
`data/prices.yaml`, stats from `tests/fixtures/workload_10.csv` (M2), SLO none,
`max_model_len 8192`: solves in under 30 s, every replica's `fit.fits` is True, cost > 0,
and the result renders a vLLM command containing `--tensor-parallel-size`.

**10.10 Mélange reproduction — SUPERSEDED (CTO, 2026-10-01).** The public melange-release repository contains only a toy input, and a comparison is not meaningful until request-size demand classes exist. Replaced by M7_DESIGN.md section 6 (brute-force exactness test plus a same-inputs cross-check). Original text kept for the record:
Using the public `melange-release` repository's GPU set, prices, and the conversational
workload profile, run the planner with the table backend seeded from Mélange's own
profiling numbers. Record the achieved cost and Mélange's reported cost in `M4_NOTES.md`.
Target: within 10%. If the data cannot be obtained, say so in the notes; do not fabricate.

**10.11 Determinism and renderer.** JSON output byte-identical across two runs; `vllm_cmd`
for a `dtype="fp8"` replica contains `--quantization fp8`.

## 10b. Carry-over items from M2/M3 (in scope for M4)
- Wire `llmplan perf estimate --trace PATH` to `load_workload` + `compute_stats` (M3 left a `TODO(M2)`).
- Move the workload stats text/JSON output from `llmplan/cli_workload.py` into the `render` package (M2 left a `TODO(M4)`), keeping output byte-identical.

## 11. Implementation order
1. Models, candidates, dominance pruning, fake backend fixture.
2. Formulation + HiGHS solve + 10.1, 10.2, 10.3, 10.4.
3. SLO handling, infeasible path, baseline, binding + 10.5, 10.6, 10.7.
4. CP-SAT scaling + 10.8; determinism + 10.11.
5. Renderer, CLI, real-catalog run 10.9, Mélange notes 10.10, docs.

## 12. Allowed dependencies
Runtime adds: `ortools`. Nothing else.

## 13. Questions for founder
- Whether autoscaling by hour (using `hourly_rps`) belongs in M4 or M5. Interim: deferred
  to a later milestone; M4 sizes for the peak window only.
