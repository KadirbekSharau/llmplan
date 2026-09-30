# M4 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the implementation
order in M4_DESIGN.md section 11.

## Implementation notes

- **Step 1 — Dependency.** Runtime adds `ortools>=9.15` (section 12; tested with 9.15.6755).
  `ortools.math_opt` ships no `py.typed` marker, so `pyproject.toml` has a mypy override
  (`ignore_missing_imports` for `ortools.*`). MathOpt objects are typed `Any` inside
  `planner/model.py` and `planner/solve.py`, the only modules that import OR-Tools.
- **Step 1 — Candidate checks, in order.** (1) `tp > gpu_count` -> `tp_gt_gpus`. (2) `tp`
  that does not divide `gpu_count` -> also `tp_gt_gpus` (reason "does not divide"): the
  section 5 packing constraint counts GPUs per instance, which is exact only when every tp
  divides the instance (with 6 GPUs and tp 4, two replicas would pass `4 + 4 <= 6 * 2` but
  need two instances each holding one). No shipped row is affected (gpu_count 1 or 8).
  (3) `tp` that does not divide the model's attention heads -> `no_fit` with that reason
  (M1 `fit()` would raise `ValidationError`; a candidate is a result, not an error).
  (4) M1 `fit()` with the request's full `EngineProfile` and `context_len = max_model_len`.
  (5) `perf.estimate(...)`; `PerfError` -> `no_perf`. (6) TTFT, then TPOT against the SLO.
- **Step 1 — `ReplicaConfig` of a candidate** takes `kv_dtype`, `gpu_memory_utilization` and
  `max_num_batched_tokens` from the engine profile; only tp, dtype and `max_num_seqs` are
  choice dimensions (section 4). The perf backends run M1 fit through `fit_for()`, which
  keeps the default `fixed_overhead_bytes` and `activation_multiplier`; the planner's own
  fit check uses the request's engine profile. They agree for the CLI (default engine).
- **Step 1 — Dominance pruning.** "VRAM use" is per-GPU weights + overhead + the KV cache that
  `max_num_seqs` full-length sequences need, capped at the KV budget
  (`candidates.vram_bytes_needed`). A candidate is pruned when another on the same row and
  tp has at least its request and token capacity and at most its VRAM use, strictly better
  in one; of exact ties the first in enumeration order stays. The MILP sees only (row, tp,
  capacities), so pruning cannot change its optimum. Pruned candidates keep status
  `eligible` in `PlanResult.candidates` (the status literal has no "pruned"); the count is
  an assumption line ("dominance pruning removed N of M eligible candidates").
- **Step 1 — Internal `Column`.** An eligible candidate with its derated capacities and VRAM
  use (a frozen dataclass, not a boundary model) is what pruning, the MILP and the baseline
  consume; it avoids re-deriving capacities and `Optional` checks in three places.
- **Step 1 — Test fake.** `tests/fake_planner.py` registers `FakeBackend` under `"fake"`
  with `llmplan.perf.register` (the M3 registry); tests set capacities per
  `(gpu_id, tp)` through the `fake_perf` fixture in `tests/conftest.py`, which restores
  them after each test. Fake GPUs have 10 TB of VRAM; fit is the real M1 code.
- **Step 2 — Order.** Test 10.1 asserts the baseline and the binding constraint, so both were
  built in step 2 with the solver instead of in step 3.
- **Step 2 — Solver parameters.** Every MILP solve sets `time_limit`, `random_seed`,
  `relative_gap_tolerance=0` (HiGHS defaults to 1e-4) and `enable_output=False`, and runs
  single-threaded. MathOpt rejects the generic `threads` for HiGHS, so it is passed as the
  HiGHS option `threads=1`. HiGHS fixes its thread pool the first time it runs in a
  process; a later solve with a different thread count fails with a bare
  `HighsStatus: kError`. Consequences: `available()` never probes HiGHS or CP-SAT (they are
  built in), probes SCIP/Gurobi once per process with the same single-thread parameters,
  and a HiGHS `RuntimeError` becomes a `SolverError` that explains this and suggests a
  fresh process or `--solver cp_sat`. (Found while running the Mélange toy model and the
  planner in one process.)
- **Step 2 — Status mapping.** `OPTIMAL` -> `optimal`; `FEASIBLE` with the time limit ->
  `feasible_time_limit` (plus an assumption line); `INFEASIBLE` and
  `INFEASIBLE_OR_UNBOUNDED` -> `InfeasiblePlan` (every variable is bounded, so unbounded is
  impossible); `NO_SOLUTION_FOUND` and anything else -> `SolverError` naming the
  termination. `InfeasiblePlan` is raised before any solve when no candidate is eligible
  (llama3-8b on the shipped catalogs with `--ttft-p95-ms 0.001`: "0 of 256 candidates
  eligible: 96 slo_ttft, 40 no_perf, 120 tp_gt_gpus"), and after the solve with
  "N of M candidates eligible, but no fleet of at most K instances per price row [on a
  single price row] meets X req/s and Y output tokens/s". No price row in scope is also
  `InfeasiblePlan` ("no price rows match gpu_ids ..., providers ..., commitments ...").
- **Step 2 — Guard.** After the solve, capacities are recomputed from the rounded integers;
  a fleet below either demand (beyond 1e-6 relative) raises `SolverError` rather than being
  reported.
- **Step 2 — `n_variables`/`n_constraints`** count the MathOpt model after pruning; rows
  without an eligible column get no variable.
- **Step 2 — Replicas.** The MILP gives replicas no cost, so `m_r` may sit anywhere between
  what demand needs and what the paid GPUs hold; the solver's (deterministic) value is
  reported. `ReplicaPlan.instances` is `ceil(count * tp / gpu_count)`, the instances those
  replicas occupy on their own; with several candidates on one row, `FleetItem.instances`
  is the row's total.
- **Step 3 — Baseline.** For every eligible column alone: replicas =
  `max(ceil(D_req / cap), ceil(D_tok / tok), 1)` (ratios guarded by 1e-9 so 30/3 is 10),
  instances = `ceil(replicas / replicas_per_instance)`, skipped above
  `max_instances_per_row`; cheapest wins, first in candidate order on ties. It is a
  `PlanResult` whose `solver.backend` is `"enumeration"` (status `optimal`, bound = cost,
  0 variables), whose `candidates` is empty (they are on the main result), and whose
  binding uses the same LP rule on its single column. `baseline` is also None when no
  single column can meet demand within `max_instances_per_row` (a mixed fleet can).
- **Step 4 — CP-SAT scaling** follows section 5 (cents per day, milli-units, demand rounded
  up) except that capacities are rounded down, not to nearest (see Deviations). Test 10.8
  and a unit test with capacities 3.3333 and 4.35 agree with HiGHS.
- **Step 4 — Determinism.** Same request, same `model_dump_json()`, within and across
  processes (checked by hand and by 10.3, 10.11, and the CLI JSON test).
  `SolverInfo.solve_time_s` is wall-clock, so it is excluded from serialization (see
  Deviations); the text output still shows it.
- **Step 5 — vLLM command.** vLLM's `--dtype` takes `bfloat16`/`float16`/`float32`, not
  `bf16`; section 8's `<bf16|float16>` is read as shorthand. int8/int4 render
  `--dtype auto` under the comment line, because the pre-quantized checkpoint decides the
  activation dtype. `--kv-cache-dtype` is emitted only for fp8 (bf16/fp16 KV is vLLM's
  `auto`). `--max-num-batched-tokens` is not emitted (not in section 8); if vLLM's default
  is larger than the 8192 the fit assumed, vLLM's own profiler shrinks the KV cache rather
  than running out of memory.
- **Step 5 — CLI.** `llmplan/cli_plan.py` holds the command, registered in `cli.py` with
  one import and `app.command("plan")(plan_command)`, like the M2/M3 sub-apps. Lists are
  comma-separated; a non-integer in `--tp`/`--max-num-seqs` exits 2 naming the flag.
  Exactly one of `--trace` or `--stats-json` is required. `--stats-json` accepts the
  `workload stats --format-out json` document (its `stats` object is used) or a bare
  `WorkloadStats` object, capped at 1 MB. `--trace` uses `compute_stats`' default 60 s
  window. The engine profile is vLLM's default (no engine flags in section 9).
  `--format vllm` prints only the commands, one comment line per replica plan.
- **Explanation lines.** Besides the section 7 assumptions (derating, leftover GPUs, peak
  window only, perf confidence of the chosen replicas), every result states the service-
  time nature of the SLO check, the pruning count, the LP relaxation's tight/slack
  constraints with their shadow prices (USD/day per req/s and per output token/s), and the
  fleet's headroom over each demand. The headroom line matters for small demands: in test
  10.9 (0.1 req/s) one L4 instance is +23,332% over request demand, which the binding label
  alone does not show.
- **Carry-over items (section 10b).** (1) `llmplan perf estimate --trace PATH` runs
  `compute_stats(load_workload(PATH))`; giving statistics flags as well exits 2. (2) The
  workload stats renderers moved to `Renderer.workload_stats` (`render/workload_text.py`
  for text, the JSON renderer for JSON). Output was compared byte for byte with the old
  functions on all five fixture traces at two windows and a day-long synthetic trace.
- **Stale `TODO(M2)` in tests/acceptance/test_m3.py** (a local `WorkloadStats` copy) was
  left alone: it is M3's acceptance file and outside M4's scope.
- **Package growth (~1,500 lines).** ARCHITECTURE.md section 1 asks for a justification above
  ~300 lines. M4 is the project's core: request/result models (~245 lines incl.
  docstrings), candidate evaluation and pruning (225), the formulation (131), solving and
  the LP explanation (182), the baseline (54), the `plan()` pipeline (283), three renderer
  modules (plan text 115, vLLM command 65, workload text 57, moved from `cli_workload.py`),
  and the CLI (154). Each module has one job; none exceeds 300 lines.
- **`docs/MILESTONES.md`** is set to "ready for review"; the CTO records the merge hash.

### 10.10 Mélange reproduction

**Outcome: not reproducible from the public repository; no comparison against the paper's
conversational result is claimed.**

- Source read on 2026-10-01: https://github.com/tyler-griggs/melange-release, `main` at
  `d46ab43855bcdbfed4740a42058d2e269374ea55` (2024-06-26), full history and all branches
  and pull-request refs.
- The repository contains the solver (`melange/solver.py`, PuLP), profiling scripts
  (`melange/profiling/`), a blog post (`docs/index.md`) and one input,
  `melange/config/example.json`, which the README calls a toy example: two GPUs (A10G at
  $1.01/h, A100-80GB at $3.67/h), a 2 x 2 request-size histogram without token boundaries,
  made-up throughputs, 30 req/s. No file in any commit holds the Arena (conversational)
  size distribution, the L4/A10G/A100/H100 profiling tables, the H100 or L4 prices, or the
  per-rate costs; the blog shows those results only as images (Arena_40-1.png,
  Arena_120-1.png). The paper (arXiv 2404.14527) was outside the allowed network scope.
  So neither "Mélange's reported cost" nor "Mélange's own profiling numbers" for the
  conversational scenario could be obtained, and the 10% target could not be evaluated.
- Sanity check on the toy input only (not the conversational scenario; not committed,
  run from a scratch script): Mélange's ILP re-implemented in MathOpt reproduces the
  README's answer, 3 x A10G + 1 x A100 = $6.70/h at slice factor 1; at the README's
  recommended slice factor 4 (and 16) it gives 2 x A10G + 1 x A100 = $5.69/h. llmplan,
  given each GPU's throughput on the whole request mix (distribution-weighted harmonic
  mean of the bucket throughputs: A10G 1 / (0.2/2 + 0.1/1 + 0.5/5 + 0.2/2) = 2.5 req/s,
  A100 1 / (0.2/20 + 0.1/20 + 0.5/40 + 0.2/20) = 26.67 req/s), chooses 2 x A10G + 1 x A100
  = $5.69/h (2.5 * 2 + 26.67 = 31.67 >= 30), identical to Mélange at slice factor 4; its
  homogeneous baseline is 2 x A100 = $7.34/h. With fractional slice assignment Mélange's
  bound is 1 x A10G + 1 x A100 = $4.68/h, reached by routing a third of the (1,0) bucket
  to the A10G.
- Finding for the roadmap: llmplan sizes every replica for the same (mean) request shape,
  so it captures Mélange's rate-granularity gains but not its request-size routing (small
  requests to cheap GPUs, large ones to fast GPUs), which is the paper's main source of
  savings. Reproducing the conversational result would need per-size-bucket demand in the
  formulation (a later milestone) and the profiling data, which is not public.

## Deviations from the design doc

- **`binding` comes from the LP relaxation over the chosen candidates, not from the integer
  solution's slack.** Section 7 says zero slack (tolerance 1e-6 relative) after solving
  marks a constraint binding, but test 10.1 expects `"requests"` while its integer fleet
  has capacity 35.0 against demand 34 (slack 1 req/s, 2.9%) and no token demand; the
  literal rule gives `"none"`. MILESTONES.md describes the explanation as "derived from
  slack and duals of the LP relaxation". Implemented: fix the set of columns the fleet
  uses, relax integrality, solve that LP with GLOP, and mark a demand constraint binding
  when its slack is at most 1e-6 relative (shadow prices are reported alongside).
  Checks: 10.1, columns {A, B}: B costs $19/32 = $0.594/h per req/s against A's
  $2/3 = $0.667, so the LP buys 34/32 = 1.0625 B instances; requests tight, tokens slack
  -> `"requests"`. 10.4, column {A}: replicas = max(6/3, 900/300) = 3; tokens tight,
  requests slack (9 > 6) -> `"tokens"`. 10.7, column {B tp4}: 40/20 = 2 replicas;
  requests tight. Why restricted to the chosen columns rather than all candidates: in 10.4
  with row B present (fake default 1,000 tokens/s per replica), the unrestricted LP uses B
  fractionally (cheaper per req/s and per token/s), needs max(6/32, 900/8000) = 0.1875
  instances and reports `"requests"`, although the fleet actually bought (3 x A) is sized
  by tokens. The 10.4 test keeps both rows and passes. `"none"` occurs only if neither
  constraint is tight in that LP (e.g. instance upper bounds bind).
- **`PlanOptions.perf_backend: str = "auto"`** instead of `Literal["auto", "roofline",
  "table"]`. Section 10 requires a fake backend "registered under the name `fake`" through
  the M3 registry, which the literal would reject. `plan()` validates the key against the
  registry (`UnknownRegistryKey`, exit 2); the CLI still offers only auto/roofline/table.
- **`CandidateEval.fit: FitResult | None`.** Candidates rejected before the memory check
  (`tp_gt_gpus`, or tp not dividing the model's heads) have no fit; section 4 checks tp
  first, and M1 `fit()` raises for a tp that does not divide the heads.
- **`SolverInfo.solve_time_s` is excluded from serialization** (`Field(exclude=True)`).
  Tests 10.3 and 10.11 and DEFINITION_OF_DONE.md section 5 require byte-identical JSON;
  a wall-clock time cannot be. The value is still on the model and in the text output.
- **CP-SAT capacities are rounded down** (`floor(cap * 1000 + 1e-9)`) instead of
  `round(cap * 1000)`. With demand rounded up, rounding capacities down makes every scaled
  solution feasible in the unscaled model; rounding to nearest could overstate a replica
  by up to 0.0005 req/s. The 1e-9 guard keeps exact decimals exact (4.35 * 1000 =
  4349.999...). Costs use `round(24 * price * 100)` as specified.
- **`render/vllm_cmd.py` is two functions, not a `Renderer` registry member.** ARCHITECTURE.md
  listed `vllm_cmd` as a renderer; the `Renderer` protocol has one method per result type
  (fit, model_info, gpus, ...), and a vLLM command exists only for a replica config.
  `serve_command(model, config)` and `plan_commands(request, result)` are used by the text
  renderer and by `llmplan plan --format vllm`. ARCHITECTURE.md sections 3 and 6 updated.
- **`Renderer` gains `workload_stats` and `plan`** (carry-over item and M4 result),
  ARCHITECTURE.md section 6 updated.
- **CLI adds `--gpu-catalog PATH` and `--prices PATH`** to `llmplan plan`. ARCHITECTURE.md
  section 8 lets users override the catalogs with their own files, but section 9 of the
  design uses `--gpus` for GPU ids, so the GPU catalog file is `--gpu-catalog`.
  ARCHITECTURE.md section 8 updated.
- **`llmplan.types.Commitment`** alias added for `PlanOptions.commitments`; `PriceRow`
  now uses it (same values). ARCHITECTURE.md sections 3 and 4 updated.
- **New modules beyond section 2's list:** `render/plan_text.py` and
  `render/workload_text.py` (keep `render/text.py` under 300 lines), `cli_plan.py` (the
  M2/M3 sub-module pattern). ARCHITECTURE.md section 3 updated.

## Questions for founder

None new. Section 13's question (autoscaling by hour with `hourly_rps` in M4 or M5) keeps
its interim decision: M4 sizes for the peak window only, and every result says so. Per the
founder's rule in docs/FOUNDER_QUESTIONS.md, these CTO-decidable items were decided here and
are open to CTO review:

- **Mélange data.** 10.10 needs the Arena size distribution and the four-GPU profiling
  tables, which the public repository does not contain. Options: ask the authors, read the
  paper's tables (outside this milestone's allowed network scope), or accept the toy-model
  sanity check above.
- **Request-size routing.** Matching Mélange's savings needs per-size-bucket demand in the
  formulation; a candidate for a later milestone, not M4.
