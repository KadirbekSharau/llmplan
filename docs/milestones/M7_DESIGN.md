# M7 Design — Request-size demand classes and planner validation

Status: drafted; scheduled after M6. Branch: `m7-demand-classes`.
Prerequisites: M1 to M6 merged. Decided by the CTO on 2026-10-01 after the M4 findings.

---

## 1. Why

M4 sizes every replica for the workload's mean request shape. Real traffic mixes short chat
turns with long document prompts, and cheaper GPUs are often the best home for the short
ones. Mélange (Berkeley, 2024) shows that most of the heterogeneous-fleet saving comes from
this routing, not from GPU price differences alone. Without demand classes the planner
under-reports savings and cannot be compared to Mélange at all.

M4's Mélange acceptance test (10.10) is superseded by this milestone. The public
melange-release repository holds only a toy input, so the comparison is redefined in
section 6 as: same inputs, two independent solvers, same answer.

## 2. Goal

1. Bucket a workload into K request-size classes; compute per-class demand.
2. Extend the MILP so each replica type serves a mix of classes; output per-class routing
   weights.
3. Extend the simulation with class-aware routing so the timeline proves the plan.
4. Validate the planner exactly against brute force on small instances, and against
   Mélange's solver on shared inputs.

## 3. Workload classes (`llmplan/workload/classes.py`)

```python
class DemandClass(BaseModel, frozen=True):
    index: int
    input_lo: int; input_hi: int          # inclusive token bounds
    output_lo: int; output_hi: int
    share: float                          # fraction of requests
    peak_rps: float                       # class demand in the fleet-wide peak window
    peak_output_tokens_per_s: float
    input_tokens_mean: float; output_tokens_mean: float
    input_tokens_p95: float; output_tokens_p95: float

def classify(workload: Workload, *, input_bins: int = 2, output_bins: int = 2,
             method: Literal["quantile", "fixed"] = "quantile",
             edges: tuple[tuple[int, ...], tuple[int, ...]] | None = None) -> tuple[DemandClass, ...]
```
Quantile edges by default (median splits give a 2x2 grid); fixed edges for users who know
their traffic. Class demand is measured in the **fleet-wide** peak window so classes are
consistent in time. Classes with fewer than 1% of requests are merged into their nearest
neighbour (noted).

## 4. Formulation change (`planner/model.py`)

New parameters: classes `k ∈ K` with demand `D_k` (rps) and `T_k` (output tokens/s); for
each candidate `r` and class `k`, capacity `cap_{r,k}` and `tok_{r,k}` from the perf
backend evaluated at the class's shape (M3 already takes stats; call it per class).
Candidates whose SLO fails for a class are simply ineligible for that class (`cap_{r,k}=0`),
not rejected outright.

New continuous variables `x_{r,k} ≥ 0`: number of replica-equivalents of `r` devoted to `k`.

```
minimize    Σ_p 24 * price_p * n_p
subject to  Σ_{r∈R_p} tp_r * m_r <= g_p * n_p                     (GPU packing, unchanged)
            Σ_k x_{r,k} <= m_r                                     for each r   (replica time share)
            Σ_r cap_{r,k} * x_{r,k} >= D_k                          for each k   (class request demand)
            Σ_r tok_{r,k} * x_{r,k} >= T_k                          for each k   (class token demand)
```
Still a MILP with the same integer variables; the LP part is tiny. Routing weight for
class `k` to replica type `r` is `x_{r,k} * cap_{r,k} / D_k`. Backward compatible: K=1
reproduces M4 exactly (acceptance test).

Result additions: `PlanResult.classes`, `PlanResult.routing: tuple[RoutingRule, ...]`
(`class_index, candidate, weight`), per-class binding.

## 5. Simulation change (`simulate/routing.py`)

New routing policy `class_weighted`: classify each request by the plan's class edges, pick
a replica type by the routing weights (deterministic: lowest cumulative-deficit rule, not
random), then least-outstanding within the type. Timeline gains per-class p95 latency and
violations.

## 6. Validation

**6.1 Exactness vs brute force (replaces M4 10.10 as an acceptance test).** For randomly
generated small instances (≤ 3 price rows, ≤ 4 candidates, ≤ 2 classes, instance counts
≤ 6), enumerate every integer fleet, solve the inner LP for allocations, and assert the
MILP objective equals the enumerated optimum to 1e-6. Fifty seeds, under 10 s total.

**6.2 Mélange cross-check (recorded, not asserted).** Run melange-release's solver and
llmplan on identical inputs: their toy example, plus three synthetic scenarios we build
from our own catalogs and the NIM benchmark rows (short-chat heavy, long-document heavy,
mixed). Record both objectives in `M7_NOTES.md`; investigate any gap above 5%. Their
slice-factor parameter maps to our continuous `x_{r,k}` (which is the slice-factor-to-
infinity limit), so llmplan should be at or below their cost on every scenario.

**6.3 Real-trace saving.** On the bundled Azure 2024 conversation sample with the shipped
catalogs, report the cost with K=1 versus K=4 classes. Recorded in the notes and surfaced in
the UI as "saving from request-size routing".

## 7. CLI and UI

`llmplan plan --classes 2x2|1|3x3|fixed:<edges>`; text output gains a routing table.
UI: a "request-size classes" selector in Advanced, default 2x2, and the routing table in
the results.

## 8. Acceptance tests (to be finalized when M6 merges)

- K=1 reproduces every M4 acceptance result byte-for-byte.
- A two-class instance where a cheap GPU meets the short-class SLO but not the long-class
  SLO chooses the cheap GPU for the short class only; the routing weights sum to 1 per class.
- 6.1 exactness test passes for 50 seeds.
- Simulation with `class_weighted` routing on that plan shows no SLO violations for either
  class, while `least_outstanding` (class-blind) on the same fleet shows long-class violations.
- Determinism of JSON output.

## 8b. Carry-over items (in scope for M7)
- Simulator: incremental KV accounting (reserve input tokens at admission, grow by one
  token per decode step, release at completion); admission uses projected mean occupancy.
  Acceptance: on an overloaded single replica, steady-state concurrency rises toward the
  planner's `effective_batch`, within 15%.
- `llmplan simulate --requests-csv PATH` exporting the `RequestLog` rows.

## 9. Allowed dependencies
None new.
