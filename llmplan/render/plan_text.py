"""Plain-text rendering of a `PlanResult` (M4_DESIGN.md section 9).

Sections: summary (cost against the baseline, confidence in the performance estimates
(M8), binding constraint, solver), fleet table, replica table with vLLM command lines, with
request-size classes (M7) the comparison with the single-class plan (M8), a class table and
the routing table, the top candidates by $/hour per request/s with status and reason, and
the assumptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmplan.perf.confidence import confidence_sentence
from llmplan.planner.result import label
from llmplan.render.vllm_cmd import serve_command

if TYPE_CHECKING:
    from collections.abc import Mapping

    from llmplan.catalog.hardware import GPUSpec
    from llmplan.planner.request import PlanRequest
    from llmplan.planner.result import CandidateEval, PlanResult
    from llmplan.simulate.compare import ClassComparison

LABEL_WIDTH = 10
TOP_CANDIDATES = 10


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def fleet_sentence(result: PlanResult, gpus: Mapping[str, GPUSpec]) -> str:
    """The fleet in one sentence (M9, the web UI's answer card), e.g. `2 x H100 SXM 80GB
    (runpod h100-sxm) serving 2 replicas`: GPUs per price row by catalog name, then the
    replica count. `gpus` is the catalog the plan was made with."""
    parts = []
    for item in result.fleet:
        row = item.price_row
        name = gpus[row.gpu_id].name.removeprefix("NVIDIA ")
        parts.append(f"{item.instances * row.gpu_count} x {name} ({row.provider} {row.instance})")
    replicas = sum(r.count for r in result.replicas)
    return f"{' + '.join(parts)} serving {replicas} replica{'s' if replicas != 1 else ''}"


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def baseline_saving(saving_pct: float) -> str:
    """The saving against the baseline as a non-negative percentage, e.g. `"saving 12.5%"`;
    a fleet dearer than its baseline (possible only at a solver time limit) reads
    `"baseline 3.0% cheaper"`, so no negative percentage is ever printed (M8 section 4)."""
    if saving_pct < 0:
        return f"baseline {-saving_pct:.1f}% cheaper"
    return f"saving {saving_pct:.1f}%"


def _pct(value: float) -> str:
    return f"{value:.1f}%" if value >= 0.05 else "under 0.1%"


def class_comparison_sentence(comparison: ClassComparison) -> str:
    """The class plan against the single-class plan in words (M8_DESIGN.md section 4): what
    sizing for the mean request would cost and why it is not enough, the saving, or that
    nothing changes. Amounts are compared to the cent; percentages are never negative."""
    cost = comparison.class_cost_usd_per_day
    single = comparison.single_class_cost_usd_per_day
    if single is None:
        return (
            "No fleet sized for the mean request meets the target; the class-sized plan "
            f"costs {_usd(cost)}/day."
        )
    if round(cost, 2) > round(single, 2):
        violations = comparison.single_class_ttft_violation_pct
        if violations:
            miss = (
                "the replay shows it would miss the latency target for "
                f"{_pct(violations)} of requests"
            )
        else:
            miss = "it would under-provision the long-request class"
        return (
            f"Sized for the mean request this fleet would cost {_usd(single)}/day, but {miss}. "
            f"The class-sized plan costs {_usd(cost)}/day."
        )
    if round(cost, 2) < round(single, 2):
        saved = single - cost
        return (
            f"Request-size routing saves {_usd(saved)}/day ({_pct(saved / single * 100)}) "
            "versus sizing every replica for the mean request."
        )
    return "Request-size routing does not change the fleet for this traffic."


def _summary(request: PlanRequest, result: PlanResult) -> list[str]:
    solver = result.solver
    if result.baseline is None or result.baseline_saving_pct is None:
        versus = "homogeneous requested" if request.options.homogeneous else "no homogeneous fleet"
    else:
        versus = (
            f"baseline {_usd(result.baseline.cost_usd_per_day)}/day, "
            f"{baseline_saving(result.baseline_saving_pct)}"
        )
    bound = solver.best_bound_usd_per_day
    return [
        _line("Model", f"{request.model.id}  (max_model_len {request.options.max_model_len:,})"),
        _line(
            "Demand",
            f"{result.demand_rps:g} req/s, {result.demand_output_tokens_per_s:g} output "
            f"tokens/s (peak {request.stats.window_s:g} s window)",
        ),
        _line(
            "Capacity",
            f"{result.capacity_rps:,.2f} req/s, {result.capacity_output_tokens_per_s:,.2f} output "
            f"tokens/s (derated x {request.slo.utilization_target:g})",
        ),
        _line("Cost", f"{_usd(result.cost_usd_per_day)}/day  ({versus})"),
        f"Confidence  {confidence_sentence(result.perf_confidence, result.perf_sources)}",
        _line("Binding", result.binding),
        _line(
            "Solver",
            f"{solver.backend} {solver.status} in {solver.solve_time_s:.2f} s "
            f"({solver.n_variables} variables, {solver.n_constraints} constraints"
            + ("" if bound is None else f", bound {_usd(bound)}/day")
            + ")",
        ),
    ]


def _fleet(result: PlanResult) -> list[str]:
    lines = [
        "",
        "Fleet",
        f"  {'provider':<10}{'instance':<20}{'gpus':<22}{'count':>6}{'$/day':>14}",
    ]
    for item in result.fleet:
        row = item.price_row
        lines.append(
            f"  {row.provider:<10}{row.instance:<20}{f'{row.gpu_count} x {row.gpu_id}':<22}"
            f"{item.instances:>6}{_usd(item.usd_per_day):>14}"
        )
    return lines


def _replicas(request: PlanRequest, result: PlanResult) -> list[str]:
    lines = ["", "Replicas"]
    for replica in result.replicas:
        c = replica.candidate
        perf = f"{c.perf.backend}/{c.perf.confidence}" if c.perf is not None else "-"
        lines.append(
            f"  {replica.count} x {c.config.dtype} tp{c.config.tensor_parallel} max_num_seqs "
            f"{c.config.max_num_seqs} on {c.price_row.provider} {c.price_row.instance}  "
            f"(perf {perf})"
        )
        lines.extend(f"    {cmd}" for cmd in serve_command(request.model, c.config).splitlines())
    return lines


def _tokens(lo: int, hi: int) -> str:
    return f"{lo:,}..{hi:,}"


def _classes(result: PlanResult, comparison: ClassComparison | None) -> list[str]:
    if not result.classes:
        return []
    lines = []
    if comparison is not None:
        lines += ["", "Request-size routing", f"  {class_comparison_sentence(comparison)}"]
    lines += [
        "",
        "Classes",
        f"  {'class':>5}  {'input tokens':<16}{'output tokens':<16}{'share':>7}"
        f"{'peak req/s':>12}{'peak tok/s':>12}  binding",
    ]
    for c, binding in zip(result.classes, result.class_binding, strict=True):
        lines.append(
            f"  {c.index:>5}  {_tokens(c.input_lo, c.input_hi):<16}"
            f"{_tokens(c.output_lo, c.output_hi):<16}{c.share * 100:>6.1f}%"
            f"{c.peak_rps:>12,.3f}{c.peak_output_tokens_per_s:>12,.1f}  {binding}"
        )
    lines.extend(
        [
            "",
            "Routing (share of each class's requests per replica type)",
            f"  {'class':>5}{'weight':>9}{'replicas':>10}{'req/s':>10}  replica type",
        ]
    )
    lines.extend(
        f"  {r.class_index:>5}{r.weight * 100:>8.1f}%{r.replicas:>10.3f}"
        f"{r.capacity_rps:>10,.2f}  {label(r.candidate)}"
        for r in result.routing
    )
    return lines


def _candidate(c: CandidateEval) -> str:
    cost = "-" if c.usd_per_hour_per_rps is None else f"{c.usd_per_hour_per_rps:.4g}"
    config = f"tp{c.config.tensor_parallel} {c.config.dtype} seqs{c.config.max_num_seqs}"
    name = f"{c.price_row.provider} {c.price_row.instance} {config}"
    return f"  {c.status:<11}{cost:>10}  {name:<44}{c.reason}"


def plan_text(
    request: PlanRequest, result: PlanResult, comparison: ClassComparison | None = None
) -> str:
    """Render `llmplan plan` output as aligned plain text. `comparison` (M8) is the class
    plan against the single-class plan, shown before the class table when given."""
    shown = result.candidates[:TOP_CANDIDATES]
    lines = [
        *_summary(request, result),
        *_fleet(result),
        *_replicas(request, result),
        *_classes(result, comparison),
        "",
        f"Candidates (top {len(shown)} of {len(result.candidates)} by $/hour per req/s)",
        f"  {'status':<11}{'$/h/rps':>10}  {'candidate':<44}reason",
        *(_candidate(c) for c in shown),
        "",
        "Assumptions",
        *(f"  - {note}" for note in result.assumptions),
    ]
    return "\n".join(lines) + "\n"
