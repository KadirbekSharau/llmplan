"""Plain-text rendering of a `PlanResult` (M4_DESIGN.md section 9).

Sections: summary (cost against the baseline, confidence in the performance estimates
(M8), binding constraint, solver), fleet table,
replica table with vLLM command lines, with request-size classes (M7) a class table and the
routing table, the top candidates by $/hour per request/s with status and reason, and the
assumptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmplan.perf.confidence import confidence_sentence
from llmplan.planner.result import label
from llmplan.render.vllm_cmd import serve_command

if TYPE_CHECKING:
    from llmplan.planner.request import PlanRequest
    from llmplan.planner.result import CandidateEval, PlanResult

LABEL_WIDTH = 10
TOP_CANDIDATES = 10


def _line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}{value}"


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def _summary(request: PlanRequest, result: PlanResult) -> list[str]:
    solver = result.solver
    if result.baseline is None:
        versus = "homogeneous requested" if request.options.homogeneous else "no homogeneous fleet"
    else:
        versus = (
            f"baseline {_usd(result.baseline.cost_usd_per_day)}/day, saving "
            f"{result.baseline_saving_pct:.1f}%"
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


def _classes(result: PlanResult) -> list[str]:
    if not result.classes:
        return []
    lines = [
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


def plan_text(request: PlanRequest, result: PlanResult) -> str:
    """Render `llmplan plan` output as aligned plain text."""
    shown = result.candidates[:TOP_CANDIDATES]
    lines = [
        *_summary(request, result),
        *_fleet(result),
        *_replicas(request, result),
        *_classes(result),
        "",
        f"Candidates (top {len(shown)} of {len(result.candidates)} by $/hour per req/s)",
        f"  {'status':<11}{'$/h/rps':>10}  {'candidate':<44}reason",
        *(_candidate(c) for c in shown),
        "",
        "Assumptions",
        *(f"  - {note}" for note in result.assumptions),
    ]
    return "\n".join(lines) + "\n"
