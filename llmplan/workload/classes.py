"""Request-size demand classes (M7_DESIGN.md section 3): `DemandClass` and `classify()`.

A workload is cut into a grid of input-token bins by output-token bins (quantile edges by
default, or fixed edges), and each cell becomes a class with its own token statistics and
its demand in the fleet-wide peak windows. Cells holding under 1% of requests are merged
into their nearest neighbour, so classes stay large enough to plan for. Classes are
disjoint rectangles that tile `[1, max input] x [0, max output]`; `assign_classes` maps any
request (also one outside those bounds) to a class.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import pydantic
from pydantic import BaseModel, ConfigDict, Field

from llmplan.errors import ValidationError
from llmplan.workload.schema import Workload

ClassMethod = Literal["quantile", "fixed"]
MIN_CLASS_SHARE = 0.01  # cells with fewer requests are merged into a neighbour
MAX_BINS = 6  # per axis; 36 classes is already far more than the planner needs
IntArray = npt.NDArray[np.int64]


class DemandClass(BaseModel):
    """One request-size class of a workload.

    Token bounds are inclusive: the class holds requests with `input_lo <= input_tokens <=
    input_hi` and `output_lo <= output_tokens <= output_hi` (the lowest bins start at 1 and
    0, the highest end at the workload's maxima). `share` is its fraction of requests.
    `peak_rps` is its request count in the workload's peak request window and
    `peak_output_tokens_per_s` its output tokens in the peak output-token window (the
    windows `compute_stats` reports), both per second, so class demands add up to the
    workload's. Token means and percentiles (numpy "linear") are over the class's requests,
    which makes a `DemandClass` usable as the perf model's `StatsLike`. `notes` record
    cells merged into it and edges dropped while classifying.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    input_lo: int = Field(ge=1)
    input_hi: int = Field(ge=1)
    output_lo: int = Field(ge=0)
    output_hi: int = Field(ge=0)
    share: float = Field(gt=0, le=1)
    peak_rps: float = Field(ge=0)
    peak_output_tokens_per_s: float = Field(ge=0)
    input_tokens_mean: float = Field(ge=1)
    output_tokens_mean: float = Field(ge=0)
    input_tokens_p50: float = Field(ge=1)
    output_tokens_p50: float = Field(ge=0)
    input_tokens_p95: float = Field(ge=1)
    output_tokens_p95: float = Field(ge=0)
    notes: tuple[str, ...] = ()

    @pydantic.model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.input_lo > self.input_hi or self.output_lo > self.output_hi:
            raise ValueError(
                f"class {self.index}: bounds must satisfy lo <= hi, got inputs "
                f"{self.input_lo}..{self.input_hi}, outputs {self.output_lo}..{self.output_hi}"
            )
        return self


@dataclass
class _Group:
    """Consecutive bins `first..last` of one axis, with the requests they hold."""

    first: int
    last: int
    members: IntArray
    notes: list[str] = field(default_factory=list)


def _cuts(
    values: IntArray, bins: int, fixed: Sequence[int] | None, axis: str, notes: list[str]
) -> list[int]:
    """Upper bounds (inclusive) of every bin but the last, inside the observed range."""
    lo, hi = int(values.min()), int(values.max())
    if fixed is None:
        quantiles = np.percentile(values, [100 * i / bins for i in range(1, bins)])
        raw = sorted({math.floor(float(q)) for q in quantiles})
    else:
        raw = list(fixed)
    kept = [c for c in raw if lo <= c < hi]
    dropped = [c for c in raw if not lo <= c < hi]
    if dropped and fixed is not None:
        notes.append(f"{axis} edges {dropped} lie outside the observed {lo}..{hi} and were dropped")
    return kept


def _bounds(group: _Group, cuts: Sequence[int], floor: int, top: int) -> tuple[int, int]:
    lo = floor if group.first == 0 else cuts[group.first - 1] + 1
    hi = top if group.last >= len(cuts) else cuts[group.last]
    return lo, hi


def _merge_tiny(
    groups: list[_Group],
    values: IntArray,
    n_total: int,
    axis: tuple[str, int, Sequence[int]],
) -> list[_Group]:
    """Merge groups under `MIN_CLASS_SHARE` of all requests into the adjacent group whose
    mean token count (on this axis) is nearest, smallest first, until none is left or one
    group remains. Empty groups go to their lower neighbour (the upper one for the first).
    `axis` is (name, lowest token value, cuts), used to describe merged ranges."""
    describe, floor, cuts = axis
    groups = list(groups)
    while len(groups) > 1:
        shares = [len(g.members) / n_total for g in groups]
        tiny = [i for i, s in enumerate(shares) if s < MIN_CLASS_SHARE]
        if not tiny:
            break
        i = min(tiny, key=shares.__getitem__)
        here = groups[i]
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        if len(here.members):
            mean = float(values[here.members].mean())
            j = min(
                neighbours,
                key=lambda k: (
                    abs(float(values[groups[k].members].mean()) - mean)
                    if len(groups[k].members)
                    else math.inf
                ),
            )
            lo, hi = _bounds(here, cuts, floor, int(values.max()))
            groups[j].notes.append(
                f"{describe} {lo}..{hi} ({shares[i]:.2%} of requests, under "
                f"{MIN_CLASS_SHARE:.0%}) merged into this class"
            )
            groups[j].notes.extend(here.notes)
        else:
            j = neighbours[0]
        target = groups[j]
        target.first, target.last = min(target.first, here.first), max(target.last, here.last)
        target.members = np.sort(np.concatenate([target.members, here.members]))
        del groups[i]
    return groups


def _check_options(
    input_bins: int,
    output_bins: int,
    method: ClassMethod,
    edges: tuple[tuple[int, ...], tuple[int, ...]] | None,
    window_s: float,
) -> None:
    if not (math.isfinite(window_s) and window_s > 0):
        raise ValidationError(f"window_s must be a positive number of seconds, got {window_s}")
    if method == "quantile":
        if edges is not None:
            raise ValidationError("edges are only used with method='fixed'")
        for name, bins in (("input_bins", input_bins), ("output_bins", output_bins)):
            if not 1 <= bins <= MAX_BINS:
                raise ValidationError(f"{name} must be between 1 and {MAX_BINS}, got {bins}")
        return
    if edges is None:
        raise ValidationError("method='fixed' needs edges: (input edges, output edges)")
    for axis, cuts, low in (("input", edges[0], 1), ("output", edges[1], 0)):
        if len(cuts) >= MAX_BINS or any(c < low for c in cuts):
            raise ValidationError(
                f"{axis} edges must be fewer than {MAX_BINS} integers >= {low}, got {cuts}"
            )
        if any(b <= a for a, b in itertools.pairwise(cuts)):
            raise ValidationError(f"{axis} edges must be strictly increasing, got {cuts}")


def classify(
    workload: Workload,
    *,
    input_bins: int = 2,
    output_bins: int = 2,
    method: ClassMethod = "quantile",
    edges: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    window_s: float = 60.0,
) -> tuple[DemandClass, ...]:
    """Bucket `workload` into request-size classes, ordered by input bin, then output bin.

    `method="quantile"` cuts each axis at its `1/bins` quantiles (median splits for 2x2);
    `method="fixed"` takes `edges = (input edges, output edges)`, each the inclusive upper
    bounds of every bin but the last (e.g. `((1024,), (256,))`). Edges outside the observed
    range and duplicate quantiles are dropped, so there may be fewer bins. Input bins whose
    requests are under 1% of the workload are merged first, then cells within each input
    bin along the output axis (noted on the receiving class). Demand is measured in the
    fleet-wide peak windows of `window_s` seconds (see `DemandClass`). Raises
    `ValidationError` for bin counts outside 1..6, bad edges, or a bad window.
    """
    _check_options(input_bins, output_bins, method, edges, window_s)
    frame = workload.frame
    inputs: IntArray = frame["input_tokens"].to_numpy(dtype=np.int64)
    outputs: IntArray = frame["output_tokens"].to_numpy(dtype=np.int64)
    arrival = frame["arrival_s"].to_numpy(dtype=np.float64)
    n = len(inputs)
    notes: list[str] = []
    in_cuts = _cuts(inputs, input_bins, None if edges is None else edges[0], "input", notes)
    out_cuts = _cuts(outputs, output_bins, None if edges is None else edges[1], "output", notes)
    in_bin = np.searchsorted(in_cuts, inputs, side="left")
    out_bin = np.searchsorted(out_cuts, outputs, side="left")
    rows = _merge_tiny(
        [_Group(b, b, np.flatnonzero(in_bin == b)) for b in range(len(in_cuts) + 1)],
        inputs,
        n,
        ("inputs", 1, in_cuts),
    )
    cells: list[tuple[_Group, _Group]] = []
    for row in rows:
        row_out = out_bin[row.members]
        groups = [_Group(b, b, row.members[row_out == b]) for b in range(len(out_cuts) + 1)]
        cells.extend(
            (row, cell) for cell in _merge_tiny(groups, outputs, n, ("outputs", 0, out_cuts))
        )

    window = np.floor((arrival - arrival.min()) / window_s).astype(np.int64)
    n_windows = int(window.max()) + 1
    peak_requests = int(np.bincount(window, minlength=n_windows).argmax())
    peak_tokens = int(np.bincount(window, weights=outputs, minlength=n_windows).argmax())
    classes = []
    for index, (row, cell) in enumerate(cells):
        members = cell.members
        ins, outs, win = inputs[members], outputs[members], window[members]
        in_p50, in_p95, _ = (float(v) for v in np.percentile(ins, [50, 95, 99]))
        out_p50, out_p95, _ = (float(v) for v in np.percentile(outs, [50, 95, 99]))
        input_lo, input_hi = _bounds(row, in_cuts, 1, int(inputs.max()))
        output_lo, output_hi = _bounds(cell, out_cuts, 0, int(outputs.max()))
        classes.append(
            DemandClass(
                index=index,
                input_lo=input_lo,
                input_hi=input_hi,
                output_lo=output_lo,
                output_hi=output_hi,
                share=len(members) / n,
                peak_rps=float(np.bincount(win, minlength=n_windows)[peak_requests]) / window_s,
                peak_output_tokens_per_s=float(
                    np.bincount(win, weights=outs, minlength=n_windows)[peak_tokens]
                )
                / window_s,
                input_tokens_mean=float(ins.mean()),
                output_tokens_mean=float(outs.mean()),
                input_tokens_p50=in_p50,
                output_tokens_p50=out_p50,
                input_tokens_p95=in_p95,
                output_tokens_p95=out_p95,
                notes=tuple((notes if index == 0 else []) + row.notes + cell.notes),
            )
        )
    return tuple(classes)


SPEC_HELP = "1, <input bins>x<output bins> (e.g. 2x2), or fixed:<input edges>/<output edges>"


def _edges(text: str, spec: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise ValidationError(f"classes {spec!r}: edges must be integers ({SPEC_HELP})") from None


def classify_spec(
    workload: Workload, spec: str, *, window_s: float = 60.0
) -> tuple[DemandClass, ...]:
    """Classes for a CLI/UI spec: `"1"` gives `()` (no classes: the whole workload, as in
    M4), `"AxB"` A x B quantile bins, `"fixed:1024,4096/256"` fixed input and output edges
    (either side may be empty). Raises `ValidationError` naming the spec."""
    text = spec.strip().lower()
    if text == "1":
        return ()
    grid = re.fullmatch(r"(\d+)x(\d+)", text)
    if grid is not None:
        return classify(
            workload, input_bins=int(grid[1]), output_bins=int(grid[2]), window_s=window_s
        )
    if text.startswith("fixed:") and text.count("/") == 1:
        in_text, out_text = text.removeprefix("fixed:").split("/")
        edges = (_edges(in_text, spec), _edges(out_text, spec))
        return classify(workload, method="fixed", edges=edges, window_s=window_s)
    raise ValidationError(f"classes must be {SPEC_HELP}, got {spec!r}")


def assign_classes(
    classes: Sequence[DemandClass], input_tokens: IntArray, output_tokens: IntArray
) -> IntArray:
    """The class index of every request: the class whose bounds contain it, else the one
    with the smallest total token distance to its bounds (ties to the lowest index). All
    zeros when `classes` is empty (one class: the whole workload)."""
    best = np.zeros(len(input_tokens), dtype=np.int64)
    if not classes:
        return best
    nearest = np.full(len(input_tokens), np.iinfo(np.int64).max, dtype=np.int64)
    for c in classes:
        distance = (
            np.maximum(c.input_lo - input_tokens, 0)
            + np.maximum(input_tokens - c.input_hi, 0)
            + np.maximum(c.output_lo - output_tokens, 0)
            + np.maximum(output_tokens - c.output_hi, 0)
        )
        closer = distance < nearest
        best[closer] = c.index
        nearest[closer] = distance[closer]
    return best
