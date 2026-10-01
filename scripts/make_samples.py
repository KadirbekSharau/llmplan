"""Cut the bundled web UI trace samples (M6_DESIGN.md section 5) from the full public traces.

Usage:

    uv run llmplan traces fetch NAME --dest TRACES_DIR --yes     # once per trace, outside the repo
    uv run python scripts/make_samples.py TRACES_DIR [--out data/traces/samples]

For every trace in `SAMPLES`, the full file is parsed with `llmplan.workload.load_workload`
and two full hours are kept: the busiest (most requests) and the median-load hour (lower
median of the full hours' request counts, earliest on ties). Hours are
`[k * 3600, (k + 1) * 3600)` seconds from the first request, as in `compute_stats`; a
trailing partial hour is not a candidate. The two hours are placed back to back in
chronological order (the earlier one first) and, when they hold more than `MAX_ROWS`
requests, each is thinned by the same fraction: `floor(n_hour * MAX_ROWS / n_total)` rows
chosen uniformly at random (numpy `default_rng(SEED)`), kept in trace order. The sample is
written in the generic `csv` format with `arrival_s` relative to its first request, and
`README.md` records the source, license, method and the exact rows kept.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llmplan.workload import Workload, load_workload
from llmplan.workload.fetch import load_manifest
from llmplan.workload.formats.generic_csv import write_csv

MAX_ROWS = 20_000
SEED = 0
SECONDS_PER_HOUR = 3600.0
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "traces" / "samples"

# manifest name -> (sample file, dataset label, citation)
SAMPLES = {
    "azure2023-code": (
        "azure2023_code.csv",
        "Azure LLM inference trace 2023, code",
        "Patel et al., Splitwise, ISCA 2024",
    ),
    "azure2023-conv": (
        "azure2023_conv.csv",
        "Azure LLM inference trace 2023, conversation",
        "Patel et al., Splitwise, ISCA 2024",
    ),
    "azure2024-code": (
        "azure2024_code.csv",
        "Azure LLM inference trace 2024, code (one week)",
        "Stojkovic et al., DynamoLLM, HPCA 2025",
    ),
    "azure2024-conv": (
        "azure2024_conv.csv",
        "Azure LLM inference trace 2024, conversation (one week)",
        "Stojkovic et al., DynamoLLM, HPCA 2025",
    ),
    "burstgpt-1": (
        "burstgpt_1.csv",
        "BurstGPT release v2.0, BurstGPT_1.csv",
        "Wang et al., BurstGPT, 2024",
    ),
}


@dataclass(frozen=True)
class Cut:
    """What one sample kept: the source hours (busiest first), their request counts in the
    full trace, the rows kept from each, and the full trace's size and duration."""

    hours: tuple[int, ...]
    hour_rows: tuple[int, ...]
    kept_rows: tuple[int, ...]
    trace_rows: int
    trace_duration_s: float


def pick_hours(arrival_s: np.ndarray) -> tuple[int, ...]:
    """The busiest and the median-load full hour (one hour when they coincide)."""
    full_hours = math.floor(float(arrival_s.max()) / SECONDS_PER_HOUR)
    hour = np.floor(arrival_s / SECONDS_PER_HOUR).astype(np.int64)
    if full_hours < 1:  # shorter than an hour: the whole trace is the only window
        return (0,)
    counts = np.bincount(hour[hour < full_hours], minlength=full_hours)
    busiest = int(counts.argmax())
    by_load = sorted(range(full_hours), key=lambda h: (int(counts[h]), h))
    median = by_load[(full_hours - 1) // 2]
    return (busiest,) if median == busiest else (busiest, median)


def cut(workload: Workload, max_rows: int = MAX_ROWS, seed: int = SEED) -> tuple[Workload, Cut]:
    """The sample frame of `workload` (see the module docstring) and what it kept."""
    frame = workload.frame
    arrival = frame["arrival_s"].to_numpy()
    hours = pick_hours(arrival)
    hour_of = np.floor(arrival / SECONDS_PER_HOUR).astype(np.int64)
    in_hour = [np.flatnonzero(hour_of == h) for h in hours]
    total = sum(len(rows) for rows in in_hour)
    rng = np.random.default_rng(seed)
    parts, kept = [], []
    for slot, h in enumerate(sorted(hours)):
        rows = in_hour[hours.index(h)]
        keep = len(rows) if total <= max_rows else math.floor(len(rows) * max_rows / total)
        chosen = np.sort(rng.choice(rows, size=keep, replace=False))
        part = frame.iloc[chosen].copy()
        part["arrival_s"] = part["arrival_s"] - h * SECONDS_PER_HOUR + slot * SECONDS_PER_HOUR
        parts.append(part)
        kept.append((h, keep))
    sample = pd.concat(parts, ignore_index=True)
    shifted = sample["arrival_s"] - sample["arrival_s"].iloc[0]
    sample["arrival_s"] = shifted.round(6)  # microseconds: drop float noise of the shifts
    kept_by_hour = dict(kept)
    result = Workload(
        source=f"sample of {workload.source}",
        format="csv",
        frame=sample,
        dropped_rows=0,
    )
    return result, Cut(
        hours=hours,
        hour_rows=tuple(len(rows) for rows in in_hour),
        kept_rows=tuple(kept_by_hour[h] for h in hours),
        trace_rows=len(frame),
        trace_duration_s=float(arrival.max()),
    )


def describe(name: str, filename: str, label: str, citation: str, info: Cut) -> list[str]:
    """README lines for one sample."""
    source = load_manifest()[name]
    full_hours = math.floor(info.trace_duration_s / SECONDS_PER_HOUR)
    total = sum(info.hour_rows)
    kept = sum(info.kept_rows)
    if full_hours < 1:
        span = f"{info.trace_duration_s / 60:.1f} minutes (shorter than one hour)"
        hours = f"the whole trace ({total:,} requests)"
    else:
        span = f"{full_hours:,} full hours"
        names = ("busiest", "median-load")
        hours = "; ".join(
            f"{kind} hour {h} from the first request ({n:,} requests, {k:,} kept)"
            for kind, h, n, k in zip(
                names, info.hours, info.hour_rows, info.kept_rows, strict=False
            )
        )
    thinning = "all rows kept" if kept == total else f"thinned to {kept / total:.4%}"
    return [
        f"## `{filename}`: {label}",
        "",
        f"- Source: manifest trace `{name}`, {source.url}",
        f"- License: CC-BY-4.0 ({source.license_url}); cite {citation}.",
        f"- Full trace: {info.trace_rows:,} valid requests over {span}.",
        f"- Kept: {hours}; {thinning}; {kept:,} rows in total.",
        "",
    ]


README_HEAD = """# Bundled trace samples (web UI presets)

Small windows of public LLM inference traces, used as the web UI's traffic presets so the
app never parses a full trace at request time (M6_DESIGN.md section 5). Generated by
`scripts/make_samples.py` from the files `llmplan traces fetch` downloads (SHA-256
verified against `data/traces/manifest.yaml`); the full traces are never committed.

Method: the busiest full hour (by requests) and the median-load full hour of each trace,
counted in hours from the trace's first request, placed back to back in chronological
order; when the two hours exceed 20,000 requests each is thinned by the same fraction
(uniform random rows, numpy seed 0, trace order kept). Thinning lowers the request rate
by that fraction and keeps the token-length distribution and the busy/median contrast.
Files are in the generic `csv` format: `arrival_s` (seconds from the sample's first request,
rounded to microseconds), `input_tokens`, `output_tokens`, and `model` where the source has it.

Changes from the sources (CC-BY-4.0 requires stating them): rows selected and thinned as
above; timestamps converted to relative seconds; columns renamed; rows with invalid
token counts dropped by llmplan's parser (BurstGPT failed requests with zero tokens).

"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, help="directory with the downloaded full traces")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="samples directory")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    lines = [README_HEAD]
    for name, (filename, label, citation) in SAMPLES.items():
        url = manifest[name].url or ""
        path = args.traces / url.rsplit("/", 1)[-1]
        sample, info = cut(load_workload(path))
        write_csv(sample, args.out / filename)
        lines += describe(name, filename, label, citation, info)
        print(f"{filename}: {sum(info.kept_rows):,} rows from hours {info.hours}")
    (args.out / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
