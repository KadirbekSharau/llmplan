"""Weekly summary of the web UI usage log (M6_DESIGN.md section 10).

Usage:

    uv run python scripts/usage_summary.py USAGE_LOG [--days 7]

Reads the JSON lines written when `LLMPLAN_USAGE_LOG` is set and prints, for the last
`--days` days (by each line's `ts`), the number of plan runs, then counts by model id, by
GPU set, and by solver status. Malformed lines are counted and skipped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path


def summarize(lines: list[str], *, days: float, now: datetime) -> str:
    """The summary text for the log `lines` within `days` before `now`."""
    since = now - timedelta(days=days)
    models: Counter[str] = Counter()
    gpu_sets: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    runs = skipped = 0
    for line in lines:
        try:
            record = json.loads(line)
            ts = datetime.fromisoformat(record["ts"])
            model, gpus, status = record["model_id"], record["gpu_ids"], record["solver_status"]
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        if ts < since:
            continue
        runs += 1
        models[str(model)] += 1
        gpu_sets[",".join(sorted(str(g) for g in gpus)) or "(none)"] += 1
        statuses[str(status)] += 1
    out = [f"plan runs since {since.date()}: {runs} ({skipped} malformed lines skipped)"]
    for title, counts in (("model id", models), ("GPU set", gpu_sets), ("status", statuses)):
        out += ["", f"by {title}:", *(f"  {n:>6}  {key}" for key, n in counts.most_common())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="usage log (JSON lines)")
    parser.add_argument("--days", type=float, default=7.0, help="window in days (default 7)")
    args = parser.parse_args(argv)
    lines = args.log.read_text(encoding="utf-8").splitlines()
    print(summarize(lines, days=args.days, now=datetime.now(UTC)), end="")


if __name__ == "__main__":
    main()
