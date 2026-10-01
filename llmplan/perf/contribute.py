"""A prefilled GitHub issue for contributing benchmark rows (M8_DESIGN.md section 5.2).

The link only opens the issue form with the rows in its body; nothing is sent anywhere
until the user submits it on GitHub. Beyond `MAX_URL_CHARS` the rows are left out and the
user attaches the CSV instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import quote, urlencode

from llmplan.perf.benchmarks import BenchmarkRow
from llmplan.perf.uploads import rows_csv

REPOSITORY_URL = "https://github.com/KadirbekSharau/llmplan"
MAX_URL_CHARS = 6_000
CHECKLIST = (
    "Please confirm the source of these rows:",
    "",
    "- [ ] Engine and version (e.g. vLLM 0.30.0):",
    "- [ ] GPU driver and CUDA version:",
    "- [ ] Date measured:",
    "- [ ] Benchmark command (e.g. `vllm bench serve ...`):",
)


def _distinct(values: Sequence[object]) -> str:
    return ", ".join(dict.fromkeys(str(v) for v in values))


def _url(title: str, body: str) -> str:
    query = urlencode({"title": title, "body": body}, quote_via=quote, safe="/:,")
    return f"{REPOSITORY_URL}/issues/new?{query}"


def contribute_url(rows: Sequence[BenchmarkRow]) -> tuple[str, bool]:
    """The new-issue URL for `rows` and whether it carries them.

    The body has a model / GPU / tensor-parallel header, the rows as an llmplan CSV in a
    code block, and a checklist asking for the engine version, driver and date. When that
    URL exceeds `MAX_URL_CHARS`, the rows are dropped from the body (it asks for the CSV as
    an attachment) and the flag is False. `rows` must not be empty.
    """
    if not rows:
        raise ValueError("contribute_url needs at least one row")
    model = _distinct([r.model_id for r in rows])
    gpu = _distinct([r.gpu_id for r in rows])
    tp = _distinct([r.tensor_parallel for r in rows])
    title = f"Benchmark rows: {model} on {gpu} (tp {tp})"
    header = [f"Model: {model}", f"GPU: {gpu}", f"Tensor parallel: {tp}", f"Rows: {len(rows)}"]
    with_rows = [*header, "", "```csv", rows_csv(rows).rstrip("\n"), "```", "", *CHECKLIST]
    url = _url(title, "\n".join(with_rows))
    if len(url) <= MAX_URL_CHARS:
        return url, True
    attach = "The rows are too many for a link: please attach the CSV downloaded from llmplan."
    return _url(title, "\n".join([*header, "", attach, "", *CHECKLIST])), False
