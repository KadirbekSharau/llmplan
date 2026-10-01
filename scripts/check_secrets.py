"""Scan the full git history for committed secrets (M8_DESIGN.md section 8.4).

    uv run python scripts/check_secrets.py [REV ...]     # default: HEAD and all its history

Reads `git log -p` (every patch of every commit reachable from the given revisions) and
reports added or removed lines that look like a credential, keyed on the design's markers:
Hugging Face (`hf_...`), GitHub (`ghp_...` and the other GitHub token prefixes), AWS access
key ids (`AKIA...`), `sk-...` API keys, and a literal value assigned to something named
`token` (or a literal `Bearer` header). The bare words are not flagged on their own:
"token" is everywhere in this code base (`input_tokens`, `HF_TOKEN`), so only
secret-shaped values count. Exits 1 when anything is found; matches are printed with the
commit, file and a masked value, never the secret itself.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "hf_": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "ghp_": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}|\bgithub_pat_[A-Za-z0-9_]{22,}"),
    "AKIA": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "sk-": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    "token": re.compile(r"(?i)token['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-./+=]{16,}['\"]"),
    "bearer": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}"),
}


@dataclass(frozen=True)
class Hit:
    commit: str
    path: str
    marker: str
    masked: str


def _mask(value: str) -> str:
    return value[:6] + "..." if len(value) > 6 else "..."


def scan_patch(lines: Iterable[str]) -> Iterator[Hit]:
    """Hits in `git log -p` output: only added or removed lines of a diff are checked."""
    commit = path = ""
    for line in lines:
        if line.startswith("commit "):
            commit = line.split()[1][:12]
        elif line.startswith("+++ ") or line.startswith("--- "):
            path = line[4:].removeprefix("b/").removeprefix("a/").strip()
        elif line[:1] in "+-":
            for marker, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    yield Hit(commit, path, marker, _mask(match.group(0)))


def scan_history(repo: Path, revisions: Iterable[str] = ("HEAD",)) -> list[Hit]:
    """Every hit in the history of `revisions` in `repo` (git must be on PATH)."""
    argv = ["git", "-C", str(repo), "log", "-p", "--no-color", "--no-ext-diff", "--text"]
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        [*argv, *revisions, "--"], capture_output=True, text=True, errors="replace", check=True
    ).stdout
    return list(scan_patch(out.splitlines()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("revisions", nargs="*", default=["HEAD"])
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    hits = scan_history(repo, args.revisions)
    for hit in hits:
        print(f"{hit.commit} {hit.path}: {hit.marker} {hit.masked}")
    print(f"secrets scan: {len(hits)} hits in the history of {' '.join(args.revisions)}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
