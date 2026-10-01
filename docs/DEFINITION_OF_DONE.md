# DEFINITION_OF_DONE.md — Rules for completing a milestone

These rules apply to every milestone and every implementing agent. The CTO reviews against
this list; a milestone that misses any item is returned to the agent with the failing item
named. Nothing here is optional.

---

## 1. Scope

- Implement exactly the milestone's design doc (`docs/milestones/M<N>_DESIGN.md`). Nothing
  from later milestones, nothing "while you're here."
- Interfaces come from `docs/ARCHITECTURE.md`. If one must change, change the doc in the
  same commit and explain why in the milestone's notes file.
- Ambiguities: choose the most conservative reading, record it under "Implementation notes"
  in `docs/milestones/M<N>_NOTES.md`, and continue. Questions only the founder can answer go
  under "Questions for founder" in the same file. Never block on them.

## 2. Code quality

- `ruff check` and `ruff format --check` clean. `mypy --strict llmplan` clean with no new
  `type: ignore` unless commented with the reason.
- Frozen pydantic v2 models at boundaries; pure functions inside; units in field names;
  typed exceptions from `llmplan/errors.py`; no `utils.py`; no module over ~300 lines
  without a justification in the notes file.
- Public functions and models have a one-paragraph docstring stating inputs, outputs, and
  assumptions. No docstrings on private helpers unless non-obvious.
- No new runtime dependency outside the design doc's allowed list without a one-line reason
  in the commit message and the notes file.
- No dead code, no commented-out code, no TODOs without a milestone tag (`TODO(M3): ...`).

## 3. Tests

- Every acceptance test in the design doc exists in `tests/acceptance/test_m<N>.py` with the
  expected values unchanged. If an expected value is provably wrong, fix the test, and
  document the arithmetic in the notes file; the CTO verifies independently.
- Unit tests for every new module. Property tests where the design doc lists them.
- Line coverage of the milestone's new modules is at least 90% (`pytest --cov=llmplan
  --cov-report=term-missing`); add `pytest-cov` as a dev dependency if absent.
- No network, no GPU, no external services in tests. Fixtures live under `llmplan/data/fixtures/`
  or `tests/fixtures/` and are hand-written or tiny samples.
- The full suite runs in under 60 seconds on a laptop CPU.

## 4. Runnable and buildable

From a clean clone, these must succeed in order, and the agent must actually run them
before declaring done:

```bash
uv sync --all-extras            # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict llmplan
uv run pytest
uv run pip-audit                # or: uv audit  — no known vulnerabilities
uv build                        # produces a wheel under dist/
uv run llmplan --help           # CLI entry point resolves
```

Each milestone's CLI commands run end to end on the shipped fixtures without arguments
beyond those in the design doc's examples.

## 5. Logical state

- The milestone's public API (ARCHITECTURE.md section 5) is implemented with the documented
  signature and returns the documented model.
- Results are deterministic: same inputs give byte-identical JSON output. Solver-based
  milestones set seeds, thread counts, and time limits explicitly.
- Every catalog, benchmark, or price row carries `source_url` and `as_of`. Unverifiable
  values are `null` with a `TODO(M<N>): verify` comment. Guessed values are a review failure.
- Error paths from the design doc raise the named exception with a message that names the
  offending field, id, or value.

## 6. Documentation

- `README.md`: usage section updated for every new or changed CLI command, with one copyable
  example per command.
- `docs/ARCHITECTURE.md`: data models section updated with any new models (copied from the
  design doc once merged) and the public API table row marked implemented.
- `docs/MILESTONES.md`: milestone status set to `done` with the merge commit hash.
- `docs/milestones/M<N>_NOTES.md`: created with three sections: Implementation notes,
  Deviations from the design doc, Questions for founder.
- `CHANGELOG.md`: one entry per milestone under an `Unreleased` heading, Keep-a-Changelog
  format.

## 7. Git workflow

- Branch `m<N>-<slug>` from the current `main` (M1 was allowed to commit straight to main;
  every later milestone uses a branch).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `build:`, `chore:`), one logical
  change per commit, in the order the design doc suggests. Every commit leaves the checks in
  section 4 green.
- Commit messages end with the agent's `Co-Authored-By:` line.
- Before declaring done: rebase on `main`, rerun section 4, and leave the working tree clean
  (`git status` empty). No force-pushes to `main`, ever.
- Push and open a pull request when a remote exists. Until then, the branch stays local and
  the CTO merges it with `git merge --no-ff` after review.

## 8. Security

- `yaml.safe_load` only. No `pickle`, `eval`, `exec`, or `subprocess` in the package.
- External identifiers (Hugging Face ids, file paths from users, URLs) validated before use.
- Downloads (traces, catalogs) require an explicit `--yes` flag, verify a recorded SHA-256,
  and are size-capped.
- No secrets in the repo, in logs, or in test fixtures. Tokens come from environment
  variables only.
- `pip-audit`/`uv audit` clean.

## 9. Final report

The agent's last message must contain, in this order:
1. `git log --oneline main..<branch>` (or the commits added to main for M1).
2. The summary lines of ruff, mypy, pytest (with coverage), and the audit.
3. Deviations from the design doc, each with a reason.
4. Catalog or benchmark values left `null` and why.
5. Questions for the founder (copied from the notes file).
6. The exact commands to reproduce the checks.

## 10. CTO review gate

The CTO checks every item above, reads the diff, runs section 4 independently, and
spot-checks acceptance arithmetic. Outcomes:
- **Approved:** merged into `main` with `--no-ff`; MILESTONES.md status updated.
- **Changes requested:** the same agent (or a fresh one with the review notes) fixes on the
  same branch. Re-review from the top.
- Two consecutive failed reviews on the same item escalate to the founder in the notes file.
