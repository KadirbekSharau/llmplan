# M1 implementation notes

Decisions taken where M1_DESIGN.md or ARCHITECTURE.md was ambiguous, silent, or needed a
change. Each entry names the commit step (PR 1 to PR 5) that introduced it.

## PR 1 — scaffold

- **Ruff N818 disabled.** ARCHITECTURE.md section 7 fixes exception names
  (`UnsupportedArchitecture`, `InfeasiblePlan`, ...) without an `Error` suffix; renaming
  them would break the contract, so the naming lint is ignored project-wide.
- **Ruff excludes `*.md`.** Ruff 0.16 formats Python code blocks inside Markdown, which
  would rewrite the hand-aligned field listings in the docs.
- **`uv audit` instead of `pip-audit`.** Section 11 allows either; `uv audit` needs no extra
  dependency. It is marked experimental in uv 0.12.
- **No `llmplan.ui` package yet.** It is Streamlit-specific (M6); empty placeholders were
  created for `workload`, `perf`, `planner`, `simulate` only, with a TODO naming the milestone.
- **`UnknownRegistryKey` added to the error hierarchy.** ARCHITECTURE.md section 6 says
  registry `get()` raises it, but section 7 did not define it. Added as a direct
  `LLMPlanError` subclass (it is a usage error, not a catalog error: renderer and format
  registries use it too) mapped to CLI exit code 2. ARCHITECTURE.md section 7 updated.
