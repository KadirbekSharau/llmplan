# FOUNDER_QUESTIONS.md — decisions deferred to the founder

The CTO and implementing agents log here anything only the founder can decide. Each entry
states the interim decision taken so work never blocks. Answer inline; the CTO applies the
answer in the next milestone.

| # | Date | Question | Interim decision | Founder answer |
|---|---|---|---|---|
| 1 | 2026-09-30 | No git remote exists and `gh` is not installed, so nothing can be pushed. Please create a GitHub repository (suggested: private, named `llmplan`) and either add it as `origin` or install and authenticate `gh` so the CTO can. | Agents work on local branches; the CTO merges with `--no-ff`; everything is pushed once a remote exists. | |
| 2 | 2026-09-30 | Rename the project directory from `ultrascheduler` to `llmplan` to match the package? | Left as is; docs and package use `llmplan`. | |
| 3 | 2026-09-30 | Per-agent reasoning effort cannot be set by the CTO when spawning agents; they run at the default effort for Opus. Acceptable, or do you want to set a session-level default? | Default effort used. | |
| 4 | 2026-09-30 | M3: accept vendor-published benchmark rows measured on TensorRT-LLM rather than vLLM? | Yes, with `engine` recorded per row; the planner prefers rows matching the user's engine. | |
