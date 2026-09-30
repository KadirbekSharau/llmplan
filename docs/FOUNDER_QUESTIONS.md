# FOUNDER_QUESTIONS.md — decisions deferred to the founder

The CTO and implementing agents log here anything only the founder can decide. Each entry
states the interim decision taken so work never blocks. Answer inline; the CTO applies the
answer in the next milestone.

| # | Date | Question | Interim decision | Founder answer |
|---|---|---|---|---|
| 1 | 2026-09-30 | No git remote exists and `gh` is not installed, so nothing can be pushed. Please create a GitHub repository (suggested: private, named `llmplan`) and either add it as `origin` or install and authenticate `gh` so the CTO can. | Resolved 2026-09-30: CTO created private repo https://github.com/KadirbekSharau/llmplan using the GitHub token already stored in the macOS keychain (scopes repo, workflow) and pushed main. Later milestones use branches and pull requests. | founder asked CTO to create it |
| 2 | 2026-09-30 | Rename the project directory from `ultrascheduler` to `llmplan` to match the package? | Left as is; docs and package use `llmplan`. | Fine with `llmplan`; directory rename not needed. |
| 3 | 2026-09-30 | Per-agent reasoning effort cannot be set by the CTO when spawning agents; they run at the default effort for Opus. Acceptable, or do you want to set a session-level default? | Default effort used. | Not a concern; control agents through docs and instructions. |
| 4 | 2026-09-30 | M3: accept vendor-published benchmark rows measured on TensorRT-LLM rather than vLLM? | Yes, with `engine` recorded per row; the planner prefers rows matching the user's engine. | CTO decides. Decision: yes, as the interim says. |
| 5 | 2026-09-30 | M6 hosting: Streamlit Community Cloud (free, needs a public repo) or a paid container host with a private repo? | Docker image built either way; decide before launch. | Decide after local testing. |
| 6 | 2026-09-30 | Public app name and domain for launch posts? | "llmplan". | Check for collisions first; may rename later. Check 2026-09-30: PyPI `llmplan` is free, PyPI `llm-plan` is taken by an unrelated package, a GitHub account named `llmplan` exists, no indexed product uses the name. Working name stays; revisit before launch. |

Rule (founder, 2026-09-30): only log questions the CTO genuinely cannot decide. Obvious or
CTO-decidable items are decided and noted in the milestone notes instead.
