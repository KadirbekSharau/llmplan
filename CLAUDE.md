# Repo rules for agents

- Read docs/PLAN.md and docs/ARCHITECTURE.md before writing code. Build only the milestone
  you were assigned (docs/MILESTONES.md); its design doc's acceptance tests define done.
- Python 3.11+, `uv`, `ruff`, `mypy --strict`, `pytest`. No network in tests.
- Frozen pydantic v2 models at boundaries; pure functions inside; units in field names.
- No `utils.py`. No new dependency without a one-line reason in the PR.
- Security: `yaml.safe_load` only; no pickle/eval/exec/subprocess; HF ids validated; tokens
  only from `HF_TOKEN` env and never logged.
- Never build anything that connects to a cluster, a GPU, or a live inference server.
- If an interface in ARCHITECTURE.md must change, change the doc in the same PR and say why.
- Commit format: conventional commits. Small PRs in the order the design doc suggests.
