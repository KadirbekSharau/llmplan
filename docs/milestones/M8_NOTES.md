# M8 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the deliverables
order of M8_DESIGN.md section 2; "9b" is the carry-over item.

## Implementation notes

- **9b — Carry-over done first.** CI is two jobs: `check` (lint, format, mypy, the default
  pytest selection, audit, build, CLI entry point) and `ui` (the Streamlit AppTest suite
  with its own coverage gate on `llmplan/ui/*`, then the `slow` tests). The AppTest modules
  (`tests/unit/ui/test_app.py`, `tests/acceptance/test_m6.py`) carry
  `pytestmark = pytest.mark.ui`; `addopts` deselects `ui` (and the new `network` marker) by
  default like `slow` and `docker`. Tests over two seconds in the baseline run moved to
  `slow`: `test_vram_split_with_and_without_the_gpu_spec` (3.5 s),
  `test_8_5_determinism` (2.6 s) and `test_simulate_json_flags_override_and_png` (2.5 s).
  Locally: `uv run pytest` is the `check` selection, `uv run pytest -m ui` the AppTest
  suite, `uv run pytest -m slow --no-cov` the slow tests.

## Deviations from the design doc

## Questions for founder

None beyond docs/FOUNDER_QUESTIONS.md item 7 (PyPI account and trusted publishing).
