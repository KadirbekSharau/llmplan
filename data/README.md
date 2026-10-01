# Moved: data now lives in `llmplan/data/`

Since M8 (v0.2.0) the GPU and price catalogs, benchmark rows, model fixtures, the trace
manifest and the bundled trace samples are package data under
[`llmplan/data/`](../llmplan/data/), so they ship in the wheel and `pip install llmplan`
works without a checkout. They are located through `llmplan/paths.py`
(`importlib.resources`). The milestone records in `docs/milestones/` written before M8
refer to the old `data/` paths; read them as `llmplan/data/`.
