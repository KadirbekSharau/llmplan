# M6 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. "Step N" refers to the deliverables
order of M6_DESIGN.md section 2; "8b" is the carry-over item.

## Implementation notes

- **8b — Carry-over done first.** `ReplicaWindowRecord.vram_bytes_total` is appended as the
  last field, so every earlier field of the timeline JSON is unchanged (checked in
  `tests/unit/test_simulate_cli.py`). The plan does not carry `GPUSpec`s, so `replay()`
  takes the catalog the plan was made with (`gpus=`, see Deviations); `llmplan simulate`
  passes the shipped catalog or `--gpu-catalog PATH`. The plot's third panel is now the
  fleet's stacked VRAM: weights, KV in use (max per window) and free (total minus both),
  summed over replicas; free is drawn only when every replica's total is known.
  `render_png(timeline) -> bytes` was split out of `save_png` for the web page.
- **Step 1 — Dependency.** Runtime adds `streamlit>=1.64` (section 11; tested with 1.64.0).
  It pulls in pyarrow, altair, pydeck, starlette, uvicorn and others; `uv
  audit` reports no known vulnerabilities in the 79 locked packages.
- **Step 1 — Uploads are parsed from memory.** `load_workload` only read paths, and section
  6 forbids writing uploads to disk, so `InMemoryTrace(name, data)` is accepted by header
  detection, the size cap and the chunked parser (see Deviations). The bytes are left out
  of `repr`/`str`; messages say `uploaded.csv`, never the visitor's filename or content.
  The parsed upload lives in that session's state only (not in a cross-session cache), so
  no upload outlives the session that sent it.
- **Step 1 — "Everything recomputes on click."** Plan, replay and Hugging Face fetches run
  only on Plan. Two cheap, local lookups happen on widget change because section 3 asks
  for them: a preset's or upload's stats, and the model-info summary of a shipped fixture
  (`llmplan model-info` text). A Hugging Face model's summary appears after a plan has
  fetched its config, so no network request is made on a widget change.
- **Step 1 — Cache.** `st.cache_data(ttl=3600, max_entries=200)` on the plan-and-replay
  wrapper, keyed by the SHA-256 of `PlanRequest.model_dump_json()`, the `SimOptions` JSON
  and a digest of the workload's arrival and token columns (see Deviations). Model configs:
  `st.cache_data(ttl=86400)`; catalogs and preset workloads: `st.cache_resource` (immutable
  models, no copy per hit).
- **Step 1 — One run at a time.** HiGHS keeps a process-wide thread pool (M4_NOTES.md) and
  Streamlit serves sessions from several threads, so `state.run_plan` holds a module lock
  around plan and replay. Runs take about a second, so the queueing cost is small.
- **Step 1 — Timeline window.** The smallest 1/2/5 x 10^k seconds value of at least
  `duration / 150`, which gives 60 to 151 windows over the trace (hypothesis-tested).
  When queues make the replay run long past the last arrival (more than 200 windows), the
  replay is repeated once with a window chosen over the full span.
- **Step 1 — Price table.** The editor (`st.data_editor`, rows can be added) is seeded from
  `data/prices.yaml`; each row is validated through `PriceRow` when Plan is clicked, the
  first invalid row is reported by index and field, unknown `gpu_id`s are rejected, and
  rows whose cells are all blank are ignored. "Reset prices" restores the shipped table.
  The provider filter offers the shipped providers plus any typed into the table.
- **Step 1 — Errors.** Every `LLMPlanError` (not only the four named in section 3) renders
  as one red box with the library's message; any other exception renders "Something went
  wrong (request id ...)" and is logged with that id and its traceback on the server only.
- **Step 1 — Brake.** 30 Plan clicks per rolling hour per session, counted whether or not
  the plan succeeds; the 31st shows a warning and runs nothing.
- **Step 1 — Answer card.** Cost, baseline and saving are metrics; binding constraint and
  solver status are one line under them (five metrics truncated at laptop widths).
- **Step 1 — Candidates.** The 15 best eligible candidates by $/hour per req/s, then every
  rejected candidate greyed, with status and reason. **Assumptions** are the plan's, the
  chosen replicas' perf estimates' (duplicates dropped, first occurrence order) and the
  replay's, escaped so `$` and Markdown characters render verbatim. **Downloads** are the
  `llmplan plan --format json` document (loadable by `llmplan simulate --plan`) and the
  timeline JSON.
- **Step 2 — Presets.** The five bundled samples plus a synthetic preset (2 req/s for one
  hour, the `workload synth` default token lengths, seed 0). The default preset is the
  Azure 2024 conversation sample, the milestone's headline preset.
- **Step 3 — Samples.** All five public traces are CC-BY-4.0 (M2_NOTES.md, GitHub license
  API), which permits redistributing adapted excerpts with attribution and a statement of
  changes; `data/traces/samples/README.md` gives both. Rows committed:

  | Sample | Rows | Source hours kept |
  |---|---|---|
  | `azure2023_code.csv` | 8,819 | whole trace (57.3 min, shorter than an hour) |
  | `azure2023_conv.csv` | 19,366 | whole trace (58.4 min) |
  | `azure2024_code.csv` | 19,999 | busiest hour 139 (301,105 req) + median hour 24 (60,686), thinned to 5.53% |
  | `azure2024_conv.csv` | 19,999 | busiest hour 62 (269,238 req) + median hour 74 (168,059), thinned to 4.57% |
  | `burstgpt_1.csv` | 19,999 | busiest hour 235 (33,298 req) + median hour 905 (189), thinned to 59.72% |

  Generating them took 95 s and 2.25 GB peak RSS on this laptop (the Azure 2024
  conversation parse dominates, as M2 measured). Downsampling is proportional thinning of
  both hours, so a thinned sample's request rate is that fraction of the source's; the
  preset labels say so ("4.6% of rows"). Thinning was chosen over a contiguous sub-window
  because it keeps the busy/median contrast and the token distribution; the cost is that a
  plan on a thinned preset sizes for a proportionally smaller deployment.
- **Step 3 — `data/traces/.gitignore`** admits exactly the five sample files and their
  README, so a full trace fetched into the checkout still cannot be committed.
- **Step 4 — Usage log.** A line is written for every Plan click that reached the planner
  (workload statistics computed), including failures: `solver_status` is `optimal` or
  `feasible_time_limit` from the solver, `infeasible` for `InfeasiblePlan`, `error`
  otherwise, with null costs on failure. Clicks stopped earlier (no model id, invalid
  upload) and clicks stopped by the brake are not plan runs and are not logged.
  `gpu_ids` is the visitor's GPU selection; `slo` is the `SLO` model's JSON;
  `duration_s` is wall time of the click (rounded to ms); `ts` is UTC, to the second.
- **Step 5 — `llmplan ui`** calls Streamlit's `bootstrap.load_config_options` and
  `bootstrap.run` in-process (the package may not use `subprocess`), with the 50 MB server
  upload cap and Streamlit's browser telemetry off (`browser.gatherUsageStats=false`, also
  in `.streamlit/config.toml` for `streamlit run` and Community Cloud). Smoke-tested: the
  server answered `ok` on `/_stcore/health`.
- **Step 6 — Docker.** See "Docker image" below. `uv` is copied from its official image
  pinned to 0.12.21 (no `curl | sh`); the project is installed editable so `data/` is read
  from `/app/data` (M1_NOTES.md), and `data/` (catalogs, fixtures, benchmarks, samples) and
  `.streamlit/` are copied explicitly.
- **Tests.** `tests/acceptance/test_m6.py` holds 9.1 to 9.7 (9.7 marked `docker`). The
  default `pytest` run excludes `docker` as well as `slow` (a Docker build takes minutes and
  downloads packages): `uv run pytest -m docker --no-cov` runs 9.7 and skips it without
  Docker. A `slow` test plans every preset through the page (`-m slow -k every_preset`).
  A static test checks that no module outside `llmplan/ui/{app,views}.py` imports Streamlit
  at module level (`cli_ui.py` imports it inside the command).
- **Measured.** 9.2 (smallest sample, `azure2023_code`, H100 + L4, default SLO): plan,
  replay and page render in **0.49 s** through AppTest on this laptop (Apple Silicon,
  Python 3.11). The milestone's Azure 2024 conversation preset with all GPUs: **0.76 s**;
  every preset is under 0.8 s.
- **Package growth (~1,280 net lines under `llmplan/`).** ARCHITECTURE.md section 1 asks
  for a justification above ~300 lines: the UI is the launch deliverable. `app.py` (~390,
  under the design's 400), `views.py` (~200), `state.py` (~245), `presets.py` (~160),
  `usage_log.py` (~100), `cli_ui.py` (~45), plus ~40 lines for in-memory traces and ~40
  for the VRAM panel. Only `app.py` is over the ~300-line module guideline: it is one page
  of widgets, bounded by the design at 400 lines, with its pure logic in `state.py`.
- **`docs/MILESTONES.md`** is set to "ready for review"; the CTO records the merge hash.

### Docker image

Docker was present on this machine (Docker 20.10.12, Apple Silicon). `docker build .`
succeeded: image `llmplan:m6`, **1.36 GB** (1,362,624,664 bytes, linux/arm64), built in
74 s from cached base images. Most of the size is the locked runtime dependencies
(OR-Tools, pyarrow, pandas, matplotlib, numpy, Streamlit) on `python:3.11-slim`. The
container ran as uid 10001, `/_stcore/health` answered `ok`, Docker reported it
`healthy`, and a plan on the default preset completed in the page.

On this machine BuildKit hung indefinitely at "load metadata for
docker.io/library/python:3.11-slim" (the client's `credsStore: desktop` helper does not
answer from the agent's shell); `docker pull` worked. The recorded build therefore used
the classic builder and an empty client config:
`DOCKER_BUILDKIT=0 DOCKER_CONFIG=<empty dir> docker build -t llmplan:m6 .`, and 9.7 passed
the same way (`DOCKER_BUILDKIT=0 DOCKER_CONFIG=<empty dir> uv run pytest -m docker
--no-cov`). The Dockerfile itself needs neither (no BuildKit-only syntax).

## Deviations from the design doc

- **`replay()` gains `gpus: Mapping[str, GPUSpec] | None = None`** (8b). Section 8b derives
  `vram_bytes_total` "from the candidate's GPU spec", but a `PlanResult` holds price rows,
  not GPU specs. Loading the shipped catalog inside `replay` would make it impure and
  would fail for plans on custom or test GPUs (the M5 acceptance tests use fake GPUs and
  must pass unchanged). So the caller passes the catalog it planned with, and
  `vram_bytes_total` is `int | None`: null, with an assumption line, for a GPU the catalog
  does not hold. ARCHITECTURE.md sections 4 and 5 updated.
- **`InMemoryTrace` and `load_workload(source: str | Path | InMemoryTrace)`** (step 1).
  Section 6 requires uploads parsed in memory and never written to disk; the M2 parsers
  took paths only. The `TraceFormat` protocol's `parse` and `detect` take `Path |
  InMemoryTrace`. ARCHITECTURE.md sections 3, 5 and 6 updated.
- **`llmplan simulate --gpu-catalog PATH`** (8b): the CLI's way to pass that catalog,
  mirroring `llmplan plan --gpu-catalog`. README updated.
- **Cache key includes the workload rows' digest.** Section 4 keys the cache on
  `PlanRequest` and `SimOptions` JSON, but the request holds only workload statistics and
  the timeline depends on the rows; two traces with equal statistics would share a
  timeline. The digest of the arrival and token columns is appended to the key.
- **Upload cap is 50,000,000 bytes** (the conservative reading of "50 MB"; Streamlit's own
  server cap is set to 50 MiB, slightly above, and the app's check runs first).
- **9.4 "refused before parsing"** is asserted by patching `llmplan.workload.load_workload`
  and checking it is never called for the 51,000,000-byte upload, in addition to the
  message.
- **9.5 price edit.** `AppTest` cannot type into `st.data_editor` cells, so the test
  replaces the editor's source table in session state (`price_table`, the same frame the
  editor shows) with the H100 rows at 10x, then plans again; everything after the editor
  (validation through `PriceRow`, the request, the cache key, the plan) is the real path.
  The documented assertion is the cost increase, with H100 as the only GPU: the cost rises
  by exactly 10x (same fleet, every row scaled alike).
- **9.7 is excluded from the default test run** (marker `docker`, deselected in
  `addopts` like `slow`), because a build takes minutes and needs the network; it is
  skipped when Docker is absent as the design says.
- **Launch screenshot.** The drafts in docs/LAUNCH.md describe the screenshot to take from
  the deployed app instead of embedding one: the deployed URL and name are not decided,
  and a local screenshot would show a localhost address.

## Questions for founder

None new. Hosting (question 5) and the public name (question 6) are already logged in
docs/FOUNDER_QUESTIONS.md; docs/DEPLOY.md documents both hosting paths and docs/LAUNCH.md
uses `<URL>` and `<REPO>` placeholders until they are decided. Per the founder's rule,
these CTO-decidable items were decided here and are open to CTO review:

- **Thinned presets.** The Azure 2024 and BurstGPT presets carry 4.6% to 60% of their
  source hours' requests. If the CTO prefers presets at full rate, the alternative is a
  contiguous slice of the busiest hour (about 4 minutes of Azure 2024 traffic fits in
  20,000 rows), at the cost of the median-hour contrast.
- **Community Cloud dependencies.** docs/DEPLOY.md generates `requirements.txt` with `uv
  export` at deploy time rather than committing a second dependency list.
