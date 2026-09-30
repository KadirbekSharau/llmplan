# M6 Design — Streamlit UI and Launch

Status: ready for implementation once M5 is merged. Assignee: developer agent.
Branch: `m6-ui`. Prerequisites: docs/PLAN.md (section 4, Deliverables), docs/ARCHITECTURE.md,
docs/DEFINITION_OF_DONE.md, and the M1 to M5 public APIs.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9.

---

## 1. Goal

The public launch deliverable: a single-page web app where someone picks a model, picks or
uploads traffic, sets a latency target, adjusts prices, and gets the cheapest fleet, the
vLLM command lines, and the utilization timeline. It contains no planning logic; every
button calls the library.

## 2. Deliverables

1. `llmplan/ui/app.py`: Streamlit app (one file, under 400 lines; helpers in
   `llmplan/ui/state.py` and `llmplan/ui/views.py` if needed).
2. `llmplan/ui/presets.py`: bundled trace samples and default SLO/price presets.
3. `data/traces/samples/`: downsampled public trace windows (section 5) with license notes.
4. `llmplan/ui/usage_log.py`: anonymous input-shape logging (section 7).
5. CLI `llmplan ui` (wraps `streamlit run`).
6. `Dockerfile`, `.dockerignore`, `docs/DEPLOY.md` (Streamlit Community Cloud and Docker).
7. `docs/LAUNCH.md`: draft posts for Show HN, r/LocalLLaMA, discuss.vllm.ai; the founder
   posts them.
8. Tests with `streamlit.testing.v1.AppTest`, README, CHANGELOG, `M6_NOTES.md`.

## 3. Page layout

Sidebar (inputs), main area (results). One "Plan" button; everything recomputes on click,
never on every widget change.

Sidebar:
1. **Model**: text input for a Hugging Face id, with a dropdown of fixtures and popular
   ids; `HF_TOKEN` note for gated models; shows `model-info` summary after resolving.
2. **Traffic**: radio: preset sample / upload CSV / synthetic. Preset shows the sample's
   stats; upload accepts `.csv` up to 50 MB (UI cap, lower than the library cap) and shows
   detected format and stats; synthetic exposes rate, duration, token distributions, seed.
3. **Latency target**: TTFT p95 ms, TPOT p95 ms, utilization target (defaults 500, 50, 0.8).
4. **Hardware and prices**: multiselect of GPU ids; provider filter; editable price table
   (`st.data_editor`) seeded from `data/prices.yaml`, with `as_of` shown; a "reset" button.
5. **Advanced** (collapsed): tp choices, dtype choices, max_num_seqs choices,
   max_model_len, perf backend, solver, time limit (UI max 30 s).

Main area, after Plan:
1. **Answer card**: cost $/day, baseline $/day, saving %, binding constraint, solver status.
2. **Fleet table** and **replica table** with copyable vLLM command lines.
3. **Timeline**: the M5 four-panel figure for the chosen traffic (window auto-chosen so
   there are 60 to 200 windows).
4. **Candidates**: the top 15 by $/hour per rps with status and reason; rejected ones shown
   greyed with the reason.
5. **Assumptions**: every assumption string from the plan and perf estimates, verbatim.
6. **Download**: JSON of the plan and the timeline.

Errors (`InfeasiblePlan`, `FetchError`, `WorkloadFormatError`, `PerfError`) render as a
single red box with the library's message; never a traceback. Unknown exceptions render a
generic message and are logged with a request id.

## 4. State and caching

- Inputs are collected into a `PlanRequest` plus `SimOptions`; the cache key is the SHA-256
  of their `model_dump_json()`. `st.cache_data` on the pure `run_plan(key, request)` wrapper,
  TTL 1 hour, max 200 entries.
- Model config fetches are cached separately (TTL 24 h).
- No state is written to disk except the usage log (section 7).

## 5. Bundled trace samples

For each public trace (Azure 2023 conv/code, Azure 2024 conv/code, BurstGPT), commit one
sample: the busiest 1-hour window (by requests) plus one median-load hour, downsampled if
needed to at most 20,000 rows, in the generic `csv` format, with a `README.md` in the
folder stating the source URL, license, the sampling method, and the exact rows kept. The
agent generates these with a script committed under `scripts/make_samples.py` that reads
the full traces from a path given on the command line (the full traces are never
committed). If a dataset license does not permit redistribution of samples, do not
commit the sample; note it and offer the synthetic preset instead.

## 6. Security and limits

- Upload cap 50 MB; parsed in memory; never written to disk; never echoed back.
- Solver time limit capped at 30 s in the UI; simulation `max_requests` capped at 200,000.
- Hugging Face id validated by the M1 fetcher rules before any request; the UI never
  builds URLs itself.
- No user-provided file paths anywhere in the UI.
- Price table edits are validated through `PriceRow` before use.
- Basic abuse control: per-session counter; more than 30 plans per hour shows a polite
  message and stops. Not a security boundary, just a brake.
- `docs/DEPLOY.md` states that the app must run without any secrets; `HF_TOKEN` is optional
  and, if set, is only used server-side for gated model configs.

## 7. Usage log

Append-only JSON lines at `LLMPLAN_USAGE_LOG` (env var path; logging disabled when unset):
`{ts, request_id, model_id, gpu_ids, n_requests, peak_rps, slo, cost_usd_per_day,
baseline_usd_per_day, solver_status, duration_s}`. Never the uploaded rows, never the
price edits, never IPs. Documented in the UI footer in one sentence with the exact fields.

## 8. Deployment

- `Dockerfile`: python 3.11 slim, `uv sync --no-dev`, non-root user, `EXPOSE 8501`,
  healthcheck on `/_stcore/health`.
- `docs/DEPLOY.md`: Streamlit Community Cloud steps (free tier, from the GitHub repo) and
  Docker steps; resource expectations (1 vCPU, 1 GB is enough for fixtures and samples).

## 9. Acceptance tests (`tests/acceptance/test_m6.py`, using `AppTest`)

**9.1 Loads.** `AppTest.from_file("llmplan/ui/app.py").run()` has no exceptions and shows
the Plan button.

**9.2 Preset end to end.** Select `fixture:llama3-8b`, the smallest bundled sample (or the
synthetic preset if no samples were committable), default SLO, GPUs `h100-sxm-80gb` and
`l4-24gb`, click Plan: completes within 60 s, renders a cost, at least one vLLM command
line containing `--tensor-parallel-size`, and a timeline figure.

**9.3 Infeasible surfaces cleanly.** TTFT target 1 ms: a single error element whose text
contains "0 of" and no element contains "Traceback".

**9.4 Upload validation.** Uploading a CSV with both `arrival_s` and `timestamp` shows the
`WorkloadFormatError` message; uploading 51 MB of bytes is refused before parsing.

**9.5 Price edit flows through.** Edit the H100 price to 10x, re-plan: the chosen fleet no
longer contains H100 or the cost increases; either assertion documented.

**9.6 Usage log.** With `LLMPLAN_USAGE_LOG` set to a temp file, one plan run appends one
JSON line with exactly the fields in section 7 and no others.

**9.7 Docker build.** `docker build .` succeeds (mark `@pytest.mark.docker`, skipped when
Docker is absent; the agent must run it once and record the image size in the notes).

## 10. Launch checklist (`docs/LAUNCH.md`)
- Deployed URL responds; a plan on each preset completes under 60 s.
- Posts drafted: title, two-paragraph body, one screenshot, a "what I'd like feedback on"
  line. Founder reviews and posts.
- Usage log enabled; a weekly summary script (`scripts/usage_summary.py`) prints counts by
  model id and GPU set.

## 11. Allowed dependencies
Runtime adds: `streamlit`. Nothing else.

## 12. Questions for founder
- Hosting: Streamlit Community Cloud (free, public repo required) or a paid container host
  with the repo private? Interim: Docker image is built either way; decision needed before
  launch.
- App name and domain for the launch posts. Interim: "llmplan".
