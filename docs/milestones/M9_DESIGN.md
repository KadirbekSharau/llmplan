# M9 Design — UI components, experience, and responsiveness

Status: ready for implementation. Branch: `m9-ui-ux`. Prerequisites: docs/PLAN.md,
docs/ARCHITECTURE.md, docs/DEFINITION_OF_DONE.md, docs/milestones/M6_DESIGN.md and
M6_NOTES.md (the current UI), M7/M8 notes (classes, calibration, confidence).

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9, plus the CTO's
visual review in a real browser at desktop and phone widths (section 10).

---

## 1. Goal

The current page is a correct but dense engineer's dashboard. Make it something a visitor
from a forum link understands in ten seconds, can use on a phone, and wants to share:
a guided flow, a clear answer first, details on demand, interactive charts, and a link
that reproduces any plan. No new planning logic; the UI stays a thin adapter.

## 2. Principles

- **Answer first.** After Plan, the first screen shows: cost per day, the fleet in one
  sentence, confidence, and one primary action (copy the vLLM command). Everything else is
  below or collapsed.
- **Progressive disclosure.** Inputs are four numbered steps; advanced settings are
  collapsed; candidates, assumptions, and per-class detail live in expanders or tabs.
- **Works on a phone.** Nothing requires a wide table to be understood. Tables show the
  five columns that matter and hide the rest behind a toggle. Charts are interactive and
  readable at 390 px.
- **Nothing recomputes on a widget change.** Only the Plan button triggers work; every
  stage shows progress.
- **Shareable.** Every plan has a URL that reproduces it.
- **Small code.** `llmplan/ui/` stays under 1,500 lines total; components are small
  functions in `views.py`; no new dependency except what Streamlit already ships (Altair).

## 3. Layout

**Header**: product name, one-line promise, version, GitHub link, confidence legend icon.

**Inputs** (sidebar on desktop; on narrow screens the sidebar is collapsed by default and
the same four steps render as a top accordion in the main area, since Streamlit cannot
detect viewport width, implement this as a "Compact layout" toggle persisted in query
params with a sensible default of off; document the limitation):
1. **Model**: dropdown with popular ids grouped (Llama, Qwen, Mistral, MoE), a free-text
   field, and an inline summary chip (params, attention, KV per token) after resolution.
2. **Traffic**: three cards (Preset / Upload / Synthetic) with a short description and the
   resulting stats chip (requests, peak rps, mean tokens).
3. **Target**: TTFT, TPOT, utilization, with plain-language help ("TTFT is how long until
   the first word appears"). Presets: Chat (500/50), Batch (none), Strict (200/30).
4. **Hardware and prices**: GPU multiselect with provider filter; price editor collapsed
   behind "Edit prices"; "Reset" button.
**Advanced** (collapsed): tensor parallel, dtypes, max_num_seqs, max_model_len, classes,
perf backend, solver, time limit, calibration upload (M8).

**Primary action**: a full-width Plan button at the bottom of inputs and a sticky copy in
the main area header when results are stale (inputs changed since the last plan).

**Results**:
- **Answer card**: `$X/day` large; one sentence: "2 x H100 (RunPod) serving 4 replicas";
  baseline comparison with a coloured delta; confidence badge (measured / interpolated /
  roofline) with a tooltip linking to the calibration expander; buttons: Copy vLLM command,
  Copy share link, Download JSON.
- **Tabs**: Fleet | Routing | Timeline | Candidates | Assumptions.
  - Fleet: fleet table (provider, instance, count, $/day) and replica cards with the
    command line in a code block.
  - Routing: class table and routing weights as a horizontal bar chart (Altair), only
    when K > 1; otherwise a one-line note.
  - Timeline: interactive Altair charts (demand vs capacity; utilization; VRAM split;
    queue depth and violations) with hover tooltips and a shared x-axis; a window-size
    selector; PNG download via the existing renderer.
  - Candidates: table with status chips, sortable, "show rejected" toggle, reason on
    hover/expand; default shows the top 15 eligible.
  - Assumptions: grouped (plan / performance / simulation) as bullet lists; roofline
    constants shown.
- **Empty state** before the first plan: a three-sentence explanation and two example
  scenario buttons ("Llama 3.1 8B chat on cheap GPUs", "Qwen3-30B-A3B document
  processing") that prefill inputs and run.

**Footer**: usage-log sentence (unchanged), license, link to the docs.

## 4. Progress and errors

- `st.status` with steps: resolving model, loading traffic, estimating performance,
  solving (with the time limit), replaying. Each step shows elapsed time when done.
- Errors render in the results area as a single callout with the library message and a
  "what to change" hint derived from the error type (`InfeasiblePlan` -> "relax the latency
  target or add GPUs"; `FetchError` -> "check the Hugging Face id or set HF_TOKEN";
  `WorkloadFormatError` -> the expected columns).
- Stale indicator: when any input changed after the last plan, the answer card shows a
  subtle "inputs changed, plan again" chip.

## 5. Share links

Encode the inputs needed to reproduce a plan in query params: model id, traffic source
(preset name or synthetic parameters; uploads are not shareable and the link says so),
SLO, GPU ids, providers, advanced settings, classes. Total under 2,000 characters; values
validated on load through the same pydantic models; unknown or invalid params are ignored
with a notice. "Copy share link" uses `st.query_params` and a code block fallback.

## 6. Theme and polish

- `.streamlit/config.toml` theme: a primary colour that passes contrast on light and dark,
  base font unchanged (performance), `layout="wide"`, `initial_sidebar_state="auto"`.
- Page title "llmplan — GPU fleet planner for LLM inference", favicon from an inline SVG.
- Number formatting helpers: currency, thousands separators, ms vs s automatically.
- Consistent naming: "latency target" (not SLO) in user-facing text; "replica" explained
  once with a tooltip.

## 7. Performance

- `st.cache_data` for catalogs, presets, model configs (24 h), plan runs keyed by the
  share-link payload (1 h, max 200).
- Altair charts built from downsampled timelines (at most 500 windows) so the browser
  stays responsive on long traces.
- First-paint under 1 s on the Droplet after warm-up (measured with the AppTest run time
  as a proxy and recorded in the notes).

## 8. Out of scope

Authentication, saved plans, multi-page apps, custom React components, analytics beyond
the existing usage log, any change to planning logic or CLI output.

## 9. Acceptance tests (`tests/acceptance/test_m9.py`, AppTest unless stated)

9.1 Empty state shows the two example scenario buttons; clicking the first runs a plan
    with no exceptions and renders the answer card with a cost.
9.2 Share link round trip: run a plan on a preset, read the share link, start a fresh
    AppTest with those query params, run, and get a byte-identical plan JSON.
9.3 Invalid query params (unknown GPU id, negative TTFT) are ignored with a visible notice
    and the page still renders.
9.4 Stale indicator appears after changing the TTFT field post-plan and disappears after
    planning again.
9.5 Error hints: an infeasible target renders one callout containing "relax"; a bad upload
    renders one containing the expected column names.
9.6 Progress: the status container lists all five step labels after a plan.
9.7 Candidates tab: default shows at most 15 rows and no rejected ones; the toggle reveals
    rejected rows with their reasons.
9.8 Routing tab renders the bar chart only when classes > 1 (check the Altair chart element
    count in both cases).
9.9 Timeline tab: changing the window selector re-renders with a different number of
    windows; PNG download button present.
9.10 All five presets run without exception under the default inputs (marked `slow`).
9.11 Line budget: `llmplan/ui/` total under 1,500 lines (a test counts lines).
9.12 Existing M6, M7, M8 UI tests still pass (update only selectors/labels, never expected
     values; list every change in the notes).

## 10. Visual review (CTO, real browser)

Desktop at 1440 px and phone at 390 px: empty state, inputs flow, results on the Azure
preset, each tab, an error state, light and dark theme. Findings go back to the agent as a
change request on the same branch.

## 11. Allowed dependencies
None new. Altair is already a Streamlit dependency; import it from there.
