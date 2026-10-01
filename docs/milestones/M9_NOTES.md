# M9 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. Section numbers refer to
docs/milestones/M9_DESIGN.md.

## Implementation notes

- **Inputs live in session state (sections 3 and 5).** Every input is a widget with a key
  and no default argument; `presets.INPUTS` holds each key's default and its choices (or,
  for a number, its bounds), and `app._seed` writes the defaults into session state before
  any widget is drawn, on every run (Streamlit drops the state of a widget the previous
  run did not draw). Share links and the example scenarios write the same keys, so the
  page has one source of truth for its inputs. Three Advanced keys take
  `state.build_request`'s argument names (`tensor_parallel`, `max_num_seqs`,
  `time_limit_s`), so the request is built from `presets.OPTION_KEYS`.
- **Layout (section 3).** The four steps are expanders: open in the sidebar; in the
  compact layout an accordion above the results with step 1 open. Advanced is a sibling
  expander and holds the calibration upload (M8). Streamlit forbids nested expanders, so
  the price editor sits behind an "Edit prices" popover and the model's full text
  (`llmplan model-info`) behind a "Model info" popover; the M8 "vLLM JSON details"
  expander became four fields shown only when the upload is a JSON (a CSV carries those
  columns). The model picker labels each id with its family ("Llama · fixture:llama3-8b";
  a selectbox has no option groups); the traffic "cards" are a radio with a caption per
  source; Chat, Batch and Strict are buttons that set TTFT and TPOT (Batch empties both:
  no target; the fields accept empty).
- **Compact layout (section 3).** Streamlit cannot detect the viewport width, so a
  "Compact layout" toggle under the header switches it, default off, remembered in the URL
  as `layout=compact` (a reload or a shared link with it opens compact). With
  `initial_sidebar_state="auto"` Streamlit collapses the sidebar on narrow screens anyway;
  in compact mode the sidebar is empty and hidden.
- **Answer card and tabs (section 3).** The card shows the cost per day as a metric with
  the baseline saving as its delta (green, or red for "baseline Y% cheaper" after a
  time-limited solve; never a negative percentage, M8), the fleet in one sentence ("1 x
  L40S 48GB (aws g6e.xlarge) + 1 x L4 24GB (aws g6.xlarge) serving 2 replicas", replica
  explained in a tooltip), a confidence badge (green measured, blue interpolated, orange
  roofline or mixed; its tooltip is the legend and links to the calibration section), the
  M8 banner, the first replica type's `vllm serve` line, the share link, and "Download plan
  JSON". "Copy" is Streamlit's copy icon on each code block (no JavaScript). Tabs: Fleet
  (table, then a card per replica type with its command line), Routing, Timeline,
  Candidates, Assumptions (the plan's, the perf estimates' and the replay's assumptions,
  then the roofline constants that M8 showed in an expander under the banner).
- **Narrow tables (section 3).** Every table shows its first five columns, ordered so the
  essentials come first (fleet: instance, instances, $/day, GPUs each, provider);
  "Show all columns" adds the rest. Dataframes are sortable by clicking a header, and a
  long cell (a candidate's reason) shows in full on hover.
- **Routing (section 3).** With a comparison (any classes) the tab shows the M8 sentence
  and the class table; with two or more classes the routing weights are a horizontal
  stacked bar per class (Altair, tooltip with the class's token bounds and the weight).
  Without classes, one line says how to turn them on. The M7 routing table is replaced by
  the chart.
- **Timeline (sections 3 and 7).** One Altair `vconcat` of four panels (demand against
  capacity; mean and maximum replica utilization; the fleet's VRAM as stacked areas of
  weights, KV cache in use and free memory, from `render.plots.vram_split_gb`; queue depth
  and requests over the latency target), sharing one x axis that drags to zoom and pan
  across all panels, with hover tooltips. Replays over 500 windows are drawn every k-th
  window. The window selector offers round windows from a fifth of the automatic window
  to five times it (e.g. 10/20/50/100/200 s around 50 s); choosing one replays the plan's
  fleet on the stored trace at that window (see Deviations). "Download PNG" renders the
  M5 figure only when clicked (the button takes a callable); "Download timeline JSON" is
  unchanged.
- **Candidates (section 3).** The 15 cheapest eligible candidates by $/hour per req/s; a
  "Show the N rejected candidates" toggle appends the rejected ones with their reason.
  Status is a plain text column: coloured chips would need a custom component.
- **Progress (section 4).** `st.status` lists the five steps with their elapsed time:
  resolving the model and loading the traffic are timed in the page; `run_plan` gains a
  `progress` callback that reports the plan's wall time less the solver's ("estimating
  performance"), the solver's own time ("solving", labelled with the time limit) and the
  replay with the single-class comparison ("replaying"). A cached run reports nothing, so
  those three steps say "cached result". On success the status reads "Planned in X"; on
  failure it ends in the error state, its label naming the stage that failed.
- **Error hints (section 4).** `state.error_text` appends "What to change: ..." chosen by
  `isinstance` on the exception type, first match wins: `InfeasiblePlan` (relax the
  latency target or add GPUs), `FetchError` (check the id or set HF_TOKEN),
  `WorkloadFormatError` (the expected columns), `SolverError` (raise the time limit).
  Other types keep the library message alone (it already names the field). An upload
  error keeps its exception (it is no longer stored as text), so planning with a bad
  upload raises the same `WorkloadFormatError` and gets the column hint.
- **Share links (section 5).** After each plan the page URL is replaced by the plan's
  inputs (`model`, `traffic` as a preset key or `synthetic` with `syn_rate`,
  `syn_duration`, `syn_in`, `syn_out`, `syn_seed`, then every other input under its widget
  key, lists comma-joined, no target as `none`; plus `layout=compact` when set), and the
  answer card shows the link in a code block. The base is `st.context.url` (empty under
  AppTest, so the link is relative there). A session's first run decodes the URL: each
  parameter is checked against the page's choices or bounds and through the request
  models (`SLO` for the target, `PlanOptions` for the hardware and advanced inputs,
  `parse_distribution`/`Distribution` for token lengths, the Hugging Face fetcher's
  `is_repo_id` for a custom model, now public in `catalog.models`); unknown or invalid
  parameters are ignored and named in a warning, the valid ones are applied, and the page
  plans. Uploads and price edits are not shareable: an upload's link leaves the traffic
  out and the card says it opens with the default preset. A link over 2,000 characters
  is not offered (the inputs' formats keep real links far below; the 9.2 link is 289
  characters plus the page's address).
- **Stale indicator (section 4).** The page fingerprints every input (the session values
  of `presets.INPUTS`' keys, the uploads' file ids, the vLLM JSON details and the price
  table) when it plans; while the current inputs differ, the answer card shows an "Inputs
  changed since this plan" chip and the main area a "Plan again" button.
- **Empty state (section 3).** Three sentences (two in the explanation, then "Choose the
  inputs in the sidebar, then click Plan.", the M6 info box) and two examples that set
  their inputs over the defaults and plan: "Llama 3.1 8B chat on cheap GPUs" (the Azure
  2023 conversation sample, TTFT 500 ms and TPOT 100 ms on A10G, L4 and L40S:
  1 x L40S + 1 x L4, $63.98/day, saving 28.4% against one GPU type) and "Qwen3-30B-A3B
  document processing" (synthetic, 1 req/s for an hour, input lognormal(8.0, 0.6)
  clipped to 6,000 tokens, output lognormal(5.0, 0.6) clipped to 1,000, no latency
  target: 1 x L40S, $44.66/day).
- **Theme (section 6).** `.streamlit/config.toml` sets the primary colour `#1F6FEB` under
  `[theme.light]` and `[theme.dark]`: 4.63:1 against white (also behind a primary
  button's white label, WCAG AA) and 4.08:1 against the dark background `#0E1117` (AA for
  large text and UI components). A plain `[theme] primaryColor` was tried first; it turns
  the page into a light-only custom theme, which the browser review caught. The base font
  is unchanged. Page title "llmplan — GPU fleet planner for LLM inference", an inline SVG
  favicon (three bars in the primary colour), wide layout, automatic sidebar.
- **Formats and naming (section 6).** `state.usd` ($1,234.50) and `state.duration` (ms
  below one second with three significant digits, else seconds). User-facing text says
  "latency target"; "SLO" remains only inside library messages and assumptions.
- **Caching (section 7).** `st.cache_data` for the catalogs, model configs and presets
  (24 hours; the catalog's `MappingProxyType` is converted to a dict so it pickles), for
  plan runs (1 hour, 200 entries) and for re-windowed replays (1 hour, 200 entries, keyed
  by the plan's key and the window).
- **Altair.** `render/charts.py` imports `altair` directly; it is a dependency of
  Streamlit (6.3.0 locked here), so nothing new is installed. One zoom selection is added
  to all four panels; Altair 6 merges it and warns that it did, so that one `UserWarning`
  is silenced where the chart is built.
- **First paint (section 7).** AppTest run time of a fresh session after warm-up, as the
  design asks: median **0.35 s** (0.346 to 0.384 s over five sessions; the cold first run,
  which imports Streamlit, pandas, OR-Tools and Altair, took 9.2 s). Plan, replay and
  render of each preset with the default inputs through AppTest: azure2024-conv 0.43 s,
  azure2024-code 0.38 s, azure2023-conv 0.41 s, azure2023-code 0.33 s, burstgpt-1 0.58 s,
  synthetic 0.30 s. This laptop (8 cores) was shared with other jobs; the load average
  was 57 to 120 during these measurements, so the Droplet should do no worse after warm-up.
- **Line budget (section 9.11).** `llmplan/ui/` was **1,362** lines at `ff3a178`
  (origin/main when the branch was cut) and is **1,495** now (`wc -l llmplan/ui/*.py`;
  9.11 counts every line, blank and docstring lines included). See Deviations for the
  renderers that moved to `render/` to stay under it. Module sizes: `app.py` 390 (one page
  of widgets and the plan flow; M6 allowed this file 400), `state.py` 330 (M6's helpers
  plus the formats and hints), every other module under 300.
- **Tests.** `tests/acceptance/test_m9.py` holds 9.1 to 9.11: AppTest tests are marked
  `ui` (9.10 also `slow`, over all six presets: the five samples and the synthetic one);
  9.11 runs in the default selection. 9.12 is the M6, M7 and M8 suites themselves. New
  unit tests: `tests/unit/ui/test_share.py` (round trips, every rejection path),
  `test_formats.py` (formats, hints by type), `test_results.py` (window choices, the
  re-windowed replay, the progress stages), additions to `test_presets.py` (model
  families, defaults inside their bounds, valid examples) and
  `tests/unit/render/test_render_charts.py` (chart data, thinning to 500 windows, routing
  bars, fleet sentence, model summary). Coverage: 99.88% in the default run; the `ui` job
  gate over `llmplan/ui/*` measures 95%.
- **Existing UI tests touched (9.12).** One selector, no expected value:
  `tests/acceptance/test_m6.py::test_9_2_preset_end_to_end` looked for the timeline
  figure with `len(at.image) >= 1`; the figure is now an Altair chart, so it looks for
  `at.get("vega_lite_chart")`. `tests/unit/ui/test_presets.py` gained two tests (above);
  its existing tests are unchanged. Every other M6, M7 and M8 UI test passes unchanged:
  the M6 empty-state text, error messages, widget keys `model_choice`, `traffic_mode`,
  `preset`, `syn_rate`, `syn_duration`, `ttft`, `utilization`, `gpu_ids`, `classes`,
  `upload`, `bench_upload`, the "Cost per day" metric, the M8 banner and the
  `` `BANDWIDTH_EFFICIENCY` = 0.7 `` line were all kept.
- **`docs/MILESTONES.md`** is set to "ready for review"; the CTO records the merge hash.

## Deviations from the design doc

- **Renderers moved to `llmplan/render/` to stay under 1,500 lines (9.11).** With every
  feature in place, `llmplan/ui/` measured about 1,600 lines in this codebase's ruff style
  (the budget left 138 lines over the 1,362 it started with). The Altair specs are
  renderers of library results, like `render/plots.py` (the same four timeline panels as
  a PNG), so they live in `render/charts.py`; the fleet sentence joins
  `render/plan_text.py` (where `class_comparison_sentence` and `baseline_saving`, which the
  UI already used, live) and the model summary joins `render/text.py` (next to
  `model_info`). Only the web UI imports `render/charts.py`. If the CTO prefers them in
  `llmplan/ui/`, the budget would need about 120 more lines. ARCHITECTURE.md sections 1,
  3 and 6 updated.
- **The Timeline window selector replays (section 1 and 7, "nothing recomputes on a widget
  change").** 9.9 asks for a selector that re-renders with a different number of
  windows; windows are aggregated by the simulator, and re-aggregating them in the page
  would be simulation logic in the UI. So choosing a window repeats the replay of the
  planned fleet on the stored trace (`state.replay_window`, cached per plan and window;
  about 0.3 s on the bundled samples); the plan is never recomputed. It is the only widget
  that computes. The alternative, replaying every choice when Plan is clicked, would make
  every plan about five replays slower for a selector few visitors use.
- **Opening a share link plans (9.2).** "Start a fresh AppTest with those query params,
  run, and get a byte-identical plan JSON" is read as: the link reproduces the plan
  without a click. A session's first run applies the valid parameters and plans (counted
  by the 30-per-hour brake and cached like any plan); a link whose parameters are all
  invalid only shows the notice.
- **Plan cache key (section 7).** Runs stay keyed by `state.cache_key` (request JSON,
  replay options, trace digest, uploaded rows), not by the share payload alone: the
  payload omits price edits and uploads, so two different plans could share it. Equal
  share links give equal requests and therefore the same key.
- **"Sticky copy" of Plan (section 3).** When results are stale, a "Plan again" button sits
  at the top of the main area; it is not CSS-sticky, since the design rules out custom
  components and JavaScript.
- **Example "Llama 3.1 8B chat on cheap GPUs" uses TPOT 100 ms.** With the Chat preset's
  50 ms, no A10G, L4 or L40S candidate meets the target for the long-request class, so
  the example would open on an infeasible plan; at 100 ms it plans a mixed L40S + L4
  fleet, which also shows request-size routing.
- **Upload errors show in the sidebar and, after Plan, in the results.** M6 test 9.4
  expects the sidebar to show the library message alone as soon as a bad file is
  uploaded; the results area's callout (with the hint) appears only after Plan, so in
  that case two red boxes are on the page, one per area.
- **The theme is in `.streamlit/config.toml` only.** `llmplan ui` passes its server flags
  unchanged, because `tests/unit/test_ui_cli.py` asserts the exact flag set. The file is
  read when the app starts from the repository root, which covers `uv run llmplan ui` in a
  checkout, Docker (`WORKDIR /app`) and Streamlit Community Cloud; a pip-installed
  `llmplan ui` started elsewhere uses Streamlit's default colours.

## Questions for founder

None. These CTO-decidable items were decided here and are open to review:

- **Renderers in `render/`** (first deviation): or a larger UI budget.
- **Share links plan on open**: or prefill only and wait for Plan (one line in `_seed`).
- **Example scenarios**: the two in `presets.SCENARIOS` can be swapped without code changes
  elsewhere; both use bundled data and fixtures only.
