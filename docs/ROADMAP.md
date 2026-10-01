# ROADMAP.md — after v0.1.0

Written by the CTO on 2026-10-01, the day after v0.1.0. Read with PLAN.md (why) and
MILESTONES.md (what shipped). This file says what happens next and why, in order.

## Where we are

A working, tested planner with a UI, zero users, and an unvalidated market. The technical
risk is now small; the product risk is all of it. The next phase is therefore about
getting real traffic shapes and real benchmark numbers from real people, not features.

The one technical weakness that will decide whether people trust the tool: the
performance model. Only two models on two GPUs have measured rows; everything else is a
roofline estimate with uncalibrated constants. A user who runs Qwen or a MoE model today
gets numbers we cannot stand behind.

## Phase 0 — Launch readiness (M8, one milestone, ~1 week of agent time)

Goal: nothing on the public page embarrasses us, and every estimate says how much to trust it.

1. **CI green.** Fix the three help-text tests that fail on GitHub's narrow terminal
   (in progress on `fix-ci-help-width`). No launch with a red badge.
2. **Honest confidence, everywhere.** A banner on every result: "performance model:
   roofline (uncalibrated, expect ±30%)" or "measured (NVIDIA NIM 1.8.0, as of ...)".
   The negative "saving" headline is replaced with the two-cost sentence from
   MILESTONES.md.
3. **Calibrate with your own numbers.** CSV import of a user's vLLM benchmark results
   (the `vllm bench serve` output format) into the table backend for that session, and a
   one-click "contribute these rows" that opens a prefilled GitHub issue. This is the
   data flywheel from CTO_ASSESSMENT.md, in its cheapest possible form.
4. **MoE support.** Qwen3-MoE, Mixtral, DeepSeek-V3-style configs: total parameters for
   weights, active parameters for compute. Explicit `UnsupportedArchitecture` stays for
   anything else. Most popular self-hosted models in late 2026 are MoE; without this the
   tool looks dated on day one.
5. **Real Hugging Face ids.** One network-marked test per supported family against the
   live config.json, run manually before release, so "paste an HF id" is known to work.
6. **`pip install llmplan`.** Package `data/` into the wheel, publish to PyPI, tag v0.2.0.
   Forum users script things; a CLI they can install in one line is distribution.
7. **Repo public, Apache-2.0.** Recommendation, founder decides: open source is the
   only distribution channel we can afford, the moat was always data and workflow, and
   Streamlit Community Cloud hosting requires it anyway.

## Phase 1 — Launch and learn (weeks 2–3)

Goal: 200 plan runs and 20 comments describing real deployments (PLAN.md section 8).

- Deploy on Streamlit Community Cloud (free) with `LLMPLAN_USAGE_LOG` on.
- Post in this order, one per day: discuss.vllm.ai, r/LocalLLaMA, Show HN. Texts are
  drafted in docs/LAUNCH.md. Reply to five live sizing threads with an actual plan link.
- Add a one-question feedback box under results: "Is this close to what you run?
  yes / no / not deployed yet" plus an optional text field. Logged with the run id.
- Weekly: `scripts/usage_summary.py`, read every comment, file issues, nothing else.

Decision gate at day 30:
- Both targets hit and users ask for more models or their own prices saved: continue to
  Phase 2.
- Runs but no deployment comments: the tool is a calculator people glance at; keep it
  free, stop investing, use it as a portfolio piece and a consulting lead source.
- Neither: shut the hosted instance, keep the repo, write up the LP lessons.

## Phase 2 — The calibration loop (month 2)

Goal: estimates people trust because they come from people like them.

- Accept contributed benchmark rows via PR with a validation workflow (the physical-bound
  check already exists). Credit contributors in the UI.
- Per-engine rows: vLLM, SGLang, TensorRT-LLM side by side, user picks the engine.
- Hourly autoscaling in the planner (`hourly_rps` exists since M2): the first feature that
  changes the answer for most cloud users.
- "Connect your Prometheus" read-only import of vLLM metrics as a trace source. This is the
  observe step from the original control-plane vision, built only if users ask.

## Phase 3 — Decide what this is (month 3)

Three outcomes, chosen by the Phase 1 and 2 signals:
- **Product:** paid tier (saved plans, private price catalogs, team sharing, API). Low
  thousands per month is the realistic first-year ceiling.
- **Lead generator:** free tool, consulting on inference sizing and cost. Faster money,
  no leverage.
- **Front door to the ledger:** users with fleets ask to track actual versus planned;
  that is the waste-ledger product from CTO_ASSESSMENT.md, and it needs a team and
  hardware. Raise or partner at that point, not before.

## Engineering rules that stay

- Same workflow: design doc, delegated implementer, CTO review, PR, merge. The
  definition of done is unchanged.
- Test suite stays under 60 s; split slow acceptance runs into a nightly job before
  adding more.
- Every estimate carries its confidence and source. No number without a URL.

## Founder decisions needed for Phase 0

1. Repo public under Apache-2.0 (recommended) or stay private and pay for hosting.
2. Public name: keep `llmplan` or choose another before PyPI publication (names there
   are permanent).
