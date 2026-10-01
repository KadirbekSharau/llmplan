# LAUNCH.md — launch checklist and draft posts

Drafts for the founder to review and post (M6_DESIGN.md section 10). Nothing here has been
posted. `https://llmplan.dev` is the deployed app and `https://github.com/KadirbekSharau/llmplan` the repository link; both are open
decisions (docs/FOUNDER_QUESTIONS.md, questions 5 and 6, working name "llmplan").

## Checklist

- [ ] Hosting chosen and deployed (docs/DEPLOY.md); `curl -fsS https://llmplan.dev/_stcore/health`
      returns `ok`.
- [ ] A plan on each preset completes under 60 s on the deployed app. Locally this is
      `uv run pytest -m slow -k every_preset --no-cov -s` (each preset took under 1 s of
      plan and replay on a laptop CPU when this was written; the deployed host adds page
      load time).
- [ ] Usage log enabled: `LLMPLAN_USAGE_LOG` points at a persistent, writable path, and
      `uv run python scripts/usage_summary.py <path>` prints counts by model id and GPU set.
      Run it weekly; the launch target is 200 distinct plan runs in 30 days (PLAN.md
      section 8).
- [ ] Public name checked again for collisions (question 6) and the drafts below updated.
- [ ] Screenshot taken from the deployed app: Azure 2024 conversation preset, default
      inputs, scrolled so the answer card, the fleet table, one `vllm serve` line and the
      top of the timeline are visible (1600 px wide, light theme). Attach it to each post.
- [ ] Price catalog date checked: the app shows the `as_of` of the shipped prices; refresh
      `llmplan/data/prices.yaml` first if it is more than a month old.
- [ ] Someone is available to answer comments for the first 24 hours.

## Show HN

**Title:** Show HN: llmplan – cheapest GPU fleet for your LLM traffic and latency target

**Body:**

I kept seeing the same sizing question on the vLLM forum: "I have this model and roughly
this much traffic, how many of which GPU do I need, and what do I set max_num_seqs to?"
Online calculators answer whether the weights fit. llmplan answers the rest: it computes
exact VRAM from the model's config.json, estimates throughput and latency per GPU, tensor
parallel degree, dtype and batch size, and solves a small mixed-integer program for the
cheapest fleet across GPU types and providers that meets your p95 TTFT/TPOT target at
your peak traffic. Then it replays your traffic on that fleet so you can see utilization,
KV-cache use and queueing over time, and it prints a `vllm serve` line per replica.

It runs entirely on CPU and never touches a GPU or your cluster. Bring a trace (Azure,
BurstGPT or a simple CSV), use one of the bundled public trace samples, or generate
synthetic traffic; prices are editable in the page. Throughput numbers come from published
benchmark rows where they exist and a labelled roofline estimate where they don't, and
every result lists its assumptions. Web app: https://llmplan.dev. Code: https://github.com/KadirbekSharau/llmplan.

**What I'd like feedback on:** where the recommended fleet or settings disagree with what
you actually run in production, and which GPUs, providers or models are missing.

## r/LocalLLaMA

**Title:** I built a free planner that picks the cheapest GPU fleet + vLLM settings for
your model, traffic and latency target (CPU-only, no signup)

**Body:**

If you self-host with vLLM and have ever guessed how many GPUs you need, this is for you.
Pick a model (Hugging Face id or one of the built-ins), pick traffic (upload a CSV trace,
use a bundled Azure or BurstGPT sample, or make synthetic traffic), set a p95 TTFT/TPOT
target, and adjust GPU prices. llmplan returns the cheapest mix of instances across
providers, the per-replica tensor parallel, dtype and max_num_seqs with a copyable
`vllm serve` command, and a replay of your traffic on that fleet showing utilization, VRAM
(weights / KV in use / free) and queueing.

The VRAM side is exact arithmetic from config.json; throughput is the empirical part and
is labelled as measured, interpolated or roofline in every result. It's a planner, not a
benchmark: it never runs a model. Free, no login: https://llmplan.dev (code: https://github.com/KadirbekSharau/llmplan).

**What I'd like feedback on:** numbers that look off for hardware you own, and
architectures you need that it can't parse yet (MoE and MLA are not supported).

## discuss.vllm.ai

**Title:** A capacity planner for vLLM deployments: fleet, tensor parallel and max_num_seqs
from your traffic and SLO

**Body:**

Sizing questions come up here often, so I built a tool for them. llmplan takes a model
config, a request trace (or a bundled public sample, or synthetic traffic), a p95 TTFT and
TPOT target and a price table, and finds the cheapest fleet of instances and per-replica
vLLM settings (tensor parallel, dtype and max_num_seqs, with max_model_len and
gpu_memory_utilization in the command) that meet the peak demand. KV-cache capacity follows vLLM's memory
accounting with a documented overhead model; each candidate's rejection reason (does not
fit, TTFT or TPOT over target) is shown, along with the binding constraint of the chosen
fleet. A discrete-event replay of the trace on that fleet shows per-replica utilization,
KV in use and queue depth per window.

It is CPU-only and offline apart from fetching config.json from Hugging Face. Web app:
https://llmplan.dev; code and a CLI (`llmplan plan`, `llmplan simulate`): https://github.com/KadirbekSharau/llmplan. Throughput comes from
published benchmark rows where available and a labelled roofline estimate elsewhere, so
I'd value corrections from people with measured numbers.

**What I'd like feedback on:** does the recommended max_num_seqs / KV-cache sizing match
what vLLM's profiler gives you on the same GPU, and which benchmark sources should the
table backend read next?
