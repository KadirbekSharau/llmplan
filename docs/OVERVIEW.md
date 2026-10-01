# OVERVIEW.md — what llmplan is, in one place

Read this first. Everything else in `docs/` is detail behind one of these sections.

---

## 1. The one-sentence version

**llmplan tells an engineer who self-hosts an open-weight language model which GPUs to
rent, how many, and how to configure the serving engine, so their traffic meets a latency
target at the lowest cost — and it proves the answer by replaying the traffic.**

It runs on a CPU. It never touches a GPU, a cluster, or a live server. It is a planner, not
a runtime system.

## 2. Who it is for and the question they ask

Someone at a company running vLLM (or SGLang, TensorRT-LLM) on rented or owned GPUs, about
to deploy or resize. They ask, on the vLLM forum, r/LocalLLaMA, or in an internal Slack:

> "For Llama-3.1-70B at 20 requests per second with a 500 ms first-token target, how many
> H100s do I need, and should I use L40S instead?"

Today they answer that with a spreadsheet, a rule of thumb, or an expensive experiment.
llmplan answers it in a few seconds with the arithmetic shown.

## 3. What goes in and what comes out

**Inputs**
1. **Model**: a Hugging Face id (or a bundled fixture). We read its config.json to get the
   exact architecture numbers.
2. **Traffic**: a trace of requests (arrival time, input tokens, output tokens). Users can
   upload their own CSV, pick a bundled sample of real public traces (Microsoft Azure,
   BurstGPT), or generate synthetic traffic.
3. **Latency target**: p95 time-to-first-token and time-per-output-token, plus a
   utilization target (how much headroom to keep).
4. **Hardware and prices**: a catalog of GPUs and instance prices per cloud provider,
   editable by the user.

**Outputs**
1. **The fleet**: which instances to rent, how many, and the cost per day, with the best
   single-GPU-type fleet as a baseline.
2. **Per-replica configuration**: tensor parallelism, max concurrent sequences, context
   length, memory utilization, as a ready-to-paste `vllm serve` command line.
3. **Request-size routing**: traffic split into classes by length, with the share of each
   class that goes to each replica type.
4. **The timeline**: a replay of the traffic on that fleet showing utilization, VRAM split
   (weights / KV cache / free), queue depth, and latency-target violations per window.
5. **Candidates and assumptions**: every GPU configuration considered, ranked by cost per
   unit of capacity, with the reason any was rejected; and every modeling assumption, in
   plain words, with its confidence level.

## 4. How it works, stage by stage

Each stage is a package under `llmplan/`, built and reviewed as one milestone.

| Stage | Package | What it does | How it knows |
|---|---|---|---|
| Fit (M1) | `memory`, `catalog` | Exact bytes for weights and KV cache per token; whether a model fits on a GPU at a given tensor parallelism; how many concurrent sequences it can hold | Arithmetic from the architecture integers; the only assumptions are the engine's overhead constants |
| Workload (M2) | `workload` | Parses traces, computes peak request rate, token-length percentiles, hourly profile; generates synthetic traffic | Public trace formats verified against their READMEs |
| Performance (M3) | `perf` | Throughput and latency per replica | A first-principles roofline model (memory bandwidth and FLOPS) and a table of published benchmark rows with interpolation; every estimate carries a confidence label |
| Planner (M4, M7) | `planner` | Chooses the fleet and replica configs that minimize cost per day subject to demand and latency constraints | A mixed-integer linear program solved with OR-Tools (HiGHS); request-size classes let short and long requests go to different GPU types |
| Simulation (M5, M7) | `simulate` | Replays the trace on the chosen fleet | A discrete-event simulator with per-replica queues, incremental KV accounting, and class-aware routing |
| Interfaces (M1–M6) | `cli*.py`, `ui/`, `render/` | Command line, Streamlit web app, text/JSON/PNG renderers | Thin adapters; no logic of their own |

The flow: fit tells the planner which configurations are even possible; the performance
model prices their capacity; the planner picks the cheapest combination that meets demand;
the simulator proves it holds under the real arrival pattern.

## 5. How it is used

- **Web app** (the launch product): open the URL, choose inputs in the sidebar, click Plan.
  Run locally with `uv run llmplan ui`. Deployed as one container (one Python process that
  serves both the page and the computation).
- **CLI**: `llmplan fit`, `llmplan workload stats`, `llmplan perf estimate`, `llmplan plan`,
  `llmplan simulate`. Same library, scriptable, JSON output. Installed with `pip install
  llmplan` once published.
- **Library**: `import llmplan` for anyone embedding the planner.

## 6. What it is not

- Not a scheduler, router, proxy, or anything in the request path.
- Not a monitoring agent; nothing connects to a cluster.
- Not an inference engine; we never implement batching or kernels.
- Not a training planner; inference only.
These were deliberate choices (see CTO_ASSESSMENT.md): the runtime layers are owned by
NVIDIA, Red Hat, Google, and the Kubernetes project; the planning layer above them was empty.

## 7. Why it exists, in two sentences

GPUs for self-hosted inference are badly under-used (industry reports put average
utilization near 5%) because sizing is guesswork, and the open planning tools answer only
"will the model load", not "what is the cheapest fleet that meets my latency target under
my traffic". The founder also wanted a real, bounded, economically meaningful use of
mixed-integer programming, and this is one.

## 8. Where we are and what comes next

- **Shipped**: v0.1.0, milestones M0–M7, 593 tests at 100% coverage, verified end to end in
  a browser.
- **In progress**: M8 "launch readiness": confidence banners on every result, import of a
  user's own benchmark numbers, Mixtral and Qwen MoE models, `pip install`, release
  workflow, Apache-2.0 license, public repository.
- **Then**: deploy to a public URL, post where the sizing questions are asked, and learn
  for 30 days before deciding whether this is a product, a lead generator, or the front
  door to something larger. See ROADMAP.md.

## 9. Map of the documents

| File | Read it when you want |
|---|---|
| `docs/OVERVIEW.md` | this orientation |
| `docs/PLAN.md` | the goals, non-goals, users, principles, deliverables |
| `docs/CTO_ASSESSMENT.md` | the original market and technical reasoning, including why the vision changed twice |
| `docs/ARCHITECTURE.md` | engineering standards, package layout, data models, interfaces |
| `docs/MILESTONES.md` | what each milestone contains, status, and the deferred list |
| `docs/milestones/M<N>_DESIGN.md` | the exact spec an implementer worked from |
| `docs/milestones/M<N>_NOTES.md` | what the implementer decided, deviated on, or could not verify |
| `docs/DEFINITION_OF_DONE.md` | the rules every milestone must satisfy before merge |
| `docs/ROADMAP.md` | phases after v0.1.0 and the decision gates |
| `docs/FOUNDER_QUESTIONS.md` | decisions that needed the founder, with answers |
| `docs/DEPLOY.md`, `docs/LAUNCH.md` | hosting steps and the draft launch posts |
| `README.md` | install and usage, one example per command |
