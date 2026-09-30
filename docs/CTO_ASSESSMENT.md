# CTO Assessment: "Intelligent Inference Control Plane"

Date: 2026-09-30
Status: Pre-validation. No code, no customers, no commitments yet.

---

## 1. The one-paragraph verdict

The problem is real and the money is real. The proposed solution is aimed at the wrong layer.
The vision document describes a scheduler that NVIDIA, Red Hat, Google, and the Kubernetes
project are already shipping for free (Run:ai / KAI Scheduler, Dynamo, llm-d, Gateway API
Inference Extension, DRA, HAMi). Two founders cannot out-build that. What nobody ships is the
layer *above* it: a model-aware view of where GPU dollars are being wasted, with recommendations
a platform team can act on in an afternoon. That is the MVP. The control plane is the
year-two product, earned by first being the system of record for GPU waste.

The sequence is **Observe -> Recommend -> Act**. Every successful infra-optimization company
(Cast AI, Datadog, Kubecost, nOps) went in that order. Start read-only.

---

## 2. What the vision document gets right

- **Utilization is catastrophic.** Cast AI's 2026 State of Kubernetes Optimization report puts
  average GPU utilization at 5%. Independent measurement of a 20k-GPU fleet found it commonly
  below 60%. An idle H100 on AWS p5 on-demand is roughly $8,850 per GPU per month.
- **Kubernetes bin-packing is model-blind.** It sees "1 GPU requested" and nothing about
  weights, KV cache, batch depth, or prefix-cache hit rate.
- **The buyer wants a dollar number.** "Cut the inference bill" is a CFO-legible pitch.
- **Inference infra is hot.** Baseten $300M at $5B (Jan 2026), Fireworks $250M at $4B,
  Inferact (vLLM) $150M seed. Capital is available for credible teams in this space.

## 3. What the vision document gets wrong (and would sink us)

### 3.1 MIG is the wrong tool for LLM inference
- MIG reconfiguration requires the GPU to be idle. NVIDIA's MIG Manager will not re-partition
  a GPU with workloads running. On some clouds the node needs a reboot. "Dynamic MILP slicing"
  is therefore not dynamic. Any re-slice is a drain-and-restart event.
- Profiles are fixed (e.g. 1g.10gb, 3g.40gb on H100). There is no "give me 37 GB".
- MIG instances cannot use NVLink between each other. Tensor-parallel models span GPUs, not
  slices.
- LLM serving wants *more* VRAM per process, not less. A 70B model in FP8 is ~70 GB of weights
  plus KV cache. MIG only helps for small models (embeddings, rerankers, classifiers, small
  vision models) which are a rounding error on the bill.

Conclusion: MIG is a feature for the long tail, not the core. Drop it from the pitch.

### 3.2 A proxy cannot do iteration-level scheduling
Continuous batching happens inside the engine's step loop (vLLM, SGLang, TensorRT-LLM). A
reverse proxy in front of the engine has no control over which sequences are in the next
forward pass. Claiming iteration-level scheduling means we are writing an inference engine.
Inferact just raised $150M to do exactly that with a 500-contributor head start. We are not
doing this.

What a proxy *can* do: KV-cache-aware routing, prefix-affinity routing, queue-depth routing,
model-variant selection. The Kubernetes Gateway API Inference Extension already does all four
with a scorer plugin architecture, and llm-d adds prefill/decode disaggregation on top.

### 3.3 Latency is not the hard part
An LLM request takes 200 ms to 60 s. LiteLLM adds ~3 ms, Portkey 10-20 ms, nobody cares.
The "must be C++" constraint is solving a non-problem and would cost us months. Go or Rust
for the agent and data plane is more than enough. Algorithms start in Python with OR-Tools.

### 3.4 "Extreme technical complexity" is not a moat when the giants open-source it
Everything in the "core technical architecture" section exists as Apache-2.0 code from
NVIDIA (KAI Scheduler, Dynamo, DRA driver), the CNCF (HAMi), or the Kubernetes SIGs
(Gateway API Inference Extension). Our moat has to be data and workflow, not the scheduler:
the model-aware waste ledger, the recommendation history, the trust earned by being right.

### 3.5 The 30% -> 85% claim is the wrong shape of waste
The dominant waste is not fragmentation inside busy GPUs. It is **whole GPUs that are
allocated and idle**, and **replica counts sized for a peak that happens two hours a day**.
That waste is visible with telemetry and fixable with rightsizing and consolidation. No new
scheduler is needed to capture the first 50% of the savings. That is the MVP.

---

## 4. Competitive map (who already owns which layer)

| Layer | Who owns it | Our stance |
|---|---|---|
| Inference engine (batching, KV cache, kernels) | vLLM/Inferact, SGLang, TensorRT-LLM | Never compete. Integrate. |
| Inference-aware L7 routing | Gateway API Inference Extension, llm-d, Dynamo, agentgateway | Never rebuild. Plug into their scorer API later. |
| GPU scheduling on K8s | KAI Scheduler (NVIDIA), Volcano, DRA, HAMi | Never rebuild. Consume their state. |
| Generic GPU telemetry | DCGM exporter + Grafana, Datadog GPU monitoring | Table stakes. We correlate what they only display. |
| Generic K8s cost allocation | Kubecost/OpenCost, Cast AI | Cast AI is the real long-term threat. They do CPU rightsizing and have GPU cost monitoring but nothing model-aware. |
| Hosted API gateways | LiteLLM, Portkey, OpenRouter | Different buyer (app teams, not platform teams). |
| **Model-aware GPU waste + recommendations** | **nobody credible** | **This is the wedge.** |

The gap: DCGM tells you "GPU 3 is at 5% SM utilization". vLLM tells you "this pod has 12%
KV cache usage". Kubecost tells you "this namespace costs $40k/month". Nobody joins those
three into "these four llama-70b replicas are at p95 KV usage of 18%, they could be two
replicas, that is $17,700/month, here is the Helm diff".

---

## 5. What to build first: the MVP

### Product name for this phase: GPU Waste Ledger (working title)

**Promise to the customer:** "Install a read-only agent. In 24 hours we show you, per GPU and
per model deployment, how much VRAM and compute is idle, what it costs, and the three changes
that recover the most money. Nothing we install can break production."

### Why read-only first
- **Sales cycle.** A DaemonSet that scrapes metrics passes security review in a week. A proxy
  in the request path of production inference is a 6-9 month review at any enterprise.
- **Trust.** Recommendations we make and they apply build the evidence that lets us later
  say "let us apply them automatically."
- **Validation.** The dashboard *is* the customer interview. If they log in daily, we have
  a business. If they don't, we found out for $0 of scheduler engineering.
- **Data moat.** Every cluster we observe teaches the recommender. The scheduler product
  later ships with priors no competitor has.

### MVP scope (v0, target 6 weeks)

**Collect (agent, Go, DaemonSet + one cluster-level pod)**
- Per GPU: NVML/DCGM memory used/total, SM utilization, power, MIG geometry if any.
- Per process on each GPU: PID -> container -> pod -> deployment mapping, VRAM held.
- Per inference pod: scrape `/metrics` from vLLM, SGLang, TGI, Triton. Key series:
  `kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`,
  `gpu_cache_usage`, prefix cache hit rate, TTFT, ITL, tokens/s.
- From the K8s API: resource requests/limits, node instance type, node labels, HPA/KEDA config.
- Cost: instance-type price table (on-demand, reserved, spot) with a manual override for
  on-prem amortized $/GPU-hour.

**Store**
- Prometheus-compatible remote write into a TSDB we host (start with VictoriaMetrics or
  ClickHouse; do not build storage).

**Show (web app, TypeScript/React)**
- Fleet view: every GPU as a bar. Weights (static) / KV cache (dynamic) / free, plus
  compute utilization, over time. This is the "VRAM tracking system" you described.
- Deployment view: replicas over time vs queue depth vs KV usage. Shows over-provisioning
  visually.
- **Waste ledger:** dollars per day, broken down into: idle-allocated GPUs, over-replicated
  deployments, under-filled VRAM, wrong instance type. Trend line.
- **Recommendations:** ranked by $/month recovered, each with confidence, a risk note, and
  the concrete change (replica count, node pool, model quantization, MIG profile for small
  models, co-location candidates).

**Simulate (the "scheduling visualizer", Python first)**
- Replay 7 days of real telemetry through a bin-packing model (start with a greedy
  first-fit-decreasing on VRAM + a p95 KV headroom constraint; MILP via OR-Tools once the
  greedy version is proven wrong on a real customer).
- Output: "with this placement you needed N GPUs instead of M, with these SLO violations."
  Rendered as a timeline. This is how we prove the algorithm before it ever touches prod,
  and it is a compelling demo.

**Out of scope for v0 (explicitly)**
- Any request-path proxy. Any write access to the cluster. Any scheduler. MIG reconfiguration.
  Multi-cloud bursting. Non-NVIDIA accelerators.

### v0.0: the one-week sales tool
Before the dashboard, ship a CLI: `gpuledger scan --kubeconfig ~/.kube/config`. It runs for
one hour, scrapes DCGM + vLLM metrics, prints a Markdown report with the waste number.
Founders run it on design-partner clusters over a screen share. This is the cheapest
possible validation and it produces the first dataset.

---

## 6. Technical architecture for the MVP

```
[GPU node] ---- agent (Go, DaemonSet) ----\
   NVML / DCGM                             \
   /proc + cgroup -> pod mapping            \
   vLLM/SGLang/TGI /metrics scrape           +--> remote_write --> TSDB (VictoriaMetrics)
[K8s API] ---- collector (Go, 1 replica) ---/                          |
   pods, nodes, HPAs, prices                                           v
                                                    analyzer (Python: pandas, OR-Tools)
                                                    - waste attribution
                                                    - recommendations
                                                    - placement simulator
                                                           |
                                                           v
                                                    API (Go or FastAPI) --> web UI (React)
```

Stack decisions and why:
- **Go for agent and collector.** go-nvml and DCGM bindings exist. Single static binary,
  small footprint, fine for a DaemonSet. Not C++: nothing here is latency-critical.
- **Python for analysis.** Iteration speed on algorithms matters more than throughput.
  Port to Go only if a customer's fleet size forces it.
- **No custom storage.** Prometheus remote-write is the lingua franca; customers may even
  point us at their existing Prometheus and skip our agent.
- **Deployment: Helm chart, read-only RBAC, no CRDs.** One `helm install`.
- **Later (v1):** a Gateway API Inference Extension scorer plugin that uses our ledger to
  influence routing. That is how we enter the request path without building a proxy.
- **Later (v2):** a KAI/Volcano-compatible placement hint or a DRA-aware rightsizer that
  applies recommendations. That is the "control plane".

---

## 7. Validation plan (do this before week 3 of engineering)

Target segments, in order of likely pain:
1. **AI-native companies running 32+ GPUs of self-hosted open-weight inference** (chat
   products, coding assistants, voice, search). Fastest buyers, technical, feel the bill.
2. **Neoclouds and GPU resellers** (Nebius, CoreWeave tenants, Lambda, regional clouds).
   Utilization is literally their margin.
3. **Regulated enterprises forced to self-host** (finance, health, defense, EU sovereignty).
   Slower, larger, and the a16z 2026 survey says 80% of Global 2000 execs are now fine hosting
   with frontier labs, so this segment may be smaller than the vision assumes.

Fifteen conversations minimum. Questions that actually discriminate:
- "What was your GPU-hours-wasted number last month, in dollars? Who owns that number?"
  (If nobody owns it, there is no budget. Move on.)
- "Which inference engine and which scheduler are you on today?" (vLLM + plain K8s = ideal.
  Already on Run:ai or Dynamo = harder, they think they've solved it.)
- "Show me how you decide replica count for your biggest model." (If the answer is "we
  guessed and added 30%", we have a customer.)
- "Would you install a read-only DaemonSet from a two-person company?" (Gate for MVP shape.)
- "If we showed you $X/month recoverable, what would you pay?" (Anchor per-GPU pricing.)

Kill criteria: if fewer than 5 of 15 can name a dollar waste number or an owner for it,
the market isn't ready for a standalone product and this becomes a feature we sell to
Cast AI, Datadog, or a neocloud.

---

## 8. Business model recommendation

- **Not** percent-of-savings. Savings attribution is a permanent argument with the
  customer's finance team and it caps upside on well-run clusters.
- **Per GPU under management per month.** Order of magnitude $15-40/GPU/month, which is
  under 0.5% of an H100's cloud cost. 500 GPUs = $10-20k/month. Free tier up to 8 GPUs to
  seed the dataset and bottom-up adoption.
- Recommendation-apply automation (the control plane) becomes the premium tier.

---

## 9. Risks, honestly ranked

1. **Cast AI adds model-aware GPU rightsizing.** They have the customers, the agent, and the
   brand. Mitigation: be deeper on LLM serving semantics than a generalist can afford to be,
   and be the reference integration for vLLM/SGLang metrics.
2. **NVIDIA gives it away.** Run:ai already shows GPU fractions and utilization; Dynamo has a
   planner. Mitigation: NVIDIA optimizes for selling GPUs, not for helping you buy fewer.
   Our incentive alignment is the story.
3. **Self-hosted inference TAM shrinks.** Menlo: open-weight share fell from 19% to 11% of
   enterprise usage. Mitigation: neoclouds and AI-native companies still grow in absolute
   GPU count; the ledger also works for training and batch jobs later.
4. **We drift into building the scheduler anyway** because it is the fun problem.
   Mitigation: this document. No write-path code until 3 paying design partners.
5. **Telemetry gaps.** Per-process VRAM attribution on shared GPUs (MPS, time-slicing) is
   messy. Mitigation: start with the 1-pod-per-GPU case that is 90% of deployments.

---

## 10. 90-day plan

| Weeks | Engineering | Founder / GTM |
|---|---|---|
| 1 | `gpuledger scan` CLI: DCGM + vLLM scrape, Markdown waste report | Line up 15 interviews; 3 design partners agree to run the CLI |
| 2-3 | Go agent + collector, Helm chart, remote_write to VictoriaMetrics | Run CLI on partner clusters; collect first real datasets |
| 4-6 | Web UI: fleet VRAM view, deployment view, waste ledger v1 | Present findings to partners; get "we applied it" evidence |
| 7-9 | Recommendation engine v1 (replica rightsizing, idle-GPU reclaim, consolidation); placement simulator with timeline replay | Convert 1-2 partners to paid pilot; write the "$X recovered" case study |
| 10-12 | Hardening, SSO, multi-cluster; decide on Gateway API Inference Extension scorer plugin as v1 write-path | Pre-seed narrative: data moat + design-partner savings; decide go/no-go on control plane |

---

## 11. Decisions I need from you (founder)

1. Agree to the reorder: waste ledger first, control plane second. If you want to build the
   scheduler first regardless, say so and we plan for an 18-month runway with no revenue.
2. Which segment do we interview first? My vote: AI-native companies on vLLM + vanilla K8s.
3. Do you have access to any cluster with real inference traffic for the first dataset? If
   not, week 1 also includes renting 2-4 GPUs and generating synthetic load with vLLM.
4. Name. "ultrascheduler" promises the thing we are deliberately not building first.

---

## Sources

- Cast AI 2026 State of Kubernetes Optimization (5% GPU utilization):
  https://cast.ai/press-release/2026-state-of-kubernetes-optimization-report/
- Cast AI GPU cost monitoring ($8,850/H100/month idle):
  https://cast.ai/blog/gpu-cost-monitoring-kubernetes/
- NVIDIA Run:ai GPU fractions: https://run-ai-docs.nvidia.com/self-hosted/platform-management/runai-scheduler/resource-optimization/fractions
- NVIDIA GPU fractioning blog: https://developer.nvidia.com/blog/unlock-massive-token-throughput-with-gpu-fractioning-in-nvidia-runai/
- NVIDIA Dynamo architecture (KV-aware routing, disaggregation):
  https://docs.nvidia.com/dynamo/latest/architecture/architecture.html
- Kubernetes Gateway API Inference Extension: https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
- agentgateway + llm-d + GAIE scorer architecture: https://agentgateway.dev/blog/2026-03-19-agentgateway-llm-d-gaie-inference-serving/
- NVIDIA GPU Operator MIG (idle requirement, fixed profiles):
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.6.0/gpu-operator-mig.html
- HAMi GPU sharing: https://cozystack.io/docs/v1.6/kubernetes/gpu-sharing/
- KubeCon EU 2026 AI infra (DRA, HAMi, vLLM): https://jimmysong.io/blog/kubecon-eu-2026-day1-ai-infra/
- Inference startup funding 2026: https://lucaberton.com/blog/inference-gold-rush-ai-startups-kubernetes-2026/
- vLLM metrics design: https://docs.vllm.ai/en/v0.12.0/design/metrics/
- llm-d observability metrics: https://llm-d.ai/docs/operations/observability/metrics
- LiteLLM vs Portkey latency overhead: https://www.pkgpulse.com/guides/portkey-vs-litellm-vs-openrouter-llm-gateway-2026
- Hosted API vs self-hosted cost, Menlo/a16z survey data: https://intuitionlabs.ai/articles/hosted-ai-api-vs-self-hosted-llm-cost

---
---

# Revision 2 (2026-09-30, same day): constraints changed, so the MVP changes

New facts from the founder:
- No GPUs, and no budget to rent any.
- No network of platform teams to interview.
- This is a side project. The original motivation was "a good use case for linear
  programming / linear solvers", not "build an inference control plane company".

## 1. Was the reorder (waste ledger first) a good idea? Honest answer: not for you

It was the right call for a funded startup with hardware and design partners. It is the
wrong call for a side project with neither. The waste ledger is mostly telemetry plumbing
(NVML, DCGM, cgroup-to-pod mapping, Helm, RBAC), it cannot be developed without a GPU to
scrape, and it only pays off after enterprise sales. None of that touches a solver. Both the
original vision and my first memo silently assumed a venture-scale company. Neither fits.

What fits: the **placement simulator / capacity planner**, which was step 3 in my plan. It is
the only piece that is pure software, CPU-only, trace-driven, and genuinely a mixed-integer
program. It moves to step 1. Everything else is deferred, possibly forever.

## 2. Is linear programming actually the right tool here? Yes, for planning. No, for routing.

- **Offline planning** (which GPUs to buy/rent, how many, how to configure each replica) is a
  cost-aware bin-packing problem. Solve time of seconds to minutes is fine. MILP fits.
- **Online routing** (which replica gets this request, decided in under a millisecond) is not
  an LP problem in practice. Heuristic scorers win. The vision document put the MILP in the
  routing path; that was the wrong place.

Prior art confirms the planning formulation works and needs no hardware:
- **Mélange** (UC Berkeley, Stoica group, 2024). Formulates GPU allocation as cost-aware bin
  packing, solves it as an ILP with an off-the-shelf solver, reports up to 77% cost reduction
  in conversational workloads versus a single GPU type. Code public.
- **Vidur** (Microsoft Research, 2024). CPU-only LLM inference simulator, under 9% latency
  error, includes Vidur-Search which finds the cheapest deployment config for Llama2-70B in
  one hour on a CPU versus an estimated 42K GPU-hours of real experiments. Code public.

Both are academic artifacts, not products. Neither has a usable UI, a price catalog, a
trace uploader, or an explanation of *why* the answer is what it is. That is the gap.

## 3. The revised MVP: an LLM capacity planner

**One-sentence promise:** "Give me your model, your traffic, your latency target, and a
price list. I return the cheapest GPU fleet and per-replica configuration that meets the
target, and I show you the utilization timeline that proves it."

**Why this is the right side project**
- Zero GPUs. The performance model comes from published benchmarks and Vidur profiles.
- The core is a MILP. This is the LP use case you were looking for, in a market that pays.
- The customer question already exists in the wild. vLLM's own forum is full of "why does
  raising max-num-seqs OOM", "how do I size this", "which GPU for this model". Existing
  calculators (TensorPlan, Morph, llm-mem-planner) answer "will it load", not "what is the
  cheapest fleet that meets my SLO under my traffic". One of those sources says this outright.
- It is the front door to the bigger idea. Users who trust the planner will later paste in
  their Prometheus URL, which is the waste ledger, which is the control plane. Same company,
  earned in the right order.

**Inputs**
- Model: name, parameter count, layers, hidden size, KV heads, dtype/quantization. Pull from
  Hugging Face config.json, do not hand-enter.
- Traffic: a trace of (arrival time, input tokens, output tokens). Ship with public traces
  (Azure LLM Inference 2023 and 2024, BurstGPT: 5.29M requests over 121 days) and accept CSV.
- SLO: p95 time-to-first-token, p95 time-per-output-token, or max queue wait.
- Price catalog: $/GPU-hour per instance type, on-demand and reserved, editable. Seed from
  AWS, GCP, Azure, Lambda, RunPod public pricing.

**Model (Python, OR-Tools)**
- VRAM per replica = weights + KV cache (2 * layers * kv_heads * head_dim * dtype_bytes per
  token, times tokens in flight) + engine overhead. This is exact, not estimated.
- Throughput per replica per GPU type = lookup from a benchmark table (NVIDIA inference
  benchmarking blog, vLLM published benchmarks, InferenceMAX, Vidur profiles), interpolated
  on batch size and context length. This is the only empirical part and the biggest risk.
- Decision variables: count of each GPU type, replicas per GPU (or GPUs per replica for
  tensor parallel), max concurrent sequences per replica, optionally time-of-day scaling.
- Objective: minimize $/day. Constraints: peak-window throughput >= demand, VRAM fits,
  SLO estimate holds. Solve with CP-SAT or HiGHS via OR-Tools.

**Output**
- The fleet: "3x H100 80GB + 2x L40S, $X/day, versus $Y/day for the naive all-H100 answer."
- Per replica config: tensor parallel degree, max_num_seqs, gpu_memory_utilization,
  max_model_len, as a ready-to-paste vLLM command line.
- Timeline chart: replayed trace, per-GPU utilization, VRAM split (weights/KV/free),
  queue depth, SLO violations. This is the "GPU scheduling visualizer".
- Explanation: which constraint was binding (VRAM, throughput, or SLO) and by how much.

**Explicitly out of scope**
- Anything that runs on the customer's cluster. Any agent. Any proxy. Any scheduler.

## 4. Replacing interviews with desk research (and a launch channel)

You do not need a network. The people with this problem are writing it down in public.
Read fifty threads, tag the question type, and build the tool around the top three.

Where they are, in priority order:
1. **discuss.vllm.ai** and vLLM GitHub issues. Search: "how many GPUs", "OOM",
   "max-num-seqs", "gpu-memory-utilization", "which GPU", "throughput". These posters are the
   customer and the launch audience.
2. **r/LocalLLaMA** ("how much VRAM", "which GPU for 70B", "serving X users").
3. **Hacker News** Ask HN and comment threads on self-hosting cost.
4. **vLLM Slack, llm-d Slack, CNCF Slack (#hami, sig-scheduling), MLOps Community Slack.**
   Free to join. Lurk, then post the tool.
5. **KubeCon EU 2026 AI-infra talk speakers** and authors of "self-hosted vs API cost" blog
   posts. They already care and reply to cold email about their own topic.

Launch is the interview: post the planner as a free tool ("Show HN", r/LocalLLaMA, vLLM
forum). Log every input people type. That input log is the interview transcript, at scale.

## 5. Honest expectations for a side project

- This can become a respected free tool with a paid tier (saved plans, private price
  catalogs, team sharing) or a lead generator for consulting. Think low thousands per month
  in the first year, not the "millions" in the vision doc.
- The venture-scale version (ledger, control plane) needs a team and hardware. The planner
  is how you earn the right, the audience, and the data to raise for it later, if you want
  to.
- If the LP itself is the goal, this is one of the best-fit real problems for it: bounded,
  economically meaningful, with academic validation and no working product.

## 6. Revised stack and 4-week plan

Stack: Python only. OR-Tools (CP-SAT/HiGHS), pandas, Hugging Face `transformers` config
loader, Streamlit for the first UI (swap to React only if the tool gets traction), Vidur as
an optional high-fidelity backend. No Go, no C++, no Kubernetes.

| Week | Build | Research |
|---|---|---|
| 1 | Exact VRAM model from HF config.json, unit-tested against known numbers (Llama-3-70B FP16 ~140 GB weights, KV per token). Benchmark table scraped from NVIDIA/vLLM/InferenceMAX. | Read and tag 50 forum threads. Write down the top 3 question shapes. |
| 2 | MILP v1: choose GPU types, counts, TP degree, max_num_seqs to minimize $/day under a trace. Replay against Azure 2024 trace. Compare to the naive all-H100 answer. | Reproduce one Mélange result to validate the formulation. |
| 3 | Streamlit UI: inputs, fleet answer, timeline chart, vLLM command line. Public traces as presets. | Draft the launch post. Pick 5 threads to reply to with real answers from the tool. |
| 4 | Ship. Log inputs. Fix what breaks. | Post: Show HN, r/LocalLLaMA, discuss.vllm.ai. Read every comment. |

## 7. Decisions now

1. Confirm: planner first, nothing that touches a cluster. (Recommended.)
2. Solver preference: OR-Tools CP-SAT is my default; if you specifically want to practice
   classical LP/MILP, use HiGHS or SCIP through OR-Tools `pywraplp` instead.
3. Rename the repo to match the product (e.g. `llm-capacity-planner`). "ultrascheduler"
   describes the thing we are deferring.

## Sources for Revision 2
- Mélange paper: https://arxiv.org/abs/2404.14527v4 ; code: https://github.com/tyler-griggs/melange-release
- Vidur paper/code: https://www.microsoft.com/en-us/research/publication/vidur-a-large-scale-simulation-framework-for-llm-inference/ ; https://github.com/microsoft/vidur
- Azure LLM inference traces (2023, 2024): https://github.com/Azure/AzurePublicDataset
- BurstGPT trace: https://github.com/hpmll/burstgpt
- Existing calculators answer "will it load" not "what is cheapest under load": https://www.premai.io/blog/llm-infrastructure-sizing-from-hardware-requirements-to-production-capacity/ ; https://github.com/terrywangcode/tensorplan ; https://www.morphllm.com/dedicated-inference/calculator
- NVIDIA inference cost benchmarking: https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost
- vLLM sizing pain, in the users' own words: https://discuss.vllm.ai/t/why-does-increasing-max-num-seqs-cause-a-runtime-cuda-oom/2820 ; https://discuss.vllm.ai/t/deploy-a-big-llm-when-gpu-vram-not-enough/1354/14 ; https://github.com/vllm-project/vllm/issues/15617
- Anyscale GPU guidance for serving: https://docs.anyscale.com/llm/serving/gpu-guidance
