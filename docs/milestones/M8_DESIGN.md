# M8 Design — Launch readiness (v0.2.0)

Status: ready for implementation once `fix-ci-help-width` is merged. Branch: `m8-launch`.
Prerequisites: docs/PLAN.md, docs/ARCHITECTURE.md, docs/DEFINITION_OF_DONE.md,
docs/ROADMAP.md (Phase 0 is this milestone), and the M1 to M7 notes.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9. Release v0.2.0 is
tagged by the CTO after review; the repository is then made public under Apache-2.0.

---

## 1. Goal

Nothing on the public page embarrasses us, every estimate says how much to trust it, the
tool works for the models people actually run, and it installs in one line.

## 2. Deliverables (in implementation order)

1. **Confidence everywhere** (section 3).
2. **Results wording** (section 4).
3. **Calibrate with your own benchmarks** (section 5).
4. **Mixture-of-experts architectures** (section 6).
5. **Live Hugging Face checks** (section 7).
6. **Packaging and release** (section 8).
7. **License and public-repo readiness** (section 8.4).
8. Tests, README, CHANGELOG (v0.2.0 section), `M8_NOTES.md`, ARCHITECTURE.md updates.

## 3. Confidence everywhere

- `PlanResult.perf_confidence: Literal["measured", "interpolated", "roofline", "mixed"]`
  derived from the chosen replicas' `PerfEstimate.confidence` (`mixed` when they differ),
  plus `perf_sources: tuple[str, ...]` (distinct `source_urls`).
- CLI `plan` text output: a line directly under `Cost`:
  `Confidence  roofline (uncalibrated first-principles model; expect ±30% on throughput)`
  or `Confidence  measured (NVIDIA NIM performance page, as of 2026-09-30)`.
- UI: a `st.warning` (roofline/mixed) or `st.info` (measured/interpolated) banner above the
  cost cards with the same sentence, and a "How to calibrate" link to section 5's expander.
- `llmplan perf estimate` text output already prints the backend; add the same sentence.
- Roofline constants (`BANDWIDTH_EFFICIENCY`, `PREFILL_MFU`, `DECODE_MFU`, `P95_FACTOR`) are
  listed in the banner's expander so users see what "uncalibrated" means.

## 4. Results wording

Replace the "Saving from request-size routing: -87.5%" headline (UI and CLI) with two
sentences built from `PlanRun.single_class_cost_usd_per_day`:

- When the class plan costs more: "Sized for the mean request this fleet would cost
  $44.66/day, but the replay shows it would miss the latency target. The class-sized plan
  costs $83.76/day." (The replay number comes from M7 section 6.3's logic: if the single-
  class fleet is available, replay it and report its TTFT violation %; otherwise omit the
  clause and say "would under-provision the long-request class".)
- When the class plan costs less: "Request-size routing saves $X/day (Y%) versus sizing
  every replica for the mean request."
- When equal: "Request-size routing does not change the fleet for this traffic."
No percentage may be negative anywhere in the UI or CLI output.

## 5. Calibrate with your own benchmarks

### 5.1 Inputs accepted
- **llmplan CSV**: one row per measurement with the `BenchmarkRow` columns (section 5.1 of
  M3_DESIGN.md) minus `source_url`/`as_of`, which are filled as `user-upload` and today.
- **vLLM benchmark JSON**: the file written by `vllm bench serve --save-result` (older
  name `benchmarks/benchmark_serving.py`). Verify the field names against the vLLM source
  at implementation time and record the version checked in the notes. Expected mapping:
  `completed`, `total_input_tokens`, `total_output_tokens` (per-request means give
  `input_len`/`output_len`), `max_concurrency` (concurrency; if null use `num_prompts`
  with a note), `output_throughput` (output tokens/s), `median_ttft_ms` -> `ttft_ms_p50`,
  `median_tpot_ms` -> `tpot_ms_p50`, `p99_*` ignored, `p95_*` used when present,
  `model_id`. GPU id, tensor parallel, dtype and engine version are not in the file; the
  user supplies them in a small form (UI) or flags (CLI).
- Size cap 5 MB; at most 500 rows per upload.

### 5.2 Behaviour
- Rows are validated exactly like shipped rows (FK, physical bound). Failures are shown
  per row with the reason; valid rows are used.
- Uploaded rows are session-only: they are merged into the table backend for that plan
  run via the existing `backends=` override mechanism (M3), never written to disk, never
  logged. The confidence banner then says `measured (your upload)`.
- CLI: `llmplan plan ... --benchmarks PATH [--benchmarks-gpu ID --benchmarks-tp N
  --benchmarks-dtype D --benchmarks-engine-version V]`, same for `perf estimate`.
- **Contribute**: a button that opens
  `https://github.com/KadirbekSharau/llmplan/issues/new?title=...&body=...` in a new tab
  with the validated rows rendered as the llmplan CSV in a code block, the model/GPU/tp
  header, and a checkbox list asking for the source (engine version, driver, date). URL
  length capped at 6,000 characters; beyond that the UI offers a CSV download and the
  issue link without rows. Nothing is sent anywhere automatically.

## 6. Mixture-of-experts architectures

New registry members in `llmplan/catalog/architectures/`:

| key | HF classes | config keys used |
|---|---|---|
| `mixtral` | `MixtralForCausalLM` | `num_local_experts`, `num_experts_per_tok`, `intermediate_size` |
| `qwen_moe` | `Qwen3MoeForCausalLM`, `Qwen2MoeForCausalLM` | `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` (Qwen2-MoE only, 0 if absent), `decoder_sparse_step` (default 1), `mlp_only_layers` (default empty) |

Formulas (per layer, dense attention as in `llama_like`, including Qwen3 qk_norm):
```
expert      = 3 * H * I_moe                      # gate, up, down
experts     = E * expert
router      = H * E
shared      = 3 * H * I_shared + H               # Qwen2-MoE shared expert + its gate; 0 otherwise
moe_layer   = attention + experts + router + shared + norms (+ qk_norm)
dense_layer = attention + 3 * H * I_dense + norms   # for layers in mlp_only_layers / not on sparse step
param_count = Σ layers + embed + lm_head + final_norm
active_per_token = attention + k * expert + router + shared + norms (+ qk_norm) per MoE layer, dense layers in full
```
`ModelSpec` gains: `num_experts: int = 0`, `experts_per_token: int = 0`,
`moe_intermediate_size: int | None`, `shared_expert_intermediate_size: int = 0`,
`moe_layer_indices: tuple[int, ...]` (derived at load). `DerivedModelInfo` gains
`active_param_count`.

Memory and perf:
- Weights use `param_count` (all experts resident). KV cache unchanged (attention is dense).
- Roofline: `flops_per_step` uses `active_param_count`. Decode bytes per step read
  `non_expert_weight_bytes + expert_weight_bytes * min(1, B * k / E)` (expected fraction of
  experts touched at batch B; documented assumption). Prefill uses active params.
- Tensor parallel split of experts is treated as even (note).

Expected values (acceptance tests, section 9):

| Fixture | key figures | param_count | active_param_count | KV bytes/token bf16 |
|---|---|---|---|---|
| `mixtral-8x7b` | H 4096, L 32, A 32, K 8, d 128, I 14336, E 8, k 2, V 32000, untied | 46,702,792,704 | 12,879,925,248 | 131,072 |
| `qwen3-30b-a3b` | H 2048, L 48, A 32, K 4, d 128, I_moe 768, E 128, k 8, V 151936, untied, qk_norm | 30,532,122,624 | 3,353,032,704 | 98,304 |

(Official figures: 46.7B / 12.9B and 30.5B / 3.3B.) DeepSeek-V3-style MLA models remain
`UnsupportedArchitecture` with a message naming MLA; do not attempt them in M8.

## 7. Live Hugging Face checks

`tests/live/test_hf_configs.py`, marked `network`, excluded by default, run manually before
release: load `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-v0.3`,
`mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen3-30B-A3B`, and (skipped without
`HF_TOKEN`) `meta-llama/Llama-3.1-8B-Instruct`; assert `param_count` within 1.5% of the
official figure recorded in the test, and that `fit()` runs. Record the run's output in
the notes.

## 8. Packaging and release

### 8.1 Data in the wheel
Move `data/gpus.yaml`, `data/prices.yaml`, `data/benchmarks/`, `data/fixtures/`,
`data/traces/manifest.yaml` and `data/traces/samples/` under `llmplan/data/` and resolve
every path through one module `llmplan/paths.py` using `importlib.resources`. Keep a
`data/` symlink or a README pointer at the old location only if a doc still references it;
update docs instead where possible. The repository-relative fallback is removed.

### 8.2 Install smoke test
CI step after `uv build`: create a clean venv, `pip install dist/*.whl`, run
`llmplan fit --model fixture:llama3-8b --gpu h100-sxm-80gb --format json` and
`llmplan gpus` from a temporary directory outside the checkout. Fail if either exits
non-zero.

### 8.3 Release workflow
`.github/workflows/release.yml` on tag `v*`: build, run the smoke test, publish to PyPI via
trusted publishing (`pypa/gh-action-pypi-publish`), create a GitHub release with
CHANGELOG notes. Document in `docs/DEPLOY.md` the one-time PyPI setup the founder must do
(FOUNDER_QUESTIONS.md item 7). The workflow must succeed up to the publish step without
PyPI configured (publish step skipped with a clear message when the environment secret or
trusted publisher is absent).

### 8.4 Public-repo readiness
- `LICENSE` (Apache-2.0, copyright 2026 Kadirbek Sharau), `NOTICE` naming the bundled
  trace samples' CC-BY-4.0 sources, `pyproject.toml` license metadata and classifiers,
  README license section and CI badge.
- `SECURITY.md` (report by email to the founder; no bounty), `CONTRIBUTING.md` (the
  definition of done in short, how to contribute benchmark rows).
- A secrets scan of the full history (`git log -p | grep` for `token`, `hf_`, `ghp_`,
  `AKIA`, `sk-`) recorded in the notes; any hit stops the milestone.

## 9. Acceptance tests (`tests/acceptance/test_m8.py` unless stated)

**9.1 Confidence.** The M4 fake-backend plan reports `perf_confidence == "measured"` when
the fake declares measured and `"roofline"` for the real-catalog plan of M4 test 10.9;
`llmplan plan` text contains the line starting `Confidence` with the matching sentence.
Two replicas with different confidences give `"mixed"`.

**9.2 Wording.** For the M7 Azure sample at the UI default SLO, the plan text and the UI
(AppTest) contain "Sized for the mean request" and do not contain "-87" or any `-` followed
by digits and `%`. For a scenario where classes save money (construct with the fake
backend), the text contains "saves $".

**9.3 Benchmark CSV import.** A 3-row llmplan CSV loads; one row with impossible throughput
is reported with its index and reason and the other two are used; the resulting estimate
has `confidence == "measured"` and `source_urls == ("user-upload",)`.

**9.4 vLLM JSON import.** A fixture JSON (hand-written in the verified format) converts to
exactly one `BenchmarkRow` with `input_len == round(total_input_tokens / completed)`,
`output_tokens_per_s == output_throughput`, `ttft_ms_p50 == median_ttft_ms`.

**9.5 Contribute link.** The generated issue URL starts with the repository's
`/issues/new?`, decodes to a body containing the CSV header line, and is under 6,000
characters for 20 rows; for 500 rows the function returns the rows-free URL and a flag.

**9.6 MoE parameter counts.** Exact values from the section 6 table for both fixtures, plus
`attention == "gqa"` and `kv_bytes_per_token_total` as listed.

**9.7 MoE fit and roofline.** `qwen3-30b-a3b` bf16 on `h100-sxm-80gb` tp 1 fits (weights
61,064,245,248 bytes); on `l4-24gb` it does not (binding `weights`). Roofline decode at
batch 1 reads `non_expert + expert * min(1, 8/128)` bytes; at batch 64 reads all expert
bytes. Assert both byte counts from the formula to 1e-9 relative.

**9.8 Unsupported MLA.** A fixture with `architectures: ["DeepseekV3ForCausalLM"]` raises
`UnsupportedArchitecture` with "MLA" in the message.

**9.9 Packaging.** `importlib.resources` resolution works from an installed wheel: the CI
smoke test in 8.2 passes; locally, `tests/acceptance/test_m8.py::test_wheel_smoke` builds
the wheel, installs it into a temp venv, and runs the two commands (marked `slow`).

**9.10 Public-repo files.** `LICENSE` contains "Apache License" and "Version 2.0";
`pyproject.toml` has `license = {text = "Apache-2.0"}` or the SPDX string form; `NOTICE`
names BurstGPT and Azure; the secrets scan script in `scripts/` returns zero hits.

**9.11 Live HF (manual).** Section 7, recorded in the notes.

## 10. Allowed dependencies
None new at runtime.

## 11. Questions for founder
- None beyond FOUNDER_QUESTIONS.md item 7 (PyPI account/trusted publishing).
