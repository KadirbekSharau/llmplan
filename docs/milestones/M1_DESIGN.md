# M1 Design — Catalog and Exact VRAM Fit

Status: ready for implementation. Assignee: developer agent.
Prerequisites: read docs/PLAN.md and docs/ARCHITECTURE.md first. This document is
self-contained beyond those two.

Definition of done: every test in section 9 passes unchanged, CI is green, and the
MILESTONES.md review checklist is satisfied.

---

## 1. Goal

Answer, exactly and offline:

> "Does model M in dtype D fit on GPU G with tensor parallel T under vLLM's memory model,
> how many tokens of KV cache remain, and therefore how many concurrent sequences of
> context length L can one replica hold?"

Everything in this milestone is arithmetic from published architecture integers and
hardware specs. The only assumptions are the engine overhead constants in section 5, and
they are labeled as such in results.

## 2. Deliverables

1. Repo scaffold (M0) as described in MILESTONES.md.
2. `llmplan.catalog.models`: `ModelSpec`, `load_model()`, `ConfigFetcher` protocol with
   `HttpConfigFetcher` and `FixtureFetcher`.
3. `llmplan.catalog.architectures`: registry with the `llama_like` member.
4. `llmplan.catalog.hardware`: `GPUSpec`, `PriceRow`, `load_gpus()`, `load_prices()`, and
   the seed data files `data/gpus.yaml`, `data/prices.yaml`.
5. `llmplan.memory`: `dtypes`, `weights`, `kv_cache`, `engine`, `fit`.
6. `llmplan.render`: `text` (rich or plain table) and `json` renderers for `FitResult`.
7. `llmplan.cli`: `llmplan fit`, `llmplan model-info`, `llmplan gpus`.
8. Fixtures in `data/fixtures/model_configs/`, unit tests, and `tests/acceptance/test_m1.py`.
9. README usage section for the three commands.

## 3. Model catalog

### 3.1 Loading from Hugging Face config.json

`ConfigFetcher` protocol:

```python
class ConfigFetcher(Protocol):
    def fetch(self, repo_id: str, revision: str = "main") -> dict[str, Any]: ...
```

`HttpConfigFetcher`:
- Validates `repo_id` matches `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$` and `revision` matches
  `^[A-Za-z0-9._-]+$`; otherwise raise `FetchError` naming the bad value.
- GET `https://huggingface.co/{repo_id}/resolve/{revision}/config.json` with `httpx`,
  timeout 10 s, `follow_redirects=True` but only to `huggingface.co` hosts, response capped
  at 1 MiB, `Authorization: Bearer $HF_TOKEN` only if the env var is set. Never log the token.
- Non-200 or non-JSON raises `FetchError` with status code and repo id. 401/403 message says
  the repo may be gated and to set `HF_TOKEN`.

`FixtureFetcher(dir: Path)`: `repo_id` `"fixture:<name>"` reads
`data/fixtures/model_configs/<name>.json`. Tests use only this fetcher.

`load_model(id, *, fetcher=None)`:
- `id` starting with `fixture:` uses `FixtureFetcher` regardless of `fetcher`.
- Otherwise uses the given fetcher or `HttpConfigFetcher()`.
- Maps the raw dict to `ModelSpec` via section 3.2, then validates.

### 3.2 Key mapping (HF config.json -> ModelSpec)

| ModelSpec field | HF key | Default if absent |
|---|---|---|
| `architecture` | `architectures[0]` mapped by table 3.3 | `UnsupportedArchitecture` |
| `hidden_size` | `hidden_size` | required |
| `num_layers` | `num_hidden_layers` | required |
| `num_attention_heads` | `num_attention_heads` | required |
| `num_kv_heads` | `num_key_value_heads` | `num_attention_heads` |
| `head_dim` | `head_dim` | `hidden_size // num_attention_heads` |
| `intermediate_size` | `intermediate_size` | required |
| `vocab_size` | `vocab_size` | required |
| `tie_word_embeddings` | `tie_word_embeddings` | `False` |
| `attention_bias` | `attention_bias` | per architecture (3.3) |
| `mlp_bias` | `mlp_bias` | `False` |
| `max_position_embeddings` | `max_position_embeddings` | required |
| `sliding_window` | `sliding_window` | `None` |
| `param_count_override` | never from HF | `None` |
| `source` | `"huggingface"` or `"fixture"` | |

Any required key missing raises `UnsupportedArchitecture(field=<key>)`. Fixture files
contain exactly the HF keys above (they are hand-written, not copied upstream files).

### 3.3 Architecture registry: `llama_like`

Registered for `architectures[0]` in:
`LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`.
Anything else raises `UnsupportedArchitecture(field="architectures")` with the value seen.

Per-architecture defaults:
- `Qwen2ForCausalLM`: `attention_bias=True` (q, k, v projections have bias; o_proj does not).
- `Qwen3ForCausalLM`: `attention_bias=False`; has `q_norm` and `k_norm` of size `head_dim`
  per layer (two extra vectors of `head_dim` each).
- `LlamaForCausalLM`, `MistralForCausalLM`: `attention_bias` from config, default `False`.

Interface every architecture module implements:

```python
def count_params(spec: ModelSpec) -> int
def embedding_params(spec: ModelSpec) -> int      # embed + lm_head (0 for lm_head if tied)
def kv_heads_per_gpu(spec: ModelSpec, tensor_parallel: int) -> int
```

## 4. Exact formulas

Let `H = hidden_size`, `L = num_layers`, `A = num_attention_heads`, `K = num_kv_heads`,
`d = head_dim`, `I = intermediate_size`, `V = vocab_size`.

### 4.1 Parameter count (llama_like)

```
q_proj   = H * (A * d)   + (A*d if attention_bias else 0)
k_proj   = H * (K * d)   + (K*d if attention_bias else 0)
v_proj   = H * (K * d)   + (K*d if attention_bias else 0)
o_proj   = (A * d) * H                       # never has bias in these families
gate_proj = H * I + (I if mlp_bias else 0)
up_proj   = H * I + (I if mlp_bias else 0)
down_proj = I * H + (H if mlp_bias else 0)
norms     = 2 * H                            # input_layernorm + post_attention_layernorm (RMSNorm weight only)
qk_norm   = 2 * d  if Qwen3 else 0

per_layer = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + norms + qk_norm

embed     = V * H
lm_head   = 0 if tie_word_embeddings else V * H
final_norm = H

param_count = L * per_layer + embed + lm_head + final_norm
```

If `param_count_override` is set, `count_params` returns it and `FitResult.confidence`
becomes `"estimated"`.

Expected values (these are acceptance tests, section 9):

| Fixture | H | L | A | K | d | I | V | tie | bias | param_count |
|---|---|---|---|---|---|---|---|---|---|---|
| `llama3-70b` | 8192 | 80 | 64 | 8 | 128 | 28672 | 128256 | F | F | 70,553,706,496 |
| `llama3-8b` | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | F | F | 8,030,261,248 |
| `mistral-7b-v0.1` | 4096 | 32 | 32 | 8 | 128 | 14336 | 32000 | F | F | 7,241,732,096 |
| `qwen2.5-7b` | 3584 | 28 | 28 | 4 | 128 | 18944 | 152064 | F | T (qwen2) | 7,615,616,512 |

Also set `max_position_embeddings`: 8192 (llama3-70b, llama3-8b), 32768 (mistral-7b-v0.1,
qwen2.5-7b). `sliding_window`: 4096 for mistral-7b-v0.1, `None` for the others.

### 4.2 Weight bytes

`bytes_per_element`: `fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, int8: 1.0, int4: 0.5`.

```
if dtype in {int8, int4} and quantize_embeddings is False (default):
    weight_bytes = ceil((param_count - embedding_params) * bpe(dtype)) + embedding_params * 2
else:
    weight_bytes = ceil(param_count * bpe(dtype))
```

`quantize_embeddings` is a field on `FitRequest` (default `False`). Quantization scale and
zero-point overhead is ignored in M1; add a note `"int4/int8: scales/zeros not counted
(typically +2-5%)"` to `FitResult.notes`.

### 4.3 KV cache bytes per token

```
kv_heads_per_gpu     = ceil(K / tensor_parallel)          # vLLM replicates KV heads when TP > K
kv_bytes_per_token_per_gpu = 2 * L * kv_heads_per_gpu * d * bpe(kv_dtype)   # 2 = key and value
kv_bytes_per_token_total   = 2 * L * K * d * bpe(kv_dtype)                  # informational
```

Expected `kv_bytes_per_token_total` at `bf16`:
llama3-70b 327,680; llama3-8b 131,072; mistral-7b-v0.1 131,072; qwen2.5-7b 57,344.

Sliding window: M1 treats the KV cache as full-context (conservative; note added to
`FitResult.notes` when `sliding_window` is set).

### 4.4 Tensor-parallel weight split

`per_gpu_weight_bytes = ceil(weight_bytes / tensor_parallel)`. Uneven splits and replicated
embeddings are ignored in M1 (note added when `tensor_parallel > 1`).

Validation: `tensor_parallel` must be >= 1 and must divide `num_attention_heads`; otherwise
`ValidationError` naming both numbers.

## 5. Engine overhead model (vLLM)

`EngineProfile` defaults (from ARCHITECTURE.md): `gpu_memory_utilization=0.9`,
`max_num_batched_tokens=8192`, `fixed_overhead_bytes=1 GiB`, `activation_multiplier=16.0`,
`kv_dtype="bf16"`.

```
per_gpu_usable_bytes     = floor(gpu.vram_bytes * gpu_memory_utilization)
activation_bytes         = max_num_batched_tokens * hidden_size * 2 * activation_multiplier
per_gpu_overhead_bytes   = fixed_overhead_bytes + activation_bytes
per_gpu_kv_budget_bytes  = per_gpu_usable_bytes - per_gpu_weight_bytes - per_gpu_overhead_bytes
```

These two constants are the only non-exact inputs in M1. Document them in the module
docstring as assumptions to be calibrated in M3, and always append
`"overhead model: 1 GiB fixed + activations(max_num_batched_tokens x hidden x 2 B x 16)"`
to `FitResult.notes`. `FitResult.confidence` stays `"exact"` for the weight and KV
arithmetic; the overhead assumption is communicated through `notes`, not `confidence`.

## 6. Fit computation

```
kv_token_capacity             = max(0, per_gpu_kv_budget_bytes // kv_bytes_per_token_per_gpu)
max_concurrent_seqs_at_context = kv_token_capacity // context_len
fits                          = max_concurrent_seqs_at_context >= 1
binding = "weights"  if per_gpu_weight_bytes > per_gpu_usable_bytes
        = "overhead" if not fits (weights fit but no room for one sequence)
        = "ok"       otherwise
```

Additional validation: `context_len <= max_position_embeddings`, else `ValidationError`.
Note when `context_len > sliding_window` (informational).

## 7. Hardware catalog

### 7.1 `data/gpus.yaml`

Schema per ARCHITECTURE.md `GPUSpec`. `vram_bytes` is the `nvidia-smi` reported total
(MiB x 1,048,576), not the marketing number. Seed rows to include, with the values the
implementing agent must verify against the linked NVIDIA datasheet and record `as_of`:

| id | name | vram (MiB) | bandwidth GB/s | fp16 dense TFLOPS | fp8 dense TFLOPS | nvlink |
|---|---|---|---|---|---|---|
| `h100-sxm-80gb` | NVIDIA H100 SXM 80GB | 81559 | 3350 | 989 | 1979 | true |
| `h200-sxm-141gb` | NVIDIA H200 SXM 141GB | 143771 | 4800 | 989 | 1979 | true |
| `a100-sxm-80gb` | NVIDIA A100 SXM 80GB | 81920 | 2039 | 312 | null | true |
| `a100-sxm-40gb` | NVIDIA A100 SXM 40GB | 40960 | 1555 | 312 | null | true |
| `l40s-48gb` | NVIDIA L40S 48GB | 46068 | 864 | 362 | 733 | false |
| `l4-24gb` | NVIDIA L4 24GB | 23034 | 300 | 121 | 242 | false |
| `a10g-24gb` | NVIDIA A10G 24GB | 23028 | 600 | 125 | null | false |

`source_url` for each points at the NVIDIA product datasheet page. If a value cannot be
verified, set it to `null` and leave a `# TODO(M3): verify` comment; do not guess.

The acceptance tests pin `h100-sxm-80gb.vram_bytes == 81559 * 2**20` and
`a10g-24gb.vram_bytes == 23028 * 2**20` so the numbers in section 9 remain stable.

### 7.2 `data/prices.yaml`

Schema per `PriceRow`. Include at least these rows, values filled from the provider's public
pricing page on the day of implementation with `as_of` set to that day:

- aws `p5.48xlarge` (8x h100-sxm-80gb), on_demand, us-east-1
- aws `p4d.24xlarge` (8x a100-sxm-40gb), on_demand, us-east-1
- aws `p4de.24xlarge` (8x a100-sxm-80gb), on_demand, us-east-1
- aws `g6e.xlarge` (1x l40s-48gb), on_demand, us-east-1
- aws `g6.xlarge` (1x l4-24gb), on_demand, us-east-1
- aws `g5.xlarge` (1x a10g-24gb), on_demand, us-east-1
- lambda `gpu_1x_h100_sxm5` (1x h100-sxm-80gb), on_demand
- runpod `h100-sxm` (1x h100-sxm-80gb), on_demand (secure cloud)

Prices are not used by M1 logic; the loader and validation are what M1 delivers. Loader
rejects a row whose `gpu_id` is not in the GPU catalog, naming the row index and id.

## 8. CLI

`typer` app, entry point `llmplan`. Exit codes per ARCHITECTURE.md section 7.

```
llmplan fit --model <hf-id|fixture:name> --gpu <gpu-id> [--tp 1] [--dtype bf16]
            [--kv-dtype bf16] [--context 8192] [--gpu-mem-util 0.9]
            [--max-num-batched-tokens 8192] [--param-count-override N]
            [--gpus PATH] [--format text|json]
llmplan model-info --model <hf-id|fixture:name> [--format text|json]
llmplan gpus [--gpus PATH] [--format text|json]
```

`fit` text output (exact layout is free; content is required):

```
Model     fixture:llama3-70b  (llama_like, 70.55B params, GQA 64/8 heads)
GPU       NVIDIA H100 SXM 80GB x TP=2   usable 76.97 GB/GPU (90% of 85.52 GB)
Weights   bf16   141.11 GB total   70.55 GB/GPU
Overhead  3.22 GB/GPU  (1 GiB fixed + 2.15 GB activations)
KV cache  163,840 B/token/GPU   budget 3.19 GB/GPU   capacity 19,493 tokens
Context   8,192 tokens  ->  max 2 concurrent sequences
Result    FITS  (binding: ok)
Notes     - overhead model: 1 GiB fixed + activations(...)
          - tensor parallel: even weight split assumed
```

GB means 1e9 bytes in display; internal fields are bytes. `--format json` dumps
`FitResult.model_dump()` plus the resolved `ModelSpec`, `GPUSpec`, `EngineProfile`.

## 9. Acceptance tests (`tests/acceptance/test_m1.py`)

All use `FixtureFetcher` and the shipped `data/gpus.yaml`. No network.

**9.1 Parameter counts (exact)**
For each fixture in table 4.1: `count_params(spec) == expected`.

**9.2 KV bytes per token (exact)**
`kv_bytes_per_token_total(spec, "bf16")` equals the values in 4.3 for all four fixtures.
`kv_bytes_per_token_total(llama3-70b, "fp8") == 163_840`.

**9.3 Weight bytes**
- llama3-70b bf16: `141_107_412_992`.
- llama3-8b int4 with 16-bit embeddings: `(8_030_261_248 - 2*525_336_576) * 0.5 + 2*525_336_576 * 2`
  = `3_489_794_048 + 2_101_346_304 = 5_591_140_352`.

**9.4 Fit outcomes on h100-sxm-80gb (81559 MiB), default EngineProfile, context 8192**
| model | dtype | tp | fits | binding | kv_token_capacity in | max_concurrent |
|---|---|---|---|---|---|---|
| llama3-70b | bf16 | 1 | False | weights | == 0 | 0 |
| llama3-70b | bf16 | 2 | True | ok | [15_000, 25_000] | 2 |
| llama3-70b | bf16 | 4 | True | ok | [400_000, 520_000] | [50, 60] |
| llama3-8b | bf16 | 1 | True | ok | [400_000, 500_000] | [48, 60] |

Reference arithmetic for the tp=2 row (implementers should get exactly these; the ranges
exist only to survive future constant tweaks):
usable 76,968,728,985; weights/GPU 70,553,706,496; overhead 3,221,225,472;
kv budget 3,193,797,017; kv/token/GPU 163,840; capacity 19,493; concurrent 2.

**9.5 Fit on a10g-24gb (23028 MiB)**
llama3-8b bf16 tp=1 context 8192: fits True, capacity in [20_000, 35_000], concurrent 3.
llama3-70b bf16 tp=1: fits False, binding weights.

**9.6 Validation and errors**
- `tensor_parallel=3` with llama3-8b (32 heads) raises `ValidationError` mentioning `3` and `32`.
- `context_len=9000` with llama3-8b (max_position_embeddings 8192) raises `ValidationError`.
- A fixture with `architectures: ["GPT2LMHeadModel"]` raises `UnsupportedArchitecture` with
  `field == "architectures"`.
- `HttpConfigFetcher().fetch("evil/../x")` raises `FetchError` without performing a request
  (assert via a stubbed transport that no request was made).
- Loading a `prices.yaml` row with unknown `gpu_id` raises `CatalogError` naming the row
  index and the id.

**9.7 Property tests (hypothesis)**
- For any valid spec: `weight_bytes(fp32) >= weight_bytes(bf16) >= weight_bytes(fp8) >= weight_bytes(int4)`.
- For any spec and tp dividing heads: `per_gpu_weight_bytes` is non-increasing in tp.
- `kv_bytes_per_token_per_gpu * tp >= kv_bytes_per_token_total` (replication never loses KV).

**9.8 CLI smoke**
`llmplan fit --model fixture:llama3-70b --gpu h100-sxm-80gb --tp 2 --format json` exits 0
and the JSON has `fits == true`. `--tp 1` exits 0 with `fits == false` (a false fit is a
valid answer, not an error). Unknown `--gpu` exits 3.

## 10. Implementation order (suggested PR sequence)

1. **PR 1 — scaffold (M0).** pyproject, uv.lock, ruff/mypy/pytest config, CI, `errors.py`,
   `types.py`, package skeleton, README stub. One passing test.
2. **PR 2 — catalog.** `ModelSpec`, fetchers, key mapping, `llama_like` with
   `count_params`, fixtures, tests 9.1, 9.6 (architecture and fetch cases).
3. **PR 3 — hardware.** `GPUSpec`, `PriceRow`, YAML loaders, seed data with sources, tests
   for loaders and the price/gpu FK check.
4. **PR 4 — memory.** `dtypes`, `weights`, `kv_cache`, `engine`, `fit`; tests 9.2 to 9.5, 9.7.
5. **PR 5 — CLI and renderers.** `typer` app, text/json renderers, test 9.8, README usage.

Each PR must leave CI green. Do not combine PRs to save time; reviewability matters more.

## 11. Dependencies allowed in M1

Runtime: `pydantic>=2`, `pyyaml`, `httpx`, `typer`, `rich` (optional, for tables).
Dev: `pytest`, `hypothesis`, `ruff`, `mypy`, `types-PyYAML`, `pip-audit` (or `uv audit`).
Anything else needs a one-line justification in the PR description.

## 12. Open questions (answer by leaving a note in the PR, do not block)

- Whether `head_dim` should be validated against `hidden_size // num_attention_heads` when
  both are present and disagree (some models legitimately differ). Current answer: trust the
  explicit `head_dim`, add a note if it disagrees.
- Whether to display GiB or GB. Current answer: GB (1e9) in text output because cloud
  marketing uses it; bytes in JSON.
