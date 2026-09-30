# M1 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. Each entry names the commit step
(PR 1 to PR 5 of M1_DESIGN.md section 10) that introduced it.

## Implementation notes

- **PR 1 — Ruff N818 disabled.** ARCHITECTURE.md section 7 fixes exception names
  (`UnsupportedArchitecture`, `InfeasiblePlan`, ...) without an `Error` suffix.
- **PR 1 — Ruff excludes `*.md`.** Ruff 0.16 formats Python code blocks inside Markdown,
  which would rewrite the hand-aligned field listings in the docs.
- **PR 1 — `uv audit` instead of `pip-audit`.** Section 11 allows either; `uv audit` needs no
  extra dependency. It is marked experimental in uv 0.12.
- **PR 1 — No `llmplan.ui` package yet.** It is Streamlit-specific (M6). Placeholders exist
  for `workload`, `perf`, `planner`, `simulate`, each with a TODO naming its milestone.
- **PR 1 — uv was installed with `python3 -m pip install --user uv`** (uv 0.12.21, at
  `~/Library/Python/3.11/bin/uv`); not on PATH by default on this machine.
- **PR 2 — Repo ids and revisions containing `..` are rejected** in addition to the regexes
  of section 3.1 (the regex alone admits `../..`).
- **PR 2 — Redirects are allowed to `huggingface.co` and its subdomains, over https only**,
  enforced by an httpx request hook on every hop. A config.json served from a
  non-huggingface.co CDN would be refused; config files are small non-LFS files.
- **PR 2 — o_proj bias is never counted** (section 4.1 says these families never have it).
  In upstream transformers, Llama/Mistral/Qwen3 with `attention_bias: true` also add an
  o_proj bias. No shipped fixture sets it; the spec is followed as written.
- **PR 2 — `data/fixtures/model_configs/gpt2.json`** is the unsupported-architecture fixture
  for test 9.6.
- **PR 2 — Data files are located relative to the repo** (`<repo>/data`). The built wheel
  does not contain `data/`; see Questions.
- **PR 3 — nvidia-smi totals are not on NVIDIA datasheets.** `vram_bytes` is a required int,
  so the MiB values from section 7.1 are kept; h100 and a10g are pinned by acceptance tests;
  the other five carry `TODO(M3): verify nvidia-smi total`. Web search surfaced third-party
  nvidia-smi listings for 81559 (H100), 46068 (L40S), and 23028 (A10G), but none could be
  fetched to confirm, so none is cited as a source.
- **PR 3 — Dense TFLOPS are half the datasheet "with sparsity" figure** for H100, H200, and
  L4 (the L4 datasheet states "one-half lower without sparsity"); A100 and L40S datasheets
  print dense figures directly. The section 7.1 table truncates to whole TFLOPS (989, 362,
  242 against datasheet-derived 989.5, 362.05, 242.5); the catalog keeps the table values,
  which the datasheets confirm to that precision, and quotes the datasheet figure beside
  each. M3_DESIGN.md test 9.1 is written against 989.
- **PR 3 — AWS prices** were read on 2026-09-30 from the public JSON file that backs
  https://aws.amazon.com/ec2/pricing/on-demand/ (us-east-1, Linux, publication date
  2026-09-25). Values are stored at full published precision (e.g. 21.957642).
- **PR 3 — Lambda's pricing page does not print API instance names**; the row uses
  `gpu_1x_h100_sxm5` from the design doc with the page's 1x H100 SXM price ($4.29/GPU-hr,
  excluding sales tax). RunPod row is the Secure Cloud price ($3.49/hr).
- **PR 3 — A100 40GB SXM source is the NVIDIA A100 datasheet PDF**; the current A100 web
  page no longer lists the 40GB SXM column.

## Deviations from the design doc

- **PR 1 — `UnknownRegistryKey` added to the error hierarchy.** ARCHITECTURE.md section 6
  says registry `get()` raises it, but section 7 did not define it. Direct `LLMPlanError`
  subclass (renderer and format registries will use it too), CLI exit code 2.
  ARCHITECTURE.md section 7 updated.
- **PR 2 — `ModelSpec.qk_norm: bool = False` added.** Section 4.1 adds `2 * d` per layer for
  Qwen3, but every supported HF class maps to the same registry key `llama_like`, so
  `ModelSpec` could not know it was Qwen3. Set from the HF class, never read from config.
  ARCHITECTURE.md section 4 updated.
- **PR 2 — Architecture interface gains `hf_classes` and `embedding_params`.** Section 3.3
  requires `embedding_params`; `hf_classes` keeps the HF class table inside the
  architecture module so adding a family stays one file plus one import. ARCHITECTURE.md
  section 6 updated.
- **PR 2 — `ModelSpec` is strict and `source` is a `Literal`.** Lax pydantic turns JSON
  `true` into `1` for an int field. ARCHITECTURE.md standards ask for `Literal` over `str`.
  Also validated: `num_attention_heads % num_kv_heads == 0` (GQA grouping requires it).
- **PR 2 — Bad config values raise `CatalogError`, not `ValidationError`.** They are catalog
  data (exit code 3), like a bad YAML row; the message names the model id and field. Missing
  required keys raise `UnsupportedArchitecture(field=<key>)` as specified.
- **PR 3 — `load_prices(path, *, gpus=None)`.** The FK check in section 7.2 needs a GPU
  catalog; a user passing `--gpus` must be able to validate prices against it.
  ARCHITECTURE.md section 5 updated.
- **PR 3 — A10G `memory_bandwidth_gbps` and `fp16_dense_tflops` are `null`** (section 7.1
  table: 600 and 125). Those are NVIDIA A10 datasheet numbers; NVIDIA publishes no A10G
  datasheet and the AWS G5 page gives neither, so they cannot be verified for the A10G.
  `nvlink: false` (a required bool) carries a TODO(M3) since it is not documented either way.
- **PR 3 — `pytest-cov` added as a dev dependency** (DEFINITION_OF_DONE.md section 3
  requires 90% coverage). Not in section 11's list.

## Questions for founder

- RunPod Community Cloud H100 SXM is $2.69/hr versus $3.49 Secure Cloud. `PriceRow` has no
  field for the cloud tier; should Community Cloud be a separate `instance` row?
- Should `data/` ship inside the wheel so `pip install llmplan` works outside a checkout?
  Today the CLI finds catalogs relative to the repo. Proposed for M6 (web deploy).
- Is an nvidia-smi listing on a public page (forum post, cluster docs) an acceptable
  `source_url` for `vram_bytes`, given datasheets only give marketing GB?
