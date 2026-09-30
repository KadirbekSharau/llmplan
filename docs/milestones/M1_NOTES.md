# M1 implementation notes

Decisions taken where M1_DESIGN.md or ARCHITECTURE.md was ambiguous, silent, or needed a
change. Each entry names the commit step (PR 1 to PR 5) that introduced it.

## PR 1 — scaffold

- **Ruff N818 disabled.** ARCHITECTURE.md section 7 fixes exception names
  (`UnsupportedArchitecture`, `InfeasiblePlan`, ...) without an `Error` suffix; renaming
  them would break the contract, so the naming lint is ignored project-wide.
- **Ruff excludes `*.md`.** Ruff 0.16 formats Python code blocks inside Markdown, which
  would rewrite the hand-aligned field listings in the docs.
- **`uv audit` instead of `pip-audit`.** Section 11 allows either; `uv audit` needs no extra
  dependency. It is marked experimental in uv 0.12.
- **No `llmplan.ui` package yet.** It is Streamlit-specific (M6); empty placeholders were
  created for `workload`, `perf`, `planner`, `simulate` only, with a TODO naming the milestone.
- **`UnknownRegistryKey` added to the error hierarchy.** ARCHITECTURE.md section 6 says
  registry `get()` raises it, but section 7 did not define it. Added as a direct
  `LLMPlanError` subclass (it is a usage error, not a catalog error: renderer and format
  registries use it too) mapped to CLI exit code 2. ARCHITECTURE.md section 7 updated.

## PR 2 — catalog

- **`ModelSpec.qk_norm: bool = False` added.** Section 4.1 adds `2 * d` per layer for Qwen3,
  but every supported HF class maps to the same registry key `llama_like`, so `ModelSpec`
  had no way to know it was Qwen3. The flag is set from the HF class (never read from
  config.json). ARCHITECTURE.md section 4 updated.
- **Architecture interface gains `hf_classes` and `embedding_params`.** Section 3.3 of the
  design doc requires `embedding_params`; `hf_classes` (HF class -> per-class defaults) keeps
  the class table inside the architecture module so adding a family is still one file plus
  one import. ARCHITECTURE.md section 6 updated.
- **`ModelSpec` is strict and `source` is a `Literal`.** In lax mode pydantic turns JSON
  `true` into `1` for an int field; strict mode rejects it. `source` was `str` with the three
  allowed values in a comment; ARCHITECTURE.md standards ask for `Literal`.
- **`num_attention_heads % num_kv_heads == 0` is validated.** GQA grouping requires it; a
  config that violates it is malformed.
- **Bad config values raise `CatalogError`, not `ValidationError`.** They are catalog data
  (exit code 3), like a bad YAML row. The message names the model id and field. Missing
  required keys still raise `UnsupportedArchitecture(field=<key>)` per section 3.2.
- **Repo ids and revisions containing `..` are rejected** in addition to the regexes in
  section 3.1 (the regex alone admits `../..`).
- **Redirects are allowed to `huggingface.co` and its subdomains, over https only**, enforced
  by an httpx request hook on every hop. A config.json served from a non-huggingface.co CDN
  would be refused; config files are small non-LFS files, so this is not expected.
- **o_proj bias is never counted** (section 4.1 says these families never have it). In
  upstream transformers, Llama/Mistral/Qwen3 with `attention_bias: true` also add an o_proj
  bias. No shipped fixture sets it; followed the spec as written. Revisit if such a model
  appears.
- **`data/fixtures/model_configs/gpt2.json`** ships as the unsupported-architecture fixture
  for test 9.6.
- **Data files are located relative to the repo** (`<repo>/data`). Packaging `data/` into
  the wheel is deferred to M6 (the web deploy).
