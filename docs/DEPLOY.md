# DEPLOY.md — running the llmplan web UI

The web UI is a single Streamlit page (`llmplan/ui/app.py`) over the library. It needs no
database, no GPU, and **no secrets**. Two hosting paths are documented below; the choice is
open (docs/FOUNDER_QUESTIONS.md, question 5). Either way the same code and data run.

## What the app needs

- Python 3.11, the locked dependencies (`uv.lock`), and the repository's `data/` directory
  (GPU and price catalogs, model fixtures, benchmark rows, and the bundled trace samples
  in `data/traces/samples/`). Catalogs are read relative to the checkout, so the project
  must be installed editable or run from the checkout (M1_NOTES.md); the Docker image does
  this.
- Resources: 1 vCPU and 1 GB of memory are enough for the fixtures and the bundled
  samples. A plan on a 20,000-row sample takes a few seconds. Uploads are capped at 50 MB
  and simulations at 200,000 requests, which bounds memory per session to a few hundred MB.
  Solves and replays run one at a time per process, so one slow plan delays the next.
- Network: outbound HTTPS to `huggingface.co` only, and only when a visitor picks a
  Hugging Face model id (the shipped `fixture:` models work offline).

## Environment variables (all optional)

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Read server-side to fetch configs of gated Hugging Face models. Never shown, logged, or written to disk. Without it, gated models fail with a clear message; everything else works. |
| `LLMPLAN_USAGE_LOG` | Path of the anonymous usage log (JSON lines, M6_DESIGN.md section 7). Unset: no logging. The fields are `ts, request_id, model_id, gpu_ids, n_requests, peak_rps, slo, cost_usd_per_day, baseline_usd_per_day, solver_status, duration_s`; never uploads, price edits, or IPs. `uv run python scripts/usage_summary.py PATH` prints weekly counts. |

The app must run without any secrets. Do not put tokens in the repository, the image, or
`.streamlit/secrets.toml`.

## Run locally

```
uv sync --locked
uv run llmplan ui                      # opens http://localhost:8501
uv run llmplan ui --port 8600 --headless
```

`llmplan ui` starts Streamlit in-process with a 50 MB upload cap and Streamlit's usage
telemetry off. `uv run streamlit run llmplan/ui/app.py` also works and reads the same
settings from `.streamlit/config.toml`.

## Docker (any container host, private repository is fine)

```
docker build -t llmplan .
docker run --rm -p 8501:8501 llmplan
docker run --rm -p 8501:8501 -e LLMPLAN_USAGE_LOG=/logs/usage.jsonl -v "$PWD/logs:/logs" llmplan
```

The image (about 1.4 GB, mostly OR-Tools, pyarrow, pandas and matplotlib) is
`python:3.11-slim` with `uv sync --locked --no-dev`, runs as a non-root user
(uid 10001), exposes port 8501, and has a healthcheck on `/_stcore/health`. For the usage
log, mount a directory writable by uid 10001. On a container host (Fly.io, Render, Cloud
Run, a VM), point the service at port 8501 and the health path `/_stcore/health`; nothing
else is required.

If `docker build` stalls at "load metadata for docker.io/library/python:3.11-slim" (seen
with Docker 20.10 when the client's credential helper does not respond), run `docker pull
python:3.11-slim` first and build with the classic builder: `DOCKER_BUILDKIT=0 docker
build -t llmplan .`.

## Streamlit Community Cloud (free, needs a public GitHub repository)

1. Make the repository public (a decision for the founder; question 5).
2. Community Cloud installs dependencies from a file at the repository root and runs the
   app with `streamlit run`, which does not install the `llmplan` package itself. Commit a
   `requirements.txt` generated from the lock file, ending with the project as an editable
   install:

   ```
   uv export --locked --no-dev --no-hashes --format requirements-txt -o requirements.txt
   ```

   `uv export` writes `-e .` for the project. Regenerate it whenever `uv.lock` changes.
3. On https://share.streamlit.io choose "Create app", select the repository and branch,
   set the main file path to `llmplan/ui/app.py`, and pick Python 3.11 under
   "Advanced settings".
4. Optional: add `LLMPLAN_USAGE_LOG` (and `HF_TOKEN` for gated models) under the app's
   "Secrets" as top-level keys; Community Cloud exposes them as environment variables.
   The local disk there is not persistent, so a usage log written there is lost on
   restart; prefer the Docker path if the launch metrics (PLAN.md section 8) matter.
5. `.streamlit/config.toml` in the repository sets the upload cap and turns telemetry off.

## Checks after deploying

- `curl -fsS https://<host>/_stcore/health` returns `ok`.
- A plan on each preset completes in under 60 s (docs/LAUNCH.md checklist).
- Uploading a file over 50 MB is refused with "Upload refused".
