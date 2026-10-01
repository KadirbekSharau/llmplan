# DEPLOY.md — running the llmplan web UI

The web UI is a single Streamlit page (`llmplan/ui/app.py`) over the library. It needs no
database, no GPU, and **no secrets**. Two hosting paths are documented below; the choice is
open (docs/FOUNDER_QUESTIONS.md, question 5). Either way the same code and data run.

## What the app needs

- Python 3.11 and the locked dependencies (`uv.lock`). The GPU and price catalogs, model
  fixtures, benchmark rows and bundled trace samples are package data
  (`llmplan/data/`, read through `importlib.resources`), so a checkout, the Docker image
  and `pip install llmplan` all carry them (M8).
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
5. `.streamlit/config.toml` in the repository sets the upload cap, turns telemetry off and
   sets the primary colour of the light and dark themes.

## Publishing to PyPI (one-time setup by the founder)

`pip install llmplan` needs the package on PyPI (docs/FOUNDER_QUESTIONS.md, item 7).
`.github/workflows/release.yml` runs when a tag `v<version>` is pushed: it checks the tag
against `pyproject.toml`'s version, builds the sdist and wheel, installs the wheel into a
clean virtual environment with pip and runs `llmplan fit` and `llmplan gpus` from outside
the checkout, publishes to PyPI, and creates the GitHub release with that version's
CHANGELOG section. Until PyPI is configured the publish step is skipped with a notice
("PyPI publishing is not configured"); everything else still runs.

Trusted publishing (recommended: no token is stored anywhere):

1. Create a PyPI account at https://pypi.org (enable two-factor authentication).
2. Open https://pypi.org/manage/account/publishing/ and add a **pending publisher**:
   PyPI project name `llmplan`, owner `KadirbekSharau`, repository name `llmplan`,
   workflow name `release.yml`, environment name left empty. The first successful upload
   creates the project under your account.
3. In the GitHub repository: Settings, Secrets and variables, Actions, Variables: add the
   repository variable `PYPI_TRUSTED_PUBLISHING` with value `true`. The workflow publishes
   only when it is set.
4. Release (the CTO, after review): `git tag v0.2.0 && git push origin v0.2.0`. A tag whose
   version differs from `pyproject.toml` fails the workflow before anything is built.

Alternative, an API token: create a token on PyPI (scope it to the `llmplan` project once
the project exists) and add it as the repository secret `PYPI_API_TOKEN`; the workflow uses
it when present. Optional hardening once the repository is public: create a GitHub
environment `pypi` with required reviewers, add `environment: pypi` to the `publish` job,
and enter `pypi` as the environment name of the trusted publisher on PyPI.

Names on PyPI are permanent; the founder confirmed `llmplan` (FOUNDER_QUESTIONS.md,
item 6).

## Checks after deploying

- `curl -fsS https://<host>/_stcore/health` returns `ok`.
- A plan on each preset completes in under 60 s (docs/LAUNCH.md checklist).
- Uploading a file over 50 MB is refused with "Upload refused".

## DigitalOcean Droplet (current production, set up 2026-10-01)

Host: `137.184.154.129` (New York, 1 vCPU, 1 GB RAM, Ubuntu 24.04, hostname `llmplan`).
Public URL: https://llmplan.dev (GoDaddy DNS A records for `@` and `www` -> 137.184.154.129;
Let's Encrypt certificate via Caddy). `www.llmplan.dev` and `137-184-154-129.sslip.io` redirect there.

One-time setup that was applied (repeat on a new Droplet):
1. 2 GB swap file (`/swapfile`, in `/etc/fstab`) so the Docker build fits in 1 GB RAM.
2. `apt-get install docker.io caddy ufw`; `ufw allow OpenSSH, 80/tcp, 443/tcp`; `ufw enable`.
3. `/etc/caddy/Caddyfile`: `llmplan.dev { encode gzip; log; reverse_proxy 127.0.0.1:8501 }` plus a
   redirect block for `www.llmplan.dev` and the sslip.io name. Caddy obtains and renews
   certificates automatically. Browsers' HTTPS-first modes block plain HTTP, so HTTPS is required.
4. `/var/lib/llmplan` owned by uid 10001 (the container user) for the usage log.
5. Source is shipped as a `git archive` tarball (the repo is private); the image is built
   on the Droplet; the container runs with `--restart unless-stopped --memory 512m`, bound
   to localhost only, with `LLMPLAN_USAGE_LOG=/var/lib/llmplan/usage.jsonl`.

Redeploy the current checkout: `deploy/droplet-redeploy.sh` (about 5 minutes; the app is
down for the few seconds between container stop and health).

Measured on 2026-10-01: build 1.84 GB image in ~6 min; steady state ~450 MB used of 961 MB
with the container idle; the app peaks at ~225 MB on the heaviest preset.
