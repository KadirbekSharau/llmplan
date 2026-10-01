# llmplan web UI (M6_DESIGN.md section 8). Build: docker build -t llmplan .
# Run: docker run --rm -p 8501:8501 llmplan   (no secrets needed; HF_TOKEN is optional)
FROM python:3.11-slim

# uv from its official image, pinned to the version the lock file was made with.
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached layer), then the project. The project is installed editable,
# so catalogs and samples are read from /app/data, as from a checkout (M1_NOTES.md).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY llmplan ./llmplan
COPY data ./data
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 10001 llmplan
USER llmplan

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["/app/.venv/bin/llmplan", "ui", "--address", "0.0.0.0", "--port", "8501", "--headless"]
