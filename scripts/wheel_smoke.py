"""Install a built wheel into a clean virtual environment and run the CLI from outside the
checkout (M8_DESIGN.md section 8.2). Used by the CI `check` job, the release workflow and
tests/acceptance/test_m8.py::test_wheel_smoke.

    uv run python scripts/wheel_smoke.py dist/llmplan-*.whl [--installer pip|uv]

`pip` (the default) is what users run; `uv` installs from uv's cache and is much faster
locally. Exits non-zero when any step fails; the package must import from the new
environment, not from the checkout, and both commands must succeed there.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMMANDS = (
    ("fit", "--model", "fixture:llama3-8b", "--gpu", "h100-sxm-80gb", "--format", "json"),
    ("gpus",),
)


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> str:
    print("+", " ".join(argv), flush=True)
    done = subprocess.run(  # noqa: S603  # argv built here, no shell
        argv, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise SystemExit(f"failed ({done.returncode}): {' '.join(argv)}\n{done.stderr[-2000:]}")
    return done.stdout


def smoke(wheel: Path, workdir: Path, *, installer: str = "pip") -> list[str]:
    """Create `workdir/venv`, install `wheel` there, and run each of `COMMANDS` with the
    installed `llmplan` from `workdir/outside` (no PYTHONPATH). Returns their outputs."""
    venv, outside = workdir / "venv", workdir / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    if installer == "uv":
        uv = os.environ.get("UV") or shutil.which("uv") or "uv"
        _run([uv, "venv", "--python", sys.executable, str(venv)], outside, env)
        _run([uv, "pip", "install", "--python", str(python), str(wheel)], outside, env)
    else:
        _run([sys.executable, "-m", "venv", str(venv)], outside, env)
        _run([str(python), "-m", "pip", "install", "--quiet", str(wheel)], outside, env)
    where = _run([str(python), "-c", "import llmplan; print(llmplan.__file__)"], outside, env)
    if not Path(where.strip()).resolve().is_relative_to(venv.resolve()):
        raise SystemExit(f"llmplan was imported from {where.strip()}, not from {venv}")
    llmplan = python.parent / "llmplan"
    return [_run([str(llmplan), *command], outside, env) for command in COMMANDS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path, help="the .whl file to install")
    parser.add_argument("--installer", choices=("pip", "uv"), default="pip")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="llmplan-smoke-") as tmp:
        outputs = smoke(args.wheel.resolve(), Path(tmp), installer=args.installer)
    print(f"wheel smoke test passed: {len(outputs)} commands ran outside the checkout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
