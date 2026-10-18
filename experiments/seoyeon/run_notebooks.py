"""Execute notebook copies with the repository's Python 3.12 environment.

Run from the repository root with ``.venv/bin/python
experiments/seoyeon/run_notebooks.py``. Cell outputs remain in local executed
notebooks; the progress log deliberately excludes data rows and tracebacks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_core.utils import run_sync
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOKS = (
    "notebooks/01_preprocessing.ipynb",
    "notebooks/02_baseline_nethealth.ipynb",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str) -> str:
    """Remove credentials from dependency URLs and common secret assignments."""
    value = re.sub(r"(https?://)[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)((?:token|password|api[_-]?key)=)[^\s&]+", r"\1[REDACTED]", value)
    return re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[REDACTED]", value)


class RuntimeKernelSpecs(KernelSpecManager):
    """Bind kernel launch to this interpreter, independent of user kernelspecs."""

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        if kernel_name != "seoyeon-runtime":
            raise KeyError(kernel_name)
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Repository Python 3.12",
            language="python",
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: Path, notebooks: list[Path]) -> dict[str, str]:
    files = {path for path in notebooks if path.is_file()}
    files.update((root / "src").rglob("*.py"))
    files.update(path for path in (root / "data/nethealth").rglob("*") if path.is_file())
    if (root / "requirements.txt").is_file():
        files.add(root / "requirements.txt")
    return {str(path.relative_to(root)): sha256(path) for path in sorted(files)}


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


def new_output(root: Path, requested: str | None) -> Path:
    allowed = (root / "results/seoyeon_tuning").resolve()
    if requested:
        output = (root / requested).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = allowed / "common_replay" / stamp
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a new directory below results/seoyeon_tuning")
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_notebook(path: Path, notebook: nbformat.NotebookNode) -> None:
    with path.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)


def run_notebook(path: Path, output: Path, root: Path, timeout: int, log) -> dict:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()
    record = {"notebook": str(path.relative_to(root)), "started_utc": now(), "status": "running"}
    current_cell = None

    def cell_start(cell, cell_index, **_):
        nonlocal current_cell
        if cell.cell_type == "code":
            current_cell = cell_index
            log(f"{path.name}: cell {cell_index} started")

    def cell_end(cell, cell_index, execute_reply, **_):
        if cell.cell_type == "code":
            status = execute_reply.get("content", {}).get("status", "unknown")
            log(f"{path.name}: cell {cell_index} returned {status}")

    manager = AsyncKernelManager(
        kernel_name="seoyeon-runtime", kernel_spec_manager=RuntimeKernelSpecs()
    )
    client = NotebookClient(
        notebook, km=manager, timeout=timeout, allow_errors=False,
        record_timing=True, resources={"metadata": {"path": str(root)}},
        on_cell_start=cell_start, on_cell_executed=cell_end,
    )
    try:
        client.execute(cwd=str(root))
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", exception_type=type(error).__name__, cell_index=current_cell)
        log(f"{path.name}: FAILED ({type(error).__name__}), cell {current_cell}")
    finally:
        # An externally supplied manager is not automatically cleaned up by nbclient.
        try:
            run_sync(client._async_cleanup_kernel)()
        except Exception as error:
            record.update(status="failed", cleanup_exception_type=type(error).__name__)
            log(f"{path.name}: kernel cleanup FAILED ({type(error).__name__})")
        record.update(finished_utc=now(), elapsed_seconds=round(time.monotonic() - started, 3))
        destination = output / f"{path.stem}.executed.ipynb"
        save_notebook(destination, notebook)
        record["executed_notebook"] = destination.name
        record["executed_sha256"] = sha256(destination)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", nargs="+", default=list(DEFAULT_NOTEBOOKS))
    parser.add_argument("--output", help="New output path below results/seoyeon_tuning")
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per cell")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("Run using this repository's .venv/bin/python (Python 3.12)")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    notebooks = [(ROOT / name).resolve() for name in args.notebooks]
    if len(set(notebooks)) != len(notebooks):
        parser.error("Notebook paths must be unique")
    if any(not path.is_relative_to(ROOT / "notebooks") or not path.is_file() for path in notebooks):
        parser.error("Every notebook must exist under the repository notebooks directory")
    if len({path.stem for path in notebooks}) != len(notebooks):
        parser.error("Notebook names must be unique")
    output = new_output(ROOT, args.output)
    manifest = {
        "started_utc": now(), "command": sanitize(shlex.join([sys.executable, *sys.argv])),
        "python": sys.version, "python_executable": sys.executable, "kernel_argv": RuntimeKernelSpecs().get_kernel_spec("seoyeon-runtime").argv,
        "versions": {name: importlib.metadata.version(name) for name in ("nbclient", "nbformat", "ipykernel", "jupyter_client")},
        "working_directory": str(ROOT), "timeout_seconds": args.timeout,
        "status": "running", "notebooks": [],
    }
    code = 1
    with (output / "progress.log").open("x", encoding="utf-8") as progress:
        def log(message: str) -> None:
            line = f"{now()} {sanitize(message)}"
            progress.write(line + "\n")
            progress.flush()
            print(line, flush=True)

        before = snapshot(ROOT, notebooks)
        write_json(output / "input_hashes_before.json", before)
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
            with (output / "pip_freeze.txt").open("x", encoding="utf-8") as stream:
                stream.write(sanitize(freeze.stdout))
            log("Execution started; cell output is retained only in local notebooks")
            for path in notebooks:
                result = run_notebook(path, output, ROOT, args.timeout, log)
                manifest["notebooks"].append(result)
                if result["status"] != "passed":
                    break
            code = 0 if len(manifest["notebooks"]) == len(notebooks) and all(item["status"] == "passed" for item in manifest["notebooks"]) else 1
        except Exception as error:
            manifest["exception_type"] = type(error).__name__
            log(f"Runner FAILED ({type(error).__name__}); no data rows logged")
        finally:
            try:
                after = snapshot(ROOT, notebooks)
                write_json(output / "input_hashes_after.json", after)
                changes = changed_files(before, after)
                manifest.update(input_files_unchanged=not changes, changed_input_files=changes)
                if changes:
                    code = 1
            except Exception as error:
                manifest.update(input_files_unchanged=False, hash_check_exception_type=type(error).__name__)
                log(f"Final input hash check FAILED ({type(error).__name__})")
                code = 1
            manifest["finished_utc"] = now()
            manifest["status"] = "passed" if code == 0 else "failed"
            manifest["exit_code"] = code
            write_json(output / "execution_manifest.json", manifest)
            log(f"Execution {manifest['status']}; source/data hashes unchanged: {manifest['input_files_unchanged']}")
            log(f"Artifacts: {output.relative_to(ROOT)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
