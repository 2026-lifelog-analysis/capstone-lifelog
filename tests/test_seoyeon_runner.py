"""Checks for safe replay artifact handling and source immutability detection."""

import json
import sys
from types import SimpleNamespace

import nbformat
import pytest

from experiments.seoyeon import run_notebooks as runner
from experiments.seoyeon.run_notebooks import (
    RuntimeKernelSpecs,
    changed_files,
    new_output,
    sanitize,
    save_notebook,
    snapshot,
)


def test_output_refuses_existing_directory_and_outside_results(tmp_path):
    output = new_output(tmp_path, "results/seoyeon_tuning/replay")
    (output / "existing.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        new_output(tmp_path, "results/seoyeon_tuning/replay")
    assert (output / "existing.txt").read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError):
        new_output(tmp_path, "../outside")


def test_artifact_write_does_not_modify_source_or_overwrite_output(tmp_path):
    source = tmp_path / "source.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")])
    nbformat.write(notebook, source)
    before = snapshot(tmp_path, [source])
    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="1\n")]
    output = tmp_path / "executed.ipynb"
    save_notebook(output, notebook)
    with pytest.raises(FileExistsError):
        save_notebook(output, notebook)
    assert snapshot(tmp_path, [source]) == before
    assert nbformat.read(output, as_version=4).cells[0].outputs[0].text == "1\n"
    source.write_text("changed source", encoding="utf-8")
    assert changed_files(before, snapshot(tmp_path, [source])) == ["source.ipynb"]
    source.unlink()
    assert changed_files(before, snapshot(tmp_path, [source])) == ["source.ipynb"]


def test_failure_preserves_partial_notebook_without_logging_cell_data(tmp_path, monkeypatch):
    class FailedClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook
            self.callback = kwargs["on_cell_start"]

        def execute(self, **kwargs):
            cell = self.notebook.cells[0]
            self.callback(cell=cell, cell_index=0)
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="private row")]
            raise ValueError("private row")

        async def _async_cleanup_kernel(self):
            return None

    monkeypatch.setattr(runner, "NotebookClient", FailedClient)
    source = tmp_path / "source.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print(1)")]), source)
    output = new_output(tmp_path, "results/seoyeon_tuning/failure")
    logs = []
    record = runner.run_notebook(source, output, tmp_path, 10, logs.append)
    assert record["status"] == "failed" and record["exception_type"] == "ValueError"
    assert record["cell_index"] == 0
    assert "private row" not in "\n".join(logs)
    partial = nbformat.read(output / record["executed_notebook"], as_version=4)
    assert partial.cells[0].outputs[0].text == "private row"
    assert changed_files({"deleted": "old"}, {"added": "new"}) == ["added", "deleted"]

    # A later hash-read failure must still produce a failing final manifest.
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/check.ipynb").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    monkeypatch.setattr(runner, "run_notebook", lambda *args: {"status": "passed"})
    calls = []

    def failing_final_snapshot(*args):
        calls.append(True)
        if len(calls) == 2:
            raise PermissionError("private path")
        return {"notebooks/check.ipynb": "initial_hash"}

    monkeypatch.setattr(runner, "snapshot", failing_final_snapshot)
    status = runner.main(["--notebooks", "notebooks/check.ipynb", "--output", "results/seoyeon_tuning/hash_failure"])
    record = json.loads((tmp_path / "results/seoyeon_tuning/hash_failure/execution_manifest.json").read_text())
    assert status == 1 and record["status"] == "failed"
    assert record["input_files_unchanged"] is False
    assert record["hash_check_exception_type"] == "PermissionError"


def test_runtime_kernel_uses_current_interpreter_and_logs_redact_secrets():
    spec = RuntimeKernelSpecs().get_kernel_spec("seoyeon-runtime")
    assert spec.argv == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    text = sanitize("pkg @ https://user:secret@host/file?token=abc&x=1 github_pat_private123")
    assert "secret" not in text and "abc" not in text and "private123" not in text
    assert "x=1" in text
