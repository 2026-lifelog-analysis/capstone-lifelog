"""기존 모듈 함수와 이식본의 구조 및 실제 자료 수치 동일성을 재현합니다.

원본 모듈은 별도 로컬 폴더가 필요합니다. 개인 행은 출력하지 않으며 새 JSON만
독점 생성합니다. 전체 자료 검증이 아니라 고정 규칙으로 선택한 15명 대조입니다.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon import comparison_models as models


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)}


def _without_docstring(function: ast.FunctionDef) -> ast.FunctionDef:
    if (function.body and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
            and isinstance(function.body[0].value.value, str)):
        function.body = function.body[1:]
    return function


def compare_structure(original: Path, port: Path) -> list[str]:
    """문서문자열과 명시된 선택적 config 보완 외 함수 AST를 정확 대조합니다."""
    expected, actual = _functions(original), _functions(port)
    if set(actual) - set(expected) != {"adapt_main"}:
        raise AssertionError("Unexpected newly added functions")
    matched = []
    for name, function in expected.items():
        reference = _without_docstring(function)
        copied = _without_docstring(actual[name])
        if name == "score_models":
            allowed = ast.parse("config = {} if config is None else config").body[0]
            if ast.dump(copied.body[0]) != ast.dump(allowed):
                raise AssertionError("Unexpected config default implementation")
            copied.body = copied.body[1:]
            copied.args = reference.args
        if ast.dump(reference, include_attributes=False) != ast.dump(copied, include_attributes=False):
            raise AssertionError(f"Algorithm AST mismatch: {name}")
        matched.append(name)
    return matched


def validate(output: str | Path) -> dict:
    output = Path(output)
    destination = output / "model_port_validation.json"
    if destination.exists():
        raise FileExistsError(destination)
    original = io.ROOT.parent / models.REFERENCE_SOURCE
    port = Path(models.__file__)
    inputs = {
        "main_daily.csv.gz": io.sha256(output / "main_daily.csv.gz"),
        "split.json": io.sha256(output / "split.json"),
        "original_models.py": io.sha256(original),
        "comparison_models.py": io.sha256(port),
        "validator.py": io.sha256(Path(__file__)),
    }
    if inputs["original_models.py"] != models.REFERENCE_SHA256:
        raise ValueError("Original model source differs from the recorded source hash")
    started = time.monotonic()
    functions = compare_structure(original, port)
    frame = io.read_frame(output / "main_daily.csv.gz")
    participants = sorted(frame.pid.unique())[::max(1, frame.pid.nunique() // 15)][:15]
    daily = models.adapt_main(frame[frame.pid.isin(participants)])
    split = json.loads((output / "split.json").read_text())
    spec = importlib.util.spec_from_file_location("original_models_readonly_validation", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = models.score_models(daily, split["development_end"], split["calibration_start"])
    expected = module.score_models(daily, split["development_end"], split["calibration_start"], {})
    for left, right in zip(actual, expected):
        assert_frame_equal(left, right, check_exact=True)
    after = {
        "main_daily.csv.gz": io.sha256(output / "main_daily.csv.gz"),
        "split.json": io.sha256(output / "split.json"),
        "original_models.py": io.sha256(original),
        "comparison_models.py": io.sha256(port),
        "validator.py": io.sha256(Path(__file__)),
    }
    if inputs != after:
        raise AssertionError("A validation input changed")
    result = {
        "status": "passed", "sample_people": len(participants), "sample_person_days": len(daily),
        "score_rows": len(actual[0]), "valid_score_rows": int(actual[0].valid.sum()),
        "baseline_rows": len(actual[1]), "exact_all_cells_scores_and_baselines": True,
        "assertion": "pandas.testing.assert_frame_equal(check_exact=True), both output tables",
        "sample_rule": "Sorted string identifiers, stride max(1, floor(total_people / 15)), first 15",
        "comparison_scope": "15-person actual-data sample; not all 579 participants",
        "ast_function_count": len(functions), "ast_identical_functions": functions,
        "allowed_port_changes": ["Docstrings", "adapt_main", "score_models optional config defaults to {}"],
        "input_hashes_before": inputs, "input_hashes_after": after, "inputs_preserved": True,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "elapsed_seconds": time.monotonic() - started,
    }
    io.write_json(destination, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summary = validate(arguments.output)
    print(json.dumps({key: summary[key] for key in [
        "status", "sample_people", "sample_person_days", "score_rows", "valid_score_rows",
        "ast_function_count", "exact_all_cells_scores_and_baselines", "inputs_preserved",
    ]}), flush=True)
