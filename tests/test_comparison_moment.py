"""Synthetic unit fixtures are never reported as NetHealth experiment results."""
from contextlib import nullcontext
import hashlib
import json
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.seoyeon.comparison_moment_inputs import (
    FEATURES, build_inputs, endpoint_errors, row_digest, select_eligible,
    verified_prediction_cache,
)
from experiments.seoyeon.comparison_moment import MomentEngine, infer
from experiments.seoyeon import comparison_moment


@pytest.fixture
def fixture_data():
    dates = pd.date_range("2020-01-01", periods=12)
    daily = pd.DataFrame({"participant": "p", "date": dates,
                          "steps": np.arange(12) + 10., "sleep_minutes": 400.,
                          "bedtime_minutes": 10., "complete": True})
    daily.loc[2, "complete"] = False
    daily.loc[2, "sleep_minutes"] = np.nan
    baselines = pd.DataFrame({"participant": ["p"], "model": ["fixed28"],
                             "eligible": [True], "baseline_end": [dates[1]],
        **{f"center_{f}": [v] for f, v in zip(FEATURES, (10., 390., 1430.))},
        **{f"scale_{f}": [v] for f, v in zip(FEATURES, (2., 5., 10.))}})
    keys = pd.DataFrame({"participant": ["p", "p"], "date": dates[[3, 9]],
                         "split": ["calibration", "test"]})
    return daily, baselines, keys


def test_calendar_masks_circular_values_and_no_future(fixture_data):
    daily, baselines, keys = fixture_data
    arrays, audit = build_inputs(daily, baselines, keys, seq_len=8, patch_len=2)
    assert arrays["x"].shape == (2, 3, 8)
    assert arrays["x"].dtype == np.float32
    assert arrays["input_mask"][0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert arrays["observation_mask"][0].tolist() == [0, 0, 0, 0, 1, 1, 0, 1]
    np.testing.assert_allclose(arrays["x"][0, :, -1], [1.5, 2., 2.])
    assert not arrays["x"][0, :, -2].any()
    assert audit.padding_days.tolist() == [4, 0]
    assert audit.complete_patches.tolist() == [1, 3]
    changed = daily.assign(steps=np.where(daily.date > keys.date.max(), 1e8, daily.steps))
    other, _ = build_inputs(changed, baselines, keys, seq_len=8, patch_len=2)
    np.testing.assert_array_equal(arrays["x"], other["x"])


def test_missing_calendar_row_is_not_left_padding(fixture_data):
    daily, baselines, keys = fixture_data
    arrays, audit = build_inputs(daily.drop(index=2), baselines, keys, seq_len=8, patch_len=2)
    assert arrays["input_mask"][0, -2] == 1
    assert arrays["observation_mask"][0, -2] == 0
    assert audit.missing_calendar_days.tolist() == [1, 1]


@pytest.mark.parametrize("fault", ["zero_scale", "late_baseline", "missing_baseline", "target_missing"])
def test_ineligible_keys_preserved_with_reason(fixture_data, fault):
    daily, baselines, keys = fixture_data
    if fault == "zero_scale":
        baselines = baselines.assign(scale_steps=0.)
    elif fault == "late_baseline":
        baselines = baselines.assign(baseline_end=keys.date.max())
    elif fault == "missing_baseline":
        baselines = baselines.iloc[:0]
    else:
        daily.loc[daily.date.isin(keys.date), "complete"] = False
    eligible, rejected = select_eligible(daily, baselines, keys)
    assert eligible.empty
    assert len(rejected) == len(keys)
    assert rejected.reason.ne("ok").all()


def test_duplicate_keys_rejected(fixture_data):
    daily, baselines, keys = fixture_data
    with pytest.raises(ValueError, match="Duplicate"):
        select_eligible(daily, baselines, pd.concat([keys, keys.iloc[:1]]))


def test_endpoint_errors_use_inverse_scaled_circular_difference():
    errors, scores = endpoint_errors(np.array([[1000., 480., 1430.]]),
        np.array([[500., 400., 10.]]), np.array([[100., 40., 20.]]), np.array([[4., 1., 0.]]))
    np.testing.assert_allclose(errors, [[1., 1., -1.]])
    np.testing.assert_allclose(scores, [1.])


@pytest.mark.parametrize("field", ["x", "input_mask", "observation_mask", "centers", "scales", "target_raw"])
def test_exact_six_field_cache_key_detects_every_changed_field(fixture_data, field):
    arrays, _ = build_inputs(*fixture_data)
    before = row_digest(arrays, 0)
    changed = {name: values.copy() for name, values in arrays.items()}
    changed[field].reshape(len(arrays["x"]), -1)[0, -1] += 1
    assert before != row_digest(changed, 0)


class FixtureEngine(MomentEngine):
    def __init__(self):
        super().__init__(None, device="cpu", batch_size=2)
        self.calls = 0
        self.metadata = {"fixture_only": True}

    def predict(self, arrays):
        self.calls += len(arrays["x"])
        return np.zeros((len(arrays["x"]), 3), dtype=np.float32)


def test_infer_cache_only_same_run_exact_inputs_and_no_overwrite(tmp_path, fixture_data):
    engine = FixtureEngine()
    first = infer(*fixture_data, tmp_path / "first", engine=engine)
    second = infer(*fixture_data, tmp_path / "second", engine=engine)
    assert first["new_inferred_persondays"] == 2
    assert second["new_inferred_persondays"] == 0
    assert second["exact_six_field_cache_persondays"] == 2
    assert engine.calls == 2
    scores = pd.read_csv(tmp_path / "second" / "scores.csv.gz")
    assert len(scores) == 2 and scores.valid.all()
    assert scores.score_source.eq("same_process_exact_six_field_cache").all()
    with pytest.raises(FileExistsError):
        infer(*fixture_data, tmp_path / "first", engine=engine)


def test_infer_keeps_unavailable_rows_not_zero_scores(tmp_path, fixture_data):
    daily, baselines, keys = fixture_data
    daily.loc[daily.date.eq(keys.date.iloc[0]), "complete"] = False
    record = infer(daily, baselines, keys, tmp_path / "missing", engine=FixtureEngine())
    scored = pd.read_csv(tmp_path / "missing" / "scores.csv.gz")
    assert record["unavailable_persondays"] == 1
    assert len(scored) == 2
    assert np.isnan(scored.score.iloc[0]) and not scored.valid.iloc[0]
    assert scored.reason.iloc[0] == "target_not_complete"


def test_nonfinite_prediction_records_failure_and_all_requested_keys(tmp_path, fixture_data):
    engine = FixtureEngine()
    engine.predict = lambda arrays: np.full((len(arrays["x"]), 3), np.nan)
    with pytest.raises(RuntimeError, match="Nonfinite"):
        infer(*fixture_data, tmp_path / "failed", engine=engine)
    record = json.loads((tmp_path / "failed" / "execution.json").read_text())
    scored = pd.read_csv(tmp_path / "failed" / "scores.csv.gz")
    assert record["status"] == "failed"
    assert record["actual_forward_requested_persondays"] == 2
    assert record["actual_forward_completed_persondays"] == 0
    assert len(scored) == 2 and scored.score.isna().all()


def test_predict_runs_repeat_mask_controls_once(fixture_data):
    arrays, _ = build_inputs(*fixture_data)
    engine = MomentEngine(None)
    engine.model = object()
    engine._forward = lambda inputs: inputs["x"][:, :, -1]
    output = engine.predict(arrays)
    assert engine.metadata["controls"]["pass"]
    assert engine.metadata["controls"]["diagnostic_forward_persondays"] == 6
    np.testing.assert_array_equal(output, arrays["x"][:, :, -1])


def test_load_checks_identity_weights_and_freezes_all_parameters(tmp_path, monkeypatch):
    checkpoint = tmp_path / "00_원자료" / "moment_checkpoint"
    checkpoint.mkdir(parents=True)
    (tmp_path / "03_처리기록").mkdir()
    (checkpoint / "config.json").write_text('{"seq_len":512,"patch_len":8}')
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    weight_hash = hashlib.sha256(b"fixture").hexdigest()
    monkeypatch.setattr(comparison_moment, "WEIGHTS_SHA256", weight_hash)
    provenance = {"model_id": comparison_moment.MODEL_ID, "revision": comparison_moment.REVISION,
        "files": {name: {"sha256": comparison_moment.sha256(checkpoint / name)}
                  for name in ("config.json", "model.safetensors")}}
    (tmp_path / "03_처리기록" / "moment_checkpoint.json").write_text(json.dumps(provenance))

    class FixtureModel:
        training, seq_len, patch_len = True, 512, 8
        normalizer = SimpleNamespace(eps=1e-5)
        def __init__(self, config, model_kwargs):
            assert model_kwargs == {"task_name": "reconstruction"}
        def load_state_dict(self, state, strict):
            assert strict and state == {}
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])
        def init(self):
            pass
        def requires_grad_(self, value):
            assert not value
        def eval(self):
            self.training = False
        def to(self, device):
            assert device == "cpu"
        def parameters(self):
            return [SimpleNamespace(requires_grad=False, numel=lambda: 7)]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None,
        set_num_threads=lambda n: None, backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))))
    monkeypatch.setitem(sys.modules, "momentfm", SimpleNamespace(MOMENTPipeline=FixtureModel))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=lambda p: {}))
    monkeypatch.setattr(comparison_moment.metadata, "version", lambda p: "fixture")
    engine = MomentEngine(tmp_path, device="cpu")
    engine._load()
    assert engine.metadata["parameters"] == 7
    assert engine.metadata["trainable_parameters"] == 0
    assert engine.metadata["strict_load_missing_keys"] == []
    with pytest.raises(RuntimeError, match="MPS unavailable"):
        MomentEngine(tmp_path, device="mps")._load()
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        MomentEngine(tmp_path)._load()


def test_forward_passes_original_masks_and_returns_only_endpoint(fixture_data, monkeypatch):
    arrays, _ = build_inputs(*fixture_data)
    class Tensor:
        def __init__(self, array):
            self.array = array
        def to(self, device):
            return self
        def __getitem__(self, index):
            return Tensor(self.array[index])
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.array
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, from_numpy=Tensor))
    def forward(**kwargs):
        np.testing.assert_array_equal(kwargs["mask"].array, arrays["observation_mask"])
        np.testing.assert_array_equal(kwargs["input_mask"].array, arrays["input_mask"])
        return SimpleNamespace(reconstruction=kwargs["x_enc"])
    engine = MomentEngine(None, device="cpu")
    engine.model = forward
    np.testing.assert_array_equal(engine._forward(arrays), arrays["x"][:, :, -1])


def test_cli_multi_job_loads_one_engine_preserves_inputs(tmp_path, fixture_data, monkeypatch):
    files = {}
    for name, frame in zip(("daily", "baselines", "keys"), fixture_data):
        files[name] = str(tmp_path / f"{name}.csv")
        frame.to_csv(files[name], index=False)
    jobs = [{**files, "output": str(tmp_path / label)} for label in ("first", "second")]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs))
    engine = FixtureEngine()
    monkeypatch.setattr(comparison_moment, "MomentEngine", lambda *args: engine)
    monkeypatch.setattr(sys, "argv", ["comparison_moment", "--jobs", str(path), "--model-root", str(tmp_path)])
    comparison_moment.main()
    record = json.loads((tmp_path / "second" / "execution.json").read_text())
    assert record["exact_six_field_cache_persondays"] == 2
    assert json.loads((tmp_path / "second" / "source_preservation.json").read_text())["unchanged"]


def test_same_experiment_cache_rebuilds_and_checks_all_six_fields(tmp_path, fixture_data):
    hashes = {name: "fixture" for name in ("daily", "baselines", "keys")}
    infer(*fixture_data, tmp_path / "clean", engine=FixtureEngine(), source_hashes=hashes)
    cache, record = verified_prediction_cache(*fixture_data, tmp_path / "clean", hashes)
    assert len(cache) == record["verified_persondays"] == 2
    daily, bases, keys = fixture_data
    with pytest.raises(ValueError, match="six-field"):
        verified_prediction_cache(daily.assign(steps=daily.steps + 1.), bases, keys, tmp_path / "clean", hashes)
    with pytest.raises(ValueError, match="source"):
        verified_prediction_cache(*fixture_data, tmp_path / "clean", {**hashes, "daily": "different"})


def test_same_experiment_cache_rejects_corrupt_output(tmp_path, fixture_data):
    hashes = {name: "fixture" for name in ("daily", "baselines", "keys")}
    infer(*fixture_data, tmp_path / "clean", engine=FixtureEngine(), source_hashes=hashes)
    (tmp_path / "clean" / "scores.csv.gz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="score checksum"):
        verified_prediction_cache(*fixture_data, tmp_path / "clean", hashes)


def test_synthetic_ready_marker_and_committed_input_hashes(tmp_path, fixture_data):
    folder = tmp_path / "synthetic" / "S1_seed00"
    assert comparison_moment.synthetic_job_paths(tmp_path, "S1_seed00") is None
    folder.mkdir(parents=True)
    daily, baselines, keys = fixture_data
    daily.to_csv(folder / "canonical_daily.csv.gz", index=False)
    keys.to_csv(folder / "moment_keys.csv.gz", index=False)
    baselines.to_csv(tmp_path / "seoyeon_baselines.csv", index=False)
    job = {"status": "passed", "artifact_hashes": {
        name: comparison_moment.sha256(folder / name)
        for name in ("canonical_daily.csv.gz", "moment_keys.csv.gz")},
        "input_hashes": {"seoyeon_baselines.csv": comparison_moment.sha256(tmp_path / "seoyeon_baselines.csv")}}
    (folder / "job.json").write_text(json.dumps(job))
    paths, hashes = comparison_moment.synthetic_job_paths(tmp_path, "S1_seed00")
    assert hashes["daily"] == job["artifact_hashes"]["canonical_daily.csv.gz"]
    assert paths["keys"] == folder / "moment_keys.csv.gz"
    (folder / "moment_keys.csv.gz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        comparison_moment.synthetic_job_paths(tmp_path, "S1_seed00")


def test_queue_consumes_exactly_100_jobs_after_clean_process_exits(tmp_path, fixture_data, monkeypatch):
    clean = tmp_path / "moment_clean"
    clean.mkdir()
    (clean / "execution.json").write_text('{"status":"complete"}')
    paths = {}
    for name, frame in zip(("daily", "baselines", "keys"), fixture_data):
        paths[name] = tmp_path / f"{name}.csv"
        frame.to_csv(paths[name], index=False)
    hashes = {name: comparison_moment.sha256(path) for name, path in paths.items()}
    engine = FixtureEngine()
    alive = iter([True, False])
    monkeypatch.setattr(comparison_moment, "_process_alive", lambda pid: next(alive))
    monkeypatch.setattr(comparison_moment.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(comparison_moment, "MomentEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(comparison_moment, "restore_current_clean_cache", lambda engine, run: {"verified": True})
    monkeypatch.setattr(comparison_moment, "synthetic_job_paths", lambda run, name: (paths, hashes))
    record = comparison_moment.watch_synthetic(tmp_path, tmp_path, wait_pid=42)
    assert record["status"] == "complete"
    assert len(record["completed_jobs"]) == 100
    assert record["completed_jobs"][0]["job"] == "S1_seed00"
    assert record["completed_jobs"][-1]["job"] == "S5_seed19"
    assert sum(item["new_inferred_persondays"] for item in record["completed_jobs"]) == 2
    assert engine.calls == 2


def test_restored_current_clean_cache_requires_fresh_numerical_replay(tmp_path, fixture_data):
    paths = {name: tmp_path / filename for name, filename in {
        "daily": "seoyeon_daily.csv.gz", "baselines": "seoyeon_baselines.csv",
        "keys": "moment_keys.csv.gz"}.items()}
    for (name, path), frame in zip(paths.items(), fixture_data):
        frame.to_csv(path, index=False)
    hashes = {name: comparison_moment.sha256(path) for name, path in paths.items()}
    infer(*fixture_data, tmp_path / "moment_clean", engine=FixtureEngine(), source_hashes=hashes)
    archive = tmp_path / "moment_code_archive" / "clean"
    archive.mkdir(parents=True)
    for name in ("comparison_moment.py", "comparison_moment_inputs.py"):
        shutil.copyfile(Path(comparison_moment.__file__).with_name(name), archive / name)
    fresh_engine = FixtureEngine()
    fresh_engine._load = lambda: None
    verified = comparison_moment.restore_current_clean_cache(fresh_engine, tmp_path)
    assert verified["verified_persondays"] == 2
    assert verified["fresh_replay_persondays"] == 2
    assert verified["fresh_replay_max_abs"] == 0.
    assert not verified["prior_historical_experiment_used"]
    assert len(fresh_engine.cache) == 2
    result = infer(*fixture_data, tmp_path / "next_process", engine=fresh_engine)
    assert result["cross_process_current_experiment_cache_persondays"] == 2
    assert result["same_process_cache_persondays"] == 0
    scored = pd.read_csv(tmp_path / "next_process" / "scores.csv.gz")
    assert scored.score_source.eq("same_experiment_cross_process_exact_six_field_cache").all()
    assert scored.source_inference_session_id.eq(verified["source_session_id"]).all()
    fresh_engine.predict = lambda arrays: np.ones((len(arrays["x"]), 3), np.float32)
    with pytest.raises(RuntimeError, match="replay"):
        comparison_moment.restore_current_clean_cache(fresh_engine, tmp_path)


@pytest.mark.parametrize("has_record", [False, True])
def test_queue_reports_exited_clean_producer_instead_of_waiting(tmp_path, monkeypatch, has_record):
    if has_record:
        (tmp_path / "moment_clean").mkdir()
        (tmp_path / "moment_clean" / "execution.json").write_text('{"status":"running"}')
    monkeypatch.setattr(comparison_moment, "_process_alive", lambda pid: False)
    monkeypatch.setattr(comparison_moment.time, "sleep", lambda seconds: pytest.fail("must not wait for exited producer"))
    with pytest.raises(RuntimeError, match="producer exited"):
        comparison_moment.watch_synthetic(tmp_path, tmp_path, wait_pid=42)
    records = list(tmp_path.glob("moment_queue_*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["status"] == "failed"


def test_statistical_failure_marker_is_not_treated_as_pending(tmp_path):
    folder = tmp_path / "synthetic" / "S1_seed00"
    folder.mkdir(parents=True)
    (folder / "job_failure.json").write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError, match="failure marker"):
        comparison_moment.synthetic_job_paths(tmp_path, "S1_seed00")
