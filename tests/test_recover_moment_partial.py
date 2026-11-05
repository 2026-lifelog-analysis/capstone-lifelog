"""Synthetic recovery fixtures do not constitute real neural validation."""
import json
from io import StringIO
import shutil
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.seoyeon import comparison_moment as moment
from experiments.seoyeon import recover_moment_partial as recovery
from experiments.seoyeon.comparison_moment_inputs import FEATURES, build_inputs, endpoint_errors, select_eligible
from experiments.seoyeon.recover_moment_partial import restore_partial_cache


class FakeEngine(moment.MomentEngine):
    """Use deterministic endpoint scaling without any neural model."""

    def __init__(self):
        super().__init__(None, "cpu", 96)
        self.loaded = 0

    def _load(self):
        self.loaded += 1
        self.model = object()
        self.metadata = {
            "model_id": moment.MODEL_ID, "revision": moment.REVISION,
            "checkpoint_sha256": {"config.json": "fixture", "model.safetensors": moment.WEIGHTS_SHA256},
            "device": "cpu", "precision": "float32", "parameters": 341248520,
            "trainable_parameters": 0, "strict_load_missing_keys": [],
            "strict_load_unexpected_keys": [], "training_mode": False,
            "seq_len": 512, "patch_len": 8, "seed": 20261011,
            "revin_epsilon": 1e-5, "python": "fixture", "versions": {"torch": "fixture"},
        }

    def _forward(self, arrays):
        return arrays["x"][:, :, -1] * np.float32(.25)


class DecimalEngine(FakeEngine):
    """Exercise float32 CSV decimals that are not exact float64 expansions."""

    def _forward(self, arrays):
        return arrays["x"][:, :, -1] * np.float32(.12345679)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rewrite_csv(path, change):
    frame = pd.read_csv(path, dtype={"participant": str}, float_precision="round_trip")
    change(frame)
    frame.to_csv(path, index=False)


@pytest.fixture
def interrupted_run(tmp_path, request, monkeypatch):
    repo = tmp_path / "capstone-lifelog"
    run = repo / "results" / "seoyeon_tuning" / "run"
    run.mkdir(parents=True)
    dates = pd.date_range("2010-01-01", periods=1541)
    daily = pd.DataFrame({"participant": "fixture-person", "date": dates,
        "steps": np.arange(1541) + 1000., "sleep_minutes": 400.,
        "bedtime_minutes": 10., "complete": True})
    baselines = pd.DataFrame({"participant": ["fixture-person"], "model": ["fixed28"],
        "eligible": [True], "baseline_end": [dates[0]],
        **{f"center_{f}": [v] for f, v in zip(FEATURES, (1000., 390., 1430.))},
        **{f"scale_{f}": [v] for f, v in zip(FEATURES, (2., 5., 10.))}})
    keys = daily.iloc[1:][["participant", "date"]].assign(split="test")
    parameter = getattr(request, "param", False)
    if parameter is True:
        daily = pd.concat([daily, daily.assign(participant="fixture-other")], ignore_index=True)
        baselines = pd.concat([baselines, baselines.assign(participant="fixture-other")], ignore_index=True)
        keys = pd.concat([keys.iloc[:770], keys.iloc[:770].assign(participant="fixture-other")], ignore_index=True)
        keys = keys.sort_values(["date", "participant"]).reset_index(drop=True)
    paths = {"daily": run / "seoyeon_daily.csv.gz", "baselines": run / "seoyeon_baselines.csv",
             "keys": run / "moment_keys.csv.gz"}
    for name, frame in zip(paths, (daily, baselines, keys)):
        frame.to_csv(paths[name], index=False)
    inputs = {name: moment.sha256(path) for name, path in paths.items()}
    old = DecimalEngine() if parameter == "decimal" else FakeEngine()
    if parameter == "smoke":
        sample = keys.iloc[np.linspace(0, len(keys) - 1, 4, dtype=int)].reset_index(drop=True)
        moment.infer(daily, baselines, sample, run / "moment_smoke", engine=old, source_hashes=inputs)
        launcher = tmp_path / "run_clean.py"
        launcher.write_text("# Synthetic fixture: the same engine runs smoke before clean.\n")
        launcher_hash = moment.sha256(launcher)
        monkeypatch.setattr(recovery, "SMOKE_LAUNCHER_SHA256", launcher_hash, raising=False)
        executable = tmp_path / "venv/bin/python"
        write_json(run / "moment_runtime_recovery.json", {
            "script_sha256": launcher_hash, "input_sha256": inputs,
            "past_prediction_cache_imported": False, "new_inference": True,
            "python_executable": str(executable),
            "command": shlex.join([str(executable), str(launcher)])})
    record = moment.infer(daily, baselines, keys, run / "moment_clean", engine=old, source_hashes=inputs)
    for name in ("input_audit_0001536.csv.gz", "score_chunk_0001536.csv.gz", "scores.csv.gz"):
        (run / "moment_clean" / name).unlink()
    saved = pd.read_csv(run / "moment_clean/score_chunk_0000000.csv.gz")
    new_count = int(saved.score_source.eq("new_three_channel_inference").sum())
    cached_count = int(saved.score_source.eq("same_process_exact_six_field_cache").sum())
    record.update(status="running", completed_persondays=1536, new_inferred_persondays=new_count,
                  exact_six_field_cache_persondays=cached_count, cross_process_current_experiment_cache_persondays=0,
                  same_process_cache_persondays=cached_count)
    write_json(run / "moment_clean" / "execution.json", record)
    archive = run / "moment_code_archive" / "clean"
    archive.mkdir(parents=True)
    for name in record["source_sha256"]:
        shutil.copyfile(Path(moment.__file__).with_name(name), archive / name)
    rawdir = repo / "data" / "nethealth"
    rawdir.mkdir(parents=True)
    raw = {}
    for name in ("FitbitActivity(1-30-20).csv", "FitbitSleep(1-30-20).csv", "Calendar.csv"):
        path = rawdir / name
        path.write_text("synthetic fixture only\n")
        raw[str(path.relative_to(tmp_path))] = moment.sha256(path)
    write_json(run / "inputs_before.json", raw)
    return run


def snapshot(run):
    return {str(path.relative_to(run)): moment.sha256(path) for path in run.rglob("*") if path.is_file()}


def test_restores_only_verified_partial_cache_and_never_writes_original(interrupted_run):
    before = snapshot(interrupted_run)
    engine = FakeEngine()
    proof = restore_partial_cache(interrupted_run, engine)
    assert proof["verified_persondays"] == 1536
    assert proof["remaining_persondays"] == 4
    assert proof["fresh_replay_persondays"] == 8
    assert proof["checkpoint_hash_scope"] == "post_interruption_before_and_after_verification"
    assert proof["pre_interruption_chunk_hashes_available"] is False
    assert len(engine.cache) == len(engine.imported_cache_hashes) == 1536
    assert set(engine.cache_source_sessions.values()) == {proof["source_session_id"]}
    assert engine.cache_origins == [proof]
    assert before == snapshot(interrupted_run)
    assert engine.metadata["controls"]["pass"] is True


@pytest.mark.parametrize("interrupted_run", ["decimal"], indirect=True)
def test_float32_short_csv_decimals_restore_original_predictions(interrupted_run):
    before = snapshot(interrupted_run)
    scored = moment._read_tables({
        "score": interrupted_run / "moment_clean/score_chunk_0000000.csv.gz"})["score"]
    columns = [f"reconstruction_z_{feature}" for feature in FEATURES]
    stored = scored[columns].to_numpy(np.float64)
    assert not np.array_equal(stored, stored.astype(np.float32).astype(np.float64))
    engine = DecimalEngine()
    proof = restore_partial_cache(interrupted_run, engine)
    tables = moment._read_tables({"daily": interrupted_run / "seoyeon_daily.csv.gz",
        "baselines": interrupted_run / "seoyeon_baselines.csv",
        "keys": interrupted_run / "moment_keys.csv.gz"})
    eligible, _ = select_eligible(**tables)
    arrays, _ = build_inputs(tables["daily"], tables["baselines"], eligible.iloc[:1536])
    np.testing.assert_array_equal(np.stack(list(engine.cache.values())), engine._forward(arrays))
    assert proof["reconstruction_csv_float32_serialization_verified"] is True
    assert "reconstruction_float32_exact" not in proof
    assert proof["fresh_replay_max_abs"] == 0
    assert before == snapshot(interrupted_run)


def test_float32_csv_extremes_recover_exact_bits():
    original = np.array([.12345679, -.2345679, np.pi, 1e-9, 1.2345e9, 0., -0.,
        np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)
    csv = pd.DataFrame({"value": original}).to_csv(index=False)
    stored = pd.read_csv(StringIO(csv), float_precision="round_trip").to_numpy(np.float64)
    decoded = recovery._decode_float32_csv(stored).ravel()
    np.testing.assert_array_equal(decoded.view(np.uint32), original.view(np.uint32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e300, .123456789123,
    float(np.float32(.12345679))])
def test_noncanonical_float32_csv_values_fail_closed(value):
    with pytest.raises(ValueError):
        recovery._decode_float32_csv(np.array([[value]], dtype=np.float64))


@pytest.mark.parametrize("interrupted_run", [True], indirect=True)
def test_legacy_same_run_label_requires_pinned_source_and_preserves_counts(interrupted_run, monkeypatch):
    score = interrupted_run / "moment_clean/score_chunk_0000000.csv.gz"
    rewrite_csv(score, lambda frame: frame.__setitem__("score_source", frame.score_source.replace({
        "same_process_exact_six_field_cache": "same_run_exact_six_field_cache"})))
    source = interrupted_run / "moment_code_archive/clean/comparison_moment.py"
    monkeypatch.setattr(recovery, "LEGACY_SAME_RUN_SOURCE_SHA256", moment.sha256(source), raising=False)
    proof = restore_partial_cache(interrupted_run, FakeEngine())
    assert proof["original_score_source_counts"] == {
        "new_three_channel_inference": 768, "same_run_exact_six_field_cache": 768}
    assert proof["source_cache_label"] == "same_run_exact_six_field_cache"


@pytest.mark.parametrize("interrupted_run", [True], indirect=True)
def test_legacy_same_run_label_from_unpinned_source_rejected(interrupted_run):
    score = interrupted_run / "moment_clean/score_chunk_0000000.csv.gz"
    rewrite_csv(score, lambda frame: frame.__setitem__("score_source", frame.score_source.replace({
        "same_process_exact_six_field_cache": "same_run_exact_six_field_cache"})))
    with pytest.raises(ValueError, match="source disagrees"):
        restore_partial_cache(interrupted_run, FakeEngine())


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
def test_verified_same_engine_smoke_history_does_not_import_uncompleted_clean_key(interrupted_run):
    before = snapshot(interrupted_run)
    engine = FakeEngine()
    proof = restore_partial_cache(interrupted_run, engine)
    assert proof["original_score_source_counts"] == {
        "new_three_channel_inference": 1533, "same_process_exact_six_field_cache": 3}
    assert proof["smoke_history"]["verified_persondays"] == 4
    assert proof["smoke_history"]["clean_prefix_reused_persondays"] == 3
    assert proof["smoke_history"]["outside_clean_prefix_not_imported"] == 1
    assert len(engine.cache) == proof["verified_persondays"] == 1536
    assert proof["remaining_persondays"] == 4
    assert snapshot(interrupted_run) == before


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
@pytest.mark.parametrize("field,value", [("session_id", "different"), ("status", "running"),
    ("model", {}), ("input_source_sha256", {}), ("source_sha256", {}),
    ("scores_sha256", "bad"), ("new_inferred_persondays", 3),
    ("completed_utc", "2100-01-01T00:00:00+00:00"),
    ("started_utc", "2010-01-01T00:00:00"),
    ("cross_process_current_experiment_cache_persondays", 1)])
def test_smoke_execution_corruption_fails_closed(interrupted_run, field, value):
    path = interrupted_run / "moment_smoke/execution.json"
    record = json.loads(path.read_text())
    write_json(path, {**record, field: value})
    engine = FakeEngine()
    with pytest.raises(ValueError):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache and not engine.imported_cache_hashes


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
@pytest.mark.parametrize("field,value", [("script_sha256", "bad"), ("input_sha256", {}),
    ("past_prediction_cache_imported", True), ("new_inference", False),
    ("command", "unexpected")])
def test_smoke_launcher_provenance_corruption_fails_closed(interrupted_run, field, value):
    path = interrupted_run / "moment_runtime_recovery.json"
    record = json.loads(path.read_text())
    write_json(path, {**record, field: value})
    with pytest.raises(ValueError, match="provenance"):
        restore_partial_cache(interrupted_run, FakeEngine())


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
def test_smoke_input_hash_corruption_fails_closed(interrupted_run):
    rewrite_csv(interrupted_run / "moment_smoke/input_audit_0000000.csv.gz",
                lambda frame: frame.loc.__setitem__((0, "input_sha256"), "bad"))
    with pytest.raises(ValueError, match="audit fields"):
        restore_partial_cache(interrupted_run, FakeEngine())


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
def test_smoke_prediction_corruption_fails_closed(interrupted_run):
    rewrite_csv(interrupted_run / "moment_smoke/score_chunk_0000000.csv.gz",
                lambda frame: frame.loc.__setitem__((0, "reconstruction_z_steps"), 99.))
    with pytest.raises(ValueError, match="smoke chunk"):
        restore_partial_cache(interrupted_run, FakeEngine())


@pytest.mark.parametrize("interrupted_run", ["smoke"], indirect=True)
def test_smoke_provenance_mutation_during_replay_rejected(interrupted_run):
    class MutatingEngine(FakeEngine):
        def _load(self):
            super()._load()
            path = interrupted_run / "moment_smoke/execution.json"
            path.write_text(path.read_text() + " ")
    engine = MutatingEngine()
    with pytest.raises(RuntimeError, match="changed"):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


@pytest.mark.parametrize("field,value", [
    ("score", 99.), ("error_z_steps", 99.), ("reconstruction_z_steps", np.nan),
    ("reconstruction_z_steps", .123456789123), ("valid", False), ("reason", "bad"),
    ("row_index", 100), ("participant", "other"), ("split", "calibration"),
    ("model", "other"), ("score_source", "historical"),
    ("source_inference_session_id", "other-session"),
])
def test_score_corruption_fails_closed(interrupted_run, field, value):
    rewrite_csv(interrupted_run / "moment_clean" / "score_chunk_0000000.csv.gz",
                lambda frame: frame.loc.__setitem__((0, field), value))
    engine = FakeEngine()
    with pytest.raises((ValueError, RuntimeError)):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache and not engine.imported_cache_hashes


@pytest.mark.parametrize("field,value", [("input_sha256", "bad"), ("row_index", 99),
    ("participant", "other"), ("observed_complete_days", 9999), ("no_future_input", False)])
def test_audit_corruption_fails_closed(interrupted_run, field, value):
    rewrite_csv(interrupted_run / "moment_clean" / "input_audit_0000000.csv.gz",
                lambda frame: frame.loc.__setitem__((0, field), value))
    engine = FakeEngine()
    with pytest.raises((ValueError, RuntimeError)):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


@pytest.mark.parametrize("field,value", [("status", "complete"), ("completed_persondays", 1535),
    ("new_inferred_persondays", 1), ("requested_persondays", 1), ("eligible_persondays", 1),
    ("past_experiment_cache_persondays", 1), ("exact_six_field_cache_persondays", 1)])
def test_progress_mismatch_fails_closed(interrupted_run, field, value):
    path = interrupted_run / "moment_clean" / "execution.json"
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(ValueError):
        restore_partial_cache(interrupted_run, FakeEngine())


@pytest.mark.parametrize("name", ["seoyeon_baselines.csv", "seoyeon_daily.csv.gz", "moment_keys.csv.gz",
    "moment_code_archive/clean/comparison_moment.py", "moment_code_archive/clean/comparison_moment_inputs.py"])
def test_input_or_archive_checksum_change_rejected(interrupted_run, name):
    path = interrupted_run / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_raw_checksum_change_rejected(interrupted_run):
    raw = interrupted_run.parents[2] / "data/nethealth/Calendar.csv"
    raw.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_chunk_offset_or_missing_partner_rejected(interrupted_run):
    path = interrupted_run / "moment_clean" / "score_chunk_0000000.csv.gz"
    path.rename(path.with_name("score_chunk_0001536.csv.gz"))
    with pytest.raises(ValueError, match="chunk"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_runtime_change_rejected(interrupted_run):
    class ChangedEngine(FakeEngine):
        def _load(self):
            super()._load()
            self.metadata["precision"] = "float16"
    engine = ChangedEngine()
    with pytest.raises(ValueError, match="runtime"):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


def test_neural_replay_mismatch_rejected(interrupted_run):
    class ChangedEngine(FakeEngine):
        def _forward(self, arrays):
            return super()._forward(arrays) + np.float32(1.)
    engine = ChangedEngine()
    with pytest.raises(RuntimeError, match="replay"):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


def test_nonempty_engine_rejected(interrupted_run):
    engine = FakeEngine()
    engine.cache["untrusted"] = np.zeros(3, np.float32)
    with pytest.raises(ValueError, match="fresh"):
        restore_partial_cache(interrupted_run, engine)


def test_post_verification_mutation_rejected(interrupted_run):
    class MutatingEngine(FakeEngine):
        def _load(self):
            super()._load()
            path = interrupted_run / "moment_clean/execution.json"
            path.write_text(path.read_text() + " ")
    engine = MutatingEngine()
    with pytest.raises(RuntimeError, match="changed"):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


def test_original_schema_without_per_row_session_is_verified(interrupted_run):
    path = interrupted_run / "moment_clean/score_chunk_0000000.csv.gz"
    rewrite_csv(path, lambda frame: frame.drop(columns="source_inference_session_id", inplace=True))
    engine = FakeEngine()
    proof = restore_partial_cache(interrupted_run, engine)
    assert len(engine.cache) == 1536
    assert set(engine.cache_source_sessions.values()) == {proof["source_session_id"]}


@pytest.mark.parametrize("interrupted_run", [True], indirect=True)
def test_duplicate_inputs_keep_original_inference_count_and_one_prediction(interrupted_run):
    engine = FakeEngine()
    proof = restore_partial_cache(interrupted_run, engine)
    assert proof["verified_persondays"] == 1536
    assert proof["unique_six_field_inputs"] == 768
    assert proof["original_score_source_counts"] == {
        "new_three_channel_inference": 768, "same_process_exact_six_field_cache": 768}


@pytest.mark.parametrize("interrupted_run", [True], indirect=True)
def test_duplicate_input_with_conflicting_prediction_rejected(interrupted_run):
    tables = moment._read_tables({"daily": interrupted_run / "seoyeon_daily.csv.gz",
        "baselines": interrupted_run / "seoyeon_baselines.csv", "keys": interrupted_run / "moment_keys.csv.gz"})
    eligible, _ = select_eligible(**tables)
    arrays, _ = build_inputs(tables["daily"], tables["baselines"], eligible.iloc[:1536])
    def corrupt(frame):
        columns = [f"reconstruction_z_{feature}" for feature in FEATURES]
        predictions = frame[columns].to_numpy(np.float32)
        predictions[1, 0] += np.float32(1.)
        errors, scores = endpoint_errors(arrays["target_raw"], arrays["centers"], arrays["scales"], predictions)
        frame[columns] = predictions.astype(np.float64)
        frame[[f"error_z_{feature}" for feature in FEATURES]] = errors
        frame["score"] = scores
    rewrite_csv(interrupted_run / "moment_clean/score_chunk_0000000.csv.gz", corrupt)
    engine = FakeEngine()
    with pytest.raises(ValueError, match="different predictions"):
        restore_partial_cache(interrupted_run, engine)
    assert not engine.cache


def test_failed_old_model_controls_rejected(interrupted_run):
    path = interrupted_run / "moment_clean/execution.json"
    record = json.loads(path.read_text())
    record["model"]["controls"]["pass"] = False
    write_json(path, record)
    with pytest.raises(ValueError, match="controls"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_missing_raw_provenance_rejected(interrupted_run):
    write_json(interrupted_run / "inputs_before.json", {})
    with pytest.raises(ValueError, match="provenance"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_missing_audit_column_rejected(interrupted_run):
    rewrite_csv(interrupted_run / "moment_clean/input_audit_0000000.csv.gz",
                lambda frame: frame.drop(columns="no_future_input", inplace=True))
    with pytest.raises(ValueError, match="schema"):
        restore_partial_cache(interrupted_run, FakeEngine())


def test_truncated_chunk_rejected(interrupted_run):
    rewrite_csv(interrupted_run / "moment_clean/score_chunk_0000000.csv.gz",
                lambda frame: frame.drop(index=1535, inplace=True))
    with pytest.raises(ValueError, match="row count"):
        restore_partial_cache(interrupted_run, FakeEngine())
