"""Fail-closed recovery of interrupted chunks from this comparison run only.

No original files are written. Chunk checksums are first captured after the
interruption, not falsely described as pre-interruption integrity evidence.
The caller owns liveness checks, new output locations, and subsequent inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

import numpy as np
import pandas as pd

from .comparison_moment import (
    MODEL_ID, REVISION, WEIGHTS_SHA256, MomentEngine, _read_tables, sha256,
)
from .comparison_moment_inputs import (
    FEATURES, build_inputs, dated, endpoint_errors, row_digest, select_eligible,
)

CHUNK_SIZE = 1536
CODE_NAMES = ("comparison_moment.py", "comparison_moment_inputs.py")
# This archived producer used "same_run" exclusively for its own live cache.
LEGACY_SAME_RUN_SOURCE_SHA256 = "281eb9f7051517158d98c9891bd73a57c1c7c5fe192788c12265fa964da5f318"
SMOKE_LAUNCHER_SHA256 = "0f69cc37e32f972b1728bff406d7363a3c6ea9e1613c6e5bf6f90368b2190199"
RAW_NAMES = ("Calendar.csv", "FitbitActivity(1-30-20).csv", "FitbitSleep(1-30-20).csv")
MODEL_FIELDS = (
    "model_id", "revision", "checkpoint_sha256", "device", "precision", "parameters",
    "trainable_parameters", "strict_load_missing_keys", "strict_load_unexpected_keys",
    "training_mode", "seq_len", "patch_len", "seed", "revin_epsilon", "python", "versions",
)
KEYS = ["participant", "date", "split", "row_index"]


def _hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def _chunks(folder: Path, prefix: str, count: int) -> list[Path]:
    paths = sorted(folder.glob(f"{prefix}_*.csv.gz"))
    expected = [folder / f"{prefix}_{offset:07d}.csv.gz"
                for offset in range(0, count, CHUNK_SIZE)]
    if paths != expected:
        raise ValueError("Missing, extra or noncontiguous chunk offsets")
    return paths


def _raw_sources(run: Path) -> tuple[dict[str, Path], dict[str, str]]:
    repo = run.parents[2]
    recorded = json.loads((run / "inputs_before.json").read_text())
    names = [f"{repo.name}/data/nethealth/{name}" for name in RAW_NAMES]
    if not all(name in recorded for name in names):
        raise ValueError("Missing original raw input checksum provenance")
    paths = {name: repo.parent / name for name in names}
    actual = _hashes(paths)
    if actual != {name: recorded[name] for name in names}:
        raise ValueError("Raw input checksum differs")
    return paths, actual


def _equal_frame(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Partial cache {label} differ") from error


def _decode_float32_csv(stored: np.ndarray) -> np.ndarray:
    """Require the original writer's finite, shortest float32 decimal values.

    Pandas serializes float32 columns as short round-trip decimals, not the
    longer float64 expansion of each binary value. Compare that serialization
    exactly; accepting arbitrary values merely close to float32 is unsafe.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = stored.astype(np.float32)
    if not np.isfinite(stored).all() or not np.isfinite(predictions).all():
        raise ValueError("Partial reconstruction is not finite float32")
    serialized = np.array([float(str(value)) for value in predictions.flat],
                          dtype=np.float64).reshape(stored.shape)
    if not np.array_equal(stored, serialized):
        raise ValueError("Partial reconstruction differs from shortest float32 CSV decimals")
    return predictions


def _validate_chunk(tables: dict, targets: pd.DataFrame, scored: pd.DataFrame,
                    saved_audit: pd.DataFrame, cache: dict, session: str,
                    cache_label: str) -> dict[str, int]:
    scored = dated(scored)
    _equal_frame(scored[KEYS], targets[KEYS], "score keys")
    if not (scored.valid.eq(True).all() and scored.reason.eq("ok").all()
            and scored.model.eq("moment").all()):
        raise ValueError("Partial cache contains invalid or non-MOMENT rows")
    if ("source_inference_session_id" in scored
            and not scored.source_inference_session_id.eq(session).all()):
        raise ValueError("Partial cache source inference session differs")
    arrays, rebuilt_audit = build_inputs(tables["daily"], tables["baselines"], targets)
    digests = [row_digest(arrays, index) for index in range(len(targets))]
    rebuilt_audit = rebuilt_audit.assign(input_sha256=digests)
    for name in ("date", "baseline_end", "window_start", "window_end"):
        saved_audit[name] = pd.to_datetime(saved_audit[name], errors="raise")
    if set(saved_audit.columns) != set(rebuilt_audit.columns):
        raise ValueError("Partial cache audit schema differs")
    _equal_frame(saved_audit[rebuilt_audit.columns], rebuilt_audit, "all audit fields and six-input hashes")
    prediction_columns = [f"reconstruction_z_{feature}" for feature in FEATURES]
    error_columns = [f"error_z_{feature}" for feature in FEATURES]
    stored = scored[prediction_columns].to_numpy(np.float64)
    predictions = _decode_float32_csv(stored)
    errors, scores = endpoint_errors(arrays["target_raw"], arrays["centers"], arrays["scales"], predictions)
    if (not np.isfinite(scored[error_columns + ["score"]].to_numpy(float)).all()
            or not np.allclose(errors, scored[error_columns], rtol=1e-12, atol=1e-12)
            or not np.allclose(scores, scored.score, rtol=1e-12, atol=1e-12)):
        raise ValueError("Partial endpoint errors or mean MSE scores differ")
    counts = {"new_three_channel_inference": 0, cache_label: 0}
    for digest, prediction, source in zip(digests, predictions, scored.score_source):
        expected_source = cache_label if digest in cache else "new_three_channel_inference"
        if source != expected_source:
            raise ValueError("Partial score source disagrees with exact-input history")
        if digest in cache and not np.array_equal(cache[digest], prediction):
            raise ValueError("Identical partial inputs have different predictions")
        counts[source] += 1
        cache[digest] = prediction.copy()
    return counts


def _smoke_history(run: Path, tables: dict, clean: dict, cache_label: str) -> tuple[dict, dict, dict]:
    """Verify the pinned launcher's four earlier same-engine diagnostic rows."""
    folder = run / "moment_smoke"
    if not folder.exists():
        return {}, {"used": False}, {}
    runtime_path = run / "moment_runtime_recovery.json"
    runtime_before = sha256(runtime_path)
    runtime = json.loads(runtime_path.read_text())
    command = shlex.split(runtime["command"])
    if (len(command) != 2 or command[0] != runtime["python_executable"]
            or Path(command[1]).name != "run_clean.py" or not Path(command[1]).is_absolute()
            or runtime.get("script_sha256") != SMOKE_LAUNCHER_SHA256
            or sha256(Path(command[1])) != SMOKE_LAUNCHER_SHA256
            or runtime.get("input_sha256") != clean["input_source_sha256"]
            or runtime.get("past_prediction_cache_imported") is not False
            or runtime.get("new_inference") is not True):
        raise ValueError("Smoke launcher provenance differs")
    paths = {"smoke/runtime": runtime_path, "smoke/launcher": Path(command[1]),
             **{f"smoke/{name}": folder / name for name in (
                 "execution.json", "input_audit_0000000.csv.gz",
                 "score_chunk_0000000.csv.gz", "scores.csv.gz")}}
    before = _hashes(paths)
    if (before["smoke/runtime"] != runtime_before
            or before["smoke/launcher"] != SMOKE_LAUNCHER_SHA256):
        raise RuntimeError("Smoke launcher provenance changed before verification")
    record = json.loads(paths["smoke/execution.json"].read_text())
    required = {"status": "complete", "requested_persondays": 4, "eligible_persondays": 4,
                "completed_persondays": 4, "valid_persondays": 4, "unavailable_persondays": 0,
                "new_inferred_persondays": 4, "exact_six_field_cache_persondays": 0,
                "past_experiment_cache_persondays": 0, "actual_forward_requested_persondays": 4,
                "actual_forward_completed_persondays": 4}
    if (any(record.get(name) != value for name, value in required.items())
            or any(record.get(name) != clean[name] for name in (
                "session_id", "input_source_sha256", "source_sha256", "features", "model"))
            or record.get("cross_process_current_experiment_cache_persondays", 0) != 0
            or record.get("same_process_cache_persondays", 0) != 0
            or record.get("scores_sha256") != before["smoke/scores.csv.gz"]):
        raise ValueError("Smoke execution identity, counts or final-score checksum differs")
    times = [datetime.fromisoformat(value) for value in (
        record["started_utc"], record["completed_utc"], clean["started_utc"])]
    if any(value.tzinfo is None for value in times) or not times[0] <= times[1] <= times[2]:
        raise ValueError("Smoke execution is not before clean inference")
    targets = tables["keys"].iloc[np.linspace(0, len(tables["keys"]) - 1, 4, dtype=int)]
    targets = targets.reset_index(drop=True).assign(row_index=np.arange(4))
    eligible, rejected = select_eligible(tables["daily"], tables["baselines"], targets)
    if len(eligible) != 4 or len(rejected):
        raise ValueError("Smoke deterministic sample is not fully eligible")
    _chunks(folder, "score_chunk", 4)
    _chunks(folder, "input_audit", 4)
    frames = _read_tables({"score": paths["smoke/score_chunk_0000000.csv.gz"],
                          "final": paths["smoke/scores.csv.gz"],
                          "audit": paths["smoke/input_audit_0000000.csv.gz"]})
    _equal_frame(frames["score"], frames["final"], "smoke chunk and final scores")
    cache = {}
    counts = _validate_chunk(tables, eligible, frames["score"], frames["audit"],
                             cache, clean["session_id"], cache_label)
    if counts != {"new_three_channel_inference": 4, cache_label: 0}:
        raise ValueError("Smoke diagnostic rows are not four new inferences")
    if _hashes(paths) != before:
        raise RuntimeError("Smoke provenance changed during verification")
    return cache, {"used": True, "verified_persondays": 4,
        "source_session_id": clean["session_id"], "original_score_source_counts": counts,
        "selection": "Four np.linspace positions across the original requested keys",
        "artifact_sha256": before, "launcher_sha256": SMOKE_LAUNCHER_SHA256,
        "scope": "Earlier same-engine diagnostics establish cache history; only completed clean keys are imported."}, paths


def restore_partial_cache(run: Path, engine: MomentEngine) -> dict:
    """Verify partial original-run artifacts before atomically importing a cache.

Requires a fresh engine and a stopped producer (liveness checked by the caller).
All cached inputs are rebuilt; eight sampled endpoints are freshly inferred.
That sample does not establish all-row neural numerical equivalence.
"""
    run = Path(run).resolve()
    if (engine.model is not None or engine.cache or engine.imported_cache_hashes
            or engine.cache_origins or engine.cache_source_sessions
            or engine.forward_requested_persondays or engine.forward_completed_persondays):
        raise ValueError("Partial recovery requires a fresh engine")
    folder = run / "moment_clean"
    execution = folder / "execution.json"
    execution_before = sha256(execution)
    ledger_before = sha256(run / "inputs_before.json")
    record = json.loads(execution.read_text())
    count = record["completed_persondays"]
    if (record["status"] != "running" or not isinstance(count, int) or count <= 0
            or count % CHUNK_SIZE or count >= record["eligible_persondays"]
            or record.get("past_experiment_cache_persondays") != 0
            or record.get("cross_process_current_experiment_cache_persondays", 0) != 0
            or record.get("features") != list(FEATURES)
            or not isinstance(record.get("session_id"), str) or not record["session_id"]):
        raise ValueError("Only partial original-session complete chunks can be recovered")
    paths = {"daily": run / "seoyeon_daily.csv.gz", "baselines": run / "seoyeon_baselines.csv",
             "keys": run / "moment_keys.csv.gz"}
    before = _hashes(paths)
    if before != record["input_source_sha256"]:
        raise ValueError("Partial source input checksum differs")
    archive = run / "moment_code_archive" / "clean"
    code_paths = {name: archive / name for name in CODE_NAMES}
    code_before = _hashes(code_paths)
    if code_before != record["source_sha256"]:
        raise ValueError("Partial source archive checksum differs")
    cache_label = ("same_run_exact_six_field_cache"
                   if code_before["comparison_moment.py"] == LEGACY_SAME_RUN_SOURCE_SHA256
                   else "same_process_exact_six_field_cache")
    current_code_paths = {name: Path(__file__).with_name(name) for name in (*CODE_NAMES, Path(__file__).name)}
    current_code_before = _hashes(current_code_paths)
    raw_paths, raw_before = _raw_sources(run)
    audit_paths = _chunks(folder, "input_audit", count)
    score_paths = _chunks(folder, "score_chunk", count)
    artifacts = {path.name: path for path in audit_paths + score_paths}
    checkpoint_before = _hashes(artifacts)
    tables = _read_tables(paths)
    eligible, _ = select_eligible(**tables)
    if (len(tables["keys"]) != record["requested_persondays"]
            or len(eligible) != record["eligible_persondays"]):
        raise ValueError("Partial requested or eligible count differs")
    cache, smoke_proof, smoke_paths = _smoke_history(run, tables, record, cache_label)
    smoke_hashes = smoke_proof.get("artifact_sha256", {})
    smoke_digests = set(cache)
    completed_digests = {}
    smoke_reused = 0
    totals = {"new_three_channel_inference": 0, cache_label: 0}
    for offset, score_path, audit_path in zip(range(0, count, CHUNK_SIZE), score_paths, audit_paths):
        targets = eligible.iloc[offset:offset + CHUNK_SIZE]
        frames = _read_tables({"score": score_path, "audit": audit_path})
        if len(frames["score"]) != CHUNK_SIZE or len(frames["audit"]) != CHUNK_SIZE:
            raise ValueError("Partial chunk row count differs")
        counts = _validate_chunk(tables, targets, frames["score"], frames["audit"],
                                 cache, record["session_id"], cache_label)
        completed_digests.update(dict.fromkeys(frames["audit"].input_sha256))
        smoke_reused += int(frames["audit"].input_sha256.isin(smoke_digests).sum())
        totals = {name: totals[name] + counts[name] for name in totals}
    if (totals["new_three_channel_inference"] != record["new_inferred_persondays"]
            or totals[cache_label] != record["exact_six_field_cache_persondays"]
            or record.get("same_process_cache_persondays", totals[cache_label]) != totals[cache_label]
            or sum(totals.values()) != count):
        raise ValueError("Partial score source counts disagree with execution record")
    cache = {digest: cache[digest] for digest in completed_digests}
    if smoke_proof["used"]:
        smoke_proof = {**smoke_proof, "clean_prefix_reused_persondays": smoke_reused,
                       "outside_clean_prefix_not_imported": len(smoke_digests - cache.keys())}
    old_model = record["model"]
    if (any(name not in old_model for name in MODEL_FIELDS)
            or old_model["model_id"] != MODEL_ID or old_model["revision"] != REVISION
            or old_model["checkpoint_sha256"].get("model.safetensors") != WEIGHTS_SHA256
            or old_model["precision"] != "float32" or old_model["trainable_parameters"] != 0
            or old_model["training_mode"] is not False
            or old_model["strict_load_missing_keys"] or old_model["strict_load_unexpected_keys"]
            or old_model.get("controls", {}).get("pass") is not True):
        raise ValueError("Original model identity, frozen state or controls failed")
    engine._load()
    if any(engine.metadata.get(name) != old_model[name] for name in MODEL_FIELDS):
        raise ValueError("Recovery model/runtime differs from the original process")
    sample = eligible.iloc[np.linspace(0, count - 1, min(8, count), dtype=int)]
    arrays, _ = build_inputs(tables["daily"], tables["baselines"], sample)
    fresh = engine.predict(arrays)
    expected = np.stack([cache[row_digest(arrays, index)] for index in range(len(sample))])
    if (fresh.shape != expected.shape or not np.isfinite(fresh).all()
            or engine.metadata.get("controls", {}).get("pass") is not True
            or not np.allclose(fresh, expected, rtol=1e-5, atol=1e-5)):
        raise RuntimeError("Partial-cache actual-data neural replay or controls failed")
    difference = float(np.max(np.abs(fresh - expected)))
    if (_hashes(paths) != before or _hashes(code_paths) != code_before
            or _hashes(raw_paths) != raw_before or _hashes(artifacts) != checkpoint_before
            or _hashes(smoke_paths) != smoke_hashes
            or _hashes(current_code_paths) != current_code_before
            or sha256(run / "inputs_before.json") != ledger_before
            or sha256(execution) != execution_before
            or _chunks(folder, "input_audit", count) != audit_paths
            or _chunks(folder, "score_chunk", count) != score_paths):
        raise RuntimeError("Original inputs or checkpoint changed during recovery verification")
    proof = {
        "kind": "same_comparison_experiment_interrupted_clean_checkpoint",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verified_persondays": count, "remaining_persondays": len(eligible) - count,
        "requested_persondays": len(tables["keys"]), "unique_six_field_inputs": len(cache),
        "source_session_id": record["session_id"], "destination_session_id": engine.session_id,
        "current_run_directory": str(run), "source_directory": str(folder),
        "execution_sha256": execution_before, "original_status_preserved": record["status"],
        "source_sha256": before, "old_code_sha256": code_before, "raw_input_sha256": raw_before,
        "recovery_code_sha256": current_code_before, "inputs_before_ledger_sha256": ledger_before,
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_before,
        "checkpoint_hash_scope": "post_interruption_before_and_after_verification",
        "pre_interruption_chunk_hashes_available": False,
        "integrity_limit": "Chunk hashes were captured after interruption; they do not prove pre-interruption bytes.",
        "original_score_source_counts": totals, "source_cache_label": cache_label,
        "smoke_history": smoke_proof,
        "all_six_inputs_rebuilt": True,
        "all_audit_fields_rebuilt": True, "all_three_endpoint_errors_and_mse_recomputed": True,
        "reconstruction_csv_float32_serialization_verified": True,
        "reconstruction_csv_validation": "Every finite stored float64 equals float(str(float32(value))) exactly; this validates shortest float32 CSV decimal values, not float64 binary expansions.",
        "fresh_replay_persondays": len(sample),
        "fresh_replay_max_abs": difference, "fresh_replay_rtol_atol": 1e-5,
        "model_runtime_identical": True, "prior_historical_experiment_used": False,
        "original_model": old_model, "recovery_model": engine.metadata,
        "replay_scope": "Eight sampled actual rows freshly replayed; all completed inputs checked, not all-row fresh neural inference.",
    }
    engine.cache.update(cache)
    engine.imported_cache_hashes.update(cache)
    engine.cache_source_sessions.update({digest: record["session_id"] for digest in cache})
    engine.cache_origins.append(proof)
    return proof
