"""Local frozen MOMENT inference for new common-preprocessing comparisons.

Run with the separate earlier MOMENT environment. No network download, previous
prediction import, model training, or silent device substitution is performed.
The optional multi-job CLI keeps one model and an exact-input cache in memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

if __package__:
    from .comparison_moment_inputs import (
        FEATURES, build_inputs, endpoint_errors, row_digest, select_eligible,
        verified_prediction_cache,
    )
else:
    from comparison_moment_inputs import (
        FEATURES, build_inputs, endpoint_errors, row_digest, select_eligible,
        verified_prediction_cache,
    )

MODEL_ID = "AutonLab/MOMENT-1-large"
REVISION = "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc"
WEIGHTS_SHA256 = "a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572"


def sha256(path: Path) -> str:
    """Read files in bounded memory, including the large checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_progress_record(path: Path) -> dict:
    """Allow a bounded retry for a producer currently replacing JSON content."""
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)
    raise RuntimeError("Unreachable JSON read state")


class MomentEngine:
    """One immutable model configuration and process-local exact-input cache."""

    def __init__(self, model_root: Path | None, device: str = "mps", batch_size: int = 96):
        if device not in {"cpu", "mps"} or batch_size < 1:
            raise ValueError("Unsupported device or nonpositive batch size")
        self.model_root = Path(model_root) if model_root is not None else None
        self.device, self.batch_size = device, batch_size
        self.model = None
        self.cache: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}
        self.session_id = str(uuid.uuid4())
        self.load_seconds = 0.
        self.forward_requested_persondays = 0
        self.forward_completed_persondays = 0
        self.cache_origins: list[dict] = []
        self.cache_source_sessions: dict[str, str] = {}
        self.imported_cache_hashes: set[str] = set()

    def _load(self) -> None:
        if self.model_root is None:
            raise ValueError("A local model root is required for real inference")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from momentfm import MOMENTPipeline
        from safetensors.torch import load_file
        started = time.perf_counter()
        checkpoint = self.model_root / "00_원자료" / "moment_checkpoint"
        provenance = json.loads((self.model_root / "03_처리기록" / "moment_checkpoint.json").read_text())
        if provenance.get("model_id") != MODEL_ID or provenance.get("revision") != REVISION:
            raise ValueError("Unexpected checkpoint identity")
        hashes = {name: sha256(checkpoint / name) for name in ("config.json", "model.safetensors")}
        if (hashes["model.safetensors"] != WEIGHTS_SHA256 or
                any(hashes[name] != provenance["files"][name]["sha256"] for name in hashes)):
            raise ValueError("Checkpoint checksum mismatch")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no automatic device substitution")
        torch.manual_seed(20261011)
        torch.set_num_threads(6)
        config = json.loads((checkpoint / "config.json").read_text())
        model = MOMENTPipeline(config=config, model_kwargs={"task_name": "reconstruction"})
        loaded = model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
        model.init()
        model.requires_grad_(False)
        model.eval()
        model.to(self.device)
        self.model = model
        self.load_seconds = time.perf_counter() - started
        self.metadata = {"model_id": MODEL_ID, "revision": REVISION,
            "checkpoint_sha256": hashes, "device": self.device, "precision": "float32",
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "strict_load_missing_keys": list(loaded.missing_keys),
            "strict_load_unexpected_keys": list(loaded.unexpected_keys),
            "training_mode": model.training, "seq_len": model.seq_len, "patch_len": model.patch_len,
            "seed": 20261011, "revin_epsilon": model.normalizer.eps,
            "python": platform.python_version(),
            "versions": {name: metadata.version(name) for name in
                         ("torch", "momentfm", "numpy", "pandas", "transformers", "safetensors")}}

    def _forward(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        import torch
        with torch.inference_mode():
            result = self.model(
                x_enc=torch.from_numpy(arrays["x"]).to(self.device),
                input_mask=torch.from_numpy(arrays["input_mask"].astype(np.int64)).to(self.device),
                mask=torch.from_numpy(arrays["observation_mask"].astype(np.int64)).to(self.device))
            return result.reconstruction[:, :, -1].detach().cpu().numpy()

    def predict(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        """Actually forward new inputs; controls are separate from scored rows."""
        if self.model is None:
            self._load()
        if "controls" not in self.metadata:
            selected = {name: value[:4] for name, value in arrays.items()}
            first, second = self._forward(selected), self._forward(selected)
            changed = {**selected, "x": np.where(selected["observation_mask"][:, None, :] == 1,
                                                selected["x"], np.float32(123.456))}
            placeholder = self._forward(changed)
            check = {"repeat_max_abs": float(np.max(np.abs(first - second))),
                     "masked_placeholder_max_abs": float(np.max(np.abs(first - placeholder))),
                     "diagnostic_forward_persondays": 3 * len(first), "rtol": 1e-5, "atol": 1e-5}
            check["pass"] = bool(np.allclose(first, second, rtol=1e-5, atol=1e-5)
                                 and np.allclose(first, placeholder, rtol=1e-5, atol=1e-5))
            self.metadata["controls"] = check
            if not check["pass"]:
                raise RuntimeError("Actual-data repeat/masked-placeholder control failed")
        return self._forward(arrays)

    def reconstruct(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], list[str], int]:
        """Only exact six-field hashes from this engine may reuse a prediction."""
        hashes = [row_digest(arrays, i) for i in range(len(arrays["x"]))]
        first_unseen = {}
        sources = []
        for index, digest in enumerate(hashes):
            if digest in self.imported_cache_hashes:
                sources.append("same_experiment_cross_process_exact_six_field_cache")
            elif digest in self.cache or digest in first_unseen:
                sources.append("same_process_exact_six_field_cache")
            else:
                first_unseen[digest] = index
                sources.append("new_three_channel_inference")
        needed = list(first_unseen.values())
        for offset in range(0, len(needed), self.batch_size):
            ids = needed[offset:offset + self.batch_size]
            self.forward_requested_persondays += len(ids)
            prediction = self.predict({name: value[ids] for name, value in arrays.items()})
            if prediction.shape != (len(ids), 3) or not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite or wrongly shaped MOMENT output")
            self.forward_completed_persondays += len(ids)
            self.cache.update({hashes[i]: pred.copy() for i, pred in zip(ids, prediction)})
            self.cache_source_sessions.update({hashes[i]: self.session_id for i in ids})
        return np.stack([self.cache[digest] for digest in hashes]), sources, hashes, len(needed)


def infer(daily: pd.DataFrame, baselines: pd.DataFrame, keys: pd.DataFrame,
          outputdir: Path, *, engine: MomentEngine | None = None,
          model_root: Path | None = None, device: str = "mps", batch_size: int = 96,
          source_hashes: dict | None = None) -> dict:
    """Score requested keys, preserving unavailable rows and immutable inputs."""
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    engine = engine or MomentEngine(model_root, device, batch_size)
    forward_before = (engine.forward_requested_persondays, engine.forward_completed_persondays)
    keys = keys.reset_index(drop=True).assign(row_index=np.arange(len(keys)))
    eligible, rejected = select_eligible(daily, baselines, keys)
    scored = keys.assign(model="moment", score=np.nan, valid=False, reason="not_executed",
                         score_source="unavailable", source_inference_session_id="unavailable")
    scored.loc[rejected.row_index, "reason"] = rejected.reason.to_numpy()
    new_count, cached_count, completed, imported_count = 0, 0, 0, 0
    audits = []
    record = {"status": "running", "session_id": engine.session_id,
        "started_utc": datetime.now(timezone.utc).isoformat(), "requested_persondays": len(keys),
        "eligible_persondays": len(eligible), "features": list(FEATURES),
        "input_source_sha256": source_hashes or {}, "past_experiment_cache_persondays": 0,
        "same_experiment_seed_cache": engine.cache_origins,
        "normalization": "Frozen personal fixed28; three channels, 512 trailing calendar days",
        "mask_policy": "Calendar padding and complete-three-feature observation masks kept distinct",
        "score": "Mean squared standardized endpoint error; circular bedtime inverse residual",
        "cache_policy": "Exact six-field cache; distinguish same-process from independently verified current-experiment cross-process reuse",
        "execution_only_protocol_extension": "Cross-process current-experiment cache is explicitly distinguished; original same-engine protocol is not rewritten",
        "source_sha256": {Path(__file__).name: sha256(Path(__file__)),
            "comparison_moment_inputs.py": sha256(Path(__file__).with_name("comparison_moment_inputs.py"))}}
    write_record(outputdir / "execution.json", record)
    try:
        for start in range(0, len(eligible), 1536):
            targets = eligible.iloc[start:start + 1536]
            arrays, audit = build_inputs(daily, baselines, targets)
            predictions, sources, hashes, newly_inferred = engine.reconstruct(arrays)
            errors, scores = endpoint_errors(arrays["target_raw"], arrays["centers"], arrays["scales"], predictions)
            ids = targets.row_index.to_numpy()
            scored.loc[ids, "score"] = scores
            scored.loc[ids, "valid"] = np.isfinite(scores)
            scored.loc[ids, "reason"] = np.where(np.isfinite(scores), "ok", "nonfinite_score")
            scored.loc[ids, "score_source"] = sources
            scored.loc[ids, "source_inference_session_id"] = [engine.cache_source_sessions[digest] for digest in hashes]
            for j, feature in enumerate(FEATURES):
                scored.loc[ids, f"reconstruction_z_{feature}"] = predictions[:, j]
                scored.loc[ids, f"error_z_{feature}"] = errors[:, j]
            audit = audit.assign(input_sha256=hashes)
            audits.append(audit)
            audit.to_csv(outputdir / f"input_audit_{start:07d}.csv.gz", index=False)
            scored.loc[ids].to_csv(outputdir / f"score_chunk_{start:07d}.csv.gz", index=False)
            new_count += newly_inferred
            cached_count += len(targets) - newly_inferred
            imported_count += sources.count("same_experiment_cross_process_exact_six_field_cache")
            completed += len(targets)
            progress = {"completed_persondays": completed, "new_inferred_persondays": new_count,
                        "exact_six_field_cache_persondays": cached_count,
                        "cross_process_current_experiment_cache_persondays": imported_count,
                        "same_process_cache_persondays": cached_count - imported_count,
                        "elapsed_seconds": time.perf_counter() - started}
            write_record(outputdir / "execution.json", {**record, **progress, "model": engine.metadata})
            print(json.dumps({"job": outputdir.name, **progress}), flush=True)
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        scored.to_csv(outputdir / "scores.csv.gz", index=False)
        record.update(model=engine.metadata, new_inferred_persondays=new_count,
                      exact_six_field_cache_persondays=cached_count, completed_persondays=completed,
                      cross_process_current_experiment_cache_persondays=imported_count,
                      same_process_cache_persondays=cached_count - imported_count,
                      valid_persondays=int(scored.valid.sum()), unavailable_persondays=int((~scored.valid).sum()),
                      elapsed_seconds=time.perf_counter() - started, model_load_seconds=engine.load_seconds,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      scores_sha256=sha256(outputdir / "scores.csv.gz"))
        record["actual_forward_requested_persondays"] = engine.forward_requested_persondays - forward_before[0]
        record["actual_forward_completed_persondays"] = engine.forward_completed_persondays - forward_before[1]
        if audits:
            audit_all = pd.concat(audits, ignore_index=True)
            record["no_complete_patch_persondays"] = int(audit_all.no_complete_patch.sum())
            record["last_patch_incomplete_persondays"] = int((~audit_all.last_patch_complete).sum())
        write_record(outputdir / "execution.json", record)
    return record


def synthetic_job_paths(run: Path, name: str) -> tuple[dict, dict] | None:
    """Accept only fully committed statistical jobs and check both input files."""
    folder = Path(run) / "synthetic" / name
    if (folder / "job_failure.json").exists():
        raise RuntimeError(f"Statistical failure marker exists: {name}")
    marker = folder / "job.json"
    if not marker.exists():
        return None
    job = read_progress_record(marker)
    if job.get("status") != "passed":
        raise ValueError(f"Statistical job is not passed: {name}")
    paths = {"daily": folder / "canonical_daily.csv.gz",
             "keys": folder / "moment_keys.csv.gz", "baselines": Path(run) / "seoyeon_baselines.csv"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    expected = {"daily": job["artifact_hashes"]["canonical_daily.csv.gz"],
                "keys": job["artifact_hashes"]["moment_keys.csv.gz"],
                "baselines": job["input_hashes"]["seoyeon_baselines.csv"]}
    if hashes != expected:
        raise ValueError(f"Synthetic committed input checksum differs: {name}")
    return paths, hashes


def _process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _read_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path, dtype={"participant": str}, float_precision="round_trip")
            for name, path in paths.items()}


def restore_current_clean_cache(engine: MomentEngine, run: Path) -> dict:
    """Verify newly inferred clean data and numerical replay before reuse."""
    paths = {"daily": run / "seoyeon_daily.csv.gz", "baselines": run / "seoyeon_baselines.csv",
             "keys": run / "moment_keys.csv.gz"}
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    tables = _read_tables(paths)
    cache, checked = verified_prediction_cache(**tables, cache_dir=run / "moment_clean",
                                                source_hashes=source_hashes)
    old_code = run / "moment_code_archive" / "clean"
    if any(sha256(old_code / name) != digest for name, digest in checked["old_code_sha256"].items()):
        raise ValueError("Clean inference source archive checksum differs")
    engine._load()
    fields = ("model_id", "revision", "checkpoint_sha256", "device", "precision", "parameters",
              "trainable_parameters", "training_mode", "seq_len", "patch_len", "seed",
              "revin_epsilon", "python", "versions")
    if any(engine.metadata.get(key) != checked["old_model"].get(key) for key in fields):
        raise ValueError("New process differs from this experiment's clean model/runtime")
    eligible, _ = select_eligible(**tables)
    sample = eligible.iloc[np.linspace(0, len(eligible) - 1, min(8, len(eligible)), dtype=int)]
    arrays, _ = build_inputs(tables["daily"], tables["baselines"], sample)
    fresh = engine.predict(arrays)
    expected = np.stack([cache[row_digest(arrays, i)] for i in range(len(sample))])
    replay_difference = float(np.max(np.abs(fresh - expected)))
    if not np.allclose(fresh, expected, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Actual-data clean cache replay failed")
    engine.cache.update(cache)
    engine.imported_cache_hashes.update(cache)
    engine.cache_source_sessions.update({digest: checked["source_session_id"] for digest in cache})
    origin = {"kind": "same_comparison_experiment_new_clean_predictions",
              **checked, "fresh_replay_persondays": len(sample), "fresh_replay_max_abs": replay_difference,
              "fresh_replay_rtol_atol": 1e-5, "model_runtime_identical": True,
              "source_sha256": source_hashes, "prior_historical_experiment_used": False,
              "source_session_id": checked["source_session_id"], "destination_session_id": engine.session_id,
              "process_changed": checked["source_session_id"] != engine.session_id,
              "current_run_directory": str(run.resolve()), "source_directory": str((run / "moment_clean").resolve()),
              "source_code_archive": str(old_code),
              "replay_scope": "Only eight representative actual rows freshly replayed; all eligible rows have exact six-field hash checks, not all-row fresh neural replay"}
    engine.cache_origins.append(origin)
    return origin


def watch_synthetic(run: Path, model_root: Path, *, device: str = "mps", batch_size: int = 96,
                    wait_pid: int | None = None, poll_seconds: float = 5.) -> dict:
    """Wait for clean completion, then consume S1-S5 x seeds 0-19 exactly once.

    The first process is allowed to finish naturally. The second loads no neural
    model until that process exits, avoiding concurrent MPS models.
    """
    if poll_seconds <= 0 or poll_seconds > 60:
        raise ValueError("Polling interval must be in (0, 60]")
    run = Path(run)
    engine = MomentEngine(model_root, device=device, batch_size=batch_size)
    state_path = run / f"moment_queue_{engine.session_id}.json"
    state = {"status": "waiting_for_clean", "session_id": engine.session_id,
             "expected_jobs": 100, "completed_jobs": [], "wait_pid": wait_pid,
             "batch_size": batch_size, "device": device,
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "runner_code_sha256": sha256(Path(__file__))}
    write_record(state_path, state)
    try:
        clean_record = run / "moment_clean" / "execution.json"
        while True:
            clean = read_progress_record(clean_record) if clean_record.exists() else {}
            if clean.get("status") == "failed":
                raise RuntimeError("Clean inference failed; no synthetic inference started")
            producer_alive = _process_alive(wait_pid)
            if clean.get("status") == "complete" and not producer_alive:
                break
            if wait_pid is not None and not producer_alive:
                raise RuntimeError("Clean producer exited without a complete inference record")
            time.sleep(poll_seconds)
        state["cache_verification"] = restore_current_clean_cache(engine, run)
        state["status"] = "running"
        write_record(state_path, state)
        for seed in range(20):
            for scenario in range(1, 6):
                name = f"S{scenario}_seed{seed:02d}"
                state["waiting_or_running_job"] = name
                write_record(state_path, state)
                resolved = synthetic_job_paths(run, name)
                while resolved is None:
                    time.sleep(poll_seconds)
                    resolved = synthetic_job_paths(run, name)
                paths, hashes = resolved
                output = run / "synthetic" / name / "moment"
                record = infer(**_read_tables(paths), outputdir=output, engine=engine, source_hashes=hashes)
                if hashes != {key: sha256(path) for key, path in paths.items()}:
                    raise RuntimeError("Synthetic input changed during inference")
                write_record(output / "source_preservation.json", {"unchanged": True, "sha256": hashes})
                state["completed_jobs"].append({"job": name, "status": record["status"],
                    "requested_persondays": record["requested_persondays"],
                    "new_inferred_persondays": record["new_inferred_persondays"],
                    "exact_six_field_cache_persondays": record["exact_six_field_cache_persondays"],
                    "execution_sha256": sha256(output / "execution.json")})
                write_record(state_path, state)
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_record(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("daily", "baselines", "keys", "output", "jobs", "model-root", "watch-run"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if not args.model_root:
        parser.error("--model-root is required")
    if args.watch_run:
        watch_synthetic(args.watch_run, args.model_root, device=args.device, batch_size=args.batch_size,
                        wait_pid=args.wait_pid)
        return
    if args.jobs:
        jobs = json.loads(args.jobs.read_text())
    elif all(getattr(args, name) for name in ("daily", "baselines", "keys", "output")):
        jobs = [{name: str(getattr(args, name)) for name in ("daily", "baselines", "keys", "output")}]
    else:
        parser.error("Specify --jobs or all of --daily --baselines --keys --output")
    engine = MomentEngine(args.model_root, args.device, args.batch_size)
    for job in jobs:
        paths = {name: Path(job[name]) for name in ("daily", "baselines", "keys")}
        before = {name: sha256(path) for name, path in paths.items()}
        tables = {name: pd.read_csv(path, dtype={"participant": str}, float_precision="round_trip")
                  for name, path in paths.items()}
        record = infer(**tables, outputdir=Path(job["output"]), engine=engine, source_hashes=before)
        after = {name: sha256(path) for name, path in paths.items()}
        if before != after:
            raise RuntimeError("Input file changed during inference")
        write_record(Path(job["output"]) / "source_preservation.json", {"unchanged": True, "sha256": after})
        print(json.dumps({key: record[key] for key in
                          ("status", "requested_persondays", "new_inferred_persondays",
                           "exact_six_field_cache_persondays", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
