"""Calendar inputs matching the earlier frozen three-channel MOMENT protocol.

This adapter changes the input cohort, not the prior model's 512-calendar-position
mask, fixed-28 normalization or circular endpoint reconstruction definitions.
No historical predictions are imported here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

FEATURES = ("steps", "sleep_minutes", "bedtime_minutes")
CACHE_FIELDS = ("x", "input_mask", "observation_mask", "centers", "scales", "target_raw")


def dated(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical date and participant types without changing the caller."""
    result = frame.assign(participant=frame.participant.astype(str),
                          date=pd.to_datetime(frame.date, errors="raise"))
    if result.date.isna().any() or not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Missing or non-midnight date")
    return result


def _checked(daily: pd.DataFrame, baselines: pd.DataFrame,
             keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily, keys = dated(daily), dated(keys).reset_index(drop=True)
    if daily.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate daily participant-date")
    if keys.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate target participant-date")
    if "row_index" not in keys:
        keys = keys.assign(row_index=np.arange(len(keys), dtype=np.int64))
    if "split" not in keys:
        keys = keys.assign(split="unspecified")
    if keys.row_index.duplicated().any():
        raise ValueError("Duplicate target row_index")
    values = daily.loc[daily.complete.eq(True), list(FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Complete daily rows contain nonfinite features")
    baselines = baselines.assign(participant=baselines.participant.astype(str))
    selected = baselines.loc[baselines.model.eq("fixed28")]
    if selected.participant.duplicated().any():
        raise ValueError("Duplicate fixed28 baseline")
    return daily, selected, keys


def _baseline_reason(row: pd.Series | None, first_target: pd.Timestamp) -> str:
    if row is None or row.eligible != True:
        return "missing_or_ineligible_fixed28_baseline"
    values = row[[f"{kind}_{feature}" for kind in ("center", "scale")
                  for feature in FEATURES]].to_numpy(float)
    if not np.isfinite(values).all() or (values[3:] <= 0).any():
        return "nonfinite_or_nonpositive_fixed28_baseline"
    end = pd.Timestamp(row.baseline_end)
    if pd.isna(end) or end >= first_target:
        return "baseline_not_strictly_before_target"
    return "ok"


def select_eligible(daily: pd.DataFrame, baselines: pd.DataFrame,
                    keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain every requested key; unavailable is never counted as no alarm."""
    daily, baselines, keys = _checked(daily, baselines, keys)
    base = {row.participant: row for _, row in baselines.iterrows()}
    complete = daily.set_index(["participant", "date"]).complete
    reasons = np.full(len(keys), "ok", dtype=object)
    for participant, targets in keys.groupby("participant", sort=False):
        reason = _baseline_reason(base.get(participant), targets.date.min())
        if reason != "ok":
            reasons[targets.index] = reason
        else:
            observed = complete.reindex(pd.MultiIndex.from_frame(targets[["participant", "date"]]))
            reasons[targets.index] = np.where(observed.eq(True), "ok", "target_not_complete")
    checked = keys.assign(reason=reasons)
    return (checked.loc[checked.reason.eq("ok")].copy(),
            checked.loc[checked.reason.ne("ok")].copy())


def build_inputs(daily: pd.DataFrame, baselines: pd.DataFrame, keys: pd.DataFrame,
                 seq_len: int = 512, patch_len: int = 8) -> tuple[dict, pd.DataFrame]:
    """Right-align days through the observed target; keep gaps as masked days."""
    if seq_len < 1 or patch_len < 1 or seq_len % patch_len:
        raise ValueError("Sequence length must be positive and divisible by patch length")
    daily, baselines, keys = _checked(daily, baselines, keys)
    eligible, rejected = select_eligible(daily, baselines, keys)
    if len(rejected) or eligible.empty:
        raise ValueError("Input keys include unavailable target or baseline")
    n = len(keys)
    x = np.zeros((n, 3, seq_len), dtype=np.float32)
    padding, observation = np.zeros((n, seq_len), np.uint8), np.zeros((n, seq_len), np.uint8)
    centers_out, scales_out, target_raw = (np.empty((n, 3), float) for _ in range(3))
    audits = []
    people = {p: group for p, group in daily.groupby("participant", sort=False)}
    base = baselines.set_index("participant")
    for participant, targets in keys.groupby("participant", sort=False):
        person, row = people[participant], base.loc[participant]
        centers = row[[f"center_{f}" for f in FEATURES]].to_numpy(float)
        scales = row[[f"scale_{f}" for f in FEATURES]].to_numpy(float)
        first = person.date.min()
        dates = pd.date_range(first, targets.date.max())
        dense = person.set_index("date").reindex(dates)
        raw, observed = dense[list(FEATURES)].to_numpy(float), dense.complete.eq(True).to_numpy()
        signed = raw - centers
        signed[:, 2] = (signed[:, 2] + 720.) % 1440. - 720.
        standardized = np.where(observed[:, None], signed / scales, 0.)
        positions = (targets.date - first).dt.days.to_numpy()
        grid = positions[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
        real, clipped = grid >= 0, np.maximum(grid, 0)
        seen = real & observed[clipped]
        indices = targets.index.to_numpy()
        windows = np.where(seen[:, :, None], standardized[clipped], 0.)
        x[indices] = windows.transpose(0, 2, 1).astype(np.float32)
        padding[indices], observation[indices] = real, seen
        centers_out[indices], scales_out[indices], target_raw[indices] = centers, scales, raw[positions]
        full = seen.reshape(len(indices), -1, patch_len).all(axis=2)
        audits.append(targets.assign(
            baseline_end=pd.Timestamp(row.baseline_end),
            window_start=targets.date - pd.Timedelta(days=seq_len - 1), window_end=targets.date,
            real_calendar_days=real.sum(axis=1), padding_days=seq_len - real.sum(axis=1),
            observed_complete_days=seen.sum(axis=1), missing_calendar_days=real.sum(axis=1)-seen.sum(axis=1),
            attention_patches=real.reshape(len(indices), -1, patch_len).all(axis=2).sum(axis=1),
            complete_patches=full.sum(axis=1), last_patch_complete=full[:, -1],
            no_complete_patch=~full.any(axis=1), no_future_input=True))
    if not np.isfinite(x).all() or not observation[:, -1].all():
        raise ValueError("Invalid finite input or unobserved target")
    arrays = {"x": x, "input_mask": padding, "observation_mask": observation,
              "centers": centers_out, "scales": scales_out, "target_raw": target_raw}
    return arrays, pd.concat(audits).sort_index()


def endpoint_errors(actual: np.ndarray, centers: np.ndarray, scales: np.ndarray,
                    predicted_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-transform before shortest circular bedtime error; average MSE."""
    predicted_raw = centers + scales * predicted_z
    difference = actual - predicted_raw
    difference[:, 2] = (difference[:, 2] + 720.) % 1440. - 720.
    errors = difference / scales
    return errors, np.mean(errors ** 2, axis=1)


def row_digest(arrays: Mapping[str, np.ndarray], index: int) -> str:
    """Hash all six input/scoring arrays including names, dtype and shape."""
    digest = hashlib.sha256()
    for name in CACHE_FIELDS:
        value = np.ascontiguousarray(arrays[name][index])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_prediction_cache(daily: pd.DataFrame, baselines: pd.DataFrame,
                              keys: pd.DataFrame, cache_dir: Path,
                              source_hashes: dict[str, str]) -> tuple[dict, dict]:
    """Rebuild every six-field input before importing this experiment's output.

    This does not authorize using a historical experiment. The caller must also
    verify the model identity/runtime and bind the cache to its current run.
    """
    cache_dir = Path(cache_dir)
    record = json.loads((cache_dir / "execution.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Cannot cache an incomplete inference job")
    if not source_hashes or source_hashes != record["input_source_sha256"]:
        raise ValueError("Cache source input hashes differ")
    if _file_digest(cache_dir / "scores.csv.gz") != record["scores_sha256"]:
        raise ValueError("Cache score checksum differs")
    scored = dated(pd.read_csv(cache_dir / "scores.csv.gz", dtype={"participant": str},
                               float_precision="round_trip"))
    targets = dated(keys).reset_index(drop=True).assign(row_index=np.arange(len(keys)))
    key_columns = ["participant", "date", "split", "row_index"]
    if not scored[key_columns].reset_index(drop=True).equals(targets[key_columns]):
        raise ValueError("Cache requested keys differ")
    eligible, _ = select_eligible(daily, baselines, targets)
    valid = scored.loc[scored.valid.eq(True)].set_index("row_index")
    if set(valid.index) != set(eligible.row_index) or len(valid) != record["valid_persondays"]:
        raise ValueError("Cache valid keys differ")
    paths = sorted(cache_dir.glob("input_audit_*.csv.gz"))
    if not paths:
        raise ValueError("Cache has no saved input audit")
    audits = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if audits.row_index.duplicated().any() or set(audits.row_index) != set(valid.index):
        raise ValueError("Cache audit keys differ")
    hashes = audits.set_index("row_index").input_sha256
    prediction_columns = [f"reconstruction_z_{feature}" for feature in FEATURES]
    cache = {}
    for start in range(0, len(eligible), 1536):
        selected = eligible.iloc[start:start + 1536]
        arrays, _ = build_inputs(daily, baselines, selected)
        expected = hashes.loc[selected.row_index].tolist()
        actual = [row_digest(arrays, index) for index in range(len(selected))]
        if actual != expected:
            raise ValueError("Cached six-field input hashes differ")
        predictions = valid.loc[selected.row_index, prediction_columns].to_numpy(np.float32)
        _, recomputed = endpoint_errors(arrays["target_raw"], arrays["centers"], arrays["scales"], predictions)
        if (not np.isfinite(predictions).all() or
                not np.allclose(recomputed, valid.loc[selected.row_index, "score"], rtol=1e-12, atol=1e-12)):
            raise ValueError("Cached endpoint scores fail independent recomputation")
        for digest, prediction in zip(actual, predictions):
            if digest in cache and not np.array_equal(cache[digest], prediction):
                raise ValueError("Identical cached input has different predictions")
            cache[digest] = prediction.copy()
    return cache, {"verified_persondays": len(eligible), "unique_six_field_inputs": len(cache),
                   "source_session_id": record["session_id"],
                   "execution_sha256": _file_digest(cache_dir / "execution.json"),
                   "scores_sha256": record["scores_sha256"], "all_six_inputs_rebuilt": True,
                   "endpoint_scores_recomputed": True, "old_model": record["model"],
                   "old_code_sha256": record["source_sha256"]}
