"""기존 모듈 1의 다섯 모델을 메인 전처리 결과에 적용하는 독립 이식본.

원본: 모듈1_NetHealth/02_코드/models.py. 원본 파일은 수정하지 않습니다.
알고리즘은 원본과 같고, adapt_main 및 선택적 config 기본값만 추가합니다.
1.4826은 정규분포 아래 MAD의 척도 보정값이며 임상 정상범위가 아닙니다.
원형 잔차에도 이 상수를 적용하는 기존 설계는 별도 타당성 보증이 없습니다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REFERENCE_SHA256 = "c7900f051515f4ecad5526a942d958e64d7e12ee52f16116185055f2eceabdbe"
REFERENCE_SOURCE = "모듈1_NetHealth/02_코드/models.py"
FEATURES = ("steps", "sleep_minutes", "bedtime_minutes")
MODELS = ("fixed14", "fixed28", "rolling28", "cusum28", "group")
SCORE_COLUMNS = [
    "participant", "date", "model", "score", "valid", "reason", "baseline_n",
    *[f"{prefix}_{feature}" for prefix in ("center", "scale", "z", "cplus", "cminus")
      for feature in FEATURES],
]
AUDIT_COLUMNS = [
    "participant", "model", "baseline_start", "baseline_end", "baseline_n",
    "eligible", "reason",
    *[f"{prefix}_{feature}" for prefix in ("center", "scale") for feature in FEATURES],
]


def adapt_main(df: pd.DataFrame) -> pd.DataFrame:
    """열과 시각 기준만 변환하며 결측 행 및 메인 전처리의 수면 선택을 보존합니다.

    메인 bed_min은 정오부터 경과한 분, 기존 모델은 자정부터 분입니다.
    세 특징 모두 유한할 때만 complete=True이며 추가 착용률 필터는 없습니다.
    """
    required = {"pid", "date", "steps", "sleep_min", "bed_min"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing main input columns: {sorted(required - set(df.columns))}")
    frame = df.loc[:, ["pid", "date", "steps", "sleep_min", "bed_min"]].rename(columns={
        "pid": "participant", "sleep_min": "sleep_minutes", "bed_min": "bedtime_minutes",
    }).copy()
    bedtime = frame.bedtime_minutes.to_numpy(dtype=float)
    finite_bedtime = np.where(np.isfinite(bedtime), bedtime, np.nan)
    frame["bedtime_minutes"] = (finite_bedtime + 720.0) % 1440.0
    frame["complete"] = np.isfinite(frame[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    return frame


def angular_difference(values: np.ndarray, center: Any) -> np.ndarray:
    """Signed shortest clock-time difference, with range [-720, 720)."""
    return (np.asarray(values, dtype=float) - center + 720.0) % 1440.0 - 720.0


def circular_median(values: np.ndarray) -> float:
    """Observed-sample circular L1 median; ties choose earliest clock time."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)] % 1440.0
    if not finite.size:
        return float("nan")
    candidates, counts = np.unique(finite, return_counts=True)
    size = len(candidates)
    angles = np.concatenate([candidates, candidates + 1440.0])
    weights = np.concatenate([counts, counts])
    weight_sum = np.concatenate([[0.0], np.cumsum(weights)])
    angle_sum = np.concatenate([[0.0], np.cumsum(weights * angles)])
    starts = np.arange(size)
    ends = starts + size
    mids = np.searchsorted(angles, candidates + 720.0, side="right")
    left_count = weight_sum[mids] - weight_sum[starts]
    right_count = weight_sum[ends] - weight_sum[mids]
    costs = (angle_sum[mids] - angle_sum[starts] - candidates * left_count
             + (1440.0 + candidates) * right_count
             - (angle_sum[ends] - angle_sum[mids]))
    return float(candidates[np.argmin(costs)])


def _statistics(values: np.ndarray, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    if not len(values):
        return np.full(3, np.nan), np.full(3, np.nan)
    centers = np.array([np.median(values[:, 0]), np.median(values[:, 1]),
                        circular_median(values[:, 2])])
    deviations = np.abs(values - centers)
    deviations[:, 2] = np.abs(angular_difference(values[:, 2], centers[2]))
    return centers, multiplier * np.median(deviations, axis=0)


def _reason(n: int, minimum: int, scales: np.ndarray) -> str:
    if n < minimum:
        return "insufficient_baseline"
    if not np.isfinite(scales).all():
        return "nonfinite_baseline"
    if (scales <= 0).any():
        return "zero_mad"
    return "ok"


def _audit(participant: str, model: str, start: pd.Timestamp, end: pd.Timestamp,
           n: int, centers: np.ndarray, scales: np.ndarray, reason: str) -> dict:
    return {
        "participant": participant, "model": model, "baseline_start": start,
        "baseline_end": end, "baseline_n": n, "eligible": reason == "ok",
        "reason": reason,
        **{f"center_{f}": centers[i] for i, f in enumerate(FEATURES)},
        **{f"scale_{f}": scales[i] for i, f in enumerate(FEATURES)},
    }


def _score_frame(person: pd.DataFrame, model: str, values: np.ndarray,
                 centers: np.ndarray, scales: np.ndarray, ns: np.ndarray,
                 reasons: np.ndarray) -> pd.DataFrame:
    n = len(person)
    center_rows = np.broadcast_to(centers, (n, 3))
    scale_rows = np.broadcast_to(scales, (n, 3))
    valid = (reasons == "ok") & person.complete.to_numpy() & np.isfinite(values).all(axis=1)
    output_reasons = np.where((reasons == "ok") & ~valid, "missing_current", reasons)
    deviations = values - center_rows
    deviations[:, 2] = angular_difference(values[:, 2], center_rows[:, 2])
    z_values = np.divide(deviations, scale_rows, out=np.full((n, 3), np.nan),
                         where=np.isfinite(scale_rows) & (scale_rows > 0))
    scores = np.full(n, np.nan)
    scores[valid] = np.abs(z_values[valid]).max(axis=1)
    return pd.DataFrame({
        "participant": person.participant.to_numpy(), "date": person.date.to_numpy(),
        "model": model, "score": scores, "valid": valid,
        "reason": output_reasons, "baseline_n": ns,
        **{f"center_{f}": center_rows[:, i] for i, f in enumerate(FEATURES)},
        **{f"scale_{f}": scale_rows[:, i] for i, f in enumerate(FEATURES)},
        **{f"z_{f}": z_values[:, i] for i, f in enumerate(FEATURES)},
        **{f"{prefix}_{f}": np.full(n, np.nan)
           for prefix in ("cplus", "cminus") for f in FEATURES},
    })[SCORE_COLUMNS]


def _fixed(person: pd.DataFrame, model: str, days: int, minimum: int,
           cutoff: pd.Timestamp, multiplier: float) -> tuple[pd.DataFrame, dict]:
    start = person.date.iloc[0]
    end = start + pd.Timedelta(days=days - 1)
    values = person[list(FEATURES)].to_numpy(dtype=float)
    mask = person.date.le(end).to_numpy() & person.complete.to_numpy()
    n = int(mask.sum())
    centers, scales = _statistics(values[mask], multiplier)
    reason = _reason(n, minimum, scales)
    if end >= cutoff:
        reason = "baseline_not_before_calibration"
    reasons = np.full(len(person), reason, dtype=object)
    if reason == "ok":
        reasons[person.date.le(end).to_numpy()] = "baseline_period"
    result = _score_frame(person, model, values, centers, scales,
                          np.full(len(person), n), reasons)
    audit = _audit(str(person.participant.iloc[0]), model, start, end, n,
                   centers, scales, reason)
    return result, audit


def _window_statistics(windows: np.ndarray, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    """Batch complete-case rolling windows; circular cost matrices are bounded."""
    centers = np.full((len(windows), 3), np.nan)
    scales = np.full_like(centers, np.nan)
    for start in range(0, len(windows), 1024):
        stop = start + 1024
        batch = windows[start:stop]
        batch_centers = np.nanmedian(batch, axis=1)
        clocks = np.sort(batch[:, :, 2], axis=1)
        differences = np.abs(angular_difference(clocks[:, :, None], clocks[:, None, :]))
        costs = np.nansum(differences, axis=2)
        costs[~np.isfinite(clocks)] = np.inf
        batch_centers[:, 2] = clocks[np.arange(len(clocks)), np.argmin(costs, axis=1)]
        distances = np.abs(batch - batch_centers[:, None, :])
        distances[:, :, 2] = np.abs(angular_difference(batch[:, :, 2], batch_centers[:, None, 2]))
        centers[start:stop] = batch_centers
        scales[start:stop] = multiplier * np.nanmedian(distances, axis=1)
    return centers, scales


def _rolling(person: pd.DataFrame, days: int, minimum: int, multiplier: float) -> pd.DataFrame:
    values = person[list(FEATURES)].to_numpy(dtype=float)
    positions = (person.date - person.date.iloc[0]).dt.days.to_numpy()
    dense = np.full((int(positions[-1]) + 1, 3), np.nan)
    dense[positions] = np.where(person.complete.to_numpy()[:, None], values, np.nan)
    padded = np.vstack([np.full((days, 3), np.nan), dense])
    windows = np.lib.stride_tricks.sliding_window_view(padded, days, axis=0)
    observed_windows = windows[positions].transpose(0, 2, 1)
    ns = np.isfinite(observed_windows).all(axis=2).sum(axis=1)
    sufficient = ns >= minimum
    centers = np.full((len(person), 3), np.nan)
    scales = np.full_like(centers, np.nan)
    if sufficient.any():
        centers[sufficient], scales[sufficient] = _window_statistics(observed_windows[sufficient], multiplier)
    reasons = np.full(len(person), "insufficient_baseline", dtype=object)
    reasons[sufficient] = "ok"
    reasons[sufficient & (scales <= 0).any(axis=1)] = "zero_mad"
    return _score_frame(person, "rolling28", values, centers, scales, ns, reasons)


def _cusum(fixed: pd.DataFrame, allowance: float) -> pd.DataFrame:
    z_values = fixed[[f"z_{f}" for f in FEATURES]].to_numpy()
    dates = fixed.date.to_numpy(dtype="datetime64[D]")
    plus = np.full_like(z_values, np.nan)
    minus = np.full_like(z_values, np.nan)
    current_plus = np.zeros(3)
    current_minus = np.zeros(3)
    valid = fixed.valid.to_numpy()
    for i in range(len(fixed)):
        gap = i > 0 and (dates[i] - dates[i - 1]) != np.timedelta64(1, "D")
        if gap or not valid[i]:
            current_plus = np.zeros(3)
            current_minus = np.zeros(3)
        if valid[i]:
            current_plus = np.maximum(0, current_plus + z_values[i] - allowance)
            current_minus = np.maximum(0, current_minus - z_values[i] - allowance)
            plus[i], minus[i] = current_plus, current_minus
    scores = np.full(len(fixed), np.nan)
    scores[valid] = np.concatenate([plus[valid], minus[valid]], axis=1).max(axis=1)
    return fixed.assign(
        model="cusum28", score=scores,
        **{f"cplus_{f}": plus[:, i] for i, f in enumerate(FEATURES)},
        **{f"cminus_{f}": minus[:, i] for i, f in enumerate(FEATURES)},
    )


def _validated(daily: pd.DataFrame, development_end: Any, calibration_start: Any,
               config: dict) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    required = {"participant", "date", "complete", *FEATURES}
    if not required.issubset(daily.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(daily.columns))}")
    development = pd.Timestamp(development_end).normalize()
    calibration = pd.Timestamp(calibration_start).normalize()
    if development >= calibration:
        raise ValueError("development_end must precede calibration_start")
    if config.get("mad_scale", 1.4826) <= 0 or config.get("cusum_k", 0.5) < 0:
        raise ValueError("MAD multiplier must be positive and CUSUM k nonnegative")
    for name, days, minimum in (("fixed14", 14, 10), ("fixed28", 28, 20), ("rolling", 28, 20)):
        width = config.get(f"{name}_days", days)
        count = config.get(f"{name}_min_valid", minimum)
        if not isinstance(width, int) or not isinstance(count, int) or not 1 <= count <= width:
            raise ValueError(f"Invalid {name} window/minimum")
    frame = daily.loc[:, ["participant", "date", *FEATURES, "complete"]].copy()
    if frame.participant.isna().any():
        raise ValueError("Participant identifiers cannot be missing")
    frame["participant"] = frame.participant.astype(str)
    frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    if frame.date.isna().any():
        raise ValueError("Dates cannot be missing")
    if frame.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate participant/date rows")
    numeric = frame[list(FEATURES)].to_numpy(dtype=float)
    frame["complete"] = frame.complete.eq(True).to_numpy() & np.isfinite(numeric).all(axis=1)
    return frame.sort_values(["participant", "date"]).reset_index(drop=True), development, calibration


def score_models(daily: pd.DataFrame, development_end: Any, calibration_start: Any,
                 config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본과 같은 일별 점수와 고정 기준선 표를 반환하며 분할은 변경하지 않습니다.

    개인 고정 기준선은 최초 제공일부터 시작하며 보정 시작 전에 끝나야 합니다.
    집단 기준선은 개발 기간의 완전 인일을 합칩니다(사람별 동일 가중치 아님).
    이동 기준선은 해당 날 이전 달력 28일이며 세 특징 모두 있는 날만 사용합니다.
    CUSUM은 결측/달력 공백에 초기화하지만 경보 발생으로는 초기화하지 않습니다.
    """
    config = {} if config is None else config
    frame, development, calibration = _validated(daily, development_end, calibration_start, config)
    if frame.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS), pd.DataFrame(columns=AUDIT_COLUMNS)
    multiplier = float(config.get("mad_scale", 1.4826))
    group_values = frame.loc[frame.complete & frame.date.le(development), list(FEATURES)].to_numpy(dtype=float)
    group_centers, group_scales = _statistics(group_values, multiplier)
    group_n = len(group_values)
    group_reason = _reason(group_n, 1, group_scales)
    if "group_reference" in config:
        reference = config["group_reference"]
        group_centers = np.array([reference[f]["center"] for f in FEATURES], dtype=float)
        group_scales = np.array([reference[f]["scale"] for f in FEATURES], dtype=float)
        if not np.isfinite(group_centers).all() or not np.isfinite(group_scales).all() or (group_scales <= 0).any():
            raise ValueError("External group_reference must have finite centers and positive finite scales")
        group_n = int(config.get("group_reference_n", -1))
        group_reason = "ok"
    audits = [_audit("__pooled__", "group", frame.date.min(), development,
                      group_n, group_centers, group_scales, group_reason)]
    frames = []
    for _, person in frame.groupby("participant", sort=True):
        person = person.reset_index(drop=True)
        fixed14, audit14 = _fixed(person, "fixed14", config.get("fixed14_days", 14),
                                 config.get("fixed14_min_valid", 10), calibration, multiplier)
        fixed28, audit28 = _fixed(person, "fixed28", config.get("fixed28_days", 28),
                                 config.get("fixed28_min_valid", 20), calibration, multiplier)
        rolling = _rolling(person, config.get("rolling_days", 28),
                           config.get("rolling_min_valid", 20), multiplier)
        cusum = _cusum(fixed28, float(config.get("cusum_k", 0.5)))
        group_reasons = np.full(len(person), group_reason, dtype=object)
        if group_reason == "ok":
            group_reasons[person.date.le(development).to_numpy()] = "development_period"
        group = _score_frame(person, "group", person[list(FEATURES)].to_numpy(dtype=float),
                             group_centers, group_scales, np.full(len(person), group_n), group_reasons)
        frames.extend([fixed14, fixed28, rolling, cusum, group])
        audits.extend([audit14, audit28, {**audit28, "model": "cusum28"}])
    return (pd.concat(frames, ignore_index=True).sort_values(["participant", "date", "model"]).reset_index(drop=True),
            pd.DataFrame(audits, columns=AUDIT_COLUMNS))


def apply_alarm_rules(scores: pd.DataFrame, thresholds: dict, persistence_days: int = 2,
                      inclusive: bool = True) -> pd.DataFrame:
    """원본의 동일 특징·동일 방향 달력 연속 확인 규칙을 보존합니다.

    기존 원점수 임계값은 >= h-1e-12, 보정 분위수선은 inclusive=False에서 > h.
    event_start는 최초 이탈일이 아니라 연속성 확인이 처음 성립하는 날짜입니다.
    """
    if not isinstance(persistence_days, int) or persistence_days < 1:
        raise ValueError("persistence_days must be a positive integer")
    output = scores.copy().reset_index(drop=True)
    if output.empty:
        return output.assign(alarm=pd.Series(dtype=bool), persistent=pd.Series(dtype=bool),
                             event_start=pd.Series(dtype=bool))
    missing = set(output.model.unique()) - set(thresholds)
    if missing:
        raise ValueError(f"Missing threshold for models: {sorted(missing)}")
    limits = output.model.map(thresholds).to_numpy(dtype=float)
    if np.isnan(limits).any() or (limits < 0).any():
        raise ValueError("Each threshold must be nonnegative, possibly positive infinity")
    directions = np.full((len(output), 6), np.nan)
    cusum_mask = output.model.eq("cusum28").to_numpy()
    for i, feature in enumerate(FEATURES):
        directions[~cusum_mask, i * 2] = output.loc[~cusum_mask, f"z_{feature}"]
        directions[~cusum_mask, i * 2 + 1] = -output.loc[~cusum_mask, f"z_{feature}"]
        if cusum_mask.any():
            directions[cusum_mask, i * 2] = output.loc[cusum_mask, f"cplus_{feature}"]
            directions[cusum_mask, i * 2 + 1] = output.loc[cusum_mask, f"cminus_{feature}"]
    flags = directions >= limits[:, None] - 1e-12 if inclusive else directions > limits[:, None]
    flags &= output.valid.to_numpy(dtype=bool)[:, None]
    persistent = np.zeros(len(output), dtype=bool)
    event_start = np.zeros(len(output), dtype=bool)
    for _, group in output.groupby(["participant", "model"], sort=False):
        ordered = group.sort_values("date")
        indices = ordered.index.to_numpy()
        dates = pd.to_datetime(ordered.date).to_numpy(dtype="datetime64[D]")
        runs = np.zeros(6, dtype=int)
        previous_confirmed = False
        for i, index in enumerate(indices):
            if i == 0 or dates[i] - dates[i - 1] != np.timedelta64(1, "D"):
                runs = np.zeros(6, dtype=int)
                previous_confirmed = False
            runs = np.where(flags[index], runs + 1, 0)
            confirmed = bool((runs >= persistence_days).any())
            persistent[index] = confirmed
            event_start[index] = confirmed and not previous_confirmed
            previous_confirmed = confirmed
    return output.assign(
        alarm=flags.any(axis=1), persistent=persistent, event_start=event_start,
        **{f"{direction}_{feature}": flags[:, i * 2 + j]
           for i, feature in enumerate(FEATURES) for j, direction in enumerate(("above", "below"))},
    )
