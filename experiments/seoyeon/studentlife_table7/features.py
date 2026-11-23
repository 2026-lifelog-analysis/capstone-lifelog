"""
features.py — StudentLife 센서 기록에서 하루 단위 특징을 만든다 (Wang 외 2014 표 7 재현용)

논문이 정한 것
  - 활동 시간: 10분 구간마다 비정지 추론 비율이 문턱값보다 크면 "움직임" 구간, 그 구간들을 하루 단위로 더함
  - 이동 거리: 주기적 GPS 표본 사이의 거리
  - 실내 이동: 건물 안에 있을 때의 걷기 추론으로 계산한 시간
  - 시간 구간: 낮 9~18시, 저녁 18~24시, 밤 0~9시 (미국 동부 시간)

논문에 없어서 인자로 열어 둔 것
  - 문턱값, "알 수 없음" 추론을 분모에 넣을지, GPS 정확도·간격 한도, near를 건물 안으로 볼지

모든 함수는 입력 표를 바꾸지 않고 새 표를 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"
BIN_SECONDS = 600
BIN_MINUTES = BIN_SECONDS // 60
STATIONARY, WALKING, RUNNING, UNKNOWN = 0, 1, 2, 3
EPOCH_HOURS = {"night": (0, 9), "day": (9, 18), "evening": (18, 24)}
EARTH_RADIUS_M = 6_371_000.0
INDOOR_MIN_SHARE = 0.5          # 구간 안 와이파이 표본 중 건물 안 판정 비율이 이 이상이면 실내 구간
INDOOR_METHODS = ("ratio", "active")


# ---------- 시간 ----------

def to_local(seconds: pd.Series) -> pd.Series:
    """유닉스 초를 미국 동부 시각으로 바꾼다."""
    return pd.to_datetime(pd.to_numeric(seconds, errors="coerce"), unit="s", utc=True).dt.tz_convert(TZ)


def epoch_of_hour(hour: int) -> str:
    """시(0~23)가 속한 논문의 시간 구간 이름."""
    for name, (start, end) in EPOCH_HOURS.items():
        if start <= hour < end:
            return name
    raise ValueError(f"시가 0~23 밖임: {hour}")


def _local_parts(bin_start: pd.Series) -> pd.DataFrame:
    """구간 시작 시각의 현지 날짜와 시간 구간."""
    local = to_local(bin_start)
    return pd.DataFrame({"date": local.dt.date.to_numpy(),
                         "epoch": local.dt.hour.map(epoch_of_hour).to_numpy()}, index=bin_start.index)


def _to_bin(seconds: pd.Series) -> pd.Series:
    return (seconds // BIN_SECONDS * BIN_SECONDS).astype("int64")


# ---------- 활동 시간 ----------

def activity_bins(activity: pd.DataFrame) -> pd.DataFrame:
    """
    활동 추론(timestamp, inference)을 10분 구간별 개수로 줄인다.
    반환 열: bin_start(유닉스 초), n_total, n_moving(걷기+뛰기), n_walk, n_unknown
    """
    t = pd.to_numeric(activity["timestamp"], errors="coerce")
    v = pd.to_numeric(activity["inference"], errors="coerce")
    ok = t.notna() & v.notna()
    v = v[ok].astype("int64")
    frame = pd.DataFrame({"bin_start": _to_bin(t[ok]), "moving": v.isin((WALKING, RUNNING)),
                          "walk": v.eq(WALKING), "unknown": v.eq(UNKNOWN)})
    grouped = frame.groupby("bin_start")
    return pd.DataFrame({"n_total": grouped.size(), "n_moving": grouped["moving"].sum(),
                         "n_walk": grouped["walk"].sum(), "n_unknown": grouped["unknown"].sum()}).reset_index()


def _denominator(bins: pd.DataFrame, include_unknown: bool) -> pd.Series:
    denom = bins["n_total"] if include_unknown else bins["n_total"] - bins["n_unknown"]
    return denom.where(denom > 0)


def daily_activity(bins: pd.DataFrame, threshold: float, include_unknown: bool = True) -> pd.DataFrame:
    """
    하루 활동 시간(분). 비정지 비율이 threshold보다 큰 10분 구간을 더한다.
    반환 열: date, act_all, act_day, act_evening, act_night, n_bins(그날 기록된 구간 수)
    """
    ratio = bins["n_moving"] / _denominator(bins, include_unknown)
    minutes = (ratio > threshold).astype("int64") * BIN_MINUTES
    frame = _local_parts(bins["bin_start"]).assign(minutes=minutes.to_numpy())
    out = frame.groupby("date").agg(act_all=("minutes", "sum"), n_bins=("minutes", "size"))
    for name in EPOCH_HOURS:
        out[f"act_{name}"] = frame.loc[frame["epoch"] == name].groupby("date")["minutes"].sum()
    out = out.fillna(0)
    return out.reset_index()[["date", "act_all", "act_day", "act_evening", "act_night", "n_bins"]]


# ---------- 이동 거리 ----------

def haversine_m(lat1, lon1, lat2, lon2):
    """두 좌표 사이의 대원 거리(m)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def daily_distance(gps: pd.DataFrame, max_accuracy: float | None = None,
                   max_gap: float | None = None) -> pd.DataFrame:
    """
    하루 이동 거리(m). 같은 날 안에서 이어진 GPS 표본 사이 거리를 더한다.
    dist_day는 두 끝점이 모두 낮(9~18시)인 구간만 더한다.
    max_accuracy: 정확도(m)가 이보다 나쁜 표본을 뺌. max_gap: 표본 간격(초)이 이보다 긴 구간을 뺌.
    반환 열: date, dist_all, dist_day, n_points
    """
    df = gps[["time", "latitude", "longitude", "accuracy"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude"]).sort_values("time")
    if max_accuracy is not None:
        df = df[df["accuracy"] <= max_accuracy]
    local = to_local(df["time"])
    df = df.assign(date=local.dt.date.to_numpy(), hour=local.dt.hour.to_numpy())
    prev = df.shift(1)
    step = haversine_m(prev["latitude"], prev["longitude"], df["latitude"], df["longitude"])
    valid = prev["time"].notna() & (prev["date"] == df["date"])
    if max_gap is not None:
        valid &= (df["time"] - prev["time"]) <= max_gap
    start, end = EPOCH_HOURS["day"]
    in_day = df["hour"].between(start, end - 1) & prev["hour"].between(start, end - 1)
    df = df.assign(seg_all=step.where(valid, 0.0), seg_day=step.where(valid & in_day, 0.0))
    out = df.groupby("date").agg(dist_all=("seg_all", "sum"), dist_day=("seg_day", "sum"),
                                 n_points=("time", "size"))
    return out.reset_index()


# ---------- 실내 이동 ----------

def wifi_bins(wifi: pd.DataFrame) -> pd.DataFrame:
    """
    와이파이 위치(time, location)를 10분 구간별 개수로 줄인다.
    location은 'in[건물]' 또는 'near[건물; ...]' 형식이다.
    반환 열: bin_start, n_total, n_in, n_near
    """
    t = pd.to_numeric(wifi["time"], errors="coerce")
    ok = t.notna()
    loc = wifi.loc[ok, "location"].astype(str)
    frame = pd.DataFrame({"bin_start": _to_bin(t[ok]), "is_in": loc.str.startswith("in["),
                          "is_near": loc.str.startswith("near[")})
    grouped = frame.groupby("bin_start")
    return pd.DataFrame({"n_total": grouped.size(), "n_in": grouped["is_in"].sum(),
                         "n_near": grouped["is_near"].sum()}).reset_index()


def daily_indoor_mobility(act_bins: pd.DataFrame, wifi: pd.DataFrame, near_as_in: bool = False,
                          method: str = "ratio", threshold: float = 0.5,
                          include_unknown: bool = True) -> pd.DataFrame:
    """
    하루 실내 이동(분). 건물 안으로 판정된 10분 구간의 걷기 시간을 더한다.
    method='ratio'  : 구간마다 10분 × 걷기 추론 비율
    method='active' : 걷기 비율이 threshold보다 큰 구간을 10분으로 셈
    near_as_in      : 'near[...]' 표본도 건물 안으로 볼지
    반환 열: date, indoor_all, indoor_day, n_bins
    """
    if method not in INDOOR_METHODS:
        raise ValueError(f"method는 {INDOOR_METHODS} 중 하나여야 함: {method}")
    w = wifi.rename(columns={"n_total": "n_wifi"})
    merged = act_bins.merge(w, on="bin_start", how="left")
    inside = merged["n_in"].fillna(0) + (merged["n_near"].fillna(0) if near_as_in else 0)
    indoor = (inside / merged["n_wifi"].where(merged["n_wifi"] > 0)) >= INDOOR_MIN_SHARE
    walk_ratio = merged["n_walk"] / _denominator(merged, include_unknown)
    share = walk_ratio.fillna(0.0) if method == "ratio" else (walk_ratio > threshold).astype("float64")
    minutes = (share * BIN_MINUTES).where(indoor, 0.0)
    frame = _local_parts(merged["bin_start"]).assign(minutes=minutes.to_numpy())
    out = frame.groupby("date").agg(indoor_all=("minutes", "sum"), n_bins=("minutes", "size"))
    out["indoor_day"] = frame.loc[frame["epoch"] == "day"].groupby("date")["minutes"].sum()
    out = out.fillna(0)
    return out.reset_index()[["date", "indoor_all", "indoor_day", "n_bins"]]
