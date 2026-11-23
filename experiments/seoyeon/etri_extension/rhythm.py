"""
rhythm.py — 화면 사용으로 본 하루 리듬 지표

리듬의 하루는 05:00에 시작해 다음 날 05:00에 끝난다.
근거: LOV3 논문(ICTC 2026 C6-2) 그림 1이 00–05시를 "core sleep hours"로 두고,
우리 화면 센서에서도 사용 비율이 04~05시에 가장 낮다(0.04, 2026-11-20 확인).

지표는 넷이다.
- onset_min : 05:00부터 그 리듬 하루의 첫 화면 사용까지의 분
- offset_min: 05:00부터 그 리듬 하루의 마지막 화면 사용까지의 분 (자정을 넘길 수 있다)
  근거: LOV3 표 2의 Intra-day bins에 "screen onset/offset"이 있다. 논문에 정의는 없다.
- onset_sd7, offset_sd7: 위 두 값의 그날 포함 최근 7일 표준편차
  근거: 팀의 김민준 님 regularity.py의 수면 중앙 시각 SD
  (t−6 ~ t, ddof=1, 7일 중 5일 이상 기록이 있고 그날 값이 있을 때만).

기록이 리듬 하루의 첫 1시간에 없으면 onset을, 마지막 1시간에 없으면 offset을 비운다.
기록이 도중에 시작하거나 끝난 날의 시각을 리듬으로 잘못 읽지 않으려는 것이다(이 규칙에는 근거가 없다).
"""
from __future__ import annotations

import pandas as pd

DAY_START_HOUR = 5
EDGE_MINUTES = 60
DAY_MINUTES = 24 * 60
SD_WINDOW = 7
SD_MIN_DAYS = 5
TIMING_COLUMNS = ["onset_min", "offset_min"]
FEATURE_COLUMNS = ["onset_min", "offset_min", "onset_sd7", "offset_sd7"]


def screen_timing(records: pd.DataFrame) -> pd.DataFrame:
    """사람·리듬 날짜별 onset_min, offset_min. records 열: subject_id, timestamp, m_screen_use."""
    shifted = records["timestamp"] - pd.Timedelta(hours=DAY_START_HOUR)
    date = shifted.dt.normalize()
    minute = ((shifted - date) / pd.Timedelta(minutes=1)).astype(int)
    frame = pd.DataFrame({"subject_id": records["subject_id"], "date": date, "minute": minute,
                          "in_use": records["m_screen_use"] == 1})
    keys = ["subject_id", "date"]
    span = frame.groupby(keys)["minute"].agg(["min", "max"])
    used = frame[frame["in_use"]].groupby(keys)["minute"].agg(["min", "max"])
    used = used.reindex(span.index)
    onset = used["min"].where(span["min"] < EDGE_MINUTES)
    offset = used["max"].where(span["max"] >= DAY_MINUTES - EDGE_MINUTES)
    return pd.DataFrame({"onset_min": onset, "offset_min": offset})


def _rolling_sd_person(values: pd.Series) -> pd.Series:
    dates = values.index.get_level_values("date")
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = pd.Series(values.to_numpy(), index=dates).reindex(calendar)
    sd = daily.rolling(SD_WINDOW, min_periods=SD_MIN_DAYS).std(ddof=1).where(daily.notna())
    return pd.Series(sd.reindex(dates).to_numpy(), index=values.index)


def rolling_sd(values: pd.Series) -> pd.Series:
    """(subject_id, date) 색인의 값에 대해, 달력 기준 최근 7일 표준편차."""
    parts = [_rolling_sd_person(group) for _, group in values.groupby(level="subject_id")]
    return pd.concat(parts).reindex(values.index)


def rhythm_features(records: pd.DataFrame) -> pd.DataFrame:
    """사람·리듬 날짜별 지표 네 개."""
    timing = screen_timing(records)
    return timing.assign(onset_sd7=rolling_sd(timing["onset_min"]),
                         offset_sd7=rolling_sd(timing["offset_min"]))[FEATURE_COLUMNS]


def features_for_days(features: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """days(subject_id, lifelog_date)의 각 행에 지표를 맞춘다. 없는 날은 빈칸."""
    keys = pd.MultiIndex.from_frame(days[["subject_id", "lifelog_date"]])
    aligned = features.reindex(keys)
    return pd.DataFrame(aligned.to_numpy(dtype=float), index=days.index, columns=FEATURE_COLUMNS)


def missing_share(table: pd.DataFrame) -> pd.Series:
    """지표별 빈칸 비율."""
    return table[FEATURE_COLUMNS].isna().mean()

