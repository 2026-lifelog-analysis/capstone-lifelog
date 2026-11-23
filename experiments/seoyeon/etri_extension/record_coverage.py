"""
record_coverage.py — 센서의 하루 기록률

기록률 = 그 구간에서 기록이 한 번이라도 있는 분(minute)의 수 ÷ 구간의 전체 분 수.
1분 간격으로 기록되는 센서에만 쓴다(기록 간격 중앙값 60초인 7종).

구간은 둘이다.
- day: 그 날짜의 00:00 ~ 24:00
- night: 그 날짜의 22:00 ~ 다음 날 10:00. 저녁이 속한 날짜에 붙인다.
  근거: LOV3 논문(ICTC 2026 C6-2) 표 2 설명 "Night denotes 22–10 h".
"""
from __future__ import annotations

import pandas as pd

DAY_MINUTES = 24 * 60
NIGHT_START_HOUR = 22
NIGHT_MINUTES = 12 * 60
WINDOWS = ("day", "night")


def daily_coverage(records: pd.DataFrame, window: str) -> pd.Series:
    """사람·날짜별 기록률. 기록이 하나도 없는 날은 결과에 없다."""
    if window not in WINDOWS:
        raise ValueError(f"구간은 {WINDOWS} 중 하나여야 함: {window}")
    minutes = records[["subject_id"]].assign(minute=records["timestamp"].dt.floor("min"))
    minutes = minutes.drop_duplicates()
    if window == "day":
        dated = minutes.assign(date=minutes["minute"].dt.normalize())
        total = DAY_MINUTES
    else:
        shifted = minutes["minute"] - pd.Timedelta(hours=NIGHT_START_HOUR)
        offset = shifted - shifted.dt.normalize()
        inside = offset < pd.Timedelta(minutes=NIGHT_MINUTES)
        dated = minutes.assign(date=shifted.dt.normalize())[inside]
        total = NIGHT_MINUTES
    return dated.groupby(["subject_id", "date"]).size() / total


def coverage_for_days(records: pd.DataFrame, days: pd.DataFrame, window: str) -> pd.Series:
    """days(subject_id, lifelog_date)의 각 행에 대한 기록률. 기록이 없는 날은 0."""
    coverage = daily_coverage(records, window)
    keys = pd.MultiIndex.from_frame(days[["subject_id", "lifelog_date"]])
    values = coverage.reindex(keys).fillna(0.0).to_numpy()
    return pd.Series(values, index=days.index)
