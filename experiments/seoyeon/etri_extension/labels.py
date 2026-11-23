"""
labels.py — ETRI 2024 라이프로그 라벨(ch2025_metrics_train.csv) 읽기

라벨 6개(Q1, Q2, Q3, S1, S2, S3)를 사람·날짜별 이진 값으로 돌려준다.
S1은 원자료가 0·1·2 세 값이라 1 이상을 양성으로 묶는다(2026-11-20 결정).
날짜 키는 lifelog_date(센서를 기록한 날)이다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGETS = ("Q1", "Q2", "Q3", "S1", "S2", "S3")
KEY = ["subject_id", "lifelog_date"]
S1_RAW_VALUES = {0, 1, 2}
S1_POSITIVE_FROM = 1
BINARY_VALUES = {0, 1}


def binarize_s1(raw: pd.Series) -> pd.Series:
    """S1 원값(0·1·2)을 1 이상이면 1, 아니면 0으로 바꾼다."""
    unknown = sorted(set(raw.unique()) - S1_RAW_VALUES)
    if unknown:
        raise ValueError(f"S1에 알 수 없는 값: {unknown}")
    return (raw >= S1_POSITIVE_FROM).astype(int)


def load_labels(path: str | Path) -> pd.DataFrame:
    """라벨 CSV를 읽어 사람·날짜순으로 정렬한 이진 라벨 표를 돌려준다."""
    raw = pd.read_csv(path, parse_dates=["sleep_date", "lifelog_date"])
    if raw.duplicated(KEY).any():
        raise ValueError("같은 사람·날짜의 라벨 행이 둘 이상 있음")
    labels = raw.assign(S1=binarize_s1(raw["S1"]))
    for target in TARGETS:
        unknown = sorted(set(labels[target].unique()) - BINARY_VALUES)
        if unknown:
            raise ValueError(f"{target}에 0·1이 아닌 값: {unknown}")
    return labels.sort_values(KEY).reset_index(drop=True)
