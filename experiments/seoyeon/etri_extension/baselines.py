"""
baselines.py — 비율 기준선 두 개

B0 전체 비율: 모든 사람의 학습 구간 양성 비율 하나를 평가 날 전부에 낸다.
B1 고정 개인 비율: 그 사람의 학습 구간 양성 비율을 그 사람의 평가 날에 낸다.

비율이 0이나 1이면 log loss가 무한대가 되므로 PROB_FLOOR ~ 1 − PROB_FLOOR로 자른다.
자른 날은 clipped 열에 표시해 몇 번 잘렸는지 보고할 수 있게 한다.

PROB_FLOOR 근거: LOV3 논문(Structure over Scale, ICTC 2026 C6-2) III절
"probabilities are clipped to [ε, 1 − ε] with ε = 10^-6".
"""
from __future__ import annotations

import pandas as pd

PROB_FLOOR = 1e-6


def as_prediction(test: pd.DataFrame, target: str, raw_prob: pd.Series) -> pd.DataFrame:
    prob = raw_prob.clip(PROB_FLOOR, 1 - PROB_FLOOR)
    return pd.DataFrame({"subject_id": test["subject_id"], "y_true": test[target],
                         "prob": prob, "clipped": prob != raw_prob}, index=test.index)


def global_rate_prediction(split: pd.DataFrame, target: str) -> pd.DataFrame:
    """B0. 반환 열: subject_id, y_true, prob, clipped (평가 행만)."""
    train = split[split["part"] == "train"]
    test = split[split["part"] == "test"]
    raw_prob = pd.Series(train[target].mean(), index=test.index)
    return as_prediction(test, target, raw_prob)


def personal_rate_prediction(split: pd.DataFrame, target: str) -> pd.DataFrame:
    """B1. 반환 열: subject_id, y_true, prob, clipped (평가 행만)."""
    train = split[split["part"] == "train"]
    test = split[split["part"] == "test"]
    rates = train.groupby("subject_id")[target].mean()
    missing = sorted(set(test["subject_id"]) - set(rates.index))
    if missing:
        raise ValueError(f"학습 구간이 없는 사람: {missing}")
    raw_prob = test["subject_id"].map(rates)
    return as_prediction(test, target, raw_prob)
