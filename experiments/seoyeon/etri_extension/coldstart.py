"""
coldstart.py — 기록이 짧은 사람을 흉내 낸 예측 방법 세 가지

평가하는 사람을 "새로 온 사람"으로 보고, 그 사람의 학습 구간 마지막 k일만 공개한다.
- 전체 비율(G): 그 사람을 뺀 나머지 사람들의 학습 구간 양성 비율
- 개인 비율(P_k): 공개된 k일의 양성 비율
- 섞는 방식(M_k): G와 P_k를 기록 일수에 따라 섞은 비율. 사전 강도 m도 나머지 사람들로 추정

표의 행은 사람 안에서 시간순이어야 한다(split.assign_split이 보장한다).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from baselines import as_prediction
from shrinkage import estimate_prior_strength, shrunk_rate

Predictor = Callable[[pd.DataFrame, str], pd.DataFrame]


def _train_test(split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split[split["part"] == "train"], split[split["part"] == "test"]


def _revealed(train: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """사람별 학습 구간의 마지막 k일. k가 None이면 전부."""
    if k is None:
        return train
    return train.groupby("subject_id").tail(k)


def _others_global_rate(train: pd.DataFrame, target: str) -> pd.Series:
    """사람별로, 그 사람을 뺀 나머지의 학습 구간 양성 비율."""
    sums = train.groupby("subject_id")[target].sum()
    counts = train.groupby("subject_id")[target].count()
    return (sums.sum() - sums) / (counts.sum() - counts)


def leave_one_out_global_prediction(split: pd.DataFrame, target: str) -> pd.DataFrame:
    """G. 반환 열: subject_id, y_true, prob, clipped (평가 행만)."""
    train, test = _train_test(split)
    raw_prob = test["subject_id"].map(_others_global_rate(train, target))
    return as_prediction(test, target, raw_prob)


def make_personal_predictor(k: int | None) -> Predictor:
    """P_k를 내는 함수를 만든다. k는 1 이상이거나 None(전부)."""
    if k is not None and k < 1:
        raise ValueError("개인 비율은 기록이 하루 이상 있어야 함")

    def predict(split: pd.DataFrame, target: str) -> pd.DataFrame:
        train, test = _train_test(split)
        rates = _revealed(train, k).groupby("subject_id")[target].mean()
        return as_prediction(test, target, test["subject_id"].map(rates))

    return predict


def _prior_strength_without(train: pd.DataFrame, target: str, person: str) -> float:
    others = train[train["subject_id"] != person].groupby("subject_id")[target]
    return estimate_prior_strength(others.mean().to_numpy(), others.count().to_numpy())


def make_mixed_predictor(k: int | None,
                         strengths: Mapping[str, float] | None = None) -> Predictor:
    """M_k를 내는 함수를 만든다. k가 0이면 전체 비율과 같다.

    strengths를 주면 라벨별 사전 강도 m을 그 값으로 고정하고,
    주지 않으면 그 사람을 뺀 나머지 사람들로 추정한다.
    """

    def strength_for(train: pd.DataFrame, target: str, person: str) -> float:
        if strengths is None:
            return _prior_strength_without(train, target, person)
        if target not in strengths:
            raise ValueError(f"{target}의 사전 강도가 주어지지 않음")
        return strengths[target]

    def predict(split: pd.DataFrame, target: str) -> pd.DataFrame:
        train, test = _train_test(split)
        global_rates = _others_global_rate(train, target)
        revealed = train.iloc[0:0] if k == 0 else _revealed(train, k)
        successes = revealed.groupby("subject_id")[target].sum()
        n_days = revealed.groupby("subject_id")[target].count()
        mixed = {person: shrunk_rate(successes.get(person, 0), n_days.get(person, 0),
                                     global_rates[person],
                                     strength_for(train, target, person))
                 for person in test["subject_id"].unique()}
        return as_prediction(test, target, test["subject_id"].map(mixed))

    return predict
