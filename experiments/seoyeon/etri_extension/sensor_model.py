"""
sensor_model.py — 섞은 비율 위에 센서 피처를 더하는 예측

기준값은 방향 1의 섞는 방식(학습 구간 전부, 사전 강도는 나머지 사람들로 추정)이다.
그 로짓을 offset으로 두고 ℓ2 로지스틱이 피처의 기여만 배운다.

- 피처는 사람별로 표준화한다. 근거: LOV3(C6-2) III절 "per-subject z-normalized copies ...
  computed over all of that subject's days without reference to labels".
- ℓ2 세기는 평가 구간을 보지 않고 고른다. 학습 구간을 사람별 연속 블록으로 나눠,
  한 블록을 빼고 적합한 뒤 그 블록의 log loss를 잰다. 근거: LOV3 IV절
  "splits each subject into contiguous temporal blocks", III절 "3 fold-splits".
  log loss가 같으면 더 큰 벌점(기준값에 가까운 쪽)을 고른다.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from baselines import PROB_FLOOR, as_prediction
from coldstart import _others_global_rate, _prior_strength_without
from offset_logit import fit_offset_logistic, predict_offset_logistic
from shrinkage import shrunk_rate

Predictor = Callable[[pd.DataFrame, str], pd.DataFrame]


def zscore_within_person(table: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """피처를 사람별 평균 0, 표준편차 1로 바꾼 새 표. 값이 일정한 피처는 0이 된다."""
    grouped = table.groupby("subject_id")[list(features)]
    std = grouped.transform("std").replace(0.0, np.nan)
    scaled = ((table[list(features)] - grouped.transform("mean")) / std).fillna(0.0)
    return table.assign(**{feature: scaled[feature] for feature in features})


def assign_temporal_blocks(train: pd.DataFrame, n_blocks: int) -> pd.Series:
    """사람별로 시간순 연속 블록 번호(0 ~ n_blocks − 1)를 붙인다."""
    position = train.groupby("subject_id").cumcount()
    size = train.groupby("subject_id")["subject_id"].transform("size")
    return (position * n_blocks // size).astype(int)


def _mixed_rates(fit: pd.DataFrame, target: str) -> pd.Series:
    """fit 행만으로 구한 사람별 섞은 비율."""
    global_rates = _others_global_rate(fit, target)
    sums = fit.groupby("subject_id")[target].sum()
    counts = fit.groupby("subject_id")[target].count()
    rates = {person: shrunk_rate(sums[person], counts[person], global_rates[person],
                                 _prior_strength_without(fit, target, person))
             for person in sums.index}
    return pd.Series(rates).clip(PROB_FLOOR, 1 - PROB_FLOOR)


def _offset(rows: pd.DataFrame, rates: pd.Series) -> np.ndarray:
    rate = rows["subject_id"].map(rates).to_numpy(dtype=float)
    return np.log(rate / (1 - rate))


def _fit_predict(fit: pd.DataFrame, held: pd.DataFrame, target: str,
                 features: Sequence[str], l2: float) -> np.ndarray:
    rates = _mixed_rates(fit, target)
    model = fit_offset_logistic(fit[list(features)].to_numpy(), fit[target].to_numpy(),
                                _offset(fit, rates), l2)
    return predict_offset_logistic(model, held[list(features)].to_numpy(), _offset(held, rates))


def _log_loss_sum(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, PROB_FLOOR, 1 - PROB_FLOOR)
    return float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).sum())


def select_l2(split: pd.DataFrame, target: str, features: Sequence[str],
              l2_grid: Sequence[float], n_blocks: int) -> float:
    """학습 구간의 연속 블록 검증으로 ℓ2 세기를 고른다."""
    train = split[split["part"] == "train"]
    blocks = assign_temporal_blocks(train, n_blocks)
    losses = {}
    for l2 in l2_grid:
        losses[l2] = sum(
            _log_loss_sum(train.loc[blocks == b, target].to_numpy(),
                          _fit_predict(train[blocks != b], train[blocks == b], target, features, l2))
            for b in range(n_blocks))
    best = min(losses.values())
    return max(l2 for l2, loss in losses.items() if loss == best)


def make_sensor_predictor(features: Sequence[str], l2_grid: Sequence[float],
                          n_blocks: int) -> Predictor:
    """섞은 비율 + 센서 피처 예측 함수를 만든다."""
    if not features:
        raise ValueError("피처가 하나 이상 있어야 함")

    def predict(split: pd.DataFrame, target: str) -> pd.DataFrame:
        train = split[split["part"] == "train"]
        test = split[split["part"] == "test"]
        l2 = select_l2(split, target, features, l2_grid, n_blocks)
        prob = _fit_predict(train, test, target, features, l2)
        return as_prediction(test, target, pd.Series(prob, index=test.index))

    return predict
