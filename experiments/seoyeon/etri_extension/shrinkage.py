"""
shrinkage.py — 개인 비율을 전체 비율 쪽으로 당기는 섞는 방식(베타-이항 수축)

섞은 비율 = (개인 양성 수 + m × 전체 비율) / (개인 기록 일수 + m)

m(사전 강도)은 "전체 비율을 며칠치 기록만큼 믿는가"이다. 기록이 0일이면 전체 비율이 되고,
기록이 m일이면 개인 비율과 전체 비율의 한가운데가 된다.

m은 다른 사람들의 비율이 서로 얼마나 다른지로 추정한다(적률법).
사람 간 분산에서 표본이 작아 생기는 분산을 뺀 것이 실제 사람 간 차이이고,
그 차이가 클수록 m이 작아진다(개인 기록을 빨리 믿는다).
실제 차이가 0 이하로 추정되면 m을 PRIOR_STRENGTH_CAP으로 두어 전체 비율만 쓴다.
"""
from __future__ import annotations

import numpy as np

PRIOR_STRENGTH_CAP = 1e6


def shrunk_rate(successes: float, n_days: float, global_rate: float, strength: float) -> float:
    """섞은 비율. strength와 n_days가 모두 0이면 정의되지 않으므로 거부한다."""
    if n_days + strength <= 0:
        raise ValueError("개인 기록 일수와 사전 강도가 모두 0임")
    return (successes + strength * global_rate) / (n_days + strength)


def strength_from_icc(icc: float) -> float:
    """급내상관(ICC)에서 사전 강도 m을 구한다: m = 1 / ICC − 1.

    베타-이항 모형에서 사람 간 분산은 μ(1−μ)/(m+1), 전체 분산은 μ(1−μ)이므로
    사람 간 분산의 비중(ICC)이 1/(m+1)이다.
    """
    if not 0 < icc < 1:
        raise ValueError("ICC는 0 초과 1 미만이어야 함")
    return 1 / icc - 1


def estimate_prior_strength(rates: np.ndarray, counts: np.ndarray) -> float:
    """사람별 비율(rates)과 그 비율을 낸 일수(counts)로 사전 강도 m을 추정한다."""
    rates = np.asarray(rates, dtype=float)
    counts = np.asarray(counts, dtype=float)
    pooled = float((rates * counts).sum() / counts.sum())
    binomial_variance = pooled * (1 - pooled)
    between = float(np.mean((rates - pooled) ** 2))
    sampling = binomial_variance * float(np.mean(1 / counts))
    true_between = between - sampling
    if true_between <= 0:
        return PRIOR_STRENGTH_CAP
    return float(min(PRIOR_STRENGTH_CAP, max(0.0, binomial_variance / true_between - 1)))
