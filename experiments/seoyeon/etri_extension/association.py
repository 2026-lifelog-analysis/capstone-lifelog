"""
association.py — 사람 내 연관

각 사람의 평균을 뺀 값끼리의 상관(사람 내 상관)을 구한다.
사람 간 수준 차이는 지워지고, 같은 사람 안에서 날마다 달라지는 부분만 남는다.

근거: 춘자 논문(ICTC 2026 C6-1) IV절의 sensor-linkage가
사람 내 잔차(y − 개인 비율)와 센서 피처의 상관으로 정의돼 있다.
구간은 팀 코드와 같은 사람 단위 부트스트랩 백분위 구간이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTERVAL_LEVEL = 0.95


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def within_person_correlation(frame: pd.DataFrame, feature: str, target: str,
                              n_boot: int, seed: int = 0) -> dict[str, float]:
    """사람 내 상관 r, 사람 단위 부트스트랩 구간, 양측 p값."""
    grouped = frame.groupby("subject_id")
    centered = pd.DataFrame({"subject_id": frame["subject_id"],
                             "x": frame[feature] - grouped[feature].transform("mean"),
                             "y": frame[target] - grouped[target].transform("mean")})
    people = [group[["x", "y"]].to_numpy() for _, group in centered.groupby("subject_id")]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(people), size=len(people))
        stacked = np.vstack([people[i] for i in picks])
        samples.append(_correlation(stacked[:, 0], stacked[:, 1]))
    valid = np.array([s for s in samples if not np.isnan(s)])
    r = _correlation(centered["x"].to_numpy(), centered["y"].to_numpy())
    base = {"r": r, "n_people": len(people), "n_days": len(frame)}
    if np.isnan(r) or len(valid) == 0:
        return {**base, "low": float("nan"), "high": float("nan"), "p_value": float("nan")}
    tail = (1 - INTERVAL_LEVEL) / 2
    low, high = np.quantile(valid, [tail, 1 - tail])
    below = (np.sum(valid <= 0) + 1) / (len(valid) + 1)
    above = (np.sum(valid >= 0) + 1) / (len(valid) + 1)
    return {**base, "low": float(low), "high": float(high),
            "p_value": float(min(1.0, 2 * min(below, above)))}
