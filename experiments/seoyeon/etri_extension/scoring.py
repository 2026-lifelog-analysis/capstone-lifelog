"""
scoring.py — 채점과 불확실성

주 지표는 log loss(자연로그, nats), 보조 지표는 AUROC다.
두 방법의 차이는 사람 단위 부트스트랩으로 구간을 낸다.
사람을 복원추출한 뒤 뽑힌 사람들의 날을 전부 모아 평균을 낸다(날 단위 합산 평균).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_BOOT = 2000
INTERVAL_LEVEL = 0.95


def daily_log_loss(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """날마다의 log loss. 확률이 0 초과 1 미만이 아니면 거부한다."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if not ((p > 0) & (p < 1)).all():
        raise ValueError("확률은 0 초과 1 미만이어야 함")
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auroc(y_true: np.ndarray, prob: np.ndarray) -> float:
    """순위 기반 AUROC. 동점은 평균 순위로 처리한다. 한 클래스뿐이면 NaN."""
    y = np.asarray(y_true)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(prob, dtype=float)).rank(method="average").to_numpy()
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_person_bootstrap(losses: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
                            seed: int = 0) -> dict[str, float]:
    """loss_a − loss_b의 날 단위 평균과 사람 단위 부트스트랩 구간·양측 p값.

    losses 열: subject_id, loss_a, loss_b. diff가 양수면 b가 더 낫다(손실이 낮다).
    """
    per_person = (losses.assign(delta=losses["loss_a"] - losses["loss_b"])
                  .groupby("subject_id")["delta"].agg(["sum", "count"]))
    sums = per_person["sum"].to_numpy()
    counts = per_person["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    samples = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - INTERVAL_LEVEL) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    below = (np.sum(samples <= 0) + 1) / (n_boot + 1)
    above = (np.sum(samples >= 0) + 1) / (n_boot + 1)
    return {"diff": float(sums.sum() / counts.sum()), "low": float(low), "high": float(high),
            "p_value": float(min(1.0, 2 * min(below, above)))}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm 보정 p값. 작은 값부터 (m − 순위)를 곱하고 단조 증가하도록 맞춘다."""
    ordered = sorted(p_values, key=lambda k: p_values[k])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted = {**adjusted, name: running}
    return adjusted
