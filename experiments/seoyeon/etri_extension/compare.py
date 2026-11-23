"""
compare.py — 두 예측을 라벨별로 비교해 요약 표를 만든다

각 라벨에서 a와 b의 log loss, 그 차이(a − b)의 사람 단위 부트스트랩 구간, Holm 보정 p값을 낸다.
"b가 이겼다"는 차이가 양수이고 구간이 0을 포함하지 않는 경우다(승호님 문서의 판정 기준).
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from labels import TARGETS
from scoring import auroc, daily_log_loss, holm_adjust, paired_person_bootstrap

Predictor = Callable[[pd.DataFrame, str], pd.DataFrame]


def _compare_target(split: pd.DataFrame, target: str, predict_a: Predictor,
                    predict_b: Predictor, n_boot: int, seed: int) -> dict[str, float | str]:
    a = predict_a(split, target)
    b = predict_b(split, target)
    losses = pd.DataFrame({"subject_id": a["subject_id"],
                           "loss_a": daily_log_loss(a["y_true"], a["prob"]),
                           "loss_b": daily_log_loss(b["y_true"], b["prob"])})
    boot = paired_person_bootstrap(losses, n_boot=n_boot, seed=seed)
    return {"target": target, "n_people": losses["subject_id"].nunique(), "n_days": len(losses),
            "logloss_a": losses["loss_a"].mean(), "logloss_b": losses["loss_b"].mean(),
            "diff": boot["diff"], "low": boot["low"], "high": boot["high"],
            "p_value": boot["p_value"],
            "auroc_a": auroc(a["y_true"], a["prob"]), "auroc_b": auroc(b["y_true"], b["prob"]),
            "clipped_days_a": int(a["clipped"].sum()), "clipped_days_b": int(b["clipped"].sum())}


def compare_predictors(split: pd.DataFrame, predict_a: Predictor, predict_b: Predictor,
                       n_boot: int, seed: int = 0) -> pd.DataFrame:
    """라벨 6개 각각에 대한 비교 요약. b_wins 열이 판정 결과다."""
    rows = [_compare_target(split, target, predict_a, predict_b, n_boot, seed)
            for target in TARGETS]
    summary = pd.DataFrame(rows)
    adjusted = holm_adjust(dict(zip(summary["target"], summary["p_value"])))
    return summary.assign(p_holm=summary["target"].map(adjusted),
                          b_wins=(summary["diff"] > 0) & (summary["low"] > 0))
