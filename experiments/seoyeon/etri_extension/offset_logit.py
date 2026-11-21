"""
offset_logit.py — 기준값(offset)이 있는 ℓ2 로지스틱 회귀

확률 = sigmoid(offset + 절편 + X·계수).
offset은 이미 알고 있는 기준 예측(여기서는 섞은 비율)의 로짓이고, 모델은 그 위에 더할 것만 배운다.
계수에만 ℓ2 벌점을 주고 절편에는 주지 않는다. 벌점이 아주 크면 기준 예측으로 돌아간다.

근거: 춘자 논문(ICTC 2026 C6-1) IV절이 사람 내 잔차를 ℓ2 로지스틱으로 예측하고,
LOV3 논문(C6-2) 식 (1)이 예측을 로짓 공간에서 결합한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ITERATIONS = 100
TOLERANCE = 1e-9
WEIGHT_FLOOR = 1e-9


@dataclass(frozen=True)
class OffsetLogit:
    intercept: float
    coef: np.ndarray


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fit_offset_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray,
                        l2: float) -> OffsetLogit:
    """뉴턴 방법으로 적합한다. x는 (날 수, 피처 수) 배열."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if not (len(x) == len(y) == len(offset)):
        raise ValueError("x, y, offset의 길이가 서로 다름")
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0.0] + [l2] * x.shape[1])
    beta = np.zeros(design.shape[1])
    for _ in range(MAX_ITERATIONS):
        prob = _sigmoid(offset + design @ beta)
        gradient = design.T @ (prob - y) + penalty @ beta
        weight = np.maximum(prob * (1 - prob), WEIGHT_FLOOR)
        hessian = design.T @ (design * weight[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta - step
        if np.max(np.abs(step)) < TOLERANCE:
            break
    return OffsetLogit(intercept=float(beta[0]), coef=beta[1:].copy())


def predict_offset_logistic(model: OffsetLogit, x: np.ndarray, offset: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return _sigmoid(np.asarray(offset, dtype=float) + model.intercept + x @ model.coef)
