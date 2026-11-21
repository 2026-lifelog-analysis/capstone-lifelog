import numpy as np
import pytest

from offset_logit import fit_offset_logistic, predict_offset_logistic


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def simulate(coef: float, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    offset = rng.normal(scale=0.5, size=n)
    y = (rng.random(n) < sigmoid(offset + coef * x[:, 0])).astype(float)
    return x, y, offset


def test_recovers_known_coefficient_with_weak_penalty():
    x, y, offset = simulate(coef=1.0)

    model = fit_offset_logistic(x, y, offset, l2=1e-6)

    assert model.coef[0] == pytest.approx(1.0, abs=0.1)
    assert model.intercept == pytest.approx(0.0, abs=0.1)


def test_strong_penalty_shrinks_coefficient_to_zero():
    x, y, offset = simulate(coef=1.0)

    model = fit_offset_logistic(x, y, offset, l2=1e9)

    assert abs(model.coef[0]) < 1e-3


def test_prediction_with_zero_coefficients_returns_offset_probability():
    x, y, offset = simulate(coef=0.0)

    model = fit_offset_logistic(x, y, offset, l2=1e9)
    prob = predict_offset_logistic(model, x, offset)

    assert prob == pytest.approx(sigmoid(offset + model.intercept))
    assert abs(model.intercept) < 0.1


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_offset_logistic(np.zeros((3, 1)), np.zeros(2), np.zeros(3), l2=1.0)
