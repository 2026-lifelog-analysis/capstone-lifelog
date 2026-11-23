import numpy as np
import pytest

from shrinkage import PRIOR_STRENGTH_CAP, estimate_prior_strength, shrunk_rate


def test_shrunk_rate_with_no_personal_days_is_global_rate():
    assert shrunk_rate(successes=0, n_days=0, global_rate=0.6, strength=5.0) == pytest.approx(0.6)


def test_shrunk_rate_with_zero_strength_is_personal_rate():
    assert shrunk_rate(successes=3, n_days=4, global_rate=0.2, strength=0.0) == pytest.approx(0.75)


def test_shrunk_rate_matches_hand_calculation():
    result = shrunk_rate(successes=3, n_days=4, global_rate=0.5, strength=4.0)

    assert result == pytest.approx((3 + 4 * 0.5) / (4 + 4))


def test_shrunk_rate_moves_toward_personal_rate_as_days_grow():
    few = shrunk_rate(successes=3, n_days=3, global_rate=0.5, strength=10.0)
    many = shrunk_rate(successes=30, n_days=30, global_rate=0.5, strength=10.0)

    assert 0.5 < few < many < 1.0


def test_strength_is_capped_when_people_do_not_differ_beyond_sampling_noise():
    rates = np.array([0.5, 0.5, 0.5, 0.5])
    counts = np.array([20, 20, 20, 20])

    assert estimate_prior_strength(rates, counts) == PRIOR_STRENGTH_CAP


def test_strength_is_small_when_people_differ_strongly():
    rates = np.array([0.05, 0.95, 0.1, 0.9])
    counts = np.array([50, 50, 50, 50])

    assert estimate_prior_strength(rates, counts) < 1.0


def test_strength_matches_hand_calculation():
    rates = np.array([0.3, 0.7])
    counts = np.array([10, 10])
    pooled = 0.5
    between = 0.04
    sampling = pooled * (1 - pooled) / 10
    expected = pooled * (1 - pooled) / (between - sampling) - 1

    assert estimate_prior_strength(rates, counts) == pytest.approx(expected)


def test_strength_from_icc_matches_beta_binomial_relation():
    from shrinkage import strength_from_icc

    assert strength_from_icc(0.25) == pytest.approx(3.0)
    assert strength_from_icc(0.1) == pytest.approx(9.0)


def test_strength_from_icc_rejects_values_outside_open_unit_interval():
    from shrinkage import strength_from_icc

    with pytest.raises(ValueError):
        strength_from_icc(0.0)
    with pytest.raises(ValueError):
        strength_from_icc(1.0)
