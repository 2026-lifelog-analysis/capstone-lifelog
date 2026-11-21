import numpy as np
import pandas as pd
import pytest

from scoring import auroc, daily_log_loss, holm_adjust, paired_person_bootstrap


def test_daily_log_loss_matches_hand_calculation():
    losses = daily_log_loss(np.array([1, 0]), np.array([0.8, 0.8]))

    assert losses == pytest.approx([-np.log(0.8), -np.log(0.2)])


def test_daily_log_loss_rejects_probability_outside_open_interval():
    with pytest.raises(ValueError):
        daily_log_loss(np.array([1]), np.array([1.0]))


def test_auroc_is_one_for_perfect_ranking_and_half_for_constant():
    y = np.array([0, 0, 1, 1])

    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_auroc_is_nan_when_only_one_class_present():
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.2, 0.9])))


def make_losses(diff_per_person: list[float], days: int = 5) -> pd.DataFrame:
    rows = [{"subject_id": f"id{i:02d}", "loss_a": 1.0 + d, "loss_b": 1.0}
            for i, d in enumerate(diff_per_person) for _ in range(days)]
    return pd.DataFrame(rows)


def test_bootstrap_point_estimate_is_pooled_day_mean_difference():
    frame = make_losses([0.2, 0.4])

    result = paired_person_bootstrap(frame, n_boot=200, seed=0)

    assert result["diff"] == pytest.approx(0.3)


def test_bootstrap_interval_excludes_zero_when_every_person_agrees():
    frame = make_losses([0.2, 0.3, 0.25, 0.35, 0.3, 0.2])

    result = paired_person_bootstrap(frame, n_boot=500, seed=0)

    assert result["low"] > 0
    assert result["p_value"] < 0.05


def test_bootstrap_interval_includes_zero_when_people_disagree():
    frame = make_losses([0.3, -0.3, 0.3, -0.3, 0.3, -0.3])

    result = paired_person_bootstrap(frame, n_boot=500, seed=0)

    assert result["low"] < 0 < result["high"]


def test_bootstrap_is_reproducible_with_same_seed():
    frame = make_losses([0.3, -0.1, 0.2, 0.0])

    first = paired_person_bootstrap(frame, n_boot=300, seed=7)
    second = paired_person_bootstrap(frame, n_boot=300, seed=7)

    assert first == second


def test_holm_adjust_matches_hand_calculation():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_holm_adjust_caps_at_one():
    adjusted = holm_adjust({"a": 0.6, "b": 0.9})

    assert adjusted == pytest.approx({"a": 1.0, "b": 1.0})
