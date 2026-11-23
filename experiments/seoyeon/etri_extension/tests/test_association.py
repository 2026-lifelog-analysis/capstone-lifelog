import numpy as np
import pandas as pd
import pytest

from association import within_person_correlation


def make_frame(sign: float, n_people: int = 6, days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for person in range(n_people):
        level = person * 10.0
        for _ in range(days):
            x = rng.normal()
            rows.append({"subject_id": f"id{person:02d}", "x": level + x,
                         "y": level * 5 + sign * x + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


def test_between_person_level_differences_do_not_create_correlation():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"subject_id": np.repeat(["a", "b", "c", "d"], 50),
                          "x": np.repeat([0.0, 10.0, 20.0, 30.0], 50) + rng.normal(size=200),
                          "y": np.repeat([0.0, 10.0, 20.0, 30.0], 50) + rng.normal(size=200)})

    result = within_person_correlation(frame, "x", "y", n_boot=300, seed=0)

    assert abs(result["r"]) < 0.2


def test_positive_within_person_relation_is_detected():
    result = within_person_correlation(make_frame(1.0), "x", "y", n_boot=300, seed=0)

    assert result["r"] > 0.9
    assert result["low"] > 0


def test_negative_within_person_relation_is_detected():
    result = within_person_correlation(make_frame(-1.0), "x", "y", n_boot=300, seed=0)

    assert result["r"] < -0.9
    assert result["high"] < 0


def test_constant_feature_gives_nan():
    frame = make_frame(1.0).assign(x=1.0)

    result = within_person_correlation(frame, "x", "y", n_boot=100, seed=0)

    assert np.isnan(result["r"])


def test_result_reports_people_and_days():
    result = within_person_correlation(make_frame(1.0), "x", "y", n_boot=100, seed=0)

    assert result["n_people"] == 6
    assert result["n_days"] == 180
    assert 0 <= result["p_value"] <= 1
    assert result["r"] == pytest.approx(result["r"])
