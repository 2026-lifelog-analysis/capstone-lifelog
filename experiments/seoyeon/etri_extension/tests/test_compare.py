import numpy as np
import pandas as pd

from baselines import global_rate_prediction, personal_rate_prediction
from compare import compare_predictors
from labels import TARGETS


def make_split(n_people: int = 6, days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for person in range(n_people):
        rate = 0.8 if person % 2 == 0 else 0.2
        for day in range(days):
            row = {"subject_id": f"id{person:02d}", "part": "train" if day < days // 2 else "test"}
            row.update({target: int(rng.random() < rate) for target in TARGETS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_has_one_row_per_target():
    summary = compare_predictors(make_split(), global_rate_prediction,
                                 personal_rate_prediction, n_boot=200)

    assert summary["target"].tolist() == list(TARGETS)
    assert (summary["n_people"] == 6).all()
    assert (summary["n_days"] == 360).all()


def test_personal_rate_wins_when_people_differ_strongly():
    summary = compare_predictors(make_split(), global_rate_prediction,
                                 personal_rate_prediction, n_boot=500)

    assert summary["b_wins"].all()
    assert (summary["logloss_b"] < summary["logloss_a"]).all()


def test_same_predictor_on_both_sides_never_wins():
    summary = compare_predictors(make_split(), personal_rate_prediction,
                                 personal_rate_prediction, n_boot=200)

    assert not summary["b_wins"].any()
    assert (summary["diff"] == 0).all()
