import pandas as pd
import pytest

from baselines import PROB_FLOOR
from coldstart import (leave_one_out_global_prediction, make_mixed_predictor,
                       make_personal_predictor)


def make_split() -> pd.DataFrame:
    rows = []
    train = {"id01": [0, 0, 0, 1, 1, 1], "id02": [1, 1, 1, 1, 1, 0], "id03": [0, 0, 0, 0, 0, 1]}
    for person, values in train.items():
        rows += [{"subject_id": person, "part": "train", "Q1": v} for v in values]
        rows += [{"subject_id": person, "part": "test", "Q1": v} for v in (1, 0)]
    return pd.DataFrame(rows)


def probs_of(prediction: pd.DataFrame, person: str) -> list[float]:
    return prediction.loc[prediction["subject_id"] == person, "prob"].tolist()


def test_leave_one_out_global_excludes_the_persons_own_train_rows():
    prediction = leave_one_out_global_prediction(make_split(), "Q1")

    assert probs_of(prediction, "id01") == pytest.approx([6 / 12] * 2)
    assert probs_of(prediction, "id02") == pytest.approx([4 / 12] * 2)


def test_personal_predictor_uses_only_the_last_k_train_days():
    prediction = make_personal_predictor(3)(make_split(), "Q1")

    assert probs_of(prediction, "id01") == pytest.approx([1 - PROB_FLOOR] * 2)
    assert probs_of(prediction, "id02") == pytest.approx([2 / 3] * 2)


def test_personal_predictor_with_none_uses_all_train_days():
    prediction = make_personal_predictor(None)(make_split(), "Q1")

    assert probs_of(prediction, "id01") == pytest.approx([0.5] * 2)


def test_personal_predictor_with_zero_days_is_rejected():
    with pytest.raises(ValueError):
        make_personal_predictor(0)


def test_mixed_predictor_with_zero_days_equals_leave_one_out_global():
    split = make_split()

    mixed = make_mixed_predictor(0)(split, "Q1")
    baseline = leave_one_out_global_prediction(split, "Q1")

    assert mixed["prob"].tolist() == pytest.approx(baseline["prob"].tolist())


def test_mixed_predictor_lies_between_global_and_personal():
    split = make_split()

    mixed = probs_of(make_mixed_predictor(3)(split, "Q1"), "id02")[0]
    personal = probs_of(make_personal_predictor(3)(split, "Q1"), "id02")[0]
    baseline = probs_of(leave_one_out_global_prediction(split, "Q1"), "id02")[0]

    assert min(baseline, personal) <= mixed <= max(baseline, personal)


def test_predictions_keep_test_rows_and_truth():
    split = make_split()

    prediction = make_mixed_predictor(3)(split, "Q1")

    assert prediction.index.tolist() == split.index[split["part"] == "test"].tolist()
    assert prediction["y_true"].tolist() == [1, 0] * 3


def test_mixed_predictor_with_given_strength_matches_hand_calculation():
    split = make_split()

    mixed = make_mixed_predictor(3, strengths={"Q1": 3.0})(split, "Q1")

    assert probs_of(mixed, "id02") == pytest.approx([(2 + 3.0 * (4 / 12)) / (3 + 3.0)] * 2)


def test_mixed_predictor_rejects_missing_strength_for_target():
    with pytest.raises(ValueError):
        make_mixed_predictor(3, strengths={"Q2": 3.0})(make_split(), "Q1")
