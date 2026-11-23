import pandas as pd
import pytest

from baselines import PROB_FLOOR, global_rate_prediction, personal_rate_prediction


def make_split() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["id01"] * 6 + ["id02"] * 6,
        "part": (["train"] * 4 + ["test"] * 2) * 2,
        "Q1": [1, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 1, 1],
    })


def test_global_rate_uses_only_train_rows_of_everyone():
    prediction = global_rate_prediction(make_split(), "Q1")

    assert prediction["prob"].tolist() == pytest.approx([3 / 8] * 4)
    assert prediction["clipped"].sum() == 0


def test_personal_rate_uses_only_that_persons_train_rows():
    prediction = personal_rate_prediction(make_split(), "Q1")

    assert prediction.loc[prediction["subject_id"] == "id01", "prob"].tolist() == \
        pytest.approx([0.75, 0.75])


def test_personal_rate_of_zero_is_clipped_and_flagged():
    prediction = personal_rate_prediction(make_split(), "Q1")

    second = prediction[prediction["subject_id"] == "id02"]
    assert second["prob"].tolist() == pytest.approx([PROB_FLOOR, PROB_FLOOR])
    assert second["clipped"].all()


def test_predictions_cover_exactly_the_test_rows():
    frame = make_split()

    prediction = personal_rate_prediction(frame, "Q1")

    assert prediction.index.tolist() == frame.index[frame["part"] == "test"].tolist()
    assert prediction["y_true"].tolist() == [0, 0, 1, 1]


def test_person_without_train_rows_is_rejected():
    frame = make_split()
    frame = frame[~((frame["subject_id"] == "id02") & (frame["part"] == "train"))]

    with pytest.raises(ValueError):
        personal_rate_prediction(frame, "Q1")
