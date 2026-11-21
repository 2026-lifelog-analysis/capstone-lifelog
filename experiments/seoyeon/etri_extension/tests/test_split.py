import pandas as pd
import pytest

from split import GAP_DAYS, assign_split


def person_frame(subject: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": subject, "lifelog_date": pd.to_datetime(dates)})


def consecutive(start: str, n: int) -> list[str]:
    return [str(d.date()) for d in pd.date_range(start, periods=n)]


def test_first_half_of_labeled_days_is_train():
    frame = person_frame("id01", consecutive("2024-07-01", 20))

    result = assign_split(frame)

    assert (result["part"].iloc[:10] == "train").all()
    assert (result["part"].iloc[10:] != "train").all()


def test_days_within_gap_after_train_end_are_excluded_from_test():
    frame = person_frame("id01", consecutive("2024-07-01", 30))

    result = assign_split(frame)

    train_end = result.loc[result["part"] == "train", "lifelog_date"].max()
    test_start = result.loc[result["part"] == "test", "lifelog_date"].min()
    assert (test_start - train_end).days == GAP_DAYS + 1
    assert (result["part"] == "gap").sum() == GAP_DAYS


def test_split_is_computed_per_person():
    frame = pd.concat([person_frame("id01", consecutive("2024-07-01", 20)),
                       person_frame("id02", consecutive("2024-09-01", 40))],
                      ignore_index=True)

    result = assign_split(frame)

    counts = result[result["part"] == "train"].groupby("subject_id").size()
    assert counts.to_dict() == {"id01": 10, "id02": 20}


def test_calendar_gap_in_labels_counts_toward_the_gap():
    dates = consecutive("2024-07-01", 4) + consecutive("2024-07-20", 4)
    frame = person_frame("id01", dates)

    result = assign_split(frame)

    assert result["part"].tolist() == ["train"] * 4 + ["test"] * 4


def test_input_frame_is_not_mutated():
    frame = person_frame("id01", consecutive("2024-07-01", 20))

    assign_split(frame)

    assert "part" not in frame.columns


def test_unsorted_input_is_rejected():
    frame = person_frame("id01", ["2024-07-02", "2024-07-01"])

    with pytest.raises(ValueError):
        assign_split(frame)
