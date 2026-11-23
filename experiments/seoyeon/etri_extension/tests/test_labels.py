import pandas as pd
import pytest

from labels import TARGETS, binarize_s1, load_labels

HEADER = "subject_id,sleep_date,lifelog_date,Q1,Q2,Q3,S1,S2,S3\n"


def write_csv(tmp_path, rows: str):
    path = tmp_path / "labels.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_binarize_s1_treats_one_and_two_as_positive():
    result = binarize_s1(pd.Series([0, 1, 2]))

    assert result.tolist() == [0, 1, 1]


def test_binarize_s1_rejects_unknown_value():
    with pytest.raises(ValueError):
        binarize_s1(pd.Series([0, 3]))


def test_load_labels_returns_binary_targets_sorted_by_person_and_date(tmp_path):
    path = write_csv(tmp_path, "id02,2024-07-02,2024-07-01,1,0,1,2,1,0\n"
                               "id01,2024-07-03,2024-07-02,0,1,0,1,0,1\n"
                               "id01,2024-07-02,2024-07-01,1,1,1,0,1,1\n")

    labels = load_labels(path)

    assert labels["subject_id"].tolist() == ["id01", "id01", "id02"]
    assert labels["lifelog_date"].tolist() == list(pd.to_datetime(
        ["2024-07-01", "2024-07-02", "2024-07-01"]))
    assert labels["S1"].tolist() == [0, 1, 1]
    assert set(labels[list(TARGETS)].to_numpy().ravel()) <= {0, 1}


def test_load_labels_rejects_duplicate_person_date(tmp_path):
    path = write_csv(tmp_path, "id01,2024-07-02,2024-07-01,1,1,1,0,1,1\n"
                               "id01,2024-07-02,2024-07-01,0,1,1,0,1,1\n")

    with pytest.raises(ValueError):
        load_labels(path)


def test_load_labels_rejects_non_binary_survey_value(tmp_path):
    path = write_csv(tmp_path, "id01,2024-07-02,2024-07-01,2,1,1,0,1,1\n")

    with pytest.raises(ValueError):
        load_labels(path)
