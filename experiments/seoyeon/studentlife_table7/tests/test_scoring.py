"""UCLA 외로움 채점 테스트 (가짜 응답)."""
import numpy as np
import pandas as pd
import pytest

import scoring


def _frame(answers: list[str], uid: str = "u00", kind: str = "post") -> pd.DataFrame:
    cols = ["uid", "type"] + [f"{i}. item" for i in range(1, 21)]
    return pd.DataFrame([[uid, kind] + answers], columns=cols)


def test_all_never_scores_47_because_nine_items_are_reversed():
    # 정방향 11문항 x 1점 + 역채점 9문항 x 4점
    out = scoring.ucla_scores(_frame(["Never"] * 20))
    assert out["score"].iloc[0] == 11 * 1 + 9 * 4


def test_all_often_scores_53():
    out = scoring.ucla_scores(_frame(["Often"] * 20))
    assert out["score"].iloc[0] == 11 * 4 + 9 * 1


def test_reverse_items_are_the_nine_positive_items_of_version_3():
    assert scoring.REVERSE_ITEMS == (1, 5, 6, 9, 10, 15, 16, 19, 20)


def test_least_lonely_pattern_scores_20_and_most_lonely_80():
    least = ["Often" if i in scoring.REVERSE_ITEMS else "Never" for i in range(1, 21)]
    most = ["Never" if i in scoring.REVERSE_ITEMS else "Often" for i in range(1, 21)]
    assert scoring.ucla_scores(_frame(least))["score"].iloc[0] == 20
    assert scoring.ucla_scores(_frame(most))["score"].iloc[0] == 80


def test_missing_item_gives_nan_score_instead_of_partial_sum():
    answers = ["Rarely"] * 20
    answers[3] = np.nan
    out = scoring.ucla_scores(_frame(answers))
    assert np.isnan(out["score"].iloc[0])


def test_unknown_answer_text_raises():
    answers = ["Rarely"] * 19 + ["Always"]
    with pytest.raises(ValueError):
        scoring.ucla_scores(_frame(answers))


def test_output_keeps_uid_and_type_and_does_not_mutate_input():
    df = _frame(["Sometimes"] * 20, uid="u07", kind="pre")
    before = df.copy()
    out = scoring.ucla_scores(df)
    assert list(out.columns) == ["uid", "type", "score"]
    assert out.iloc[0]["uid"] == "u07" and out.iloc[0]["type"] == "pre"
    pd.testing.assert_frame_equal(df, before)
