"""사람 단위 요약, 상관, 논문 표 7 비교 테스트 (가짜 데이터)."""
import numpy as np
import pandas as pd
import pytest

import table7 as t7


def daily(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["uid", "date", "act_all", "n_bins"])


def test_paper_values_are_the_six_rows_of_table_7():
    assert t7.PAPER_TABLE7 == {
        "act_all": (-0.388, 0.018), "act_day": (-0.326, 0.049), "act_evening": (-0.464, 0.004),
        "dist_all": (-0.338, 0.044), "dist_day": (-0.336, 0.042), "indoor_day": (-0.332, 0.045),
    }


def test_person_summary_mean_and_median():
    d = daily([("u00", "2013-04-01", 10, 144), ("u00", "2013-04-02", 20, 144), ("u00", "2013-04-03", 90, 144),
               ("u01", "2013-04-01", 30, 144)])
    assert t7.person_summary(d, ["act_all"], agg="mean")["act_all"].to_dict() == {"u00": 40.0, "u01": 30.0}
    assert t7.person_summary(d, ["act_all"], agg="median")["act_all"].to_dict() == {"u00": 20.0, "u01": 30.0}


def test_person_summary_drops_days_with_too_few_recorded_bins():
    d = daily([("u00", "2013-04-01", 10, 144), ("u00", "2013-04-02", 500, 20)])
    assert t7.person_summary(d, ["act_all"], min_bins=72)["act_all"].to_dict() == {"u00": 10.0}


def test_person_summary_rejects_unknown_aggregation():
    with pytest.raises(ValueError):
        t7.person_summary(daily([("u00", "2013-04-01", 10, 144)]), ["act_all"], agg="max")


def test_correlate_uses_only_people_with_both_values():
    feats = pd.DataFrame({"act_all": [1.0, 2.0, 3.0, 4.0, np.nan]}, index=["u00", "u01", "u02", "u03", "u04"])
    scores = pd.Series({"u00": 40.0, "u01": 30.0, "u02": 20.0, "u03": 10.0, "u04": 99.0, "u09": 50.0})
    out = t7.correlate(feats, scores).iloc[0]
    assert out["feature"] == "act_all" and out["n"] == 4
    assert out["r"] == pytest.approx(-1.0)


def test_correlate_returns_nan_when_fewer_than_three_people():
    feats = pd.DataFrame({"act_all": [1.0, 2.0]}, index=["u00", "u01"])
    out = t7.correlate(feats, pd.Series({"u00": 1.0, "u01": 2.0})).iloc[0]
    assert out["n"] == 2 and np.isnan(out["r"])


def test_compare_with_paper_reports_difference_sign_and_tolerance():
    corr = pd.DataFrame({"feature": ["act_all", "dist_all"], "n": [37, 37], "r": [-0.30, 0.10], "p": [0.07, 0.55]})
    out = t7.compare_with_paper(corr, tolerance=0.10).set_index("feature")
    assert out.loc["act_all", "paper_r"] == -0.388
    assert out.loc["act_all", "diff"] == pytest.approx(0.088)
    assert bool(out.loc["act_all", "same_sign"]) and bool(out.loc["act_all", "within_tol"])
    assert not bool(out.loc["dist_all", "same_sign"]) and not bool(out.loc["dist_all", "within_tol"])


def test_sensitivity_summary_gives_range_per_feature():
    runs = pd.DataFrame({"feature": ["act_all"] * 3 + ["dist_all"] * 2, "r": [-0.5, -0.2, -0.3, 0.1, -0.1]})
    out = t7.sensitivity_summary(runs).set_index("feature")
    assert (out.loc["act_all", "r_min"], out.loc["act_all", "r_max"], out.loc["act_all", "n_runs"]) == (-0.5, -0.2, 3)
    assert out.loc["act_all", "share_negative"] == 1.0
    assert out.loc["dist_all", "share_negative"] == 0.5


def test_sensitivity_summary_ignores_runs_whose_r_could_not_be_computed():
    runs = pd.DataFrame({"feature": ["act_all", "act_all"], "r": [-0.3, np.nan]})
    out = t7.sensitivity_summary(runs).set_index("feature")
    assert out.loc["act_all", "n_runs"] == 1
    assert out.loc["act_all", "share_negative"] == 1.0
