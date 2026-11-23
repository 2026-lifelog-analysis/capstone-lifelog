import numpy as np
import pandas as pd
import pytest

from rhythm import (SD_MIN_DAYS, SD_WINDOW, features_for_days, rhythm_features, rolling_sd,
                    screen_timing)


def full_day(subject: str, date: str, use_from: str, use_to: str) -> pd.DataFrame:
    """date의 05:00부터 다음 날 05:00까지 1분 간격 기록. use_from~use_to(시각 문자열)만 화면 사용."""
    start = pd.Timestamp(f"{date} 05:00")
    stamps = pd.date_range(start, periods=24 * 60, freq="min")
    on = (stamps >= pd.Timestamp(use_from)) & (stamps <= pd.Timestamp(use_to))
    return pd.DataFrame({"subject_id": subject, "timestamp": stamps, "m_screen_use": on.astype(int)})


def test_onset_and_offset_are_minutes_from_5am_and_cross_midnight():
    records = full_day("id01", "2024-07-01", "2024-07-01 07:30", "2024-07-02 01:00")

    timing = screen_timing(records)

    row = timing.loc[("id01", pd.Timestamp("2024-07-01"))]
    assert row["onset_min"] == 150
    assert row["offset_min"] == 20 * 60


def test_records_before_5am_belong_to_the_previous_rhythm_day():
    records = full_day("id01", "2024-07-01", "2024-07-02 02:00", "2024-07-02 02:10")

    timing = screen_timing(records)

    assert list(timing.index.get_level_values("date")) == [pd.Timestamp("2024-07-01")]


def test_onset_is_missing_when_recording_starts_after_the_first_hour():
    records = full_day("id01", "2024-07-01", "2024-07-01 13:00", "2024-07-01 23:00")
    late_start = records[records["timestamp"] >= pd.Timestamp("2024-07-01 12:00")]

    timing = screen_timing(late_start)

    row = timing.loc[("id01", pd.Timestamp("2024-07-01"))]
    assert np.isnan(row["onset_min"])
    assert row["offset_min"] == 18 * 60


def test_offset_is_missing_when_recording_stops_before_the_last_hour():
    records = full_day("id01", "2024-07-01", "2024-07-01 08:00", "2024-07-01 22:00")
    early_stop = records[records["timestamp"] < pd.Timestamp("2024-07-02 00:00")]

    timing = screen_timing(early_stop)

    row = timing.loc[("id01", pd.Timestamp("2024-07-01"))]
    assert row["onset_min"] == 180
    assert np.isnan(row["offset_min"])


def test_rolling_sd_uses_past_seven_calendar_days_including_today():
    dates = pd.date_range("2024-07-01", periods=8)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
                       index=pd.MultiIndex.from_product([["id01"], dates],
                                                        names=["subject_id", "date"]))

    result = rolling_sd(values)

    assert result.loc[("id01", dates[6])] == pytest.approx(np.std(np.arange(1.0, 8.0), ddof=1))
    assert result.loc[("id01", dates[7])] == pytest.approx(
        np.std([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0], ddof=1))
    assert SD_WINDOW == 7


def test_rolling_sd_needs_five_recorded_days_and_a_value_today():
    dates = pd.date_range("2024-07-01", periods=7)
    index = pd.MultiIndex.from_product([["id01"], dates], names=["subject_id", "date"])
    four = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, 3.0, 4.0], index=index)
    today_missing = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan], index=index)

    assert np.isnan(rolling_sd(four).loc[("id01", dates[6])])
    assert np.isnan(rolling_sd(today_missing).loc[("id01", dates[6])])
    assert SD_MIN_DAYS == 5


def test_rolling_sd_counts_calendar_gaps_as_missing_days():
    dates = list(pd.date_range("2024-07-01", periods=4)) + [pd.Timestamp("2024-07-20")]
    index = pd.MultiIndex.from_arrays([["id01"] * 5, dates], names=["subject_id", "date"])
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

    assert np.isnan(rolling_sd(values).loc[("id01", pd.Timestamp("2024-07-20"))])


def test_rhythm_features_and_alignment_to_labeled_days():
    records = pd.concat([full_day("id01", f"2024-07-0{d}", f"2024-07-0{d} 07:0{d}",
                                  f"2024-07-0{d} 23:00") for d in range(1, 8)])

    features = rhythm_features(records)
    days = pd.DataFrame({"subject_id": ["id01", "id01"],
                         "lifelog_date": pd.to_datetime(["2024-07-07", "2024-07-30"])})
    aligned = features_for_days(features, days)

    assert list(aligned.columns) == ["onset_min", "offset_min", "onset_sd7", "offset_sd7"]
    assert aligned.loc[0, "onset_min"] == 127
    assert aligned.loc[0, "onset_sd7"] == pytest.approx(np.std(np.arange(1.0, 8.0), ddof=1))
    assert aligned.loc[1].isna().all()
    assert aligned.index.equals(days.index)
