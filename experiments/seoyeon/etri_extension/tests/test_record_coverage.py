import pandas as pd
import pytest

from record_coverage import DAY_MINUTES, NIGHT_MINUTES, coverage_for_days, daily_coverage


def stream(subject: str, start: str, minutes: int) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": subject,
                         "timestamp": pd.date_range(start, periods=minutes, freq="min")})


def test_day_coverage_is_recorded_minutes_over_full_day():
    records = stream("id01", "2024-07-01 00:00", 720)

    result = daily_coverage(records, "day")

    assert result.loc[("id01", pd.Timestamp("2024-07-01"))] == pytest.approx(720 / DAY_MINUTES)


def test_duplicate_records_in_the_same_minute_count_once():
    records = pd.concat([stream("id01", "2024-07-01 00:00", 60)] * 2)

    result = daily_coverage(records, "day")

    assert result.loc[("id01", pd.Timestamp("2024-07-01"))] == pytest.approx(60 / DAY_MINUTES)


def test_night_window_runs_from_22h_to_10h_and_belongs_to_the_evening_date():
    records = stream("id01", "2024-07-01 21:00", 14 * 60)

    result = daily_coverage(records, "night")

    assert result.loc[("id01", pd.Timestamp("2024-07-01"))] == pytest.approx(1.0)
    assert len(result) == 1
    assert NIGHT_MINUTES == 720


def test_daytime_records_do_not_count_toward_night_coverage():
    records = stream("id01", "2024-07-01 12:00", 120)

    assert daily_coverage(records, "night").empty


def test_unknown_window_is_rejected():
    with pytest.raises(ValueError):
        daily_coverage(stream("id01", "2024-07-01 00:00", 10), "week")


def test_coverage_for_days_fills_days_without_records_with_zero():
    records = stream("id01", "2024-07-01 00:00", 1440)
    days = pd.DataFrame({"subject_id": ["id01", "id01"],
                         "lifelog_date": pd.to_datetime(["2024-07-01", "2024-07-02"])})

    result = coverage_for_days(records, days, "day")

    assert result.tolist() == pytest.approx([1.0, 0.0])
    assert result.index.equals(days.index)
