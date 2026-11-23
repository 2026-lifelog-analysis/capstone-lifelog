"""일별 특징 계산 테스트 (가짜 데이터). 실제 StudentLife 파일은 쓰지 않는다."""
import numpy as np
import pandas as pd
import pytest

import features as ft

DAY = "2013-04-01"


def ts(clock: str, day: str = DAY) -> int:
    """미국 동부 시각 문자열을 유닉스 초로 바꾼다."""
    return int(pd.Timestamp(f"{day} {clock}", tz=ft.TZ).timestamp())


def activity(rows: list[tuple[str, list[int]]], day: str = DAY) -> pd.DataFrame:
    """(구간 시작 시각, 추론값 목록)들로 활동 추론 표를 만든다. 추론은 3초 간격."""
    out = []
    for clock, values in rows:
        start = ts(clock, day)
        out += [(start + 3 * i, v) for i, v in enumerate(values)]
    return pd.DataFrame(out, columns=["timestamp", "inference"])


# ---------- 시간 ----------

@pytest.mark.parametrize("hour,expected", [(0, "night"), (8, "night"), (9, "day"), (17, "day"),
                                             (18, "evening"), (23, "evening")])
def test_epoch_boundaries_follow_the_paper(hour, expected):
    assert ft.epoch_of_hour(hour) == expected


def test_to_local_uses_us_eastern_time():
    local = ft.to_local(pd.Series([ts("09:00:00")]))
    assert local.dt.hour.iloc[0] == 9 and str(local.dt.date.iloc[0]) == DAY


# ---------- 활동 시간 ----------

def test_activity_bins_count_each_inference_kind():
    bins = ft.activity_bins(activity([("10:00:00", [0, 1, 1, 2, 3, 0])]))
    row = bins.iloc[0]
    assert len(bins) == 1
    assert (row["n_total"], row["n_moving"], row["n_walk"], row["n_unknown"]) == (6, 3, 2, 1)
    assert row["bin_start"] == ts("10:00:00")


def test_bin_is_active_only_when_ratio_is_strictly_greater_than_threshold():
    acts = activity([("10:00:00", [1, 1, 1, 1, 0, 0]),      # 4/6 > 0.5 -> 움직임
                     ("11:00:00", [1, 1, 1, 0, 0, 0])])     # 3/6 = 0.5 -> 아님
    daily = ft.daily_activity(ft.activity_bins(acts), threshold=0.5)
    assert daily["act_all"].iloc[0] == 10


def test_unknown_inferences_can_be_left_out_of_the_denominator():
    bins = ft.activity_bins(activity([("10:00:00", [1, 3, 3, 0])]))
    with_unknown = ft.daily_activity(bins, threshold=0.4, include_unknown=True)
    without_unknown = ft.daily_activity(bins, threshold=0.4, include_unknown=False)
    assert with_unknown["act_all"].iloc[0] == 0        # 1/4 = 0.25
    assert without_unknown["act_all"].iloc[0] == 10    # 1/2 = 0.5 > 0.4


def test_active_minutes_are_split_into_paper_epochs():
    moving = [1, 1, 1, 1]
    acts = activity([(c, moving) for c in ("00:00:00", "08:50:00", "09:00:00", "17:50:00", "18:00:00", "23:50:00")])
    daily = ft.daily_activity(ft.activity_bins(acts), threshold=0.5).iloc[0]
    assert (daily["act_night"], daily["act_day"], daily["act_evening"], daily["act_all"]) == (20, 20, 20, 60)
    assert daily["n_bins"] == 6


def test_days_are_split_at_local_midnight():
    acts = pd.concat([activity([("23:50:00", [1, 1])]), activity([("00:00:00", [1, 1])], day="2013-04-02")])
    daily = ft.daily_activity(ft.activity_bins(acts), threshold=0.5)
    assert [str(d) for d in daily["date"]] == ["2013-04-01", "2013-04-02"]
    assert list(daily["act_all"]) == [10, 10]


def test_activity_functions_do_not_mutate_input():
    acts = activity([("10:00:00", [1, 0])])
    before = acts.copy()
    ft.daily_activity(ft.activity_bins(acts), threshold=0.1)
    pd.testing.assert_frame_equal(acts, before)


# ---------- 이동 거리 ----------

def gps(rows: list[tuple[str, float, float, float]], day: str = DAY) -> pd.DataFrame:
    return pd.DataFrame([(ts(c, day), lat, lon, acc) for c, lat, lon, acc in rows],
                        columns=["time", "latitude", "longitude", "accuracy"])


ONE_STEP = 111.19  # 적도에서 경도 0.001도의 거리(m)


def test_haversine_matches_known_distance():
    assert ft.haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_195, abs=50)
    assert ft.haversine_m(43.7, -72.28, 43.7, -72.28) == 0


def test_distance_sums_consecutive_points_and_day_epoch_needs_both_ends_inside():
    points = gps([("10:00:00", 0, 0.000, 20), ("10:10:00", 0, 0.001, 20), ("19:00:00", 0, 0.002, 20)])
    daily = ft.daily_distance(points).iloc[0]
    assert daily["dist_all"] == pytest.approx(2 * ONE_STEP, abs=0.5)
    assert daily["dist_day"] == pytest.approx(ONE_STEP, abs=0.5)   # 10:10 -> 19:00 구간은 낮 밖에서 끝남
    assert daily["n_points"] == 3


def test_day_epoch_distance_is_zero_when_the_segment_starts_before_nine():
    points = gps([("08:50:00", 0, 0.000, 20), ("09:10:00", 0, 0.001, 20)])
    daily = ft.daily_distance(points).iloc[0]
    assert daily["dist_day"] == 0
    assert daily["dist_all"] == pytest.approx(ONE_STEP, abs=0.5)


def test_inaccurate_points_are_dropped_when_a_limit_is_given():
    points = gps([("10:00:00", 0, 0.000, 20), ("10:10:00", 0, 0.050, 500), ("10:20:00", 0, 0.001, 20)])
    assert ft.daily_distance(points)["dist_all"].iloc[0] > 10 * ONE_STEP
    assert ft.daily_distance(points, max_accuracy=100)["dist_all"].iloc[0] == pytest.approx(ONE_STEP, abs=0.5)


def test_long_gaps_are_skipped_when_a_gap_limit_is_given():
    points = gps([("10:00:00", 0, 0.000, 20), ("10:10:00", 0, 0.001, 20), ("15:00:00", 0, 0.002, 20)])
    assert ft.daily_distance(points, max_gap=3600)["dist_all"].iloc[0] == pytest.approx(ONE_STEP, abs=0.5)


def test_segments_across_midnight_are_not_counted():
    points = pd.concat([gps([("23:50:00", 0, 0.000, 20)]), gps([("00:05:00", 0, 0.001, 20)], day="2013-04-02")])
    daily = ft.daily_distance(points)
    assert list(daily["dist_all"]) == [0, 0]


# ---------- 실내 이동 ----------

def wifi(rows: list[tuple[str, str]], day: str = DAY) -> pd.DataFrame:
    return pd.DataFrame([(ts(c, day), loc) for c, loc in rows], columns=["time", "location"])


def test_wifi_bins_count_in_and_near_samples():
    bins = ft.wifi_bins(wifi([("10:00:00", "in[kemeny]"), ("10:00:20", "near[kemeny; sudikoff; ]"), ("10:00:40", "in[sudikoff]")]))
    row = bins.iloc[0]
    assert (row["n_total"], row["n_in"], row["n_near"]) == (3, 2, 1)


def test_indoor_mobility_counts_walking_share_only_in_indoor_day_bins():
    acts = activity([("10:00:00", [1, 1, 1, 0, 0, 0]),     # 건물 안, 걷기 3/6 -> 5분
                     ("11:00:00", [1, 1, 1, 1, 1, 1]),     # near뿐 -> 기본 설정에서는 제외
                     ("19:00:00", [1, 1, 1, 1, 1, 1])])    # 건물 안이지만 저녁
    places = wifi([("10:00:00", "in[kemeny]"), ("11:00:00", "near[kemeny; ]"), ("19:00:00", "in[kemeny]")])
    a, w = ft.activity_bins(acts), ft.wifi_bins(places)
    assert ft.daily_indoor_mobility(a, w)["indoor_day"].iloc[0] == pytest.approx(5.0)
    assert ft.daily_indoor_mobility(a, w, near_as_in=True)["indoor_day"].iloc[0] == pytest.approx(15.0)


def test_indoor_mobility_active_method_counts_whole_bins_over_threshold():
    acts = activity([("10:00:00", [1, 1, 1, 1, 0, 0]), ("10:10:00", [1, 0, 0, 0, 0, 0])])
    places = wifi([("10:00:00", "in[kemeny]"), ("10:10:00", "in[kemeny]")])
    daily = ft.daily_indoor_mobility(ft.activity_bins(acts), ft.wifi_bins(places), method="active", threshold=0.5)
    assert daily["indoor_day"].iloc[0] == 10


def test_running_is_not_walking_for_indoor_mobility():
    acts = activity([("10:00:00", [2, 2, 1, 0])])
    places = wifi([("10:00:00", "in[gym]")])
    daily = ft.daily_indoor_mobility(ft.activity_bins(acts), ft.wifi_bins(places))
    assert daily["indoor_day"].iloc[0] == pytest.approx(2.5)   # 걷기 1/4


def test_bins_without_any_wifi_sample_are_not_indoor():
    acts = activity([("10:00:00", [1, 1, 1, 1])])
    places = wifi([("12:00:00", "in[kemeny]")])
    daily = ft.daily_indoor_mobility(ft.activity_bins(acts), ft.wifi_bins(places))
    assert daily["indoor_day"].iloc[0] == 0


def test_unknown_indoor_method_raises():
    acts, places = activity([("10:00:00", [1])]), wifi([("10:00:00", "in[a]")])
    with pytest.raises(ValueError):
        ft.daily_indoor_mobility(ft.activity_bins(acts), ft.wifi_bins(places), method="other")
