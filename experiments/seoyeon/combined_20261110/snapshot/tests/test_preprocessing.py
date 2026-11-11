"""전처리 규칙 테스트 (실제 데이터 없이 작은 가짜 표로 확인)."""
import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp


def _sleep(pieces, pid=1, date='2016-03-01'):
    """(잠든 시각 'HH:MM:SS', 누워 있던 분) 목록 → 수면 원본 모양의 표."""
    return pd.DataFrame({'egoid': pid, 'dataDate': date,
                         'timetobed': [t for t, _ in pieces],
                         'bedtimedur': [d for _, d in pieces],
                         'minsasleep': [d for _, d in pieces]})


def _activity(rows):
    """(날짜, 착용률, 걸음) 목록 → 활동 원본 모양의 표."""
    return pd.DataFrame({'egoid': 1, 'datadate': [r[0] for r in rows],
                         'complypercent': [r[1] for r in rows], 'meanrate': 70.0,
                         'steps': [r[2] for r in rows], 'lightlyactiveminutes': 100,
                         'fairlyactiveminutes': 10, 'veryactiveminutes': 5})


@pytest.mark.parametrize('t, expected', [('23:30:00', 690), ('01:00:00', 780),
                                         ('12:00:00', 0), ('00:00:00', 720)])
def test_bed_min_unit_is_minutes_from_noon(t, expected):
    assert pp.hms_to_min(t) == expected


def test_bed_min_single_sleep():
    assert pp.merge_sleep(_sleep([('23:30:00', 450)]))['bed_min'].iloc[0] == 690
    assert pp.merge_sleep(_sleep([('01:00:00', 360)]))['bed_min'].iloc[0] == 780


def test_bed_min_wake_in_the_night_is_one_sleep():
    # 23:30~02:00 (150분), 02:40~07:00 (260분): 40분 깸 → 한 덩어리 → 잠든 시각 23:30
    d = pp.merge_sleep(_sleep([('23:30:00', 150), ('02:40:00', 260)]))
    assert len(d) == 1
    assert d['bed_min'].iloc[0] == 690
    assert d['sleep_min'].iloc[0] == 150 + 260
    assert d['n_episodes'].iloc[0] == 2
    assert d['wake_min'].iloc[0] == 780          # 07:00 = 18시부터 780분


def test_sleep_total_sums_all_pieces_and_bed_uses_longest_block():
    # 낮잠(14:00, 60분) + 밤잠(23:00, 420분): 총 수면은 합산, 잠든 시각은 긴 밤잠의 시작
    d = pp.merge_sleep(_sleep([('14:00:00', 60), ('23:00:00', 420)]))
    assert d['sleep_min'].iloc[0] == 480
    assert d['bed_min'].iloc[0] == 660


def test_sleep_gap_argument():
    # 2시간 깸: 기본(90분)이면 두 덩어리 → 더 긴 두 번째 덩어리의 시작, gap=180이면 한 덩어리
    pieces = [('23:00:00', 120), ('03:00:00', 240)]
    assert pp.merge_sleep(_sleep(pieces))['bed_min'].iloc[0] == pp.hms_to_min('03:00:00')
    assert pp.merge_sleep(_sleep(pieces), gap=180)['bed_min'].iloc[0] == 660


def test_low_wear_day_is_nan_not_zero():
    act = pp.build_activity(_activity([('2016-03-01', 95, 8000), ('2016-03-02', 50, 3000)]))
    low = act[act['date'] == '2016-03-02'].iloc[0]
    assert np.isnan(low['steps'])
    assert np.isnan(low['steps_per_hour']) and np.isnan(low['active_min']) and np.isnan(low['meanrate'])
    assert low['complypercent'] == 50           # 줄은 남아 있음 (수면은 따로 유지)
    ok = act[act['date'] == '2016-03-01'].iloc[0]
    assert ok['steps'] == 8000
    assert ok['steps_per_hour'] == pytest.approx(8000 / (24 * 0.95))


def test_duplicates_keep_row_with_fewest_blanks():
    A = _activity([('2016-03-01', 95, np.nan), ('2016-03-01', 95, 8000)])
    act = pp.build_activity(A)
    assert len(act) == 1 and act['steps'].iloc[0] == 8000


def test_filter_min_days():
    dates = pd.date_range('2016-01-01', periods=50)
    d = pd.DataFrame({'pid': ['a'] * 50 + ['b'] * 41,
                      'date': list(dates) + list(dates[:41]),
                      'steps': 1.0, 'sleep_min': np.nan})
    assert set(pp.filter_min_days(d)['pid']) == {'a'}


def test_split_study_phase():
    # 기본 경계 2017-08-13: 전날까지 1~2학년, 그날부터 3~4학년
    d = pd.DataFrame({'pid': 'a', 'date': pd.to_datetime(['2017-08-12', '2017-08-13'])})
    assert pp.PHASE_CUTOFF == '2017-08-13'
    assert list(pp.split_study_phase(d, 'before')['date']) == [pd.Timestamp('2017-08-12')]
    assert list(pp.split_study_phase(d, 'after')['date']) == [pd.Timestamp('2017-08-13')]
    assert len(pp.split_study_phase(d, 'all')) == 2
