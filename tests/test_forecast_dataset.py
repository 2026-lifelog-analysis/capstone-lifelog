"""예측용 밤·정답·입력 테스트 (실제 데이터 없이 작은 가짜 원본으로)."""
import numpy as np
import pandas as pd
import pytest

from src import forecast_dataset as fd
from src import preprocessing as pp


def _sleep(rows, pid=1):
    """(시작 'YYYY-MM-DD HH:MM', 종료 'YYYY-MM-DD HH:MM', minsasleep) → FitbitSleep 모양.
    원본처럼 dataDate = 종료일, bedtimedur = 시계 차이 + 1분."""
    out = []
    for s, e, asleep in rows:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        out.append({'egoid': pid, 'dataDate': e.strftime('%Y-%m-%d'),
                    'timetobed': s.strftime('%H:%M:%S'), 'timeoutofbed': e.strftime('%H:%M:%S'),
                    'bedtimedur': int((e - s) / pd.Timedelta(minutes=1)) + 1, 'minsasleep': asleep})
    return pd.DataFrame(out)


def _activity(start, n, pid=1, steps=lambda i: 1000 * (i + 1), wear=95):
    d0 = pd.Timestamp(start)
    return pd.DataFrame({'egoid': pid,
                         'datadate': [(d0 + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)],
                         'complypercent': wear, 'meanrate': 70.0,
                         'steps': [steps(i) for i in range(n)], 'lightlyactiveminutes': 100,
                         'fairlyactiveminutes': 10, 'veryactiveminutes': 5})


def _night(d, h_bed='23:00', h_wake='07:00', asleep=450):
    """밤 d: d h_bed ~ (d+1) h_wake 한 조각. h_bed가 12시 전이면 d+1의 그 시각."""
    D = pd.Timestamp(d)
    bed_day = D if int(h_bed[:2]) >= 12 else D + pd.Timedelta(days=1)
    return (f'{bed_day.date()} {h_bed}', f'{(D + pd.Timedelta(days=1)).date()} {h_wake}', asleep)


def _row(table, date, pid='1'):
    r = table[(table['pid'] == pid) & (table['date'] == pd.Timestamp(date))]
    assert len(r) == 1, f'{date} 줄이 없음'
    return r.iloc[0]


# ---------- 시각 복원 (dataDate = 종료일) ----------

def test_episode_crossing_midnight_starts_previous_day():
    ep, dropped = fd.sleep_episodes(_sleep([('2016-03-01 23:30', '2016-03-02 07:00', 420)]))
    assert ep['start'].iloc[0] == pd.Timestamp('2016-03-01 23:30')
    assert ep['end'].iloc[0] == pd.Timestamp('2016-03-02 07:00')
    assert sum(dropped.values()) == 0


def test_uncertain_and_duplicate_records_are_dropped_and_counted():
    S = _sleep([('2016-03-01 23:30', '2016-03-02 07:00', 420),
                ('2016-03-01 23:30', '2016-03-02 07:00', 420),       # 똑같은 조각
                ('2016-03-02 23:30', '2016-03-03 07:00', 420),
                ('2016-03-03 23:30', '2016-03-04 07:00', 420)])
    S.loc[2, 'timeoutofbed'] = np.nan
    S.loc[3, 'bedtimedur'] = 300                                     # 시계 차이(450)와 안 맞음
    ep, dropped = fd.sleep_episodes(S)
    assert len(ep) == 1
    assert dropped['duplicate'] == 1 and dropped['no_end_time'] == 1 and dropped['clock_mismatch'] == 1


# ---------- 밤 배정 ----------

@pytest.mark.parametrize('start, end, night', [
    ('2016-03-01 23:00', '2016-03-02 07:00', '2016-03-01'),   # 자정 전 시작
    ('2016-03-02 02:00', '2016-03-02 09:00', '2016-03-01'),   # 자정 후 시작도 같은 밤
    ('2016-03-02 11:50', '2016-03-02 12:30', '2016-03-01'),   # 다음 정오 직전 시작
    ('2016-03-02 12:00', '2016-03-02 13:00', '2016-03-02'),   # 정오부터는 다음 밤
    ('2016-03-01 12:30', '2016-03-01 14:00', '2016-03-01'),   # 그날 낮
])
def test_night_assignment_by_start(start, end, night):
    ep = fd.assign_nights(fd.sleep_episodes(_sleep([(start, end, 30)]))[0])
    assert ep['date'].iloc[0] == pd.Timestamp(night)


def test_both_before_and_after_midnight_are_targets():
    S = _sleep([_night('2016-03-01', '23:00', '07:00', 450),    # 밤 03-01: 23시 시작
                _night('2016-03-02', '02:00', '09:00', 400)])   # 밤 03-02: 03-03 02시 시작
    t = fd.build_forecast_table(S).table
    assert list(t['date']) == [pd.Timestamp('2016-03-01'), pd.Timestamp('2016-03-02')]
    assert list(t['target_sleep_min']) == [450, 400]


def test_long_sleep_past_next_noon_is_not_cut():
    S = _sleep([('2016-03-02 03:00', '2016-03-02 13:30', 600)])
    r = _row(fd.build_forecast_table(S).table, '2016-03-01')
    assert r['target_wake_at'] == pd.Timestamp('2016-03-02 13:30')
    assert r['target_waketime'] == 25.5 * 60                    # 03-01 12:00부터 1530분 (나머지 연산 없음)


def test_each_piece_in_exactly_one_night():
    S = _sleep([('2016-03-01 23:00', '2016-03-02 07:00', 400), ('2016-03-02 11:30', '2016-03-02 12:40', 60),
                ('2016-03-02 13:00', '2016-03-02 13:40', 30), ('2016-03-03 01:00', '2016-03-03 08:00', 380)])
    ep = fd.assign_nights(fd.sleep_episodes(S)[0])
    b = fd.sleep_blocks(ep)
    assert b['n_pieces'].sum() == len(ep) == 4
    assert b.groupby(['pid', 'date'])['sleep_min'].sum().sum() == S['minsasleep'].sum()


# ---------- 병합·주수면 ----------

def test_pieces_within_90_min_are_one_block_even_across_midnight():
    # 21:30~23:30 (120), 00:30~07:00 (380): 60분 깸 → 한 덩어리. 종료일이 달라도 같은 밤
    S = _sleep([('2016-03-01 21:30', '2016-03-01 23:30', 110), ('2016-03-02 00:30', '2016-03-02 07:00', 370)])
    r = _row(fd.build_forecast_table(S).table, '2016-03-01')
    assert r['target_n_pieces'] == 2 and r['target_n_blocks'] == 1
    assert r['target_bed_at'] == pd.Timestamp('2016-03-01 21:30')
    assert r['target_wake_at'] == pd.Timestamp('2016-03-02 07:00')
    assert r['target_sleep_min'] == 110 + 370


def test_main_sleep_is_block_with_largest_asleep_sum():
    # 22:00~01:00 (170) | 2시간 깸 | 03:00~05:00, 05:30~09:00 (110 + 200 = 310) → 두 번째 덩어리
    S = _sleep([('2016-03-01 22:00', '2016-03-02 01:00', 170),
                ('2016-03-02 03:00', '2016-03-02 05:00', 110), ('2016-03-02 05:30', '2016-03-02 09:00', 200)])
    r = _row(fd.build_forecast_table(S).table, '2016-03-01')
    assert r['target_sleep_min'] == 310 and r['target_n_blocks'] == 2
    assert r['target_bed_at'] == pd.Timestamp('2016-03-02 03:00')


def test_main_sleep_tie_goes_to_earlier_block():
    S = _sleep([('2016-03-01 20:00', '2016-03-01 23:00', 170), ('2016-03-02 03:00', '2016-03-02 06:00', 170)])
    r = _row(fd.build_forecast_table(S).table, '2016-03-01')
    assert r['target_bed_at'] == pd.Timestamp('2016-03-01 20:00')


def test_total_sleep_is_sum_of_minsasleep_not_end_minus_start():
    # 23:00~07:00 = 480분 누워 있었지만 minsasleep은 300 + 100
    S = _sleep([('2016-03-01 23:00', '2016-03-02 03:00', 200), ('2016-03-02 03:30', '2016-03-02 07:00', 200)])
    S['minsasleep'] = [300 - 100, 200]
    r = _row(fd.build_forecast_table(S).table, '2016-03-01')
    assert r['target_sleep_min'] == 400
    assert (r['target_wake_at'] - r['target_bed_at']) / pd.Timedelta(minutes=1) == 480


def test_anomaly_daily_merge_sleep_is_unchanged():
    # 이상 탐지 하루 표는 여전히 dataDate별 '모든 조각 합'
    S = pd.DataFrame({'egoid': 1, 'dataDate': '2016-03-01', 'timetobed': ['14:00:00', '23:00:00'],
                      'bedtimedur': [60, 420], 'minsasleep': [60, 420]})
    d = pp.merge_sleep(S)
    assert d['sleep_min'].iloc[0] == 480 and d['bed_min'].iloc[0] == 660


# ---------- 시각 표현 ----------

def test_bedtime_is_continuous_minutes_from_noon():
    S = _sleep([('2016-03-01 23:50', '2016-03-02 07:00', 400), ('2016-03-03 00:10', '2016-03-03 07:00', 380)])
    t = fd.build_forecast_table(S).table
    assert list(t['target_bedtime']) == [710, 730]               # 같은 밤 기준이라 차이 20분
    assert abs(t['target_bedtime'].iloc[0] - t['target_bedtime'].iloc[1]) == 20


# ---------- 예측 기준 시각 ----------

def test_main_sleep_starting_before_forecast_is_excluded_and_counted():
    S = _sleep([('2016-03-01 17:00', '2016-03-02 06:00', 700),     # 18시 전 시작
                _night('2016-03-02', '22:00', '06:00', 450)])
    data = fd.build_forecast_table(S)
    assert list(data.table['date']) == [pd.Timestamp('2016-03-02')]
    assert data.info['night_reasons']['started_before_forecast'] == 1
    assert pd.Timestamp('2016-03-01') in set(fd.build_forecast_table(S, forecast_hour=16).table['date'])


def test_nights_without_sleep_have_no_target_row_and_are_counted():
    S = _sleep([_night('2016-03-01'), _night('2016-03-04')])
    data = fd.build_forecast_table(S)
    assert len(data.table) == 2 and data.table['target_sleep_min'].notna().all()
    assert data.info['night_reasons']['no_sleep_record'] == 2


# ---------- 입력: 달력 7밤 ----------

def test_lags_follow_calendar_nights_and_keep_gaps():
    nights = ['2016-03-01', '2016-03-02', '2016-03-03', '2016-03-05', '2016-03-06', '2016-03-07',
              '2016-03-08', '2016-03-09']                           # 03-04 밤 빠짐
    S = _sleep([_night(d, asleep=400 + i) for i, d in enumerate(nights)])
    r = _row(fd.build_forecast_table(S).table, '2016-03-08')
    lags = [r[f'sleep_lag{k}'] for k in range(1, 8)]
    # D-1=03-07(405) D-2=03-06(404) D-3=03-05(403) D-4=03-04(없음) D-5=03-03(402) D-6=03-02(401) D-7=03-01(400)
    assert lags[:3] == [405, 404, 403] and np.isnan(lags[3]) and lags[4:] == [402, 401, 400]


def test_early_nights_are_kept_with_nan_inputs():
    S = _sleep([_night('2016-03-01'), _night('2016-03-02')])
    t = fd.build_forecast_table(S).table
    first = _row(t, '2016-03-01')
    assert all(np.isnan(first[f'sleep_lag{k}']) for k in range(1, 8))
    assert len(fd.build_forecast_table(S, min_nights=3).table) == 0   # 옵션으로만 제외


def test_bed_and_wake_lags_use_their_own_night_reference():
    S = _sleep([_night('2016-03-01', '23:30', '07:15'), _night('2016-03-02', '01:00', '08:00')])
    r = _row(fd.build_forecast_table(S).table, '2016-03-02')
    assert r['bedtime_lag1'] == 690 and r['waketime_lag1'] == 19.25 * 60
    assert r['target_bedtime'] == 780


# ---------- 입력: 예측 시점 이후 정보 제외 ----------

def test_inputs_do_not_change_when_target_or_future_changes():
    D = '2016-03-09'
    S = _sleep([_night(f'2016-03-{d:02d}') for d in range(1, 12)])
    A = _activity('2016-03-01', 12)
    base = fd.build_forecast_table(S, A)
    b = fd.add_features(base, fd.feat_steps_lags)

    S2 = S.copy()
    S2.loc[S2['dataDate'] >= '2016-03-10', 'minsasleep'] = 1      # 밤 D(정답)와 그 뒤
    A2 = A.copy()
    A2.loc[A2['datadate'] >= D, 'steps'] = 999_999                 # D 당일과 그 뒤 걸음
    new = fd.build_forecast_table(S2, A2)
    n = fd.add_features(new, fd.feat_steps_lags)

    inputs = [c for c in b.columns if '_lag' in c] + ['dow']
    rb, rn = _row(b, D), _row(n, D)
    pd.testing.assert_series_equal(rb[inputs], rn[inputs])
    assert rn['target_sleep_min'] == 1 and rb['target_sleep_min'] == 450
    assert rb['steps_lag1'] == 8000                                 # 03-08 걸음, 당일(03-09) 아님


def test_previous_night_not_finished_by_forecast_time_is_masked():
    # 밤 03-01이 03-02 19:00에 끝남 → 03-02 18시 예측 때는 모름
    S = _sleep([('2016-03-02 02:00', '2016-03-02 19:00', 900), _night('2016-03-02', '23:30')])
    data = fd.build_forecast_table(S)
    r = _row(data.table, '2016-03-02')
    assert np.isnan(r['sleep_lag1']) and np.isnan(r['bedtime_lag1']) and np.isnan(r['waketime_lag1'])
    assert data.info['lag_nights_masked_not_known_yet'] == 1


def test_no_same_day_aggregate_in_table():
    S = _sleep([_night('2016-03-01')])
    t = fd.build_forecast_table(S, _activity('2016-03-01', 3)).table
    for c in ['steps', 'meanrate', 'complypercent', 'sleep_lag0', 'steps_lag0', 'sleep_min']:
        assert c not in t.columns


# ---------- 입력 열 관리 ----------

def test_base_features_and_forbidden_columns():
    assert fd.base_features('sleep') == [f'sleep_lag{k}' for k in range(1, 8)] + ['dow']
    for bad in ['pid', 'date', 'target_sleep_min', 'target_bed_at', 'forecast_at', 'target_anything']:
        with pytest.raises(ValueError):
            fd.check_features(['sleep_lag1', bad])


def test_add_features_interface():
    S = _sleep([_night(f'2016-03-{d:02d}') for d in range(1, 6)])
    data = fd.build_forecast_table(S)
    before = data.table.copy()

    def my_feat(d):
        return pd.DataFrame({'n_sleep_obs': d.table[fd.base_features('sleep')[:-1]].notna().sum(axis=1)},
                            index=d.table.index)
    t = fd.add_features(data, my_feat)
    assert list(t['n_sleep_obs']) == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(data.table, before)              # 원래 표는 그대로
    with pytest.raises(ValueError):
        fd.add_features(data, lambda d: pd.DataFrame({'target_leak': d.table['target_sleep_min']}))
    with pytest.raises(ValueError):
        fd.add_features(data, lambda d: pd.DataFrame({'dow': d.table['dow']}))


# ---------- 겹치는 조각 ----------

def _reasons(data):
    return data.candidates.set_index('date')['reason'].to_dict()


def test_exact_duplicate_is_dropped_and_night_kept():
    S = _sleep([_night('2016-03-01', '23:00', '07:00', 450), _night('2016-03-01', '23:00', '07:00', 450)])
    data = fd.build_forecast_table(S)
    assert data.info['dropped_episodes']['duplicate'] == 1
    assert data.info['overlap']['pieces'] == 0
    assert _row(data.table, '2016-03-01')['target_sleep_min'] == 450          # 두 번 더하지 않음


def test_same_span_different_value_is_overlap_not_duplicate():
    S = _sleep([_night('2016-03-01', '23:00', '07:00', 450), _night('2016-03-01', '23:00', '07:00', 300)])
    data = fd.build_forecast_table(S)
    assert data.info['dropped_episodes']['duplicate'] == 0
    assert data.info['overlap']['pairs_by_kind']['same_span'] == 1
    assert len(data.table) == 0 and _reasons(data)[pd.Timestamp('2016-03-01')] == 'overlap_conflict'


@pytest.mark.parametrize('second, kind', [
    (('2016-03-02 03:00', '2016-03-02 09:00', 300), 'partial'),      # 23:00~07:00과 일부 겹침
    (('2016-03-02 01:00', '2016-03-02 02:00', 50), 'contained'),     # 안에 들어감
    (('2016-03-01 23:00', '2016-03-02 01:00', 100), 'contained'),    # 시작이 같고 안에 들어감
])
def test_overlapping_night_is_excluded_whole(second, kind):
    S = _sleep([_night('2016-03-01', '23:00', '07:00', 450), second,
                ('2016-03-01 13:00', '2016-03-01 14:00', 55)])             # 같은 밤의 겹치지 않는 낮잠
    data = fd.build_forecast_table(S)
    o = data.info['overlap']
    assert o['pairs_by_kind'][kind] == 1 and o['pieces'] == 2
    assert o['nights_excluded'] == 1 and o['people_with_excluded_nights'] == 1
    # 낮잠만 남겨 다른 주수면으로 바꾸지 않음 → 그 밤은 통째로 없음
    assert pd.Timestamp('2016-03-01') not in set(data.table['date'])
    assert data.info['night_reasons']['overlap_conflict'] == 1


def test_overlapping_night_is_not_used_as_past_input():
    S = _sleep([_night('2016-03-01', '23:00', '07:00', 450), ('2016-03-02 03:00', '2016-03-02 09:00', 300),
                _night('2016-03-02', '23:00', '07:00', 420)])
    data = fd.build_forecast_table(S)
    r = _row(data.table, '2016-03-02')
    assert np.isnan(r['sleep_lag1']) and np.isnan(r['bedtime_lag1']) and np.isnan(r['waketime_lag1'])
    assert data.info['lag_nights_masked_overlap'] == 1


def test_touching_pieces_are_not_overlap():
    # 23:00~03:00 다음 조각이 정확히 03:00 시작 → 겹침 아님, 한 덩어리
    S = _sleep([('2016-03-01 23:00', '2016-03-02 03:00', 230), ('2016-03-02 03:00', '2016-03-02 07:00', 220)])
    data = fd.build_forecast_table(S)
    assert data.info['overlap']['pieces'] == 0
    r = _row(data.table, '2016-03-01')
    assert r['target_sleep_min'] == 450 and r['target_n_pieces'] == 2


def test_overlap_flags_symmetric_and_open_interval():
    ep = pd.DataFrame({'pid': '1', 'start': pd.to_datetime(['2016-03-01 22:00', '2016-03-01 23:00',
                                                            '2016-03-02 05:00', '2016-03-02 08:00']),
                       'end': pd.to_datetime(['2016-03-02 06:00', '2016-03-02 01:00',
                                              '2016-03-02 08:00', '2016-03-02 09:00'])})
    # 0이 1을 품고(포함), 0과 2가 일부 겹침, 2와 3은 08:00에 맞닿기만 함
    assert list(fd.overlap_flags(ep)) == [True, True, True, False]


def test_date_assumption_check_prefers_end_date_on_synthetic():
    from src import forecast_synthetic as fsyn
    d = fd.date_assumption_check(fsyn.make_raw(n_people=5, n_days=30)[1])
    assert d['end_date']['rate'] < d['start_date']['rate']
    assert d['end_date']['pieces'] == d['start_date']['pieces']
    assert d['clock_matches_bedtimedur'] == 1
