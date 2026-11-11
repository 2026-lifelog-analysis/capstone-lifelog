"""기준선 함수 테스트: 미래 정보를 쓰지 않는지, 규칙대로 계산하는지."""
import numpy as np
import pandas as pd

from src import baseline as bl
from src import evaluation as ev


def _series(n=80, seed=0, pid='a', start='2016-01-04'):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'pid': pid, 'date': pd.date_range(start, periods=n),
                         'steps': rng.normal(8000, 1500, n)})


def test_personal_z_does_not_use_future():
    d = _series()
    z0 = bl.personal_z(d, 'steps')
    today = 50
    d2 = d.copy()
    d2.loc[today + 1:, 'steps'] = d2.loc[today + 1:, 'steps'] * 10 + 99999   # 미래만 바꿈
    z1 = bl.personal_z(d2, 'steps')
    pd.testing.assert_series_equal(z0.iloc[:today + 1], z1.iloc[:today + 1])


def test_personal_z_dow_adjust_does_not_use_future():
    d = _series()
    z0 = bl.personal_z(d, 'steps', dow_adjust=True)
    d2 = d.copy()
    d2.loc[41:, 'steps'] = -5000
    z1 = bl.personal_z(d2, 'steps', dow_adjust=True)
    pd.testing.assert_series_equal(z0.iloc[:41], z1.iloc[:41])


def test_personal_z_uses_calendar_days_not_rows():
    # 기록이 띄엄띄엄(이틀에 한 번)이면 28일 창 안 기록은 14개 → 딱 계산 가능,
    # 사흘에 한 번이면 10개 → min_periods(14) 미달로 빈칸
    every2 = _series(120).iloc[::2]
    every3 = _series(120).iloc[::3]
    assert bl.personal_z(every2, 'steps').notna().any()
    assert bl.personal_z(every3, 'steps').isna().all()


def test_personal_z_value_and_clip():
    d = _series(40)
    d.loc[39, 'steps'] = 1e9
    z = bl.personal_z(d, 'steps')
    assert z.iloc[39] == 8                               # ±8로 자름
    assert bl.personal_z(d, 'steps', clip=None).iloc[39] > 8
    win = d['steps'].iloc[10:38]                         # 38번째 날의 기준선 = 직전 28일
    med = win.median()
    mad = (win - med).abs().median() * 1.4826
    assert np.isclose(z.iloc[38], (d['steps'].iloc[38] - med) / mad)


def test_dow_adjust_removes_weekly_pattern():
    # 주말마다 걸음이 절반인 사람: 보정 없으면 주말 z가 크게 음수, 보정하면 0 근처
    d = _series(84)
    d['steps'] = np.where(d['date'].dt.dayofweek >= 5, 4000, 10000) + np.random.default_rng(1).normal(0, 300, 84)
    weekend = d['date'].dt.dayofweek >= 5
    z_raw = bl.personal_z(d, 'steps')[weekend].dropna()
    z_adj = bl.personal_z(d, 'steps', dow_adjust=True)[weekend].dropna()
    assert z_raw.median() < -3
    assert abs(z_adj.median()) < 1.5


def test_group_z_needs_min_people():
    d = pd.concat([_series(5, seed=i, pid=str(i)) for i in range(10)], ignore_index=True)
    assert bl.group_z(d, 'steps', min_n=30).isna().all()
    assert bl.group_z(d, 'steps', min_n=5).notna().all()


def test_bad_score_direction():
    z = pd.Series([1.0])
    assert bl.bad_score(z, 'steps').iloc[0] == -1
    assert bl.bad_score(z, 'bed_min').iloc[0] == 1


def test_cusum_accumulates_and_resets():
    d = pd.DataFrame({'pid': 'a', 'date': pd.date_range('2016-01-01', periods=6),
                      'b': [0.0, 1.0, 1.0, np.nan, 1.0, 3.0]})
    s = bl.cusum(d, 'b', k=0.5)
    assert list(s.fillna(-1)) == [0.0, 0.5, 1.0, -1, 1.5, 4.0]   # 빈 날은 빈칸, 누적은 이어짐
    s_h = bl.cusum(d, 'b', k=0.5, h=1.2)
    assert list(s_h.fillna(-1)) == [0.0, 0.5, 1.0, -1, 1.5, 2.5]  # 1.5에서 경보 → 0으로 → 0+3−0.5


def test_inject_drift_reproducible_and_only_inside_segment():
    d = pd.concat([_series(90, seed=i, pid=str(i)) for i in range(5)], ignore_index=True)
    a, seg_a = ev.inject_drift(d, 'steps', -0.3, seed=7)
    b, seg_b = ev.inject_drift(d, 'steps', -0.3, seed=7)
    pd.testing.assert_frame_equal(seg_a, seg_b)
    changed = ~np.isclose(a['steps'], d['steps'])
    inside = ev._segment_rows(d, seg_a).index
    assert set(d.index[changed]) == set(inside)
    assert len(inside) == 5 * 14
    assert np.allclose(a.loc[inside, 'steps'], d.loc[inside, 'steps'] * 0.7)


def test_inject_drift_gradual():
    d = _series(90)
    a, seg = ev.inject_drift(d, 'steps', 120, mode='absolute', shape='gradual', seed=0)
    rows = ev._segment_rows(d, seg).sort_values('date')
    diff = (a.loc[rows.index, 'steps'] - d.loc[rows.index, 'steps']).to_numpy()
    assert np.allclose(diff, 120 * np.arange(1, 15) / 14)
