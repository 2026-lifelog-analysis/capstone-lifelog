"""수면 리듬 탐지기(experiments/minjun/regularity)의 규칙 검사."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'minjun' / 'regularity'))
import regularity as R  # noqa: E402


def _daily(n_days=60, people=('a', 'b'), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in people:
        for i in range(n_days):
            bed = 780 + rng.normal(0, 30)           # 정오 기준 분 (01:00 안팎)
            wake = bed + 480 + rng.normal(0, 20)
            rows.append({'pid': p, 'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                         'steps': 8000 + rng.normal(0, 1000), 'sleep_min': 450.0,
                         'bed_min': bed, 'wake_min': (wake - 360) % 1440})
    return R.add_sleep_mid(pd.DataFrame(rows))


def test_sd_is_blank_when_fewer_than_5_of_7_days():
    x = np.array([1., 2., np.nan, np.nan, np.nan, 3., 4., 5., 6., 7.])
    sd = R.roll_sd(x)
    # 날 8: 창 = 날 2~8, 기록 4개(3,4,5,6) → 빈칸. 날 9: 창 = 날 3~9, 기록 5개(3,4,5,6,7) → 값 있음
    assert np.isnan(sd[8])
    assert np.isclose(sd[9], np.std([3, 4, 5, 6, 7], ddof=1))
    # 그날이 빈칸이면 창 안 기록이 충분해도 빈칸
    y = np.array([1., 2., 3., 4., 5., 6., np.nan, 8., 9.])
    assert np.isnan(R.roll_sd(y)[6])
    assert np.isclose(R.roll_sd(y)[5], np.std([1, 2, 3, 4, 5, 6], ddof=1))


def test_sd_window_does_not_read_future_days():
    x = np.arange(30, dtype=float)
    base = R.roll_sd(x)
    y = x.copy()
    y[20:] += 1000                                   # 날 20 이후만 바꿈
    changed = R.roll_sd(y)
    assert np.allclose(base[:20], changed[:20], equal_nan=True)


def test_jitter_keeps_mean_bedtime_and_sleep_length():
    nh = _daily()
    seg = pd.DataFrame({'pid': ['a', 'b'], 'start': [pd.Timestamp('2020-02-01')] * 2})
    seg['end'] = seg['start'] + pd.Timedelta(days=R.LENGTH - 1)
    d = R.jitter_sleep(nh, seg, sigma=90.0, seed=0)
    for p in ['a', 'b']:
        m = (nh['pid'] == p) & (nh['date'] >= seg['start'].iloc[0]) & (nh['date'] <= seg['end'].iloc[0])
        delta = d.loc[m, 'bed_min'] - nh.loc[m, 'bed_min']
        assert abs(delta.mean()) < 1e-9                                       # 구간 평균 취침 시각 유지
        assert delta.std(ddof=1) > 30                                          # 실제로 흔들렸음
        assert np.allclose(d.loc[m, 'wake_noon'] - d.loc[m, 'bed_min'],
                           nh.loc[m, 'wake_noon'] - nh.loc[m, 'bed_min'])      # 수면 길이 유지
    outside = ~((nh['date'] >= seg['start'].iloc[0]) & (nh['date'] <= seg['end'].iloc[0]))
    assert np.allclose(d.loc[outside, 'bed_min'], nh.loc[outside, 'bed_min'])  # 구간 밖은 그대로


def test_jitter_raises_mid_sd_only_inside_window():
    nh = _daily()
    seg = pd.DataFrame({'pid': ['a', 'b'], 'start': [pd.Timestamp('2020-02-01')] * 2})
    seg['end'] = seg['start'] + pd.Timedelta(days=R.LENGTH - 1)
    d = R.jitter_sleep(nh, seg, sigma=90.0, seed=0)
    s0, s1 = R.sleep_mid_sd(nh), R.sleep_mid_sd(d)
    before = nh['date'] < seg['start'].iloc[0]
    assert np.allclose(s0[before], s1[before], equal_nan=True)                 # 구간 전 지표는 변하지 않음
    inside = (nh['date'] >= seg['start'].iloc[0] + pd.Timedelta(days=6)) & (nh['date'] <= seg['end'].iloc[0])
    assert s1[inside].mean() > s0[inside].mean()                               # 구간 안에서는 SD 가 커짐


def test_seg_metrics_matches_hand_count():
    seg = pd.DataFrame({'rep': [0, 0, 0, 0, 1, 1], 'pid': ['a', 'a', 'b', 'b', 'a', 'a'],
                        'day': [0, 1, 0, 1, 0, 1]})
    a_inj = np.array([False, True, False, False, True, True])
    a_ctl = np.array([False, False, True, False, False, False])
    pp, w = R.seg_metrics(seg, a_ctl, a_inj)
    pa = pp.set_index('pid')
    assert pa.loc['a', '탐지율'] == 1.0 and pa.loc['a', '오경보율'] == 0.0 and pa.loc['a', '차이'] == 1.0
    assert pa.loc['b', '탐지율'] == 0.0 and pa.loc['b', '오경보율'] == 1.0 and pa.loc['b', '차이'] == -1.0
    assert pa.loc['a', '탐지지연'] == 0.5                                       # rep0 첫 경보 1일, rep1 0일
