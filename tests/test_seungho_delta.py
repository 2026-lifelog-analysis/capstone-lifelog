# 작성 2026-11-06 07:30 (KST)
"""① 평소 대비 최근 수면 변화량 테스트 (가짜 데이터)."""
import numpy as np
import pandas as pd
import pytest

from src import forecast_dataset as fd
from src import forecast_synthetic as fsyn
from experiments.seungho import sleep_delta as sd


@pytest.fixture(scope='module')
def data():
    A, S = fsyn.make_raw(n_people=20, n_days=120, seed=3)
    return fd.build_forecast_table(S, A)


@pytest.fixture(scope='module')
def feats(data):
    return sd.feat_delta(data)


def _past(data, row, target, lo, hi):
    """밤 D−hi ~ D−lo의 유효 값 (정의대로 직접)."""
    n = data.nights
    ok = ((n['pid'] == row['pid']) & (n['date'] <= row['date'] - pd.Timedelta(days=lo))
          & (n['date'] >= row['date'] - pd.Timedelta(days=hi))
          & (n['known_at'] <= row['forecast_at']) & ~n['overlap'] & ~n['asleep_invalid'])
    return n.loc[ok, fd.NIGHT_VALUE[target]].dropna().to_numpy(dtype=float)


def test_only_adds_columns(data):
    t = fd.add_features(data, sd.feat_delta)
    extra = fd.check_common_columns(data.table, t)
    assert all(c.startswith('sd_') for c in extra)


def test_all_experiment_columns_exist(data, feats):
    for e in sd.all_phase1().values():
        for t in fd.TARGETS:
            cols = e.features(t)
            fd.check_features(cols)
            assert set(cols) - set(data.table.columns) <= set(feats.columns), e.name


def test_no_duplicate_names():
    names = list(sd.all_phase1())
    assert len(names) == len(set(names))
    assert len(names) > 60


@pytest.mark.parametrize('target', list(fd.TARGETS))
def test_level_recent_delta_match_definition(data, feats, target):
    rng = np.random.default_rng(0)
    tb = data.table
    for i in rng.choice(len(tb), 40, replace=False):
        row = tb.iloc[i]
        past28 = _past(data, row, target, 1, 28)
        L = np.median(past28) if len(past28) >= 7 else np.nan
        r7 = _past(data, row, target, 1, 7)
        R = r7.mean() if len(r7) else np.nan
        got_L = feats[sd.col(target, sd.L0)].iloc[i]
        got_D = feats[sd.col(target, sd.D0)].iloc[i]
        np.testing.assert_allclose(got_L, L, equal_nan=True)
        np.testing.assert_allclose(got_D, R - L, equal_nan=True)
        ex = _past(data, row, target, 8, 35)
        Lex = np.median(ex) if len(ex) >= 7 else np.nan
        np.testing.assert_allclose(feats[sd.col(target, sd.lname(span='ex'))].iloc[i], Lex, equal_nan=True)


def test_future_does_not_leak(data):
    """밤 D 이후 기록(정답 포함)을 바꿔도 밤 D의 특징은 그대로."""
    tb = data.table
    pid = tb['pid'].iloc[0]
    cut = tb.loc[tb['pid'] == pid, 'date'].iloc[60]
    n2 = data.nights.copy()
    fut = (n2['pid'] == pid) & (n2['date'] >= cut)
    n2.loc[fut, ['sleep_min', 'bedtime', 'waketime']] += 333.0
    a = sd.delta_features(tb, data.nights)
    b = sd.delta_features(tb, n2)
    rows = (tb['pid'] == pid) & (tb['date'] <= cut)
    pd.testing.assert_frame_equal(a[rows], b[rows])
    later = (tb['pid'] == pid) & (tb['date'] > cut + pd.Timedelta(days=1))
    assert not a[later].equals(b[later])        # 미래 쪽은 실제로 바뀌어야 테스트가 의미 있음


def test_shapes_simple_case():
    M = np.full((1, sd.MAX_LAG), 400.0)
    M[0, :7] = [340, 350, 360, 370, 380, 390, 400]     # lag1=340 (어제가 가장 짧음)
    f = sd.quantity_features(M, 'sleep')
    assert f['slope'][0] == pytest.approx(-10.0)        # 최근으로 올수록 10분씩 줄어듦
    assert f[sd.D0][0] == pytest.approx(370 - 400)
    assert f['nextreme'][0] == 0                        # 60분 넘게 짧은 밤 없음
    w = 0.5 ** (np.arange(1, 8) / sd.EW_HALF)
    assert f['ewd'][0] == pytest.approx((w * M[0, :7]).sum() / w.sum() - 400)


# ---------- 확장 계열 (extended.py) ----------

from experiments.seungho import extended as ext


@pytest.mark.parametrize('s,wake,exp', [('11:30', False, 690), ('1 AM', False, 780), ('12am', False, 720),
                                         ('10:30 PM', False, 630), ('8:00', True, 1200), ('1:00 PM', True, 1500),
                                         ('12', True, 1440), ('abc', False, np.nan)])
def test_psqi_clock(s, wake, exp):
    np.testing.assert_equal(ext._clock(s, wake), exp)


def test_day_lags_use_only_past_days(data):
    t = data.table
    a = data.activity[['pid', 'date']].copy()
    a['steps'] = np.arange(len(a), dtype=float)
    base = ext.day_lags(t, a, ['steps'], [1])['steps_l1']
    a2 = a.copy()
    later = a2['date'] >= a2['date'].min() + pd.Timedelta(days=50)
    a2.loc[later, 'steps'] += 1e6
    new = ext.day_lags(t, a2, ['steps'], [1])['steps_l1']
    rows = (t['date'] <= a['date'].min() + pd.Timedelta(days=50)).to_numpy()   # D−1 < 컷
    np.testing.assert_array_equal(base[rows], new[rows])


def test_night_lags_match_common_masking(data):
    """부가 지표 lag의 결측 위치가 공통 sleep_lag와 같다 (같은 가림 규칙)."""
    nm = ext.night_metrics(data)
    out = ext.night_lags(data.table, nm, ['n_inbed'], [1, 2])
    for k in (1, 2):
        np.testing.assert_array_equal(np.isnan(out[f'n_inbed_l{k}']), data.table[f'sleep_lag{k}'].isna().to_numpy())


def test_feats_of_keeps_base_and_no_targets():
    for e in ext.phase3_exps().values():
        for t in fd.TARGETS:
            cols = e.features(t)
            fd.check_features(cols)
            assert len(cols) == len(set(cols)), e.name
            if not e.name.startswith(('Q_', 'L4w_', 'L6w_', 'L8w_', 'TAG', 'TODAY_', 'DOW_', 'STAB_', 'TYPE_')):
                assert cols[:8] == fd.base_features(t), e.name
