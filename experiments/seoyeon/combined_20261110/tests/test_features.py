import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'snapshot')]
from prepare_features import selected_b_features, feature_spec


def fixture():
    day = pd.Timestamp('2017-02-01')
    dates = pd.date_range(day - pd.Timedelta(days=35), periods=38)
    nights = pd.DataFrame({'pid': 'p', 'date': dates,
                           'known_at': dates + pd.Timedelta(days=1, hours=8),
                           'overlap': False, 'asleep_invalid': False,
                           'sleep_min': np.arange(38.) + 400,
                           'bedtime': np.arange(38.) + 600,
                           'waketime': np.arange(38.) + 1000})
    table = pd.DataFrame({'pid': ['p'], 'date': [day],
                          'forecast_at': [day + pd.Timedelta(hours=18)]})
    caldates = pd.date_range(day - pd.Timedelta(days=40), periods=45)
    cal = pd.DataFrame({'date': caldates})
    cal['off'] = (caldates.dayofweek >= 5).astype(float)
    for col in ['h_class', 'h_finals', 'h_break', 'h_winter', 'h_summer',
                'h_term_week', 'h_days_to_finals', 'h_fed']:
        cal[col] = 0.
    return SimpleNamespace(table=table, nights=nights), cal


def test_future_and_unfinished_nights_do_not_enter_b():
    data, cal = fixture()
    original = selected_b_features(data, cal)
    changed = data.nights.copy()
    changed.loc[changed.date >= data.table.date.iloc[0], ['sleep_min', 'bedtime', 'waketime']] = 99999.
    pd.testing.assert_frame_equal(original, selected_b_features(SimpleNamespace(table=data.table, nights=changed), cal))
    changed = data.nights.copy()
    last = changed.date == data.table.date.iloc[0] - pd.Timedelta(days=1)
    changed.loc[last, 'known_at'] = data.table.forecast_at.iloc[0] + pd.Timedelta(hours=1)
    changed.loc[last, 'sleep_min'] = 99999.
    omitted = changed.loc[~last].copy()
    pd.testing.assert_frame_equal(
        selected_b_features(SimpleNamespace(table=data.table, nights=changed), cal),
        selected_b_features(SimpleNamespace(table=data.table, nights=omitted), cal))


def test_b_matches_upstream_selected_calculation():
    from experiments.seungho import sleep_delta as sd
    from experiments.seungho import extended as ext
    data, cal = fixture()
    result = selected_b_features(data, cal)
    matrix = sd.lag_matrices(data.table, data.nights, max_lag=28)['sleep']
    day = data.table.date.iloc[0]
    offs = cal.set_index('date')['off']
    masks = np.array([offs[day - pd.Timedelta(days=k - 1)] for k in range(1, 29)])
    work = matrix[:, masks == 0]
    expected = np.nanmedian(work)
    assert result.loc[0, 'x_sleep_Lsame'] == np.float32(expected)
    assert result.loc[0, 'x_sleep_gap_same'] == np.float32(expected - sd.level(matrix, 28, 'med', 'in', 7)[0])
    original = next(e for e in ext.window28_exps() if e.name == 'L4w_lag7_offbase28_term')
    spec = feature_spec()
    assert spec['feature_sets']['B']['sleep'] == original.features('sleep')
    assert len(spec['feature_sets']['B']['sleep']) == 21


def test_feature_lists_have_no_duplicates_or_targets():
    from src import forecast_dataset as fd
    spec = feature_spec()
    for targets in spec['feature_sets'].values():
        for cols in targets.values():
            assert len(cols) == len(set(cols))
            assert not set(cols) & set(fd.NON_FEATURE_COLS)
            assert not any('_l0' in c or '_d0' in c for c in cols)
