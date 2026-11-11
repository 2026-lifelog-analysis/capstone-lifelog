"""공통 예측 표 보호 테스트: run에 넘기는 table과 add_features의 특징 함수가 공통 열을 바꾸지 못하게."""
import numpy as np
import pandas as pd
import pytest

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from src import forecast_split as fs


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp('common')
    return fr.prepare(synthetic=True, out_dir=out)


class Spy(fe.GroupMedian):
    fits = 0

    def fit(self, train, y, features):
        Spy.fits += 1
        return super().fit(train, y, features)


def _run(prepared, table, tmp_path):
    data, split, man, _ = prepared
    Spy.fits = 0
    return fr.run(data, split, man, tmp_path, experiments=[fe.Experiment('spy', Spy)], table=table,
                  targets=['sleep'], synthetic=True)


def _assert_blocked(prepared, table, tmp_path, match):
    with pytest.raises(ValueError, match=match):
        _run(prepared, table, tmp_path)
    assert Spy.fits == 0                                   # 학습 전에 멈춤
    assert not (tmp_path / 'runs').exists()                # 결과도 저장 안 됨


@pytest.mark.parametrize('col, change', [
    ('target_sleep_min', lambda s: s + 100),               # 정답
    ('target_bedtime', lambda s: s + 1),
    ('target_wake_at', lambda s: s + pd.Timedelta(minutes=1)),   # 정답 보조 열
    ('forecast_at', lambda s: s + pd.Timedelta(hours=1)),  # 예측 시각
    ('sleep_lag1', lambda s: s.fillna(0)),                 # 기본 입력 (NaN → 0)
    ('dow', lambda s: (s + 1) % 7),
    ('pid', lambda s: s.where(s.index != 0, 'other')),     # 참가자
])
def test_changing_common_column_is_blocked(prepared, tmp_path, col, change):
    data = prepared[0]
    t = data.table.copy()
    t[col] = change(t[col])
    _assert_blocked(prepared, t, tmp_path, col)


def test_dropped_column_row_reorder_and_duplicate_column_are_blocked(prepared, tmp_path):
    t = prepared[0].table
    _assert_blocked(prepared, t.drop(columns='waketime_lag3'), tmp_path, '빠짐')
    _assert_blocked(prepared, t.iloc[1:], tmp_path, '행 수')
    _assert_blocked(prepared, t.iloc[::-1], tmp_path, '바뀜')
    _assert_blocked(prepared, t.iloc[::-1].reset_index(drop=True), tmp_path, '바뀜')
    dup = pd.concat([t, t[['sleep_lag1']]], axis=1)
    _assert_blocked(prepared, dup, tmp_path, '중복 열')


def test_extra_feature_columns_and_nan_and_dtype_are_allowed(prepared, tmp_path):
    data = prepared[0]
    assert data.table['sleep_lag1'].isna().any()           # 원래 NaN이 있는 표
    t = data.table.copy()
    t['dow'] = t['dow'].astype(float)                       # 같은 값의 정수·실수 표현 차이
    t['my_feat'] = 1.0
    assert fd.check_common_columns(data.table, t) == ['my_feat']
    scores, summary, comp, rd = _run(prepared, t, tmp_path)
    assert Spy.fits == 5 and (summary['status'] == 'complete').all()


def test_changed_data_table_cannot_be_recorded_with_original_fingerprint(prepared, tmp_path):
    data, split, man, _ = prepared
    import dataclasses
    changed = dataclasses.replace(data, table=data.table.assign(target_sleep_min=data.table['target_sleep_min'] + 100))
    Spy.fits = 0
    with pytest.raises(ValueError, match='지문'):
        fr.run(changed, split, man, tmp_path, experiments=[fe.Experiment('spy', Spy)], targets=['sleep'])
    assert Spy.fits == 0 and not (tmp_path / 'runs').exists()


def test_original_table_duplicate_keys_are_detected(prepared):
    t = prepared[0].table
    bad = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='중복'):
        fd.check_common_columns(bad, bad)


def test_feature_function_mutating_table_is_blocked_and_original_protected(prepared):
    data = prepared[0]
    before = data.table.copy(deep=True)
    before_nights = data.nights.copy(deep=True)

    def sneaky(d):
        d.table['target_sleep_min'] += 100                  # 원본을 직접 고치려 함
        d.nights['sleep_min'] = 0
        return pd.DataFrame({'x': 1.0}, index=d.table.index)
    with pytest.raises(ValueError, match='직접 고쳤음'):
        fd.add_features(data, sneaky)
    pd.testing.assert_frame_equal(data.table, before)       # 원본 그대로
    pd.testing.assert_frame_equal(data.nights, before_nights)

    def adds_column_in_place(d):
        d.table['y'] = 1.0                                  # 열을 표에 직접 붙이는 것도 반환값이 아니므로 무시 + 원본 보호
        return pd.DataFrame({'z': 2.0}, index=d.table.index)
    t = fd.add_features(data, adds_column_in_place)
    assert 'z' in t.columns and 'y' not in data.table.columns


def test_normal_feature_example_still_works(prepared):
    data = prepared[0]

    def my_feat(d):
        lags = d.table[[f'sleep_lag{k}' for k in range(1, 8)]]
        return pd.DataFrame({'sleep_lag_mean': lags.mean(axis=1)}, index=d.table.index)
    t = fd.add_features(data, my_feat, fd.feat_steps_lags)
    assert fd.check_common_columns(data.table, t) == ['sleep_lag_mean'] + [f'steps_lag{k}' for k in range(1, 8)]


def test_new_split_must_keep_previous_final_group(tmp_path):
    pids = [f'p{i:03d}' for i in range(40)]
    prev = tmp_path / 'v2'
    fs.load_or_make_split(pd.DataFrame({'pid': pids}), prev, protocol='v2')
    s, m = fs.load_or_make_split(pd.DataFrame({'pid': pids}), tmp_path / 'v3', protocol='v3', previous_dir=prev)
    assert m['same_final_as_previous'] and m['same_split_as_previous']
    assert m['previous_split_sha256'] == fs.split_hash(s)
    with pytest.raises(fs.SplitMismatch, match='최종 평가'):     # 참가자가 바뀌어 최종 평가 목록이 달라지면
        fs.load_or_make_split(pd.DataFrame({'pid': pids[:-1]}), tmp_path / 'v3b', protocol='v3', previous_dir=prev)
    assert not (tmp_path / 'v3b' / fs.SPLIT_FILE).exists()       # 저장하지 않음
