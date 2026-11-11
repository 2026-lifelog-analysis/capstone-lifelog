"""③ 개인·집단 기준선 결합 테스트 (가짜 데이터)."""
import numpy as np
import pandas as pd
import pytest

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_split as fs
from src import forecast_synthetic as fsyn
from experiments.seoyeon import baseline_blend as bb


@pytest.fixture(scope='module')
def data():
    A, S = fsyn.make_raw(n_people=30, n_days=90, seed=2)
    return fd.build_forecast_table(S, A)


@pytest.fixture(scope='module')
def table(data):
    return fd.add_features(data, bb.feat_personal_history)


@pytest.fixture(scope='module')
def split(data):
    return fs.make_split(fs.participants(data.table))


def _brute(data, row, target, window=bb.WINDOW):
    """한 줄의 개인 과거 중앙값·개수를 정의대로 직접 계산 (비교용)."""
    n = data.nights
    v = fd.NIGHT_VALUE[target]
    ok = ((n['pid'] == row['pid']) & (n['date'] < row['date'])
          & (n['date'] >= row['date'] - pd.Timedelta(days=window))
          & (n['known_at'] <= row['forecast_at']) & ~n['overlap'] & ~n['asleep_invalid'])
    vals = n.loc[ok, v].dropna()
    return (float(vals.median()) if len(vals) else np.nan), int(len(vals))


# ---------- 개인 과거 특징 ----------

def test_feature_columns_added_without_touching_common_columns(data, table):
    extra = fd.check_common_columns(data.table, table)
    expected = [f'{fd.LAG_PREFIX[t]}_{s}{bb.WINDOW}' for t in fd.TARGETS for s in ('pmed', 'n')]
    assert sorted(extra) == sorted(expected)


@pytest.mark.parametrize('target', list(fd.TARGETS))
def test_personal_median_and_count_match_definition(data, table, target):
    p = fd.LAG_PREFIX[target]
    rows = table.sample(60, random_state=0)
    for _, r in rows.iterrows():
        med, cnt = _brute(data, r, target)
        assert r[f'{p}_n{bb.WINDOW}'] == cnt
        if cnt:
            assert r[f'{p}_pmed{bb.WINDOW}'] == pytest.approx(med)
        else:
            assert np.isnan(r[f'{p}_pmed{bb.WINDOW}'])


def test_feature_never_uses_night_d_or_later(data):
    # 밤 D 이후의 값을 모두 바꿔도 밤 D 줄의 특징은 그대로여야 함
    r = data.table.iloc[len(data.table) // 2]
    later = (data.nights['pid'] == r['pid']) & (data.nights['date'] >= r['date'])
    nights2 = data.nights.copy()
    nights2.loc[later, ['sleep_min', 'bedtime', 'waketime']] = 99999.0
    a = bb.personal_history(data.table, data.nights)
    b = bb.personal_history(data.table, nights2)
    i = data.table.index[(data.table['pid'] == r['pid']) & (data.table['date'] == r['date'])][0]
    pd.testing.assert_series_equal(a.loc[i], b.loc[i])


def test_first_night_of_a_person_has_zero_history(data, table):
    # 그 사람의 전체 수면 기록 중 첫 밤 (정답 표의 첫 줄과 다를 수 있음: 앞 밤이 18시 전 시작이면 표에서만 빠짐)
    first_night = data.nights.groupby('pid')['date'].min().rename('first').reset_index()
    first = table.merge(first_night, on='pid')
    first = first[first['date'] == first['first']]
    assert len(first) > 0
    for t in fd.TARGETS:
        p = fd.LAG_PREFIX[t]
        assert (first[f'{p}_n{bb.WINDOW}'] == 0).all()
        assert first[f'{p}_pmed{bb.WINDOW}'].isna().all()


# ---------- 모델 ----------

def _cols(target):
    return fd.base_features(target) + bb.blend_features(target)


def _fit_predict(model, table, target, train_mask):
    y = fd.TARGETS[target]
    feats = _cols(target)
    train = table[train_mask & table[y].notna()]
    model.fit(train, y, feats)
    valid = table[~train_mask]
    return model, model.predict(valid[fd.KEY_COLS + fd.META_COLS + feats]), valid


@pytest.mark.parametrize('target', list(fd.TARGETS))
def test_shrink_blend_limits(table, split, target):
    train_p, _ = fs.fold_pids(split, 1)
    m, pred, valid = _fit_predict(bb.ShrinkBlend(), table, target, table['pid'].isin(train_p))
    p = fd.LAG_PREFIX[target]
    assert np.isfinite(pred).all() and len(pred) == len(valid)
    no_hist = (valid[f'{p}_n{bb.WINDOW}'] == 0).to_numpy()
    assert np.allclose(pred[no_hist], m.group_)          # 기록 없음 → 집단 값
    assert m.k_ in bb.K_GRID


def test_shrink_blend_weight_formula():
    m = bb.ShrinkBlend()
    m.group_, m.k_, m.target_ = 400.0, 4.0, 'sleep'
    m.med_, m.cnt_ = bb.blend_features('sleep')
    df = pd.DataFrame({'sleep_pmed28': [500.0, 500.0, np.nan], 'sleep_n28': [4, 28, 0]})
    out = m._blend(df)
    assert out == pytest.approx([450.0, 400 + 100 * 28 / 32, 400.0])


@pytest.mark.parametrize('target', list(fd.TARGETS))
def test_fixed_blend_weight_in_grid_and_finite(table, split, target):
    train_p, _ = fs.fold_pids(split, 2)
    m, pred, _ = _fit_predict(bb.FixedBlend(), table, target, table['pid'].isin(train_p))
    assert 0.0 <= m.w_ <= 1.0 and m.w_ in bb.W_GRID
    assert np.isfinite(pred).all()


def test_personal_median_falls_back_to_group(table, split):
    train_p, _ = fs.fold_pids(split, 3)
    m, pred, valid = _fit_predict(bb.PersonalMedian(), table, 'sleep', table['pid'].isin(train_p))
    no_hist = (valid['sleep_n28'] == 0).to_numpy()
    assert np.allclose(pred[no_hist], m.group_)
    has = ~no_hist
    assert np.allclose(pred[has], valid['sleep_pmed28'].to_numpy()[has])


def test_parameters_chosen_from_training_rows_only(table, split):
    # 검증 참가자의 정답을 망가뜨려도 고른 k·w·집단값이 같아야 함
    train_p, _ = fs.fold_pids(split, 1)
    mask = table['pid'].isin(train_p)
    broken = table.copy()
    broken.loc[~mask, 'target_sleep_min'] = 99999.0
    for cls in (bb.ShrinkBlend, bb.FixedBlend):
        a, _, _ = _fit_predict(cls(), table, 'sleep', mask)
        b, _, _ = _fit_predict(cls(), broken, 'sleep', mask)
        assert a.params_() == b.params_()


def test_predict_does_not_need_target_columns(table, split):
    train_p, _ = fs.fold_pids(split, 1)
    mask = table['pid'].isin(train_p)
    m, pred, valid = _fit_predict(bb.ShrinkBlend(), table, 'bed', mask)
    assert not any(c.startswith('target_') for c in valid[fd.KEY_COLS + fd.META_COLS + _cols('bed')].columns)


# ---------- 공통 5-fold 실행 ----------

def test_run_cv_all_complete_on_same_nights(table, split):
    exps = bb.experiments()
    scores, preds = fe.run_cv(table, split, exps)
    summary = fe.summarize(scores, preds)
    assert (summary['status'] == 'complete').all()
    # 모든 실험이 같은 목표·fold에서 같은 평가 밤으로 채점됨
    assert scores.groupby(['target', 'fold'])['eval_keys_sha256'].nunique().eq(1).all()
    assert not set(preds['pid']) & fs.final_pids(split)


def test_mae_by_history_bins_counts_every_night(table, split):
    scores, preds = fe.run_cv(table, split, [bb.experiments()[0]], targets=['sleep'])
    out = bb.mae_by_history(preds, table)
    assert out['n_nights'].sum() == len(preds)
    assert set(out['bin']) <= set(bb.BIN_LABELS)
