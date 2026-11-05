"""공통 채점·5-fold 실행 테스트 (가짜 데이터)."""
import numpy as np
import pandas as pd
import pytest

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_split as fs
from src import forecast_synthetic as fsyn


@pytest.fixture(scope='module')
def data():
    A, S = fsyn.make_raw(n_people=30, n_days=60, seed=1)
    return fd.build_forecast_table(S, A)


@pytest.fixture(scope='module')
def split(data):
    return fs.make_split(fs.participants(data.table))


def _ev(rows):
    return pd.DataFrame(rows, columns=['pid', 'date', 'fold', 'y_true']).assign(
        date=lambda d: pd.to_datetime(d['date']))


def test_same_night_2350_vs_0010_error_is_20_minutes():
    # 같은 밤 D=03-01: 실제 23:50(710), 예측 다음 날 00:10(730)
    ev = _ev([('a', '2016-03-01', 1, 710.0)])
    s = fe.score(ev[['pid', 'date']].assign(y_pred=730.0), ev)
    assert s['mae'] == 20 and s['complete']


def test_missing_or_nan_prediction_makes_run_incomplete():
    ev = _ev([('a', '2016-03-01', 1, 400.0), ('a', '2016-03-02', 1, 420.0), ('b', '2016-03-01', 1, 500.0)])
    pred = ev[['pid', 'date']].iloc[:2].assign(y_pred=[400.0, np.nan])   # b 빠짐, a 둘째 NaN
    s = fe.score(pred, ev)
    assert not s['complete'] and np.isnan(s['mae'])
    assert s['n_missing'] == 1 and s['n_nonfinite'] == 1 and s['n_nights'] == 3
    assert '불완전' in s['note']


def test_duplicate_or_foreign_predictions_raise():
    ev = _ev([('a', '2016-03-01', 1, 400.0)])
    with pytest.raises(ValueError):
        fe.score(pd.concat([ev, ev])[['pid', 'date']].assign(y_pred=1.0), ev)
    with pytest.raises(ValueError):
        fe.score(_ev([('zz', '2016-03-01', 1, 0)])[['pid', 'date']].assign(y_pred=1.0), ev)


def test_eval_nights_do_not_depend_on_inputs(data, split):
    t = data.table.copy()
    t.loc[t.index[::3], [c for c in t.columns if '_lag' in c]] = np.nan   # 입력을 많이 지워도
    for target in fd.TARGETS:
        pd.testing.assert_frame_equal(fe.eval_nights(data.table, split, target),
                                      fe.eval_nights(t, split, target))


def test_all_models_scored_on_same_nights_and_final_never_used(data, split):
    seen = []

    class Spy(fe.GroupMedian):
        def fit(self, train, y, features):
            seen.append(('fit', set(train['pid'])))
            return super().fit(train, y, features)

        def predict(self, df):
            seen.append(('predict', set(df['pid'])))
            assert not any(c.startswith('target_') for c in df.columns)   # 검증 정답은 안 넘어감
            return super().predict(df)

    exps = [fe.Experiment('spy', Spy), fe.Experiment('pm', fe.PersonalMean),
            fe.Experiment('ridge', fe.RidgeModel)]
    scores, preds = fe.run_cv(data.table, split, exps)
    final = fs.final_pids(split)
    assert all(not (pids & final) for _, pids in seen)
    assert not set(preds['pid']) & final
    assert scores['complete'].all()
    # 같은 목표·fold에서 실험마다 평가 밤 목록이 같음
    for (t, f), g in preds.groupby(['target', 'fold']):
        keys = [tuple(map(tuple, x[['pid', 'date']].to_numpy())) for _, x in g.groupby('experiment')]
        assert all(k == keys[0] for k in keys)
    # 목표가 달라도 같은 참가자 배정
    folds = preds.groupby(['target', 'pid'])['fold'].first().unstack(0)
    assert (folds.nunique(axis=1) == 1).all()
    dev_fold = split.set_index('pid')['fold']
    assert (folds.iloc[:, 0] == dev_fold.loc[folds.index]).all()


def test_target_specific_missing_rows_do_not_change_assignment(data, split):
    t = data.table.copy()
    t.loc[t.index[::4], 'target_waketime'] = np.nan
    ev_w, ev_s = fe.eval_nights(t, split, 'wake'), fe.eval_nights(t, split, 'sleep')
    assert len(ev_w) < len(ev_s)
    a = ev_w.groupby('pid')['fold'].first()
    b = ev_s.groupby('pid')['fold'].first()
    assert (a == b.loc[a.index]).all()


def test_nan_predicting_model_is_flagged_not_dropped(data, split):
    class Holey(fe.GroupMedian):
        def predict(self, df):
            p = super().predict(df)
            p[::7] = np.nan
            return p
    scores, preds = fe.run_cv(data.table, split, [fe.Experiment('holey', Holey)], targets=['sleep'])
    assert not scores['complete'].any() and scores['mae'].isna().all()
    summ = fe.summarize(scores, preds)
    assert not summ['complete'].iloc[0] and np.isnan(summ['oof_mae'].iloc[0])
    assert summ['n_nights'].iloc[0] == len(fe.eval_nights(data.table, split, 'sleep'))


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
def test_nonfinite_prediction_is_not_scored(bad):
    ev = _ev([('a', '2016-03-01', 1, 400.0), ('a', '2016-03-02', 1, 420.0)])
    s = fe.score(ev[['pid', 'date']].assign(y_pred=[400.0, bad]), ev)
    assert not s['complete'] and np.isnan(s['mae']) and s['n_nonfinite'] == 1


def test_inf_predicting_model_is_flagged(data, split):
    class Inf(fe.GroupMedian):
        def predict(self, df):
            p = super().predict(df)
            p[0] = np.inf
            return p
    scores, preds = fe.run_cv(data.table, split, [fe.Experiment('inf', Inf)], targets=['bed'])
    assert not scores['complete'].any() and scores['mae'].isna().all()
    assert (fe.summarize(scores, preds)['status'] == 'incomplete').all()


def _scores_two(data, split, targets=('sleep',)):
    exps = [fe.Experiment('gm', fe.GroupMedian), fe.Experiment('pm', fe.PersonalMean)]
    return fe.run_cv(data.table, split, exps, targets=list(targets))


def test_compare_four_folds_is_partial_not_averaged(data, split):
    scores, preds = _scores_two(data, split)
    four = scores[~((scores['experiment'] == 'pm') & (scores['fold'] == 5))]
    c = fe.compare(four, 'gm', 'pm')
    m = c[c['fold'] == 'mean'].iloc[0]
    assert m['status'] == 'partial' and np.isnan(m['diff']) and np.isnan(m['n_improved'])
    assert (fe.summarize(four, preds).set_index('experiment').loc['pm', 'status']) == 'partial'
    assert np.isnan(fe.summarize(four, preds).set_index('experiment').loc['pm', 'fold_mae_mean'])
    # 일부러 fold 일부만 돌린 실행도 부분 실행으로 표시
    s3, p3 = fe.run_cv(data.table, split, [fe.Experiment('gm', fe.GroupMedian)], targets=['sleep'], folds=[1, 2])
    assert fe.summarize(s3, p3)['status'].iloc[0] == 'partial'


def test_compare_with_nan_fold_is_incomplete(data, split):
    scores, _ = _scores_two(data, split)
    bad = scores.copy()
    i = bad.index[(bad['experiment'] == 'pm') & (bad['fold'] == 3)]
    bad.loc[i, ['mae', 'complete', 'note']] = [np.nan, False, '불완전: 예측 없음 1밤, NaN/inf 예측 0밤']
    c = fe.compare(bad, 'gm', 'pm')
    m = c[c['fold'] == 'mean'].iloc[0]
    assert m['status'] == 'incomplete' and np.isnan(m['diff']) and np.isnan(m['n_improved'])
    assert c.loc[(c['fold'] == 3), 'status'].iloc[0] == 'incomplete'
    assert c.loc[(c['fold'] == 1), 'status'].iloc[0] == 'complete'


def test_compare_duplicate_fold_rows_are_not_complete(data, split):
    scores, _ = _scores_two(data, split)
    dup = pd.concat([scores, scores[(scores['experiment'] == 'pm') & (scores['fold'] == 1)]], ignore_index=True)
    c = fe.compare(dup, 'gm', 'pm')
    m = c[c['fold'] == 'mean'].iloc[0]
    assert m['status'] != 'complete' and np.isnan(m['diff']) and np.isnan(m['n_improved'])
    assert 'fold 중복' in m['note']


def test_compare_same_count_but_different_keys_raises(data, split):
    scores, _ = _scores_two(data, split)
    other = scores.copy()
    i = other.index[(other['experiment'] == 'pm') & (other['fold'] == 2)]
    other.loc[i, 'eval_keys_sha256'] = 'different-nights-same-count'
    assert (other.loc[i, 'n_nights'] == scores.loc[i, 'n_nights']).all()
    with pytest.raises(ValueError, match='평가 밤'):
        fe.compare(other, 'gm', 'pm')


def test_eval_keys_hash_changes_with_keys_not_order():
    ev = _ev([('a', '2016-03-01', 1, 1.0), ('b', '2016-03-02', 1, 1.0)])
    assert fe.keys_hash(ev) == fe.keys_hash(ev.iloc[::-1])
    moved = ev.assign(date=pd.to_datetime(['2016-03-01', '2016-03-03']))
    assert fe.keys_hash(ev) != fe.keys_hash(moved)


def test_summary_and_compare_definitions(data, split):
    exps = [fe.Experiment('gm', fe.GroupMedian), fe.Experiment('ridge', fe.RidgeModel)]
    scores, preds = fe.run_cv(data.table, split, exps, targets=['bed'])
    summ = fe.summarize(scores, preds).set_index('experiment')
    g = scores[scores['experiment'] == 'ridge']
    assert summ.loc['ridge', 'fold_mae_mean'] == pytest.approx(g['mae'].mean())
    assert summ.loc['ridge', 'fold_mae_std'] == pytest.approx(np.std(g['mae'], ddof=1))
    p = preds[preds['experiment'] == 'ridge']
    assert summ.loc['ridge', 'oof_mae'] == pytest.approx((p['y_true'] - p['y_pred']).abs().mean())
    c = fe.compare(scores, 'gm', 'ridge')
    per_fold = c[c['fold'] != 'mean']
    assert len(per_fold) == 5
    assert np.allclose(per_fold['diff'], per_fold['base_mae'] - per_fold['new_mae'])
    m = c[c['fold'] == 'mean'].iloc[0]
    assert m['n_improved'] == (per_fold['diff'] > 0).sum()


def test_models_are_new_each_fold(data, split):
    made = []

    def factory():
        m = fe.GroupMedian()
        made.append(m)
        return m
    fe.run_cv(data.table, split, [fe.Experiment('gm', factory)], targets=['sleep'])
    assert len(made) == 5 and len({id(m) for m in made}) == 5


# ---------- CatBoost ----------

def _cb():
    pytest.importorskip('catboost')
    return lambda: fe.CatBoostModel(iterations=30)


def test_catboost_fixed_seed_and_new_model_per_fold(data, split):
    mk = _cb()
    made = []

    def factory():
        m = mk()
        made.append(m)
        return m
    exps = [fe.Experiment('cb', factory)]
    s1, p1 = fe.run_cv(data.table, split, exps, targets=['sleep', 'wake'])
    s2, p2 = fe.run_cv(data.table, split, exps, targets=['sleep', 'wake'])
    assert len(made) == 2 * 2 * 5 and len({id(m) for m in made}) == 20     # 목표 × fold마다 새로
    assert s1['complete'].all()
    np.testing.assert_array_equal(p1['y_pred'].to_numpy(), p2['y_pred'].to_numpy())
    assert made[0].params['random_seed'] == fe.SEED


def test_catboost_inputs_and_categorical_dow(data, split):
    mk = _cb()
    m = mk()
    tr = data.table[data.table['pid'].isin(fs.fold_pids(split, 1)[0])]
    m.fit(tr, 'target_bedtime', fd.base_features('bed'))
    assert m.model_.feature_names_ == fd.base_features('bed')            # ID·정답·메타 없음
    assert m.cat_ == ['dow'] and m.model_.get_cat_feature_indices() == [7]
    assert m.params['nan_mode'] == 'Min'
    with pytest.raises(ValueError):
        fd.check_features(fd.base_features('bed') + ['pid'])


def test_catboost_prediction_does_not_depend_on_other_validation_rows(data, split):
    mk = _cb()
    train_p, valid_p = fs.fold_pids(split, 1)
    tr = data.table[data.table['pid'].isin(train_p)]
    va = data.table[data.table['pid'].isin(valid_p)]
    feats = fd.base_features('sleep')
    m = mk().fit(tr, 'target_sleep_min', feats)
    full = m.predict(va[feats])
    va2 = va[feats].copy()
    va2.iloc[1:, :-1] = 9999.0                                           # 다른 검증 줄을 바꿔도
    assert m.predict(va2)[0] == full[0]                                  # 첫 줄 예측은 그대로
    m2 = mk().fit(tr, 'target_sleep_min', feats)                         # 검증 줄 없이 다시 학습해도 같음
    np.testing.assert_array_equal(m2.predict(va[feats]), full)


def test_catboost_dow_is_one_hot_not_target_statistics():
    assert fe.CATBOOST_PARAMS['one_hot_max_size'] >= 7
