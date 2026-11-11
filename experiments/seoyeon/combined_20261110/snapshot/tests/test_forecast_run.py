"""공통 실행(가짜 원본) 테스트: 저장 파일, 최종 평가 미사용."""
import json

import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from src import forecast_split as fs


def test_default_run_writes_files_and_never_scores_final(tmp_path):
    data, split, man, out = fr.prepare(synthetic=True, out_dir=tmp_path)
    scores, summary, comp, rd = fr.run(data, split, man, out, synthetic=True, name='t')
    for f in ['scores.csv', 'summary.csv', 'compare.csv', 'predictions.csv', 'run_meta.json']:
        assert (rd / f).exists()
    assert (tmp_path / fs.SPLIT_FILE).exists() and (tmp_path / fs.MANIFEST_FILE).exists()

    preds = pd.read_csv(rd / 'predictions.csv', dtype={'pid': str})
    assert list(preds.columns) == ['pid', 'date', 'target', 'experiment', 'fold', 'y_true', 'y_pred']
    assert not set(preds['pid']) & fs.final_pids(split)
    assert set(preds['fold']) == set(fs.FOLDS)
    assert not preds.duplicated(['pid', 'date', 'target', 'experiment']).any()

    sc = pd.read_csv(rd / 'scores.csv')
    assert {'target', 'experiment', 'fold', 'mae', 'n_people', 'n_nights'} <= set(sc.columns)
    assert set(sc['fold']) == set(fs.FOLDS)

    meta = json.loads((rd / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['final_test_scored'] is False and meta['seed'] == 0
    assert meta['protocol'] == fd.PROTOCOL == man['protocol']
    assert meta['split_sha256'] == man['split_sha256'] == fs.split_hash(split)
    assert meta['data_fingerprint'] == man['data_fingerprint']

    # 같은 설정으로 다시 준비하면 같은 분할·같은 데이터
    data2, split2, man2, _ = fr.prepare(synthetic=True, out_dir=tmp_path)
    assert man2['split_sha256'] == man['split_sha256']
    assert fd.table_fingerprint(data2.table) == man['data_fingerprint']


def test_run_with_extra_feature_uses_same_rows(tmp_path):
    data, split, man, out = fr.prepare(synthetic=True, out_dir=tmp_path)
    table = fd.add_features(data, fd.feat_steps_lags)
    steps = [f'steps_lag{k}' for k in range(1, 8)]
    exps = [fe.Experiment('ridge_base', fe.RidgeModel),
            fe.Experiment('ridge_steps', fe.RidgeModel, lambda t: fd.base_features(t) + steps)]
    scores, summary, comp, rd = fr.run(data, split, man, out, experiments=exps, table=table,
                                       targets=['sleep'], synthetic=True)
    assert set(comp['new']) == {'ridge_steps'}
    a = scores[scores['experiment'] == 'ridge_base'].set_index('fold')['n_nights']
    b = scores[scores['experiment'] == 'ridge_steps'].set_index('fold')['n_nights']
    assert a.equals(b)


def test_outputs_are_separated_by_protocol():
    assert fr.OUT_REAL.name == fd.PROTOCOL and fr.OUT_SYNTH.name == fd.PROTOCOL
    assert fr.OUT_REAL.parent.name == 'forecast'


def test_cli_catboost_experiment_on_synthetic(tmp_path, capsys):
    import pytest
    pytest.importorskip('catboost')
    fr.main(['--synthetic', '--out', str(tmp_path), '--experiments', 'catboost_base', 'ridge_base',
             '--baseline', 'catboost_base', '--targets', 'sleep', '--name', 'cb'])
    rd = next((tmp_path / 'runs').iterdir())
    meta = json.loads((rd / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['baseline'] == 'catboost_base' and meta['final_test_scored'] is False
    assert meta['experiments']['catboost_base']['params']['random_seed'] == 0
    assert meta['experiments']['catboost_base']['features']['sleep'] == fd.base_features('sleep')
    comp = pd.read_csv(rd / 'compare.csv')
    assert set(comp['base']) == {'catboost_base'} and set(comp['new']) == {'ridge_base'}
    assert (comp.loc[comp['fold'] == 'mean', 'status'] == 'complete').all()
