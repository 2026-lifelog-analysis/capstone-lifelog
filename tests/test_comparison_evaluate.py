import numpy as np
import pandas as pd
import json
import pytest

from experiments.seoyeon.comparison_evaluate import alarm_metrics, pair_metrics, segment_metrics
from experiments.seoyeon.comparison_evaluate import MODELS, finalize_clean
from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_moment import MODEL_ID, REVISION, WEIGHTS_SHA256


@pytest.fixture
def clean_run(tmp_path):
    rows = pd.DataFrame({'participant': ['a'] * 5 + ['b'] * 3,
        'date': pd.date_range('2020-01-01', periods=8),
        'split': ['development'] + ['calibration'] * 4 + ['test'] * 3,
        'common_candidate': [True] * 8})
    for model in set(MODELS) - {'moment'}:
        rows[model] = np.arange(8, dtype=float)
    for rate in ['r05', 'r033']:
        rows.to_csv(tmp_path / f'clean_scores_{rate}.csv.gz', index=False)
    keys = rows.loc[rows.split.ne('development'), ['participant', 'date', 'split']]
    keys.to_csv(tmp_path / 'common_candidate_keys.csv.gz', index=False)
    keys.to_csv(tmp_path / 'moment_keys.csv.gz', index=False)
    rows.to_csv(tmp_path / 'seoyeon_daily.csv.gz', index=False)
    pd.DataFrame({'participant': ['a','b']}).to_csv(tmp_path / 'seoyeon_baselines.csv', index=False)
    (tmp_path / 'moment_clean').mkdir()
    moment = keys.assign(score=[0., 1., 2., 3., 100., 200., 300.], valid=True)
    moment.to_csv(tmp_path / 'moment_clean/scores.csv.gz', index=False)
    record = dict(status='complete', requested_persondays=len(keys), completed_persondays=len(keys),
        valid_persondays=len(keys), unavailable_persondays=0, eligible_persondays=len(keys),
        input_source_sha256={name: io.sha256(tmp_path / file) for name, file in
            [('daily','seoyeon_daily.csv.gz'),('baselines','seoyeon_baselines.csv'),('keys','moment_keys.csv.gz')]},
        scores_sha256=io.sha256(tmp_path / 'moment_clean/scores.csv.gz'),
        model=dict(model_id=MODEL_ID, revision=REVISION, precision='float32', trainable_parameters=0,
                   training_mode=False, checkpoint_sha256={'model.safetensors': WEIGHTS_SHA256},
                   controls={'pass': True}))
    (tmp_path / 'moment_clean/execution.json').write_text(json.dumps(record))
    thresholds = [{'id': label, 'rate': rate,
        'thresholds': {model: 3. for model in MODELS if model != 'moment'}}
        for label, rate in [('r05', .05), ('r033', 1 / 30)]]
    (tmp_path / 'clean_thresholds_without_moment.json').write_text(json.dumps(thresholds))
    return tmp_path


def test_clean_finalize_preserves_frozen_keys_and_never_uses_test_thresholds(clean_run):
    finalize_clean(clean_run)
    settings = json.loads((clean_run / 'clean_thresholds.json').read_text())
    assert all(s['thresholds']['moment'] == 3. for s in settings)
    metrics = pd.read_csv(clean_run / 'alarm_summary.csv')
    assert len(metrics) == 2 * 10 * 2 * 2
    test = metrics.query("rate_id == 'r05' and split == 'test' and scope == 'common'")
    assert test.days.eq(3).all()
    assert test.alarm_days.eq(3).all()
    record = json.loads((clean_run / 'clean_evaluation.json').read_text())
    assert record['frozen_common_rows'] == 7
    assert record['threshold_test_access'] is False
    assert record['source_before'] == record['source_after']


@pytest.mark.parametrize('kind,error', [
    ('missing', 'checksum|count|missing on frozen'), ('duplicate', 'Duplicate MOMENT'),
    ('incomplete', 'inference is incomplete')])
def test_clean_finalize_rejects_unverified_moment(clean_run, kind, error):
    path = clean_run / 'moment_clean/scores.csv.gz'
    frame = pd.read_csv(path)
    if kind == 'missing':
        frame.iloc[:-1].to_csv(path, index=False)
    elif kind == 'duplicate':
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    else:
        (clean_run / 'moment_clean/execution.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError, match=error):
        finalize_clean(clean_run)
    assert not (clean_run / 'alarm_summary.csv').exists()


@pytest.mark.parametrize('kind', ['score', 'input', 'count', 'model'])
def test_clean_provenance_rejects_unlinked_results(clean_run, kind):
    path = clean_run / 'moment_clean/execution.json'
    record = json.loads(path.read_text())
    if kind == 'score':
        record['scores_sha256'] = 'wrong'
    elif kind == 'input':
        record['input_source_sha256']['daily'] = 'wrong'
    elif kind == 'count':
        record['completed_persondays'] = 1
    else:
        record['model']['revision'] = 'wrong'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        finalize_clean(clean_run)
    assert not (clean_run / 'clean_thresholds.json').exists()


def test_alarm_metrics_counts_eligible_zero_people_and_nan():
    frame = pd.DataFrame({'participant': ['a', 'a', 'b', 'b'], 'score': [0., 3., 0., np.nan]})
    result = alarm_metrics(frame, 'score', 2.)
    assert result['days'] == 3 and result['people'] == 2
    assert result['alarm_days'] == 1 and result['zero_people'] == 1
    assert result['top10_alarm_share'] == 1.


def test_pair_overlap_uses_personal_alarm_denominator_not_jaccard():
    frame = pd.DataFrame({'a': [3., 3., 0., np.nan], 'b': [3., 0., 3., 3.]})
    result = pair_metrics(frame, 'a', 'b', 2., 2.)
    assert result['common_days'] == 3
    assert result['overlap_over_a'] == .5
    assert result['jaccard'] == 1 / 3
    assert result['disagreement_days'] == 2


def test_segment_delay_is_conditional_and_missing_not_no_alarm():
    rows = pd.DataFrame({'participant': ['a', 'a', 'b', 'b'],
                         'day_offset': [0, 3, 0, 1],
                         'clean': [0., 0., 0., 0.], 'changed': [0., 3., np.nan, np.nan]})
    result = segment_metrics(rows, 'changed', 'clean', 2.)
    a, b = result.set_index('participant').loc['a'], result.set_index('participant').loc['b']
    assert a['detected'] == 1 and a['delay_days'] == 3
    assert np.isnan(b['detected']) and b['unavailable_days'] == 2
