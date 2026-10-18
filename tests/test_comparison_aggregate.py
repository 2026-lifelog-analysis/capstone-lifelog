"""고정 분모·100회 완료 강제 및 개인/seed 집계를 합성 자료로 검증합니다."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments.seoyeon.comparison_aggregate import aggregate, exact_align
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon import comparison_io as io


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture(scope='module')
def completed_template(tmp_path_factory):
    run = tmp_path_factory.mktemp('aggregate_fixture') / 'run'
    run.mkdir()
    write_json(run / 'protocol.json', {'version': 'fixture-v1'})
    dates = pd.date_range('2020-01-29', periods=14)
    frozen = pd.DataFrame({'participant': 'p01', 'date': dates, 'split': 'test'})
    frozen.to_csv(run / 'common_candidate_keys.csv.gz', index=False)
    settings = [dict(id=name, rate=rate, thresholds={model: 1. for model in MODELS})
                for name, rate in [('r05', .05), ('r033', 1 / 30)]]
    write_json(run / 'clean_thresholds.json', settings)
    without = [dict(item, thresholds={key: value for key, value in item['thresholds'].items()
                                      if key != 'moment'}) for item in settings]
    write_json(run / 'clean_thresholds_without_moment.json', without)
    for name in ['main_daily.csv.gz', 'split.json', 'seoyeon_baselines.csv']:
        (run / name).write_text('fixture unused input')
    source_names = ['main_daily.csv.gz', 'common_candidate_keys.csv.gz', 'split.json',
                    'seoyeon_baselines.csv', 'clean_thresholds_without_moment.json']
    source_hashes = {name: io.sha256(run / name) for name in source_names}
    for setting in settings:
        clean = frozen.assign(common_candidate=True, **{model: 0. for model in MODELS})
        clean.to_csv(run / f"evaluated_scores_{setting['id']}.csv.gz", index=False)
    core_model = dict(model_id='AutonLab/MOMENT-1-large', revision='fixed-fixture-revision',
        checkpoint_sha256={'weights':'fixture'}, precision='float32', trainable_parameters=0,
        training_mode=False, seq_len=512, patch_len=8, seed=20261011, revin_epsilon=1e-5,
        versions={'torch':'fixture'}, controls={'pass':True})
    (run / 'moment_clean').mkdir()
    write_json(run / 'moment_clean/execution.json', {'status':'complete','model':core_model})
    clean_files = ['clean_thresholds.json','evaluated_scores_r05.csv.gz','evaluated_scores_r033.csv.gz']
    write_json(run / 'clean_evaluation.json', dict(status='passed',
        outputs_sha256={name:io.sha256(run/name) for name in clean_files},
        inputs_sha256={'moment_clean/execution.json':io.sha256(run/'moment_clean/execution.json')}))
    for scenario in range(1, 6):
        for seed in range(20):
            job = run / 'synthetic' / f'S{scenario}_seed{seed:02d}'
            moment = job / 'moment'
            moment.mkdir(parents=True)
            sample = frozen.assign(start=dates[0], day_offset=np.arange(14), seed=seed)
            sample.to_csv(job / 'moment_keys.csv.gz', index=False)
            sample.to_csv(job / 'canonical_daily.csv.gz', index=False)
            pd.DataFrame({'pid': ['p01'], 'start': [dates[0]], 'end': [dates[-1]],
                          'seed': [seed]}).to_csv(job / 'segments.csv', index=False)
            values = np.zeros(14)
            if seed % 2 == 0:
                values[3] = 2.
            for setting in settings:
                changed = sample.assign(**{model: values for model in MODELS if model != 'moment'})
                changed.to_csv(job / f"scores_{setting['id']}.csv.gz", index=False)
            scores = sample.assign(score=values, valid=True, row_index=np.arange(14), model='moment')
            scores.to_csv(moment / 'scores.csv.gz', index=False)
            write_json(moment / 'execution.json', dict(status='complete',
                requested_persondays=14, valid_persondays=14, unavailable_persondays=0,
                completed_persondays=14, eligible_persondays=14, model=core_model,
                input_source_sha256={'keys': io.sha256(job / 'moment_keys.csv.gz'),
                                     'daily': io.sha256(job / 'canonical_daily.csv.gz'),
                                     'baselines': io.sha256(run / 'seoyeon_baselines.csv')},
                scores_sha256=io.sha256(moment / 'scores.csv.gz')))
            write_json(job / 'job.json', dict(status='passed', scenario=f'S{scenario}', seed=seed,
                participants=1, sample_days=14, thresholds_refitted=False, settings=without,
                input_hashes=source_hashes,
                artifact_hashes={path.name: io.sha256(path) for path in job.iterdir() if path.is_file()}))
    return run


@pytest.mark.parametrize('kind', ['clean_threshold', 'synthetic_model', 'synthetic_count'])
def test_changed_producer_outputs_or_neural_identity_are_rejected(completed_run, kind):
    if kind == 'clean_threshold':
        path = completed_run / 'clean_thresholds.json'
        value = json.loads(path.read_text())
        value[0]['thresholds']['moment'] = 999.
    else:
        path = completed_run / 'synthetic/S1_seed00/moment/execution.json'
        value = json.loads(path.read_text())
        if kind == 'synthetic_model':
            value['model']['revision'] = 'different-model'
        else:
            value['completed_persondays'] = 1
    write_json(path, value)
    with pytest.raises(ValueError):
        aggregate(completed_run, completed_run / 'rejected')


@pytest.fixture
def completed_run(completed_template, tmp_path):
    run = tmp_path / 'run'
    shutil.copytree(completed_template, run)
    return run


def test_exact_align_restores_key_order_but_rejects_missing_and_duplicate():
    expected = pd.DataFrame({'participant': ['a', 'a'], 'date': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    other = expected.assign(score=[1., 2.]).iloc[::-1]
    actual = exact_align(expected, other, ['score'], 'fixture')
    assert actual.score.tolist() == [1., 2.]
    with pytest.raises(ValueError, match='missing|key'):
        exact_align(expected, other.iloc[:1], ['score'], 'fixture')
    with pytest.raises(ValueError, match='duplicate'):
        exact_align(expected, pd.concat([other, other]), ['score'], 'fixture')


def test_all_hundred_jobs_aggregate_person_and_seed_views(completed_run):
    output = completed_run / 'summary'
    result = aggregate(completed_run, output)
    assert result['status'] == 'passed'
    assert result['completed_jobs'] == 100
    records = pd.read_csv(output / 'segment_results.csv.gz')
    assert len(records) == 2000
    assert records.unavailable_days.eq(0).all()
    people = pd.read_csv(output / 'participant_summary.csv')
    assert len(people) == 100
    assert people.repetitions.eq(20).all()
    assert people.detection_rate.eq(.5).all()
    assert people.clean_alarm_rate.eq(0).all()
    assert people.delta_mean.eq(.5).all()
    assert people.delay_mean_detected.eq(3.).all()
    assert people.delay_median_detected.eq(3.).all()
    seeds = pd.read_csv(output / 'seed_summary.csv')
    assert len(seeds) == 2000
    assert seeds.loc[seeds.seed.mod(2).eq(1), 'delay_median_detected'].isna().all()
    distribution = pd.read_csv(output / 'seed_distribution_summary.csv')
    assert distribution.delay_available_units.eq(10).all()
    assert distribution.delay_mean_detected_mean.eq(3.).all()
    with pytest.raises(FileExistsError):
        aggregate(completed_run, output)


def test_incomplete_batch_is_unverified_without_summary_tables(completed_run):
    path = completed_run / 'synthetic/S5_seed19/job.json'
    value = json.loads(path.read_text())
    write_json(path, {**value, 'status': 'running'})
    output = completed_run / 'incomplete'
    with pytest.raises(ValueError, match='100|incomplete|passed'):
        aggregate(completed_run, output)
    record = json.loads((output / 'aggregate_failure.json').read_text())
    assert record['status'] == 'unverified'
    assert not (output / 'participant_summary.csv').exists()


@pytest.mark.parametrize('fault', ['nonfinite', 'missing', 'duplicate', 'wrong_offset'])
def test_fault_on_frozen_key_is_never_silently_dropped(completed_run, fault):
    job = completed_run / 'synthetic/S1_seed00'
    score = job / 'scores_r05.csv.gz'
    data = pd.read_csv(score)
    if fault == 'nonfinite':
        data.loc[0, 'fixed14'] = np.nan
    elif fault == 'missing':
        data = data.iloc[1:]
    elif fault == 'duplicate':
        data = pd.concat([data, data.iloc[:1]])
    else:
        data.loc[0, 'day_offset'] = 13
    data.to_csv(score, index=False)
    record = json.loads((job / 'job.json').read_text())
    record['artifact_hashes'][score.name] = io.sha256(score)
    write_json(job / 'job.json', record)
    output = completed_run / 'invalid'
    with pytest.raises(ValueError):
        aggregate(completed_run, output)
    assert json.loads((output / 'aggregate_failure.json').read_text())['status'] == 'unverified'
    assert not (output / 'segment_results.csv.gz').exists()


def test_moment_provenance_hash_is_required(completed_run):
    path = completed_run / 'synthetic/S1_seed00/moment/execution.json'
    record = json.loads(path.read_text())
    record['input_source_sha256']['keys'] = 'not-the-input-sha'
    write_json(path, record)
    with pytest.raises(ValueError, match='MOMENT.*hash|MOMENT.*SHA'):
        aggregate(completed_run, completed_run / 'wrong-moment')
