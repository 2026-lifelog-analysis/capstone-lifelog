"""100개 실제 주입 실행을 고정 공통 인일에서만 집계합니다.

탐지율은 합성 변화 구간의 경보 확률이며 임상 정확도가 아닙니다. 같은 구간의
무주입 경보율도 임상 위양성률이 아닙니다. 불완전 작업이나 분모 변경을 허용하지
않고, 개인별 20회 요약과 seed별 사람 평균을 서로 분리해 저장합니다.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS, segment_metrics
from experiments.seoyeon.comparison_safety import verified_outputs

VERSION = 'frozen-common-synthetic-aggregate-1'
KEYS = ['participant', 'date']
RATES = {'r05': .05, 'r033': 1 / 30}
SCENARIOS = tuple(f'S{i}' for i in range(1, 6))
EXPECTED = tuple(f'{scenario}_seed{seed:02d}' for scenario in SCENARIOS for seed in range(20))
STAT_MODELS = [name for name in MODELS if name != 'moment']
SOURCE_NAMES = ['main_daily.csv.gz', 'common_candidate_keys.csv.gz', 'split.json',
                'seoyeon_baselines.csv', 'clean_thresholds_without_moment.json']


def _keys(frame, label):
    if not set(KEYS).issubset(frame) or frame[KEYS].isna().any().any():
        raise ValueError(f'{label}: missing participant/date keys')
    data = frame.copy()
    data['participant'] = data.participant.astype(str)
    data['date'] = pd.to_datetime(data.date, errors='raise')
    if data.date.isna().any() or not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError(f'{label}: invalid calendar dates')
    if data.duplicated(KEYS).any():
        raise ValueError(f'{label}: duplicate participant/date keys')
    return data


def exact_align(expected, frame, columns, label, allow_extra=False):
    """키 순서를 명시적으로 맞추되 누락·중복 또는 예기치 않은 추가 키는 거부."""
    wanted, data = _keys(expected, 'frozen sample'), _keys(frame, label)
    if not set(columns).issubset(data):
        raise ValueError(f'{label}: missing required columns')
    aligned = wanted[KEYS].merge(data[KEYS + list(columns)], on=KEYS, how='left',
                               validate='one_to_one', indicator=True, sort=False)
    if not aligned._merge.eq('both').all():
        raise ValueError(f'{label}: missing frozen keys')
    if not allow_extra and len(data) != len(wanted):
        raise ValueError(f'{label}: extra keys outside the frozen sample')
    return aligned.drop(columns='_merge')


def _finite(frame, columns, label):
    if not np.isfinite(frame[list(columns)].to_numpy(float)).all():
        raise ValueError(f'{label}: nonfinite values on frozen keys; no rows dropped')


def _metadata(expected, actual, label, include_split=False):
    for column in ['day_offset', 'seed']:
        if not np.array_equal(expected[column].to_numpy(), actual[column].to_numpy()):
            raise ValueError(f'{label}: {column} differs from frozen segment')
    if not pd.to_datetime(actual.start).reset_index(drop=True).equals(
            pd.to_datetime(expected.start).reset_index(drop=True)):
        raise ValueError(f'{label}: start differs from frozen segment')
    if include_split and not actual.split.eq('test').all():
        raise ValueError(f'{label}: non-test keys found')


class Evidence:
    """읽은 모든 입력 파일을 실행 전후 해시로 고정합니다."""

    def __init__(self, root):
        self.root, self.hashes = root, {}

    def path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Input path must remain within the execution directory')
        path = self.root / relative
        digest = io.sha256(path)
        name = relative.as_posix()
        if name in self.hashes and self.hashes[name] != digest:
            raise ValueError(f'Input changed during aggregation: {name}')
        self.hashes[name] = digest
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def frame(self, relative):
        return io.read_frame(self.path(relative))

    def check(self, relative, expected):
        self.path(relative)
        if self.hashes[Path(relative).as_posix()] != expected:
            raise ValueError(f'Input SHA256 mismatch: {relative}')

    def unchanged(self):
        for relative, expected in self.hashes.copy().items():
            self.check(relative, expected)


def _settings(raw):
    if len(raw) != 2 or {item['id'] for item in raw} != set(RATES):
        raise ValueError('Exactly two frozen rate settings are required')
    for setting in raw:
        if setting['rate'] != RATES[setting['id']] or set(setting['thresholds']) != set(MODELS):
            raise ValueError('Frozen rate/model configuration differs')
        if not np.isfinite(list(setting['thresholds'].values())).all():
            raise ValueError('Frozen thresholds must be finite')
    return {item['id']: item for item in raw}


def _sample(test_keys, segments, seed):
    required = ['pid', 'start', 'end', 'seed']
    if not set(required).issubset(segments) or segments[required].isna().any().any():
        raise ValueError('Segment metadata is incomplete')
    parts = segments[required].rename(columns={'pid': 'participant'}).copy()
    parts['participant'] = parts.participant.astype(str)
    parts['start'], parts['end'] = pd.to_datetime(parts.start), pd.to_datetime(parts.end)
    if not parts.start.eq(parts.start.dt.normalize()).all() or not parts.end.eq(parts.end.dt.normalize()).all():
        raise ValueError('Segment boundaries must be calendar-day midnights')
    if parts.participant.duplicated().any() or not parts.seed.eq(seed).all():
        raise ValueError('Duplicate participant segments or wrong seed')
    if not parts.end.sub(parts.start).dt.days.eq(13).all():
        raise ValueError('Each injection segment must span exactly 14 calendar days')
    joined = test_keys[KEYS].merge(parts, on='participant', how='inner', validate='many_to_one')
    selected = joined[joined.date.between(joined.start, joined.end)].copy()
    if set(selected.participant) != set(parts.participant) or selected.empty:
        raise ValueError('Participant has no frozen evaluation keys')
    if selected.groupby('participant').size().lt(7).any():
        raise ValueError('Fewer than seven frozen evaluation days in a segment')
    selected['day_offset'] = selected.date.sub(selected.start).dt.days
    return selected.sort_values(KEYS).reset_index(drop=True).assign(split='test')


def _moment(evidence, prefix, expected, reference_model):
    record = evidence.json(prefix / 'moment/execution.json')
    if record.get('status') != 'complete':
        raise ValueError(f'{prefix.name}: MOMENT execution incomplete')
    model = record.get('model', {})
    identity = ['model_id', 'revision', 'checkpoint_sha256', 'precision', 'trainable_parameters',
                'training_mode', 'seq_len', 'patch_len', 'seed', 'revin_epsilon', 'versions']
    if (any(key not in reference_model or model.get(key) != reference_model[key] for key in identity)
            or model.get('controls', {}).get('pass') is not True
            or model.get('training_mode') is not False or model.get('trainable_parameters') != 0):
        raise ValueError(f'{prefix.name}: MOMENT model differs from verified clean execution')
    keys_path = prefix / 'moment_keys.csv.gz'
    if record.get('input_source_sha256', {}).get('keys') != io.sha256(evidence.path(keys_path)):
        raise ValueError(f'{prefix.name}: MOMENT keys SHA256 mismatch')
    for name, relative in [('daily', prefix / 'canonical_daily.csv.gz'),
                           ('baselines', Path('seoyeon_baselines.csv'))]:
        if record.get('input_source_sha256', {}).get(name) != io.sha256(evidence.path(relative)):
            raise ValueError(f'{prefix.name}: MOMENT {name} SHA256 mismatch')
    score_path = prefix / 'moment/scores.csv.gz'
    if record.get('scores_sha256') != io.sha256(evidence.path(score_path)):
        raise ValueError(f'{prefix.name}: MOMENT scores SHA256 mismatch')
    data = evidence.frame(score_path)
    if (len(data) != record.get('requested_persondays') or len(data) != record.get('valid_persondays')
            or len(data) != record.get('completed_persondays') or len(data) != record.get('eligible_persondays')
            or record.get('unavailable_persondays') != 0 or not data.model.eq('moment').all()
            or not data.row_index.eq(np.arange(len(data))).all()):
        raise ValueError(f'{prefix.name}: MOMENT row coverage differs')
    columns = ['score', 'valid', 'start', 'day_offset', 'seed', 'split']
    aligned = exact_align(expected, data, columns, f'{prefix.name} MOMENT')
    _metadata(expected, aligned, 'MOMENT', include_split=True)
    if not aligned.valid.eq(True).all():
        raise ValueError(f'{prefix.name}: MOMENT invalid frozen keys')
    _finite(aligned, ['score'], 'MOMENT')
    return aligned.score.to_numpy(float)


def _summaries(records):
    group = ['scenario', 'rate_id', 'model']
    person = records.groupby(group + ['participant'], sort=True).agg(
        repetitions=('seed', 'nunique'), detection_rate=('detected', 'mean'),
        clean_alarm_rate=('clean_alarm', 'mean'), delta_mean=('delta', 'mean'),
        detected_repetitions=('detected', 'sum'), evaluation_days=('evaluation_days', 'sum'),
        delay_mean_detected=('delay_days', 'mean'),
        delay_median_detected=('delay_days', 'median')).reset_index()
    if not person.repetitions.eq(20).all():
        raise ValueError('Every participant must have exactly 20 completed repetitions')
    seeds = records.groupby(group + ['seed'], sort=True).agg(
        participants=('participant', 'nunique'), detection_rate=('detected', 'mean'),
        clean_alarm_rate=('clean_alarm', 'mean'), delta_mean=('delta', 'mean'),
        detected_participants=('detected', 'sum'), evaluation_days=('evaluation_days', 'sum'),
        delay_mean_detected=('delay_days', 'mean'),
        delay_median_detected=('delay_days', 'median')).reset_index()
    distributions = {}
    for name, table, unit in [('participant', person, 'participant'), ('seed', seeds, 'seed')]:
        rows = []
        for keys, subset in table.groupby(group, sort=True):
            row = dict(zip(group, keys), units=int(subset[unit].nunique()),
                       delay_available_units=int(subset.delay_mean_detected.notna().sum()))
            for metric in ['detection_rate', 'clean_alarm_rate', 'delta_mean', 'delay_mean_detected']:
                values = subset[metric]
                row.update({f'{metric}_{stat}': float(value) for stat, value in {
                    'mean': values.mean(), 'median': values.median(),
                    'q25': values.quantile(.25), 'q75': values.quantile(.75),
                    'min': values.min(), 'max': values.max()}.items()})
            rows.append(row)
        distributions[f'{name}_distribution_summary.csv'] = pd.DataFrame(rows)
    return {'participant_summary.csv': person, 'seed_summary.csv': seeds, **distributions}


def _collect(evidence, settings, clean, test_keys, jobs, reference_model):
    records, fixed_by_seed = [], {}
    for name in EXPECTED:
        prefix, job = Path('synthetic') / name, jobs[name]
        scenario, seed = name[:2], int(name[-2:])
        if job.get('scenario') != scenario or job.get('seed') != seed or job.get('thresholds_refitted') is not False:
            raise ValueError(f'{name}: job metadata differs from frozen protocol')
        if not set(SOURCE_NAMES).issubset(job.get('input_hashes', {})):
            raise ValueError(f'{name}: input hashes incomplete')
        for relative, digest in job['input_hashes'].items():
            evidence.check(relative, digest)
        artifacts = {'segments.csv', 'moment_keys.csv.gz', 'canonical_daily.csv.gz',
                     'scores_r05.csv.gz', 'scores_r033.csv.gz'}
        if not artifacts.issubset(job.get('artifact_hashes', {})):
            raise ValueError(f'{name}: artifact hashes incomplete')
        for relative, digest in job.get('artifact_hashes', {}).items():
            evidence.check(prefix / relative, digest)
        expected = _sample(test_keys, evidence.frame(prefix / 'segments.csv'), seed)
        if len(expected) != job.get('sample_days') or expected.participant.nunique() != job.get('participants'):
            raise ValueError(f'{name}: recorded sample size differs')
        if seed in fixed_by_seed and not expected.equals(fixed_by_seed[seed]):
            raise ValueError(f'{name}: scenarios do not use the same frozen person-days')
        fixed_by_seed[seed] = expected
        keys = exact_align(expected, evidence.frame(prefix / 'moment_keys.csv.gz'),
                           ['start', 'day_offset', 'seed', 'split'], name)
        _metadata(expected, keys, name, include_split=True)
        moment = _moment(evidence, prefix, expected, reference_model)
        job_settings = {item['id']: item for item in job.get('settings', [])}
        if set(job_settings) != set(RATES):
            raise ValueError(f'{name}: job rate settings incomplete')
        for rate_id, setting in settings.items():
            frozen = setting['thresholds']
            if job_settings[rate_id]['rate'] != setting['rate'] or any(
                    job_settings[rate_id]['thresholds'].get(model) != frozen[model] for model in STAT_MODELS):
                raise ValueError(f'{name}: thresholds changed after freezing')
            changed = exact_align(expected, evidence.frame(prefix / f'scores_{rate_id}.csv.gz'),
                                  STAT_MODELS + ['start', 'day_offset', 'seed'], name)
            _metadata(expected, changed, name)
            changed['moment'] = moment
            control = exact_align(expected, clean[rate_id], MODELS, 'clean scores', allow_extra=True)
            _finite(changed, MODELS, name)
            _finite(control, MODELS, 'clean scores')
            for model in MODELS:
                frame = expected.assign(changed=changed[model].to_numpy(), clean=control[model].to_numpy())
                result = segment_metrics(frame, 'changed', 'clean', frozen[model])
                if result.unavailable_days.ne(0).any() or result.detected.isna().any():
                    raise ValueError(f'{name}: an evaluation outcome is unavailable')
                records.append(result.assign(scenario=scenario, seed=seed, rate_id=rate_id,
                    rate=setting['rate'], model=model, threshold=frozen[model],
                    delta=result.detected - result.clean_alarm))
    return pd.concat(records, ignore_index=True)


def aggregate(run, output):
    """100개 완료가 필수입니다. 실패하면 새 출력 경로에 미검증 기록만 남깁니다."""
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, evidence = time.perf_counter(), Evidence(run)
    code_paths = [Path(__file__), Path(__file__).with_name('comparison_evaluate.py'),
                  Path(__file__).with_name('comparison_io.py'), Path(__file__).with_name('comparison_safety.py')]
    base = dict(version=VERSION, code_sha256={path.name: io.sha256(path) for path in code_paths},
                python=platform.python_version(), versions={name: version(name) for name in ['numpy', 'pandas']},
                definitions={'detected': 'Any strict score > frozen threshold on frozen segment days',
                    'clean_alarm': 'Any clean-data alarm on exactly the same person-days; not clinical FPR',
                    'delta': 'detected minus clean_alarm; not clinical accuracy',
                    'delay_days': 'First calendar day offset, conditional on detection; otherwise missing',
                    'summaries': 'Separate participant-over-20-seeds and seed-over-participants summaries',
                    'delay_distribution': 'Equally weighted conditional unit means; unavailable units excluded and counted',
                    'distribution_quantiles': 'Pandas linear interpolation; descriptive, no independence claim'})
    try:
        actual = {path.name for path in (run / 'synthetic').glob('S*_seed*') if path.is_dir()}
        if actual != set(EXPECTED):
            raise ValueError(f'Exactly 100 completed jobs required; missing={len(set(EXPECTED)-actual)}, extra={len(actual-set(EXPECTED))}')
        jobs = {name: evidence.json(Path('synthetic') / name / 'job.json') for name in EXPECTED}
        if any(job.get('status') != 'passed' for job in jobs.values()):
            raise ValueError('All 100 synthetic jobs must have passed status; batch incomplete')
        verified = verified_outputs(run, 'clean_evaluation.json',
            ['clean_thresholds.json', 'evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz'])
        evidence.json('clean_evaluation.json')
        clean_moment = evidence.json('moment_clean/execution.json')
        if verified.get('inputs_sha256', {}).get('moment_clean/execution.json') != io.sha256(
                run / 'moment_clean/execution.json'):
            raise ValueError('Clean MOMENT metadata differs from verified producer record')
        settings = _settings(evidence.json('clean_thresholds.json'))
        protocol = evidence.json('protocol.json')
        frozen = _keys(evidence.frame('common_candidate_keys.csv.gz'), 'frozen common keys')
        if not frozen.split.isin(['calibration', 'test']).all():
            raise ValueError('Frozen keys contain unexpected split labels')
        clean = {}
        for rate_id in settings:
            table = _keys(evidence.frame(f'evaluated_scores_{rate_id}.csv.gz'), 'clean scores')
            selected = table[table.common_candidate.eq(True)]
            aligned = exact_align(frozen, selected, MODELS + ['split'], 'clean frozen common keys')
            if not aligned.split.equals(frozen.split.reset_index(drop=True)):
                raise ValueError('Clean split labels differ from frozen keys')
            _finite(aligned, MODELS, 'clean frozen common keys')
            clean[rate_id] = table
        records = _collect(evidence, settings, clean, frozen[frozen.split.eq('test')], jobs, clean_moment['model'])
        tables = _summaries(records)
        evidence.unchanged()
        if any(io.sha256(path) != base['code_sha256'][path.name] for path in code_paths):
            raise ValueError('Aggregation code changed during execution')
        records.to_csv(output / 'segment_results.csv.gz', index=False)
        for name, table in tables.items():
            table.to_csv(output / name, index=False)
        result = dict(base, status='passed', completed_jobs=100, completed_scenarios=5, seeds=20,
                      protocol_version=protocol.get('version'), models=MODELS, segment_records=len(records),
                      input_sha256=evidence.hashes, seconds=time.perf_counter() - started,
                      outputs_sha256={path.name: io.sha256(path) for path in output.iterdir() if path.is_file()})
        io.write_json(output / 'aggregate_manifest.json', result)
        return result
    except Exception as error:
        io.write_json(output / 'aggregate_failure.json', dict(base, status='unverified',
            reason=f'{type(error).__name__}: {error}', input_sha256=evidence.hashes,
            seconds=time.perf_counter() - started, completed_claim=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    aggregate(arguments.run, arguments.output)
