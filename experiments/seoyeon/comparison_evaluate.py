"""Matched denominators and explicit unavailable outcomes for the comparison."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_safety import assert_new_outputs, source_hashes, verify_sources

MODELS = ['fixed14', 'fixed28', 'rolling28', 'cusum28', 'group', 'moment',
          'sh_base', 'sh_dow', 'sh_common', 'sh_final']


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def alarm_metrics(frame, model, threshold):
    data = frame.loc[np.isfinite(frame[model]), ['participant', model]]
    alarm = data[model] > threshold
    counts = alarm.groupby(data.participant).sum()
    rates = alarm.groupby(data.participant).mean()
    top = max(1, int(len(counts) * .1))
    return dict(days=len(data), people=len(counts), unavailable_days=len(frame) - len(data),
                alarm_days=int(alarm.sum()), alarm_rate=_ratio(alarm.sum(), len(data)),
                zero_people=int(counts.eq(0).sum()), top_people=top if len(counts) else 0,
                top10_alarm_share=_ratio(counts.nlargest(top).sum(), alarm.sum()),
                person_alarm_rate_mean=float(rates.mean()),
                person_alarm_rate_median=float(rates.median()),
                person_alarm_rate_q25=float(rates.quantile(.25)),
                person_alarm_rate_q75=float(rates.quantile(.75)))


def pair_metrics(frame, a, b, ta, tb):
    valid = np.isfinite(frame[[a, b]]).all(axis=1)
    aa, bb = frame.loc[valid, a] > ta, frame.loc[valid, b] > tb
    overlap, union = int((aa & bb).sum()), int((aa | bb).sum())
    return dict(common_days=int(valid.sum()), a_alarm_days=int(aa.sum()),
                b_alarm_days=int(bb.sum()), overlap_days=overlap,
                overlap_over_a=_ratio(overlap, aa.sum()), jaccard=_ratio(overlap, union),
                disagreement_days=int((aa != bb).sum()))


def segment_metrics(frame, changed, clean, threshold):
    records = []
    for pid, part in frame.groupby('participant', sort=True):
        valid = np.isfinite(part[[changed, clean]]).all(axis=1)
        missing = int((~valid).sum())
        if missing or not len(part):
            detected = control = delay = np.nan
        else:
            hit = part[changed] > threshold
            detected = float(hit.any())
            control = float((part[clean] > threshold).any())
            delay = float(part.loc[hit, 'day_offset'].min()) if hit.any() else np.nan
        records.append(dict(participant=pid, evaluation_days=len(part),
                            unavailable_days=missing, detected=detected,
                            clean_alarm=control, delay_days=delay))
    return pd.DataFrame(records)


def verify_clean_moment(run, moment, record):
    """Bind the completed predictions to this run's inputs, keys, and checkpoint."""
    from experiments.seoyeon.comparison_moment import MODEL_ID, REVISION, WEIGHTS_SHA256
    expected = {name: io.sha256(run / file) for name, file in
        [('daily', 'seoyeon_daily.csv.gz'), ('baselines', 'seoyeon_baselines.csv'),
         ('keys', 'moment_keys.csv.gz')]}
    if record.get('input_source_sha256') != expected:
        raise ValueError('MOMENT input checksum differs from this run')
    if record.get('scores_sha256') != io.sha256(run / 'moment_clean/scores.csv.gz'):
        raise ValueError('MOMENT output checksum mismatch')
    keys = io.read_frame(run / 'moment_keys.csv.gz')
    key_columns = ['participant', 'date']
    if keys.duplicated(key_columns).any() or len(moment) != len(keys):
        raise ValueError('MOMENT requested key count or uniqueness mismatch')
    expected_keys = keys[key_columns].sort_values(key_columns).reset_index(drop=True)
    actual_keys = moment[key_columns].sort_values(key_columns).reset_index(drop=True)
    if not actual_keys.equals(expected_keys):
        raise ValueError('MOMENT output keys differ from requested keys')
    counts = ['requested_persondays', 'eligible_persondays', 'completed_persondays', 'valid_persondays']
    if any(record.get(name) != len(keys) for name in counts) or record.get('unavailable_persondays') != 0:
        raise ValueError('MOMENT completion count mismatch')
    if not moment.valid.eq(True).all() or not np.isfinite(moment.score).all():
        raise ValueError('MOMENT requested outputs are not all finite and valid')
    model = record.get('model', {})
    if (model.get('model_id') != MODEL_ID or model.get('revision') != REVISION
            or model.get('checkpoint_sha256', {}).get('model.safetensors') != WEIGHTS_SHA256
            or model.get('precision') != 'float32' or model.get('trainable_parameters') != 0
            or model.get('training_mode') is not False or model.get('controls', {}).get('pass') is not True):
        raise ValueError('MOMENT model identity, frozen state, or controls mismatch')


def finalize_clean(run):
    run = Path(run)
    assert_new_outputs(run, ['evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
        'clean_thresholds.json', 'alarm_summary.csv', 'coverage.csv', 'pairwise.csv', 'clean_evaluation.json'])
    source = source_hashes(['comparison_evaluate.py', 'comparison_io.py', 'comparison_safety.py'])
    inputs = ['moment_clean/execution.json', 'moment_clean/scores.csv.gz',
        'seoyeon_daily.csv.gz', 'seoyeon_baselines.csv', 'moment_keys.csv.gz',
        'common_candidate_keys.csv.gz', 'clean_thresholds_without_moment.json',
        'clean_scores_r05.csv.gz', 'clean_scores_r033.csv.gz']
    before = {name: io.sha256(run / name) for name in inputs}
    moment = io.read_frame(run / 'moment_clean/scores.csv.gz')
    record = json.loads((run / 'moment_clean/execution.json').read_text())
    if record['status'] != 'complete':
        raise ValueError('MOMENT clean inference is incomplete')
    if moment.duplicated(['participant', 'date']).any():
        raise ValueError('Duplicate MOMENT keys')
    verify_clean_moment(run, moment, record)
    moment['moment'] = moment.score.where(moment.valid.eq(True))
    frozen = io.read_frame(run / 'common_candidate_keys.csv.gz')
    checked = frozen.merge(moment[['participant', 'date', 'moment']],
                           on=['participant', 'date'], how='left', validate='one_to_one')
    if not np.isfinite(checked.moment).all():
        raise ValueError('MOMENT missing on frozen common keys; no subset recalibration allowed')
    settings = json.loads((run / 'clean_thresholds_without_moment.json').read_text())
    metrics, pairs, coverage = [], [], []
    for setting in settings:
        data = io.read_frame(run / f"clean_scores_{setting['id']}.csv.gz").merge(
            moment[['participant', 'date', 'moment']], on=['participant', 'date'],
            how='left', validate='one_to_one')
        common = data.common_candidate.eq(True)
        cal = data.split.eq('calibration') & common
        test = data.split.eq('test') & common
        thresholds = dict(setting['thresholds'])
        thresholds['moment'] = io.quantile_threshold(data.moment, cal, setting['rate'])
        setting.update(thresholds=thresholds, moment_status='actual_new_inference_complete')
        if not np.isfinite(data.loc[cal | test, MODELS]).all().all():
            raise ValueError('Frozen common sample is not complete across ten methods')
        data.to_csv(run / f"evaluated_scores_{setting['id']}.csv.gz", index=False)
        for model in MODELS:
            for split in ['calibration', 'test']:
                period = data.split.eq(split)
                for scope, mask in [('common', period & common), ('available', period)]:
                    metrics.append(dict(rate_id=setting['id'], rate=setting['rate'],
                        model=model, split=split, scope=scope, threshold=thresholds[model],
                        **alarm_metrics(data.loc[mask], model, thresholds[model])))
                coverage.append(dict(rate_id=setting['id'], model=model, split=split,
                    total_period_days=int(period.sum()),
                    finite_days=int((period & np.isfinite(data[model])).sum()),
                    common_days=int((period & common).sum())))
        for a, b in combinations(MODELS, 2):
            pairs.append(dict(rate_id=setting['id'], a=a, b=b,
                         **pair_metrics(data.loc[test], a, b, thresholds[a], thresholds[b])))
    io.write_json(run / 'clean_thresholds.json', settings)
    pd.DataFrame(metrics).to_csv(run / 'alarm_summary.csv', index=False)
    pd.DataFrame(coverage).to_csv(run / 'coverage.csv', index=False)
    pd.DataFrame(pairs).to_csv(run / 'pairwise.csv', index=False)
    if before != {name: io.sha256(run / name) for name in inputs}:
        raise ValueError('Clean evaluation inputs changed during execution')
    io.write_json(run / 'clean_evaluation.json', dict(status='passed',
        **verify_sources(source),
        inputs_sha256=before,
        outputs_sha256={name: io.sha256(run / name) for name in
            ['evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
             'clean_thresholds.json', 'alarm_summary.csv', 'coverage.csv', 'pairwise.csv']},
        frozen_common_rows=len(frozen), calibration_rows=int(frozen.split.eq('calibration').sum()),
        test_rows=int(frozen.split.eq('test').sum()), threshold_test_access=False,
        all_requested_common_moment_finite=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    finalize_clean(args.output)
