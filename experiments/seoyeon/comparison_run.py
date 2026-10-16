"""Run the approved comparison into a new, ignored local result directory."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src import data_loader
from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.audit import preparation
from experiments.seoyeon.comparison_safety import assert_new_outputs, source_hashes, verify_sources


def prepare(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'results/seoyeon_tuning'):
        raise ValueError('Use a new local results/seoyeon_tuning directory')
    output.mkdir(parents=True, exist_ok=False)
    source = source_hashes(['comparison_run.py', 'comparison_io.py', 'comparison_safety.py'])
    io.assert_main_source()
    io.write_json(output / 'protocol.json', io.protocol())
    before = io.preserved_inputs()
    io.write_json(output / 'inputs_before.json', before)
    start = time.monotonic()
    daily = data_loader.load_nethealth()
    if daily.duplicated(['pid', 'date']).any():
        raise ValueError('Duplicate preprocessed person-date')
    daily.to_csv(output / 'main_daily.csv.gz', index=False)
    counts, quality = preparation(daily)
    counts.to_csv(output / 'preprocessing_counts.csv', index=False)
    io.write_json(output / 'split.json', io.calendar_split(daily.date))
    after = io.preserved_inputs()
    if before != after:
        raise ValueError('An original input changed during preprocessing')
    io.write_json(output / 'preprocessing_execution.json', dict(
        status='passed', elapsed_seconds=time.monotonic() - start,
        rows=len(daily), people=daily.pid.nunique(), quality=quality,
        original_source_preserved=True, **verify_sources(source),
        daily_sha256=io.sha256(output / 'main_daily.csv.gz'),
        python=sys.version, versions={p: importlib.metadata.version(p) for p in
                                    ['numpy', 'pandas', 'scipy', 'scikit-learn']}))
    print(json.dumps(dict(stage='preprocessing', rows=len(daily), people=daily.pid.nunique(),
                          seconds=time.monotonic() - start)), flush=True)


def score(output):
    from experiments.seoyeon.comparison_models import adapt_main, score_models
    output = Path(output)
    assert_new_outputs(output, ['seoyeon_daily.csv.gz', 'seoyeon_scores.csv.gz',
        'seoyeon_baselines.csv', 'seoyeon_wide.csv.gz', 'moment_keys.csv.gz', 'seoyeon_execution.json'])
    source = source_hashes(['comparison_run.py', 'comparison_models.py',
                            'comparison_io.py', 'comparison_safety.py'])
    start = time.monotonic()
    main = io.read_frame(output / 'main_daily.csv.gz')
    daily = adapt_main(main)
    split = json.loads((output / 'split.json').read_text())
    daily.to_csv(output / 'seoyeon_daily.csv.gz', index=False)
    scores, baselines = score_models(daily, pd.Timestamp(split['development_end']),
                                    pd.Timestamp(split['calibration_start']))
    scores.to_csv(output / 'seoyeon_scores.csv.gz', index=False)
    baselines.to_csv(output / 'seoyeon_baselines.csv', index=False)
    wide = scores.pivot(index=['participant', 'date'], columns='model', values='score').reset_index()
    wide['split'] = io.label_split(wide.date, split)
    wide.to_csv(output / 'seoyeon_wide.csv.gz', index=False)
    keys = wide.loc[wide.split.ne('development') & np.isfinite(wide.fixed28), ['participant', 'date', 'split']]
    keys.to_csv(output / 'moment_keys.csv.gz', index=False)
    io.write_json(output / 'seoyeon_execution.json', dict(status='passed',
        **verify_sources(source),
        elapsed_seconds=time.monotonic() - start, rows=len(scores),
        moment_requested_rows=len(keys), valid_scores=int(scores.valid.sum())))
    print(json.dumps(dict(stage='seoyeon_models', seconds=time.monotonic() - start,
                          moment_rows=len(keys))), flush=True)


def fit(output):
    from experiments.seoyeon.comparison_seungho import directional_scores, fit_h, score_with_h
    output = Path(output)
    assert_new_outputs(output, ['seungho_directional.csv.gz', 'common_candidate_keys.csv.gz',
        'clean_scores_r05.csv.gz', 'clean_scores_r033.csv.gz',
        'clean_thresholds_without_moment.json', 'fit_execution.json'])
    source = source_hashes(['comparison_run.py', 'comparison_seungho.py',
                            'comparison_io.py', 'comparison_safety.py'])
    start = time.monotonic()
    main = io.read_frame(output / 'main_daily.csv.gz')
    own = io.read_frame(output / 'seoyeon_wide.csv.gz')
    names = ['fixed14', 'fixed28', 'rolling28', 'cusum28', 'group']
    ordered = main[['pid', 'date']].rename(columns={'pid': 'participant'}).merge(
        own, on=['participant', 'date'], how='left', validate='one_to_one')
    bad = directional_scores(main)
    bad.to_csv(output / 'seungho_directional.csv.gz', index=False)
    common = np.isfinite(ordered[names]).all(axis=1) & np.isfinite(bad).all(axis=1)
    cal = common & ordered.split.eq('calibration')
    test = common & ordered.split.eq('test')
    ordered.loc[cal | test, ['participant', 'date', 'split']].to_csv(
        output / 'common_candidate_keys.csv.gz', index=False)
    thresholds = []
    for label, rate in zip(['r05', 'r033'], io.RATES):
        h = fit_h(main, bad, cal, rate)
        sh = score_with_h(main, bad, h)
        scores = pd.concat([ordered, sh], axis=1)
        scores['common_candidate'] = common
        scores.to_csv(output / f'clean_scores_{label}.csv.gz', index=False)
        thresholds.append(dict(id=label, rate=rate, thresholds={
            **{name: io.quantile_threshold(scores[name], cal, rate) for name in names}, **h},
            common_calibration_days=int(cal.sum()), common_test_days=int(test.sum()),
            moment_status='awaiting_actual_inference_on_fixed_keys'))
    io.write_json(output / 'clean_thresholds_without_moment.json', thresholds)
    io.write_json(output / 'fit_execution.json', dict(status='passed',
        **verify_sources(source),
        elapsed_seconds=time.monotonic() - start, calibration_days=int(cal.sum()),
        test_days=int(test.sum()), mask_uses_no_test_outcomes=True))
    print(json.dumps(dict(stage='fit', seconds=time.monotonic() - start,
                          calibration_days=int(cal.sum()), test_days=int(test.sum()))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'score', 'fit'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'score': score, 'fit': fit}[args.stage](args.output)
