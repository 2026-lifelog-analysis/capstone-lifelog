import pandas as pd
import pytest
import json

from experiments.seoyeon.comparison_report import alarm_table, synthetic_table, generate
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon import comparison_io as io


def test_alarm_table_uses_observed_counts_and_percent():
    frame = pd.DataFrame([dict(rate_id='r05', scope='common', model='fixed14', split=s,
                               days=100, people=2, alarm_days=a, alarm_rate=a / 100,
                               zero_people=0, top10_alarm_share=.5)
                          for s, a in [('calibration', 5), ('test', 9)]])
    text = alarm_table(frame, 'r05', models=['fixed14'])
    assert '9.0000%' in text and '|9|' in text and '5.0000%' in text


def test_synthetic_table_reports_actual_seed_means_not_claimed_accuracy():
    frame = pd.DataFrame([dict(rate_id='r05', model='fixed14', scenario=s,
                               detection_rate_mean=.4, delta_mean_mean=.1)
                          for s in ['S1', 'S2', 'S3', 'S4', 'S5']])
    text = synthetic_table(frame, 'r05', models=['fixed14'])
    assert text.count('40.00% / +10.00%p') == 5
    with pytest.raises(KeyError):
        synthetic_table(frame.iloc[:-1], 'r05', models=['fixed14'])


def test_report_requires_verified_files_and_preserves_existing_output(tmp_path, monkeypatch):
    run, aggregate = tmp_path / 'run', tmp_path / 'run/aggregate'
    aggregate.mkdir(parents=True)
    for name in ['original_seungho', 'replay_verification', 'figures', 'moment_clean']:
        (run / name).mkdir()
    alarms = pd.DataFrame([dict(rate_id=r, model=m, split=s, scope='common', days=100,
                               people=20, alarm_days=5, alarm_rate=.05, zero_people=1,
                               top10_alarm_share=.25)
        for r in ['r05','r033'] for m in MODELS for s in ['calibration','test']])
    alarms.to_csv(run / 'alarm_summary.csv', index=False)
    pd.DataFrame([dict(rate_id='r05', model=m, split='test', finite_days=150, common_days=100)
                  for m in MODELS]).to_csv(run / 'coverage.csv', index=False)
    pd.DataFrame([dict(rate_id=r, model=m, scenario=s, detection_rate_mean=.5, delta_mean_mean=.2)
        for r in ['r05','r033'] for m in MODELS for s in ['S1','S2','S3','S4','S5']]).to_csv(
            aggregate / 'seed_distribution_summary.csv', index=False)
    def manifest(directory, filename, files):
        io.write_json(directory / filename, dict(status='passed',
            outputs_sha256={file: io.sha256(directory / file) for file in files}))
    manifest(run, 'clean_evaluation.json', ['alarm_summary.csv','coverage.csv'])
    manifest(aggregate, 'aggregate_manifest.json', ['seed_distribution_summary.csv'])
    io.write_json(run / 'original_seungho/manifest.json', dict(status='passed',
        dev_people=14, holdout_people=6, holdout_rows=100, n_reps=20))
    manifest(run / 'replay_verification', 'verification.json', [])
    io.write_json(run / 'model_port_validation.json', dict(status='passed',
        ast_function_count=13, sample_people=15, sample_person_days=1000, score_rows=5000))
    pd.DataFrame([dict(stage=stage, rows=100, people=20) for stage in
        ['raw_activity','raw_sleep','activity_dedup','sleep_days','outer_union','after_min42']]).to_csv(
            run / 'preprocessing_counts.csv', index=False)
    figures = ['daily_alarm_rates.png','synthetic_detection.png','synthetic_paired_delta.png']
    for name in figures:
        (run / 'figures' / name).write_bytes(b'fixture-not-research-figure')
    manifest(run / 'figures', 'plot_manifest.json', figures)
    io.write_json(run / 'moment_clean/execution.json', dict(requested_persondays=100,
        completed_persondays=100,new_inferred_persondays=100,exact_six_field_cache_persondays=0))
    io.write_json(run / 'split.json', io.calendar_split(pd.date_range('2020-01-01', periods=100)))
    io.write_json(run / 'protocol.json', {})
    io.write_json(run / 'inputs_before.json', {})
    monkeypatch.setattr(io, 'preserved_inputs', lambda: {})
    output = tmp_path / 'report.md'
    original = (aggregate / 'seed_distribution_summary.csv').read_bytes()
    with (aggregate / 'seed_distribution_summary.csv').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='checksum'):
        generate(run, aggregate, output)
    assert not output.exists()
    (aggregate / 'seed_distribution_summary.csv').write_bytes(original)
    generate(run, aggregate, output)
    assert '정답 상태 라벨' in output.read_text()
    record = json.loads((run / 'report_manifest.json').read_text())
    assert record['output_sha256'] == io.sha256(output)
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        generate(run, aggregate, output)
    assert output.read_bytes() == previous
