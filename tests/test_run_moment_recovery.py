"""Isolated recovery orchestration; these tests never load a neural model."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.seoyeon import run_moment_recovery as recovery


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def original(tmp_path, monkeypatch):
    run = tmp_path / 'original'
    run.mkdir()
    frame = pd.DataFrame({'participant': ['fixture-a', 'fixture-b'],
                          'date': ['2020-01-01', '2020-01-02']})
    for name in ['seoyeon_daily.csv.gz', 'moment_keys.csv.gz', 'seoyeon_baselines.csv']:
        frame.to_csv(run / name, index=False)
    write_json(run / 'inputs_before.json', {'fixture': 'same'})
    write_json(run / 'moment_clean/execution.json', {'status': 'running'})
    (run / 'moment_clean/score_chunk_0000000.csv.gz').write_bytes(b'original partial')
    for name in recovery.JOBS:
        job = run / 'synthetic' / name
        job.mkdir(parents=True)
        for filename in ['canonical_daily.csv.gz', 'moment_keys.csv.gz']:
            frame.to_csv(job / filename, index=False)
        write_json(job / 'job.json', {'status': 'passed', 'artifact_hashes': {
            name: recovery.io.sha256(job / name)
            for name in ['canonical_daily.csv.gz', 'moment_keys.csv.gz']},
            'input_hashes': {'seoyeon_baselines.csv':
                            recovery.io.sha256(run / 'seoyeon_baselines.csv')}})
    monkeypatch.setattr(recovery.io, 'preserved_inputs', lambda: {'fixture': 'same'})
    monkeypatch.setattr(recovery, 'CLEAN_REQUESTS', 2)
    monkeypatch.setattr(recovery, 'SYNTHETIC_REQUESTS', 200)
    return run


@pytest.fixture
def options(original, tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    return dict(run=original, recovery_dir=original / 'recovery_fixture',
                model_root=model, finalizer_python=Path('/bin/echo'),
                report=tmp_path / 'comparison.md')


@pytest.fixture
def fake_execution(monkeypatch):
    calls = []
    engine = SimpleNamespace(session_id='fixture-session', forward_requested_persondays=0,
                             forward_completed_persondays=0)
    monkeypatch.setattr(recovery, 'MomentEngine', lambda *a, **k: engine)

    def restore(run, supplied):
        assert supplied is engine
        calls.append('restore')
        return {'verified': True, 'fresh_replay_persondays': 8}

    def infer(**kwargs):
        assert kwargs['engine'] is engine
        calls.append(kwargs['outputdir'].parent.name)
        output = kwargs['outputdir']
        output.mkdir()
        count = len(kwargs['keys'])
        record = dict(status='complete', requested_persondays=count,
                      completed_persondays=count, valid_persondays=count,
                      unavailable_persondays=0, new_inferred_persondays=count - 1,
                      exact_six_field_cache_persondays=1)
        write_json(output / 'execution.json', record)
        return record

    def finalizer(command, **kwargs):
        calls.append('finalizer')
        assert command[1:3] == ['-m', 'experiments.seoyeon.comparison_finish']
        view = Path(command[command.index('--run') + 1])
        if not view.is_absolute():
            view = Path(kwargs['cwd']) / view
        report = Path(command[command.index('--report') + 1])
        report.write_text('Fixture summary without participant data.')
        artifacts = {
            'clean_evaluation.json': ['evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
                'clean_thresholds.json', 'alarm_summary.csv', 'coverage.csv', 'pairwise.csv'],
            'aggregate/aggregate_manifest.json': ['segment_results.csv.gz', 'seed_summary.csv',
                'participant_summary.csv', 'seed_distribution_summary.csv',
                'participant_distribution_summary.csv'],
            'figures/plot_manifest.json': [f'{name}.{extension}' for name in [
                'daily_alarm_rates', 'synthetic_detection', 'synthetic_paired_delta']
                for extension in ['png', 'pdf']],
        }
        for name, filenames in artifacts.items():
            parent = (view / name).parent
            parent.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                (parent / filename).write_text('fixture aggregate')
            write_json(view / name, {'status': 'passed', 'outputs_sha256': {
                filename: recovery.io.sha256(parent / filename) for filename in filenames}})
        write_json(view / 'report_manifest.json', {'status': 'passed',
                    'output_sha256': recovery.io.sha256(report)})
        write_json(view / 'completion_status.json', {'status': 'complete',
                    'completed_claim': True, 'report_sha256': recovery.io.sha256(report)})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recovery, 'restore_partial_cache', restore)
    monkeypatch.setattr(recovery, 'infer', infer)
    monkeypatch.setattr(recovery.subprocess, 'run', finalizer)
    return calls


def test_view_has_new_outputs_and_only_file_links_inside_synthetic(original):
    for name in recovery.EXCLUDED:
        if name not in ['synthetic', 'moment_clean']:
            path = original / name
            if not path.exists():
                path.write_text('old output')
    destination = original / 'recovery_fixture'
    destination.mkdir()
    (destination / 'external_review.log').write_text('permitted existing log')
    view = recovery.create_run_view(original, destination)
    assert (view / 'seoyeon_daily.csv.gz').is_symlink()
    assert not (view / 'synthetic').is_symlink()
    assert not (view / 'synthetic/S1_seed00').is_symlink()
    assert (view / 'synthetic/S1_seed00/job.json').is_symlink()
    assert not any((view / name).exists() for name in recovery.EXCLUDED - {'synthetic'})
    assert not (view / destination.name).exists()
    with pytest.raises(FileExistsError):
        recovery.create_run_view(original, destination)


@pytest.mark.parametrize('problem', ['moment', 'unexpected', 'missing', 'directory'])
def test_view_rejects_preexisting_neural_work_and_unexpected_layout(original, problem):
    job = original / 'synthetic/S1_seed00'
    if problem == 'moment':
        (job / 'moment').mkdir()
    elif problem == 'unexpected':
        (original / 'synthetic/S6_seed00').mkdir()
    elif problem == 'missing':
        (job / 'job.json').unlink()
    else:
        (job / 'nested').mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        recovery.create_run_view(original, original / 'recovery_fixture')
    assert not (original / 'recovery_fixture/run_view').exists()


def test_recovery_reuses_one_engine_and_preserves_all_original_files(options, fake_execution):
    before = recovery.original_snapshot(options['run'])
    state = recovery.run_recovery(**options)
    assert state['status'] == 'complete' and state['completed_claim']
    assert state['completed_jobs'] == list(recovery.JOBS)
    assert state['new_inferred_persondays'] == 101
    assert state['exact_six_field_cache_persondays'] == 101
    assert fake_execution[0] == 'restore' and fake_execution[-1] == 'finalizer'
    assert len(fake_execution) == 103
    assert recovery.original_snapshot(options['run']) == before
    recovery_dir = options['recovery_dir']
    assert (recovery_dir / 'original_before.json').exists()
    assert (recovery_dir / 'original_after.json').exists()
    assert (recovery_dir / 'cache_provenance.json').exists()
    assert (recovery_dir / 'run_view/synthetic/S5_seed19/moment/source_preservation.json').exists()


@pytest.mark.parametrize('problem', ['report', 'missing_parent', 'existing_state', 'bad_dir',
                                     'changed_initial', 'missing_python', 'missing_model'])
def test_preflight_fails_before_inference(options, fake_execution, problem, monkeypatch):
    if problem == 'report':
        options['report'].write_text('retain')
    elif problem == 'missing_parent':
        options['report'] = options['report'].parent / 'absent/report.md'
    elif problem == 'existing_state':
        write_json(options['recovery_dir'] / 'recovery_status.json', {'status': 'running'})
    elif problem == 'bad_dir':
        options['recovery_dir'] = options['run']
    elif problem == 'changed_initial':
        monkeypatch.setattr(recovery.io, 'preserved_inputs', lambda: {'fixture': 'different'})
    elif problem == 'missing_python':
        options['finalizer_python'] = options['report'].parent / 'absent_python'
    else:
        options['model_root'] = options['report'].parent / 'absent_model'
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        recovery.run_recovery(**options)
    assert not fake_execution


@pytest.mark.parametrize('problem', ['restore', 'incomplete', 'mutation', 'finalizer', 'source'])
def test_failures_never_claim_complete(options, fake_execution, monkeypatch, problem):
    if problem == 'restore':
        def fail(*args):
            raise ValueError('fixture restore failure')
        monkeypatch.setattr(recovery, 'restore_partial_cache', fail)
    elif problem == 'finalizer':
        monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **k: None)
    elif problem == 'source':
        def changed(*args):
            raise ValueError('source changed')
        monkeypatch.setattr(recovery, 'verify_sources', changed)
    else:
        original_infer = recovery.infer
        def bad_infer(**kwargs):
            record = original_infer(**kwargs)
            if problem == 'incomplete':
                return {**record, 'completed_persondays': 1}
            (options['run'] / 'moment_clean/execution.json').write_text('modified fixture')
            return record
        monkeypatch.setattr(recovery, 'infer', bad_infer)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        recovery.run_recovery(**options)
    state = json.loads((options['recovery_dir'] / 'recovery_status.json').read_text())
    assert state['status'] == 'failed' and not state['completed_claim']
    assert (options['recovery_dir'] / 'original_after.json').exists()


def test_snapshot_excludes_recovery_and_checks_deleted_files(original):
    before = recovery.original_snapshot(original)
    write_json(original / 'recovery_other/temporary.json', {'ignored': True})
    assert recovery.original_snapshot(original) == before
    target = original / 'moment_clean/execution.json'
    target.unlink()
    with pytest.raises(ValueError, match='preservation'):
        recovery.check_original_preservation(before, recovery.original_snapshot(original))


def test_cli_parses_required_paths_and_forwards(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(recovery, 'run_recovery', lambda **kwargs: calls.append(kwargs))
    args = []
    for name in ['run', 'recovery-dir', 'model-root', 'finalizer-python', 'report']:
        args.extend(['--' + name, str(tmp_path / name)])
    recovery.main(args)
    assert calls[0]['device'] == 'mps' and calls[0]['batch_size'] == 96


def test_finalizer_keeps_virtual_environment_python_symlink(options, fake_execution, monkeypatch):
    launcher = options['report'].parent / 'venv_python'
    launcher.symlink_to('/bin/echo')
    options['finalizer_python'] = launcher
    invoke = recovery.subprocess.run
    commands = []
    def observed(command, **kwargs):
        commands.append(command)
        return invoke(command, **kwargs)
    monkeypatch.setattr(recovery.subprocess, 'run', observed)
    recovery.run_recovery(**options)
    assert commands[0][0] == str(launcher)


def test_failed_state_survives_preservation_snapshot_error(options, fake_execution, monkeypatch):
    snapshot = recovery.original_snapshot
    seen = []
    def interrupted_snapshot(run):
        seen.append(True)
        if len(seen) > 1:
            raise OSError('fixture snapshot read error')
        return snapshot(run)
    def failed_restore(*args):
        raise ValueError('fixture cache mismatch')
    monkeypatch.setattr(recovery, 'original_snapshot', interrupted_snapshot)
    monkeypatch.setattr(recovery, 'restore_partial_cache', failed_restore)
    with pytest.raises(ValueError, match='cache mismatch'):
        recovery.run_recovery(**options)
    state = json.loads((options['recovery_dir'] / 'recovery_status.json').read_text())
    assert state['status'] == 'failed' and not state['completed_claim']
    assert state['preservation_snapshot_error_type'] == 'OSError'


def test_finalizer_uses_repository_relative_run_path(options, fake_execution, monkeypatch):
    monkeypatch.setattr(recovery.io, 'ROOT', options['run'].parent)
    invoke = recovery.subprocess.run
    commands = []
    def observed(command, **kwargs):
        commands.append(command)
        return invoke(command, **kwargs)
    monkeypatch.setattr(recovery.subprocess, 'run', observed)
    recovery.run_recovery(**options)
    value = commands[0][commands[0].index('--run') + 1]
    assert value == 'original/recovery_fixture/run_view'


@pytest.mark.parametrize('problem', ['report_hash', 'missing_outputs', 'output_hash'])
def test_finalizer_rejects_artifact_manifest_mismatch(options, fake_execution, monkeypatch, problem):
    invoke = recovery.subprocess.run
    def inconsistent(command, **kwargs):
        result = invoke(command, **kwargs)
        view = Path(command[command.index('--run') + 1])
        if problem == 'report_hash':
            write_json(view / 'report_manifest.json', {'status': 'passed', 'output_sha256': 'bad'})
        elif problem == 'missing_outputs':
            write_json(view / 'clean_evaluation.json', {'status': 'passed', 'outputs_sha256': {}})
        else:
            path = view / 'figures/plot_manifest.json'
            manifest = json.loads(path.read_text())
            manifest['outputs_sha256']['daily_alarm_rates.png'] = 'bad'
            write_json(path, manifest)
        return result
    monkeypatch.setattr(recovery.subprocess, 'run', inconsistent)
    with pytest.raises(ValueError):
        recovery.run_recovery(**options)
    state = json.loads((options['recovery_dir'] / 'recovery_status.json').read_text())
    assert not state['completed_claim']
