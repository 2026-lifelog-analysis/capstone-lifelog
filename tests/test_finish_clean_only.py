"""Clean-only release boundaries use fixtures, not claims of real inference."""
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.seoyeon import finish_clean_only as finish


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    rec = tmp_path / 'run/recovery_fixture'
    view = rec / 'run_view'
    for job in finish.JOBS:
        (view / 'synthetic' / job).mkdir(parents=True)
    put(view / 'synthetic/S1_seed00/moment', {
        'kind': 'user_scope_boundary_guard_not_inference_output',
        'synthetic_moment': 'not_executed_by_user_scope_change',
        'do_not_remove_to_resume_synthetic': True})
    state = dict(status='failed', phase='synthetic', error_type='FileExistsError',
                 completed_claim=False, current_job='S1_seed00', completed_jobs=[],
                 expected_jobs=100, session_id='current')
    put(rec / 'recovery_status.json', state)
    before = {f'file_{i}': str(i) for i in range(871)}
    put(rec / 'original_before.json', before)
    put(rec / 'original_after.json', before)
    put(rec.parent / 'inputs_before.json', {'fixture': 'digest'})
    monkeypatch.setattr(finish, 'original_snapshot', lambda _: before.copy())
    monkeypatch.setattr(finish.io, 'preserved_inputs', lambda: {'fixture': 'digest'})
    return rec


def test_boundary_accepts_authorized_guard(recovery):
    assert finish.validate_boundary(recovery)['status'] == 'failed'


@pytest.mark.parametrize('key,value', [
    ('status', 'running'), ('error_type', 'ValueError'),
    ('current_job', 'S2_seed00'), ('completed_jobs', ['S1_seed00']),
    ('completed_claim', True), ('phase', 'clean')])
def test_boundary_rejects_unexpected_stop(recovery, key, value):
    path = recovery / 'recovery_status.json'
    state = json.loads(path.read_text())
    put(path, {**state, key: value})
    with pytest.raises(ValueError, match='boundary'):
        finish.validate_boundary(recovery)


def test_boundary_rejects_synthetic_output(recovery):
    (recovery / 'run_view/synthetic/S2_seed00/moment').mkdir()
    with pytest.raises(ValueError, match='Synthetic'):
        finish.validate_boundary(recovery)


def test_boundary_rejects_guard_change(recovery):
    put(recovery / 'run_view/synthetic/S1_seed00/moment', {'kind': 'not_guard'})
    with pytest.raises(ValueError, match='guard'):
        finish.validate_boundary(recovery)


def test_preservation_checks_current_files(recovery, monkeypatch):
    assert finish.validate_preservation(recovery) == 871
    monkeypatch.setattr(finish, 'original_snapshot', lambda _: {})
    with pytest.raises(ValueError, match='preservation'):
        finish.validate_preservation(recovery)


def test_preservation_rejects_input_change(recovery, monkeypatch):
    monkeypatch.setattr(finish.io, 'preserved_inputs', lambda: {})
    with pytest.raises(ValueError, match='inputs'):
        finish.validate_preservation(recovery)


@pytest.fixture
def evaluated(recovery):
    view = recovery / 'run_view'
    rows = [dict(rate_id=rate, model=model, split=split, scope=scope,
                 people=people, days=days, unavailable_days=0)
            for rate in ('r05', 'r033') for model in finish.MODELS
            for split, people, days in [('calibration', 181, 36260), ('test', 135, 23721)]
            for scope in ('common', 'available')]
    pd.DataFrame(rows).to_csv(view / 'alarm_summary.csv', index=False)
    outputs = {name: finish.io.sha256(view / name) for name in ['alarm_summary.csv']}
    put(view / 'clean_evaluation.json', dict(status='passed', calibration_rows=36260,
        test_rows=23721, frozen_common_rows=59981, threshold_test_access=False,
        all_requested_common_moment_finite=True, outputs_sha256=outputs))
    return view


def test_evaluation_fixed_denominators(evaluated):
    assert finish.validate_evaluation(evaluated)['test_rows'] == 23721


@pytest.mark.parametrize('column,value', [('people', 134), ('days', 23720),
                                        ('unavailable_days', 1)])
def test_evaluation_rejects_changed_denominator(evaluated, column, value):
    frame = pd.read_csv(evaluated / 'alarm_summary.csv')
    frame.loc[0, column] = value
    frame.to_csv(evaluated / 'alarm_summary.csv', index=False)
    manifest = json.loads((evaluated / 'clean_evaluation.json').read_text())
    manifest['outputs_sha256']['alarm_summary.csv'] = finish.io.sha256(evaluated / 'alarm_summary.csv')
    put(evaluated / 'clean_evaluation.json', manifest)
    with pytest.raises(ValueError, match='denominator'):
        finish.validate_evaluation(evaluated)


def test_evaluation_detects_output_tamper(evaluated):
    (evaluated / 'alarm_summary.csv').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        finish.validate_evaluation(evaluated)


def test_hash_manifest_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match='relative'):
        finish.check_hashes(tmp_path, {'../escape': 'digest'})


def test_refuses_existing_report_before_mutation(recovery):
    report = recovery.parent.parent / 'report.md'
    report.write_text('keep')
    with pytest.raises(FileExistsError):
        finish.finish_clean_only(recovery, report)
    assert report.read_text() == 'keep'
    assert not (recovery / 'clean_only_source_snapshot').exists()


def test_main_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(finish, 'finish_clean_only', lambda *args: calls.append(args))
    finish.main(['--recovery-dir', str(tmp_path), '--report', str(tmp_path / 'report.md')])
    assert calls == [(tmp_path, tmp_path / 'report.md')]


@pytest.fixture
def completed_clean(recovery, monkeypatch):
    view = recovery / 'run_view'
    proof = dict(verified_persondays=78336, remaining_persondays=9550,
        requested_persondays=87886, unique_six_field_inputs=78336,
        all_six_inputs_rebuilt=True, all_audit_fields_rebuilt=True,
        all_three_endpoint_errors_and_mse_recomputed=True,
        reconstruction_csv_float32_serialization_verified=True,
        fresh_replay_persondays=8, fresh_replay_max_abs=0,
        model_runtime_identical=True, prior_historical_experiment_used=False,
        destination_session_id='current', recovery_model={'controls': {
            'pass': True, 'repeat_max_abs': 0, 'masked_placeholder_max_abs': 0}})
    put(recovery / 'cache_provenance.json', proof)
    record = dict(status='complete', requested_persondays=87886, eligible_persondays=87886,
        completed_persondays=87886, valid_persondays=87886, unavailable_persondays=0,
        new_inferred_persondays=9550, exact_six_field_cache_persondays=78336,
        cross_process_current_experiment_cache_persondays=78336,
        same_process_cache_persondays=0, past_experiment_cache_persondays=0,
        session_id='current', same_experiment_seed_cache=[proof], model=proof['recovery_model'],
        input_source_sha256={'fixture': 'digest'}, source_sha256={'fixture.py': 'digest'})
    put(view / 'moment_clean/execution.json', record)
    put(view / 'moment_clean/source_preservation.json',
        dict(unchanged=True, sha256={'fixture': 'digest'}))
    frame = pd.DataFrame({'score_source': ['new_three_channel_inference'] * 9550
        + ['same_experiment_cross_process_exact_six_field_cache'] * 78336})
    frame.to_csv(view / 'moment_clean/scores.csv.gz', index=False)
    state = json.loads((recovery / 'recovery_status.json').read_text())
    state = {**state, 'clean': record, 'source_sha256': {'fixture.py': 'digest'}}
    put(recovery / 'recovery_status.json', state)
    monkeypatch.setattr(finish, 'verify_clean_moment', lambda *args: None)
    monkeypatch.setattr(finish, 'verify_sources', lambda _: None)
    original_check = finish.check_hashes
    monkeypatch.setattr(finish, 'check_hashes', lambda directory, hashes:
                        None if directory.name == 'source_snapshot' else original_check(directory, hashes))
    return recovery, state


def test_clean_provenance_passes(completed_clean):
    recovery, state = completed_clean
    assert finish.validate_clean(recovery, state)['completed_persondays'] == 87886


@pytest.mark.parametrize('key,value', [('completed_persondays', 87000),
    ('new_inferred_persondays', 87886), ('exact_six_field_cache_persondays', 0)])
def test_clean_rejects_incorrect_counts(completed_clean, key, value):
    recovery, state = completed_clean
    path = recovery / 'run_view/moment_clean/execution.json'
    record = json.loads(path.read_text())
    put(path, {**record, key: value})
    with pytest.raises(ValueError, match='counts'):
        finish.validate_clean(recovery, state)


def test_clean_rejects_state_mismatch(completed_clean):
    recovery, state = completed_clean
    with pytest.raises(ValueError, match='Runner'):
        finish.validate_clean(recovery, {**state, 'session_id': 'other'})


def test_clean_rejects_score_source_mismatch(completed_clean):
    recovery, state = completed_clean
    pd.DataFrame({'score_source': ['new_three_channel_inference']}).to_csv(
        recovery / 'run_view/moment_clean/scores.csv.gz', index=False)
    with pytest.raises(ValueError, match='score provenance'):
        finish.validate_clean(recovery, state)


def test_clean_rejects_missing_preservation(completed_clean):
    recovery, state = completed_clean
    put(recovery / 'run_view/moment_clean/source_preservation.json', {'unchanged': False})
    with pytest.raises(ValueError, match='source preservation'):
        finish.validate_clean(recovery, state)


@pytest.mark.parametrize('key,value', [('fresh_replay_max_abs', .1),
    ('fresh_replay_persondays', 0), ('all_six_inputs_rebuilt', False)])
def test_clean_rejects_unverified_cache(completed_clean, key, value):
    recovery, state = completed_clean
    path = recovery / 'cache_provenance.json'
    proof = json.loads(path.read_text())
    put(path, {**proof, key: value})
    with pytest.raises(ValueError, match='partial-cache'):
        finish.validate_clean(recovery, state)


def test_clean_rejects_unbound_cache(completed_clean):
    recovery, state = completed_clean
    path = recovery / 'cache_provenance.json'
    proof = json.loads(path.read_text())
    put(path, {**proof, 'destination_session_id': 'other'})
    with pytest.raises(ValueError, match='not bound'):
        finish.validate_clean(recovery, state)


def test_finish_writes_only_clean_scope(completed_clean, monkeypatch, tmp_path):
    recovery, state = completed_clean
    report = tmp_path / 'report.md'
    frozen = (recovery / 'recovery_status.json').read_bytes()
    source_dir = tmp_path / 'source_fixture'
    source_dir.mkdir()
    for name in ['finish_clean_only.py', 'clean_only_report.py', 'comparison_evaluate.py',
                 'comparison_io.py', 'comparison_safety.py', 'run_moment_recovery.py']:
        (source_dir / name).write_text('# Test fixture only')
    monkeypatch.setattr(finish, '__file__', str(source_dir / 'finish_clean_only.py'))
    monkeypatch.setattr(finish, 'source_hashes', lambda names: {
        name: finish.io.sha256(Path(finish.__file__).parent / name) for name in names})
    monkeypatch.setattr(finish, 'validate_evaluation', lambda _: {
        'calibration_rows': 36260, 'test_rows': 23721})
    def fake_evaluate(view):
        for name in finish.CLEAN_OUTPUTS:
            (view / name).write_text('fixture')
    def fake_report(view, report, figures):
        figures.mkdir()
        report.write_text('fixture clean only')
        (figures / 'image.png').write_bytes(b'fixture')
        put(figures / 'plot_manifest.json', {'fixture': True})
        put(view / 'clean_only_report_manifest.json', {'fixture': True})
        return dict(status='passed', report_sha256=finish.io.sha256(report),
                    outputs_sha256={'image.png': finish.io.sha256(figures / 'image.png')})
    monkeypatch.setattr(finish, 'finalize_clean', fake_evaluate)
    monkeypatch.setattr(finish, 'generate_report', fake_report)
    result = finish.finish_clean_only(recovery, report)
    assert result['status'] == 'complete'
    assert result['scope'] == 'clean_only'
    assert result['full_comparison_completed_claim'] is False
    assert result['synthetic_moment_jobs_completed'] == 0
    assert (recovery / 'recovery_status.json').read_bytes() == frozen
    assert json.loads((recovery / 'clean_only_completion.json').read_text()) == result
    with pytest.raises(FileExistsError):
        finish.finish_clean_only(recovery, report)


def test_empty_hash_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match='nonempty'):
        finish.check_hashes(tmp_path, {})


def test_boundary_rejects_missing_job(recovery):
    (recovery / 'run_view/synthetic/S5_seed19').rmdir()
    with pytest.raises(ValueError, match='inventory'):
        finish.validate_boundary(recovery)


def test_boundary_rejects_guard_symlink(recovery):
    guard = recovery / 'run_view/synthetic/S1_seed00/moment'
    content = guard.read_text()
    target = recovery / 'guard_copy'
    target.write_text(content)
    guard.unlink()
    guard.symlink_to(target)
    with pytest.raises(ValueError, match='regular file'):
        finish.validate_boundary(recovery)


def test_preservation_requires_all_original_files(recovery):
    put(recovery / 'original_before.json', {'missing': 'rest'})
    with pytest.raises(ValueError, match='871'):
        finish.validate_preservation(recovery)


def test_evaluation_rejects_failed_status(evaluated):
    path = evaluated / 'clean_evaluation.json'
    manifest = json.loads(path.read_text())
    put(path, {**manifest, 'status': 'failed'})
    with pytest.raises(ValueError, match='verification'):
        finish.validate_evaluation(evaluated)


def test_evaluation_rejects_missing_model(evaluated):
    path = evaluated / 'alarm_summary.csv'
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    manifest = json.loads((evaluated / 'clean_evaluation.json').read_text())
    manifest['outputs_sha256']['alarm_summary.csv'] = finish.io.sha256(path)
    put(evaluated / 'clean_evaluation.json', manifest)
    with pytest.raises(ValueError, match='grid'):
        finish.validate_evaluation(evaluated)


def test_report_must_remain_outside_original_run(recovery):
    with pytest.raises(ValueError, match='outside'):
        finish.finish_clean_only(recovery, recovery / 'report.md')
