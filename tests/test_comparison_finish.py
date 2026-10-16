import json
import pytest

from experiments.seoyeon.comparison_finish import progress
from experiments.seoyeon import comparison_finish as finisher


def test_missing_work_is_pending_not_complete(tmp_path):
    state = progress(tmp_path, None)
    assert not state['clean_ready'] and not state['synthetic_ready']
    assert state['completed_statistical_jobs'] == 0


def test_progress_requires_exact_hundred_successful_jobs(tmp_path):
    clean = tmp_path / 'moment_clean'
    clean.mkdir()
    (clean / 'execution.json').write_text(json.dumps({'status':'complete','completed_persondays':10}))
    for seed in range(20):
        for scenario in ['S1','S2','S3','S4','S5']:
            job = tmp_path / 'synthetic' / f'{scenario}_seed{seed:02d}'
            (job / 'moment').mkdir(parents=True)
            (job / 'job.json').write_text('{"status":"passed"}')
            (job / 'moment/execution.json').write_text('{"status":"complete"}')
    state = progress(tmp_path, None)
    assert state['clean_ready'] and state['synthetic_ready']
    assert state['completed_moment_jobs'] == 100
    (job / 'moment/execution.json').write_text('{"status":"running"}')
    assert not progress(tmp_path, None)['synthetic_ready']


@pytest.mark.parametrize('path', ['moment_clean/execution.json', 'synthetic/S1_seed00/job_failure.json', 'queue.json'])
def test_explicit_failures_stop_finalization(tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError):
        progress(tmp_path, tmp_path / 'queue.json' if path == 'queue.json' else None)


def test_advance_defers_until_actual_completion(tmp_path, monkeypatch):
    assert not finisher.advance(tmp_path, tmp_path / 'report', {'clean_ready':False})
    from experiments.seoyeon import comparison_evaluate as evaluate
    from experiments.seoyeon import comparison_aggregate as aggregate
    from experiments.seoyeon import comparison_plot as plot
    from experiments.seoyeon import comparison_report as report
    calls = []
    monkeypatch.setattr(evaluate, 'finalize_clean', lambda run: calls.append('clean'))
    monkeypatch.setattr(aggregate, 'aggregate', lambda *args: calls.append('aggregate'))
    monkeypatch.setattr(plot, 'plot', lambda *args: calls.append('plot'))
    monkeypatch.setattr(report, 'generate', lambda *args: calls.append('report'))
    assert not finisher.advance(tmp_path, tmp_path / 'report', {'clean_ready':True,'synthetic_ready':False})
    calls.clear()
    assert finisher.advance(tmp_path, tmp_path / 'report', {'clean_ready':True,'synthetic_ready':True})
    assert calls == ['clean','aggregate','plot','report']


def test_watch_records_true_completion_and_blocks_reentry(tmp_path, monkeypatch):
    output = tmp_path / 'report.md'
    monkeypatch.setattr(finisher, 'progress', lambda *args: {'clean_ready':True,'synthetic_ready':True})
    def complete(run, report, state):
        report.write_text('fixture verified report')
        return True
    monkeypatch.setattr(finisher, 'advance', complete)
    result = finisher.watch(tmp_path, output)
    assert result['completed_claim'] and result['status'] == 'complete'
    assert (tmp_path / 'finalizer_source_snapshot/comparison_finish.py').exists()
    with pytest.raises(FileExistsError):
        finisher.watch(tmp_path, output)


def test_watch_timeout_is_unverified_without_killing_producers(tmp_path, monkeypatch):
    monkeypatch.setattr(finisher.time, 'monotonic', iter([0., 2., 3.]).__next__)
    with pytest.raises(TimeoutError):
        finisher.watch(tmp_path, tmp_path / 'report', max_hours=.0001)
    state = json.loads((tmp_path / 'completion_status.json').read_text())
    assert state['status'] == 'unverified' and not state['completed_claim']
