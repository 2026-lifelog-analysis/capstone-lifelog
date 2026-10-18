"""Finish this single running experiment after its actual producers complete."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources

EXPECTED = {f'{scenario}_seed{seed:02d}' for seed in range(20)
            for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']}


def read_progress(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # Producers update small progress files; an interrupted read is not success.
        return {}


def progress(run, queue):
    run = Path(run)
    clean = read_progress(run / 'moment_clean/execution.json')
    queue_record = read_progress(queue) if queue else {}
    if clean.get('status') == 'failed' or queue_record.get('status') == 'failed':
        raise RuntimeError('An actual MOMENT producer failed; results are incomplete')
    statistical, moments = 0, 0
    for name in EXPECTED:
        job = run / 'synthetic' / name
        if (job / 'job_failure.json').exists():
            raise RuntimeError('An actual statistical job failed; results are incomplete')
        record = read_progress(job / 'job.json')
        neural = read_progress(job / 'moment/execution.json')
        if record.get('status') == 'failed' or neural.get('status') == 'failed':
            raise RuntimeError('An actual synthetic job failed; results are incomplete')
        statistical += record.get('status') == 'passed'
        moments += neural.get('status') == 'complete'
    actual = {path.name for path in (run / 'synthetic').glob('S*_seed*') if path.is_dir()}
    if actual - EXPECTED:
        raise RuntimeError('Unexpected synthetic job directory')
    return dict(clean_ready=clean.get('status') == 'complete',
        clean_completed_persondays=clean.get('completed_persondays', 0),
        completed_statistical_jobs=statistical, completed_moment_jobs=moments,
        synthetic_ready=statistical == 100 and moments == 100)


def advance(run, report, state):
    """Each stage performs its own provenance checks and rejects existing outputs."""
    if not state['clean_ready']:
        return False
    if not (run / 'clean_evaluation.json').exists():
        from experiments.seoyeon.comparison_evaluate import finalize_clean
        finalize_clean(run)
    if not state['synthetic_ready']:
        return False
    from experiments.seoyeon.comparison_aggregate import aggregate
    from experiments.seoyeon.comparison_plot import plot
    from experiments.seoyeon.comparison_report import generate
    if not (run / 'aggregate/aggregate_manifest.json').exists():
        aggregate(run, run / 'aggregate')
    if not (run / 'figures/plot_manifest.json').exists():
        plot(run, run / 'aggregate', run / 'figures')
    generate(run, run / 'aggregate', report)
    return True


def watch(run, report, queue=None, max_hours=12.):
    if not 0 < max_hours <= 24:
        raise ValueError('Wait bound must be positive and at most 24 hours')
    run, report = Path(run), Path(report)
    if report.exists() or (run / 'completion_status.json').exists():
        raise FileExistsError('Finalizer or final report already exists')
    paths = sorted(Path(__file__).parent.glob('comparison_*.py'))
    paths.append(Path(__file__).with_name('validate_model_port.py'))
    source = source_hashes([path.name for path in paths])
    archive = run / 'finalizer_source_snapshot'
    archive.mkdir(exist_ok=False)
    for path in paths:
        shutil.copyfile(path, archive / path.name)
        if io.sha256(archive / path.name) != source[path.name]:
            raise ValueError('Finalizer source copy mismatch')
    record = dict(status='waiting', started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source, report=str(report), queue=str(queue) if queue else None,
        max_hours=max_hours, completed_claim=False)
    status_path = run / 'completion_status.json'
    io.write_json(status_path, record)
    started, previous = time.monotonic(), None
    try:
        while time.monotonic() - started < max_hours * 3600:
            verify_sources(source)
            state = progress(run, queue)
            record.update(progress=state, elapsed_seconds=time.monotonic() - started)
            status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            if state != previous:
                print(json.dumps(state), flush=True)
                previous = state
            if advance(run, report, state):
                record.update(status='complete', completed_claim=True,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    report_sha256=io.sha256(report), **verify_sources(source))
                status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                return record
            time.sleep(30)
        raise TimeoutError('Finalization wait bound reached; actual runs are not marked complete')
    except BaseException as error:
        record.update(status='unverified', error_type=type(error).__name__,
                      completed_claim=False, elapsed_seconds=time.monotonic() - started)
        status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['run', 'report']:
        parser.add_argument('--' + option, required=True, type=Path)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--max-hours', type=float, default=12.)
    args = parser.parse_args()
    watch(args.run, args.report, args.queue, args.max_hours)
