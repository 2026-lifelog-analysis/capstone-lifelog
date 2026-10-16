"""Finish interrupted MOMENT work in an isolated, non-overwriting run view.

Original artifacts are linked only as inputs. New inference and finalizer outputs
belong to this recovery directory; no existing producer is started or stopped.
An operator must first establish that no other neural producer is running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_moment import (
    MomentEngine, _read_tables, infer, synthetic_job_paths,
)
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources

JOBS = tuple(f'S{scenario}_seed{seed:02d}' for seed in range(20)
             for scenario in range(1, 6))
CLEAN_REQUESTS = 87886
SYNTHETIC_REQUESTS = 150910
EXCLUDED = frozenset({
    'moment_clean', 'synthetic', 'completion_status.json', 'finalizer.log',
    'finalizer_source_snapshot', 'report_manifest.json', 'clean_evaluation.json',
    'evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
    'alarm_summary.csv', 'coverage.csv', 'pairwise.csv', 'clean_thresholds.json',
    'aggregate', 'figures',
})
OWNED = ('run_view', 'recovery_status.json', 'original_before.json',
         'original_after.json', 'cache_provenance.json', 'source_snapshot',
         'preservation_result.json', 'finalizer.log')
FINAL_OUTPUTS = {
    'clean_evaluation.json': frozenset({
        'evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
        'clean_thresholds.json', 'alarm_summary.csv', 'coverage.csv', 'pairwise.csv'}),
    'aggregate/aggregate_manifest.json': frozenset({
        'segment_results.csv.gz', 'seed_summary.csv', 'participant_summary.csv',
        'seed_distribution_summary.csv', 'participant_distribution_summary.csv'}),
    'figures/plot_manifest.json': frozenset(
        f'{name}.{extension}' for name in
        ('daily_alarm_rates', 'synthetic_detection', 'synthetic_paired_delta')
        for extension in ('png', 'pdf')),
}


def restore_partial_cache(run: Path, engine: MomentEngine) -> dict:
    """Import lazily so command help and fake-engine tests need no model runtime."""
    from experiments.seoyeon.recover_moment_partial import restore_partial_cache as restore
    return restore(run, engine)


def original_snapshot(run: Path) -> dict[str, str]:
    """Hash all old files without following the recovery view back into itself."""
    result = {}
    for directory, children, files in os.walk(run, followlinks=False):
        children[:] = [name for name in children
                       if not name.startswith(('recovery_', 'nodecopies'))]
        for name in sorted(files):
            path = Path(directory) / name
            result[str(path.relative_to(run))] = io.sha256(path)
    return dict(sorted(result.items()))


def check_original_preservation(before: dict, after: dict) -> None:
    """Additional old-run files are permitted; every preexisting file is immutable."""
    if any(after.get(name) != digest for name, digest in before.items()):
        raise ValueError('Original artifact preservation failed')


def validate_jobs(run: Path) -> None:
    source = run / 'synthetic'
    names = {path.name for path in source.iterdir()}
    if names != set(JOBS):
        raise ValueError('Expected exactly the one hundred statistical job directories')
    for name in JOBS:
        job = source / name
        if not job.is_dir() or job.is_symlink():
            raise ValueError('Statistical job must be a real directory')
        if (job / 'moment').exists() or (job / 'moment').is_symlink():
            raise FileExistsError('Existing synthetic neural work requires separate review')
        if any(not path.is_file() for path in job.iterdir()):
            raise ValueError('Only individual input files may be linked into a synthetic job')
        if synthetic_job_paths(run, name) is None:
            raise ValueError('Missing committed statistical job')


def create_run_view(run: Path, recovery_dir: Path) -> Path:
    """Materialize output parents, linking only unchanged original inputs."""
    view = recovery_dir / 'run_view'
    if view.exists() or view.is_symlink():
        raise FileExistsError('Recovery view already exists; it is not resumable in place')
    validate_jobs(run)
    recovery_dir.mkdir(parents=False, exist_ok=True)
    view.mkdir()
    for path in sorted(run.iterdir()):
        if path.name in EXCLUDED or path.name.startswith(('recovery_', 'nodecopies')):
            continue
        (view / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (view / 'synthetic').mkdir()
    for name in JOBS:
        destination = view / 'synthetic' / name
        destination.mkdir()
        for path in sorted((run / 'synthetic' / name).iterdir()):
            (destination / path.name).symlink_to(path.resolve())
    return view


def preflight(run: Path, recovery_dir: Path, model_root: Path,
              finalizer_python: Path, report: Path) -> None:
    if not run.is_dir() or not model_root.is_dir():
        raise FileNotFoundError('Original run and model root must exist')
    if (recovery_dir.is_symlink() or recovery_dir.parent != run
            or not recovery_dir.name.startswith('recovery_')):
        raise ValueError('Recovery directory must be a separate recovery_ child of the original run')
    if report.exists() or report.is_symlink():
        raise FileExistsError('Final report already exists')
    if not report.parent.is_dir():
        raise FileNotFoundError('Final report parent must already exist')
    if report.is_relative_to(run):
        raise ValueError('Final report must be outside the preserved original run')
    if not finalizer_python.is_file() or not os.access(finalizer_python, os.X_OK):
        raise FileNotFoundError('Finalizer Python executable is missing or not executable')
    if any((recovery_dir / name).exists() or (recovery_dir / name).is_symlink()
           for name in OWNED):
        raise FileExistsError('Recovery-owned output already exists')
    initial = json.loads((run / 'inputs_before.json').read_text())
    if initial != io.preserved_inputs():
        raise ValueError('An original tracked file or input changed before recovery')
    validate_jobs(run)


def archive_sources(recovery_dir: Path) -> dict[str, str]:
    folder = Path(__file__).parent
    paths = sorted(folder.glob('comparison_*.py')) + [
        folder / 'validate_model_port.py', Path(__file__),
    ]
    partial = folder / 'recover_moment_partial.py'
    if partial.exists():
        paths.append(partial)
    hashes = source_hashes([path.name for path in paths])
    destination = recovery_dir / 'source_snapshot'
    destination.mkdir()
    for path in paths:
        shutil.copyfile(path, destination / path.name)
        if io.sha256(destination / path.name) != hashes[path.name]:
            raise ValueError('Recovery source snapshot differs')
    return hashes


def save_state(path: Path, state: dict, **changes: Any) -> dict:
    """Only update the status file exclusively created by this invocation."""
    if path.is_symlink() or not path.is_file():
        raise ValueError('Recovery status must remain an owned regular file')
    updated = {**state, **changes, 'updated_utc': datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return updated


def run_inference(paths: dict, hashes: dict, output: Path,
                  engine: MomentEngine, sources: dict, expected: int | None = None) -> dict:
    verify_sources(sources)
    tables = _read_tables(paths)
    count = len(tables['keys'])
    if expected is not None and count != expected:
        raise ValueError('Requested clean sample differs from the fixed experiment')
    record = infer(**tables, outputdir=output, engine=engine, source_hashes=hashes)
    required = ('requested_persondays', 'completed_persondays', 'valid_persondays')
    if (record.get('status') != 'complete' or record.get('unavailable_persondays') != 0
            or any(record.get(name) != count for name in required)):
        raise ValueError('MOMENT inference did not complete the entire fixed sample')
    if hashes != {name: io.sha256(path) for name, path in paths.items()}:
        raise ValueError('Inference input preservation failed')
    verify_sources(sources)
    io.write_json(output / 'source_preservation.json', {'unchanged': True, 'sha256': hashes})
    return record


def add_counts(state: dict, record: dict) -> dict:
    names = ('new_inferred_persondays', 'exact_six_field_cache_persondays',
             'requested_persondays', 'completed_persondays')
    return {**state, **{name: state.get(name, 0) + record[name] for name in names}}


def finalize(view: Path, report: Path, python: Path, state_path: Path, log: Path) -> None:
    display_run = view.relative_to(io.ROOT) if view.is_relative_to(io.ROOT) else view
    command = [str(python), '-m', 'experiments.seoyeon.comparison_finish',
               '--run', str(display_run), '--report', str(report), '--queue', str(state_path),
               '--max-hours', '12']
    with log.open('x', encoding='utf-8') as output:
        subprocess.run(command, cwd=io.ROOT, stdout=output, stderr=subprocess.STDOUT,
                       check=True)
    completion = json.loads((view / 'completion_status.json').read_text())
    if (completion.get('status') != 'complete' or completion.get('completed_claim') is not True
            or completion.get('report_sha256') != io.sha256(report)):
        raise ValueError('Finalizer did not verify report completion')
    report_manifest = json.loads((view / 'report_manifest.json').read_text())
    if (report_manifest.get('status') != 'passed'
            or report_manifest.get('output_sha256') != io.sha256(report)):
        raise ValueError('Report manifest checksum differs')
    for manifest, required in FINAL_OUTPUTS.items():
        path = view / manifest
        record = json.loads(path.read_text())
        if record.get('status') != 'passed':
            raise ValueError('Final artifact manifest is not passed')
        output_hashes = record.get('outputs_sha256', {})
        if not required.issubset(output_hashes):
            raise ValueError('Final artifact manifest omits required outputs')
        for name, digest in output_hashes.items():
            if io.sha256(path.parent / name) != digest:
                raise ValueError('Final artifact checksum differs')


def run_recovery(*, run: Path, recovery_dir: Path, model_root: Path,
                 finalizer_python: Path, report: Path, device: str = 'mps',
                 batch_size: int = 96) -> dict:
    run, model_root, finalizer_python = map(Path, (run, model_root, finalizer_python))
    run, model_root = run.resolve(), model_root.resolve()
    # Resolving a venv interpreter symlink would silently bypass that environment.
    finalizer_python = finalizer_python.absolute()
    recovery_dir, report = Path(recovery_dir).absolute(), Path(report).absolute()
    preflight(run, recovery_dir, model_root, finalizer_python, report)
    before = original_snapshot(run)
    recovery_dir.mkdir(exist_ok=True)
    io.write_json(recovery_dir / 'original_before.json', before)
    state_path = recovery_dir / 'recovery_status.json'
    state = dict(status='running', phase='preparing', completed_claim=False,
                 completed_jobs=[], expected_jobs=len(JOBS), device=device, batch_size=batch_size,
                 started_utc=datetime.now(timezone.utc).isoformat())
    io.write_json(state_path, state)
    try:
        view = create_run_view(run, recovery_dir)
        sources = archive_sources(recovery_dir)
        state = save_state(state_path, state, phase='cache_verification', source_sha256=sources)
        engine = MomentEngine(model_root, device=device, batch_size=batch_size)
        provenance = restore_partial_cache(run, engine)
        io.write_json(recovery_dir / 'cache_provenance.json', provenance)
        verify_sources(sources)
        state = save_state(state_path, state, phase='clean', cache_provenance=provenance,
                           session_id=engine.session_id)
        paths = {'daily': view / 'seoyeon_daily.csv.gz',
                 'baselines': view / 'seoyeon_baselines.csv', 'keys': view / 'moment_keys.csv.gz'}
        hashes = {name: io.sha256(path) for name, path in paths.items()}
        clean = run_inference(paths, hashes, view / 'moment_clean', engine, sources, CLEAN_REQUESTS)
        state = save_state(state_path, add_counts(state, clean), clean=clean)
        synthetic_requested = 0
        for name in JOBS:
            state = save_state(state_path, state, phase='synthetic', current_job=name)
            resolved = synthetic_job_paths(view, name)
            if resolved is None:
                raise ValueError('A prevalidated synthetic job disappeared')
            paths, hashes = resolved
            record = run_inference(paths, hashes, view / 'synthetic' / name / 'moment', engine, sources)
            synthetic_requested += record['requested_persondays']
            state = save_state(state_path, add_counts(state, record),
                               completed_jobs=[*state['completed_jobs'], name])
        if synthetic_requested != SYNTHETIC_REQUESTS:
            raise ValueError('Synthetic request total differs from the fixed experiment')
        state = save_state(state_path, state, phase='finalization')
        verify_sources(sources)
        finalize(view, report, finalizer_python, state_path, recovery_dir / 'finalizer.log')
        verify_sources(sources)
        after = original_snapshot(run)
        io.write_json(recovery_dir / 'original_after.json', after)
        check_original_preservation(before, after)
        if json.loads((run / 'inputs_before.json').read_text()) != io.preserved_inputs():
            raise ValueError('Original tracked file or input preservation failed')
        io.write_json(recovery_dir / 'preservation_result.json',
                      dict(status='passed', original_files_checked=len(before),
                           source_preserved=True, original_inputs_preserved=True))
        state = save_state(state_path, state, status='complete', phase='complete',
                           completed_claim=True, report_sha256=io.sha256(report))
    except BaseException as error:
        state = save_state(state_path, state, status='failed', completed_claim=False,
                           error_type=type(error).__name__)
        after_path = recovery_dir / 'original_after.json'
        try:
            if not after_path.exists():
                io.write_json(after_path, original_snapshot(run))
        except OSError as snapshot_error:
            state = save_state(state_path, state,
                               preservation_snapshot_error_type=type(snapshot_error).__name__)
        raise
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'recovery-dir', 'model-root', 'finalizer-python', 'report'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--batch-size', type=int, default=96)
    run_recovery(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
