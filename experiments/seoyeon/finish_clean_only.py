"""Verify and finalize the explicitly authorized clean-only MOMENT comparison.

The interrupted full-scope status remains failed. This module never runs neural
inference and never declares the excluded synthetic experiment complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS, finalize_clean, verify_clean_moment
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources
from experiments.seoyeon.run_moment_recovery import (
    JOBS, check_original_preservation, original_snapshot,
)

CLEAN_OUTPUTS = ('evaluated_scores_r05.csv.gz', 'evaluated_scores_r033.csv.gz',
                 'clean_thresholds.json', 'alarm_summary.csv', 'coverage.csv',
                 'pairwise.csv', 'clean_evaluation.json')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def check_hashes(directory: Path, expected: dict) -> None:
    if not expected:
        raise ValueError('Expected nonempty checksum manifest')
    for name, digest in expected.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Manifest paths must remain relative to their directory')
        if io.sha256(directory / path) != digest:
            raise ValueError('Artifact checksum mismatch')


def validate_boundary(recovery: Path) -> dict:
    state = read_json(recovery / 'recovery_status.json')
    expected = dict(status='failed', phase='synthetic', error_type='FileExistsError',
                    completed_claim=False, current_job='S1_seed00', completed_jobs=[],
                    expected_jobs=100)
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError('Full-scope runner did not stop at the authorized boundary')
    synthetic = recovery / 'run_view/synthetic'
    if {path.name for path in synthetic.iterdir()} != set(JOBS):
        raise ValueError('Synthetic input job inventory changed')
    for name in JOBS:
        output = synthetic / name / 'moment'
        if name == 'S1_seed00':
            if output.is_symlink() or not output.is_file():
                raise ValueError('Scope guard must be an owned regular file')
            guard = read_json(output)
            if (guard.get('kind') != 'user_scope_boundary_guard_not_inference_output'
                    or guard.get('synthetic_moment') != 'not_executed_by_user_scope_change'
                    or guard.get('do_not_remove_to_resume_synthetic') is not True):
                raise ValueError('Scope guard contents differ')
        elif output.exists() or output.is_symlink():
            raise ValueError('Synthetic inference output exists despite scope exclusion')
    return state


def validate_preservation(recovery: Path) -> int:
    before = read_json(recovery / 'original_before.json')
    if len(before) != 871:
        raise ValueError('Original preservation inventory differs from 871 files')
    check_original_preservation(before, read_json(recovery / 'original_after.json'))
    check_original_preservation(before, original_snapshot(recovery.parent))
    if read_json(recovery.parent / 'inputs_before.json') != io.preserved_inputs():
        raise ValueError('Original tracked files or inputs changed')
    return len(before)


def validate_clean(recovery: Path, state: dict) -> dict:
    view = recovery / 'run_view'
    record = read_json(view / 'moment_clean/execution.json')
    required = dict(status='complete', requested_persondays=87886, eligible_persondays=87886,
                    completed_persondays=87886, valid_persondays=87886, unavailable_persondays=0,
                    new_inferred_persondays=9550, exact_six_field_cache_persondays=78336,
                    cross_process_current_experiment_cache_persondays=78336,
                    same_process_cache_persondays=0, past_experiment_cache_persondays=0)
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError('Clean inference completion or provenance counts differ')
    if state.get('clean') != record or state.get('session_id') != record.get('session_id'):
        raise ValueError('Runner and clean inference record differ')
    moment = io.read_frame(view / 'moment_clean/scores.csv.gz')
    verify_clean_moment(view, moment, record)
    counts = moment.score_source.value_counts().to_dict()
    if counts != {'new_three_channel_inference': 9550,
                  'same_experiment_cross_process_exact_six_field_cache': 78336}:
        raise ValueError('Actual score provenance counts differ')
    preservation = read_json(view / 'moment_clean/source_preservation.json')
    if (preservation.get('unchanged') is not True
            or preservation.get('sha256') != record['input_source_sha256']):
        raise ValueError('Clean source preservation record differs')
    proof = read_json(recovery / 'cache_provenance.json')
    checks = dict(verified_persondays=78336, remaining_persondays=9550,
                  requested_persondays=87886, unique_six_field_inputs=78336,
                  all_six_inputs_rebuilt=True, all_audit_fields_rebuilt=True,
                  all_three_endpoint_errors_and_mse_recomputed=True,
                  reconstruction_csv_float32_serialization_verified=True,
                  fresh_replay_persondays=8, fresh_replay_max_abs=0,
                  model_runtime_identical=True, prior_historical_experiment_used=False)
    if any(proof.get(key) != value for key, value in checks.items()):
        raise ValueError('Verified partial-cache provenance differs')
    if (record.get('same_experiment_seed_cache') != [proof]
            or proof.get('destination_session_id') != record.get('session_id')
            or proof.get('recovery_model') != record['model']):
        raise ValueError('Cached provenance is not bound to this frozen model and session')
    controls = record['model'].get('controls', {})
    if (controls.get('pass') is not True or controls.get('repeat_max_abs') != 0
            or controls.get('masked_placeholder_max_abs') != 0):
        raise ValueError('Actual neural controls differ from verified zero-difference results')
    verify_sources(record['source_sha256'])
    verify_sources(state['source_sha256'])
    check_hashes(recovery / 'source_snapshot', state['source_sha256'])
    return record


def validate_evaluation(view: Path) -> dict:
    manifest = read_json(view / 'clean_evaluation.json')
    required = dict(status='passed', calibration_rows=36260, test_rows=23721,
                    frozen_common_rows=59981, threshold_test_access=False,
                    all_requested_common_moment_finite=True)
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError('Clean evaluation verification differs')
    check_hashes(view, manifest.get('outputs_sha256', {}))
    summary = pd.read_csv(view / 'alarm_summary.csv')
    common = summary.loc[summary.scope.eq('common')]
    expected = {(rate, model, split) for rate in ('r05', 'r033')
                for model in MODELS for split in ('calibration', 'test')}
    keys = list(common[['rate_id', 'model', 'split']].itertuples(index=False, name=None))
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Ten-method two-rate evaluation grid differs')
    for split, people, days in [('calibration', 181, 36260), ('test', 135, 23721)]:
        part = common.loc[common.split.eq(split)]
        if not (part.people.eq(people) & part.days.eq(days) & part.unavailable_days.eq(0)).all():
            raise ValueError('Frozen common denominator differs')
    return manifest


def generate_report(view: Path, report: Path, figures: Path) -> dict:
    from experiments.seoyeon.clean_only_report import generate_clean_report
    return generate_clean_report(view, report, figures)


def finish_clean_only(recovery_dir: Path, report: Path) -> dict:
    recovery, report = recovery_dir.absolute(), report.absolute()
    view, figures = recovery / 'run_view', recovery / 'run_view/clean_only_figures'
    destination = recovery / 'clean_only_completion.json'
    snapshot = recovery / 'clean_only_source_snapshot'
    outputs = [destination, snapshot, report, figures, view / 'clean_only_report_manifest.json',
               *(view / name for name in CLEAN_OUTPUTS)]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError('Clean-only output exists; no existing artifact is overwritten')
    if not report.parent.is_dir() or report.is_relative_to(recovery.parent):
        raise ValueError('Report must have an existing parent outside the original run')
    state = validate_boundary(recovery)
    original_count = validate_preservation(recovery)
    record = validate_clean(recovery, state)
    evidence = [recovery / name for name in ('recovery_status.json', 'cache_provenance.json',
                'original_before.json', 'original_after.json')]
    evidence += [view / 'moment_clean' / name for name in
                 ('execution.json', 'scores.csv.gz', 'source_preservation.json')]
    evidence += [view / 'synthetic/S1_seed00/moment']
    before = {str(path.relative_to(recovery)): io.sha256(path) for path in evidence}
    names = ['finish_clean_only.py', 'clean_only_report.py', 'comparison_evaluate.py',
             'comparison_io.py', 'comparison_safety.py', 'run_moment_recovery.py']
    sources = source_hashes(names)
    snapshot.mkdir()
    for name in names:
        shutil.copyfile(Path(__file__).parent / name, snapshot / name)
    check_hashes(snapshot, sources)
    finalize_clean(view)
    evaluation = validate_evaluation(view)
    generated = generate_report(view, report, figures)
    if generated.get('status') != 'passed' or generated.get('report_sha256') != io.sha256(report):
        raise ValueError('Clean-only report verification failed')
    check_hashes(figures, generated.get('outputs_sha256', {}))
    check_hashes(recovery, before)
    verify_sources(sources)
    validate_boundary(recovery)
    validate_preservation(recovery)
    output_paths = [view / name for name in CLEAN_OUTPUTS]
    output_paths += [view / 'clean_only_report_manifest.json', figures / 'plot_manifest.json']
    output_paths += [figures / name for name in generated['outputs_sha256']]
    completion = dict(status='complete', scope='clean_only', clean_completed_claim=True,
        full_comparison_completed_claim=False, synthetic_moment_jobs_completed=0,
        synthetic_scope='excluded_by_user_request', full_scope_status_preserved='failed',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        requested_persondays=87886, new_inferred_persondays=record['new_inferred_persondays'],
        verified_reused_persondays=record['exact_six_field_cache_persondays'],
        diagnostic_replay_persondays=8, original_files_checked=original_count,
        calibration_rows=evaluation['calibration_rows'], test_rows=evaluation['test_rows'],
        evidence_sha256=before, source_sha256=sources, report_sha256=io.sha256(report),
        outputs_sha256={str(path.relative_to(recovery)): io.sha256(path) for path in output_paths})
    io.write_json(destination, completion)
    return completion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    finish_clean_only(args.recovery_dir, args.report)


if __name__ == '__main__':
    main()
