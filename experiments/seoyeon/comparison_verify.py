"""Independently rerun preprocessing and nine statistical methods without overwrites."""
import argparse
import json
from pathlib import Path
import time

import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon import comparison_run as runner
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources

TABLES = ['main_daily.csv.gz', 'preprocessing_counts.csv', 'seoyeon_daily.csv.gz',
          'seoyeon_scores.csv.gz', 'seoyeon_baselines.csv', 'seoyeon_wide.csv.gz',
          'moment_keys.csv.gz', 'seungho_directional.csv.gz', 'common_candidate_keys.csv.gz',
          'clean_scores_r05.csv.gz', 'clean_scores_r033.csv.gz']
RECORDS = ['protocol.json', 'split.json', 'clean_thresholds_without_moment.json']


def compare_tables(first, second):
    left, right = io.read_frame(first), io.read_frame(second)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    return dict(rows=len(left), columns=len(left.columns), all_cells_exact=True)


def verify(run, replay):
    run, replay = Path(run), Path(replay)
    if replay.exists():
        raise FileExistsError('Replay directory must be new')
    before = io.preserved_inputs()
    initial = json.loads((run / 'inputs_before.json').read_text())
    if before != initial:
        raise ValueError('Original input differs from initial run snapshot')
    source = source_hashes(['comparison_verify.py', 'comparison_run.py', 'comparison_io.py',
                            'comparison_models.py', 'comparison_seungho.py', 'comparison_safety.py'])
    start = time.monotonic()
    runner.prepare(replay)
    try:
        runner.score(replay)
        runner.fit(replay)
        tables = {name: compare_tables(run / name, replay / name) for name in TABLES}
        for name in RECORDS:
            if json.loads((run / name).read_text()) != json.loads((replay / name).read_text()):
                raise ValueError(f'Replay settings differ: {name}')
        after = io.preserved_inputs()
        if initial != after:
            raise ValueError('Original input changed during replay')
        io.write_json(replay / 'verification.json', dict(status='passed', tables=tables,
            settings_exact=RECORDS, original_inputs_preserved=True,
            initial_run_code_before_hash='Not recorded for root prepare/score/fit; not retroactively asserted',
            new_replay_provenance=verify_sources(source), seconds=time.monotonic() - start))
    except Exception as error:
        io.write_json(replay / 'verification_failure.json', dict(status='failed',
            error_type=type(error).__name__, seconds=time.monotonic() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay-output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.replay_output)
