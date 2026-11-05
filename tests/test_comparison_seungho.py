"""승호 원식 동등성, 보정 누수 방지, 원설정 실행 격리를 검증합니다."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.seoyeon.comparison_seungho import (
    K_VALUES, directional_scores, fit_h, score_with_h,
)
from experiments.seoyeon.comparison_original import OriginalScoreCache, reference_context, run
from experiments.seoyeon.audit import ROOT
from src import baseline as bl, evaluation as ev


@pytest.fixture
def local_reference():
    folder = ROOT / '_archive' / 'seungho_1806f0a' / 'experiments'
    if not all((folder / name).is_file() for name in ['common.py', 'final.py', 'rule_baseline.py']):
        pytest.skip('Fixed-commit reference archive is required for integration verification.')


@pytest.fixture
def sample():
    rng = np.random.default_rng(12)
    dates = pd.date_range('2020-01-01', periods=85)
    df = pd.DataFrame({'pid': np.repeat(np.arange(36), len(dates)),
                       'date': np.tile(dates, 36)})
    df['dow'] = df.date.dt.dayofweek
    df['steps'] = 6500 + rng.normal(0, 1700, len(df)) + df.dow * 210
    df['sleep_min'] = 390 + rng.normal(0, 45, len(df))
    df['bed_min'] = 680 + rng.normal(0, 50, len(df))
    df.loc[df.index % 37 == 0, 'steps'] = np.nan
    return df


def test_directional_scores_equal_fixed_reference(sample, local_reference):
    before = sample.copy(deep=True)
    with reference_context() as (_, final, _):
        ref = sample.iloc[:0]
        expected = pd.DataFrame({
            name: final.personal_bad(sample, ref, *settings[:2])
            for name, settings in zip(K_VALUES, final.STEPS.values())})
    pd.testing.assert_frame_equal(directional_scores(sample), expected)
    pd.testing.assert_frame_equal(sample, before)


def test_all_period_calibration_equals_original_threshold(sample):
    bad = directional_scores(sample)
    actual = fit_h(sample, bad, pd.Series(True, index=sample.index), 1 / 30)
    for name, k in K_VALUES.items():
        expected = bl.cusum_threshold(sample.assign(b=bad[name]), 'b', k=k, rate=1 / 30)
        assert actual[name] == expected


def test_test_values_cannot_change_calibrated_h(sample):
    mask = sample.date.between('2020-02-01', '2020-02-25')
    bad = directional_scores(sample)
    expected = fit_h(sample, bad, mask, .05)
    changed = sample.copy()
    changed.loc[changed.date > '2020-02-25', 'steps'] = 1e9
    actual = fit_h(changed, directional_scores(changed), mask, .05)
    assert actual == expected


def test_cusum_matches_original_missing_and_gap_semantics():
    df = pd.DataFrame({'pid': [1] * 5, 'date': pd.to_datetime([
        '2020-01-01', '2020-01-02', '2020-01-10', '2020-01-11', '2020-01-12'])})
    bad = pd.DataFrame({name: [1., np.nan, 1., 1., 1.] for name in K_VALUES})
    scores = score_with_h(df, bad, {name: 1. for name in K_VALUES})
    np.testing.assert_allclose(scores.sh_base, [.5, np.nan, 1., 1.5, .5], equal_nan=True)
    for name, k in K_VALUES.items():
        pd.testing.assert_series_equal(scores[name], bl.cusum(df.assign(b=bad[name]), 'b', k=k, h=1.),
                                       check_names=False)


def test_empty_calibration_is_rejected(sample):
    with pytest.raises(ValueError, match='calibration'):
        fit_h(sample, directional_scores(sample), np.zeros(len(sample), dtype=bool), .05)


@pytest.mark.parametrize('rate', [0, 1, -1])
def test_invalid_calibration_rates_are_rejected(sample, rate):
    with pytest.raises(ValueError, match='rate'):
        fit_h(sample, directional_scores(sample), np.ones(len(sample), dtype=bool), rate)


def test_no_finite_calibration_is_rejected(sample):
    scores = pd.DataFrame(np.nan, index=sample.index, columns=K_VALUES)
    with pytest.raises(ValueError, match='finite calibration'):
        fit_h(sample, scores, np.ones(len(sample), dtype=bool), .05)


def test_mismatched_indices_are_rejected(sample):
    scores = directional_scores(sample).reset_index(drop=True)
    scores.index = scores.index + 10
    with pytest.raises(ValueError, match='same unique index'):
        score_with_h(sample, scores, {name: 1. for name in K_VALUES})


def test_invalid_thresholds_are_rejected(sample):
    scores = directional_scores(sample)
    with pytest.raises(ValueError, match='nonnegative thresholds'):
        score_with_h(sample, scores, {name: -1. for name in K_VALUES})


def test_reference_context_restores_injection_on_error(local_reference):
    original = ev.inject_drift
    with pytest.raises(RuntimeError, match='deliberate'):
        with reference_context():
            assert ev.inject_drift is not original
            raise RuntimeError('deliberate')
    assert ev.inject_drift is original


def test_reference_context_rejects_nested_use(local_reference):
    with reference_context():
        with pytest.raises(RuntimeError, match='nested'):
            with reference_context():
                pass


def test_cached_original_scores_and_injection_equal_original_functions(sample, local_reference):
    with reference_context() as (common, final, _):
        dev, hold = common.split_pids(sample)
        reference = sample[sample.pid.isin(dev)].reset_index(drop=True)
        held = sample[sample.pid.isin(hold)].reset_index(drop=True)
        cache = OriginalScoreCache(final, reference)
        cached = cache.score_fn(held, .05)
        original = final.make_score_fn(held, reference, .05)
        assert cached.h == original.h
        pd.testing.assert_frame_equal(cached(held), original(held))
        effects, shape, _ = final.SCEN['S5']
        (col, effect, mode), *extra = effects
        common.INJECT_EXTRA[:] = extra
        arguments = dict(col=col, effect=effect, mode=mode, shape=shape, n_reps=2, rate=.05)
        actual = ev.evaluate_injection(held, score_fn=cached, **arguments)
        expected = ev.evaluate_injection(held, score_fn=original, **arguments)
        for key in ['per_person', 'segments']:
            pd.testing.assert_frame_equal(actual[key], expected[key])
        for key in ['threshold', 'day_fa_all', 'pr_auc', 'setting']:
            assert actual[key] == expected[key]
        assert cache.computed_inputs == 3
        assert cache.hits > 0


def test_original_runner_saves_all_conditions_only_under_new_output(sample, local_reference, tmp_path):
    source = tmp_path / 'synthetic.csv'
    sample.assign(event_tag='none', term_tag='term').to_csv(source, index=False)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / 'original'
    manifest = run(output, source, n_reps=1)
    assert manifest['status'] == 'passed'
    assert len(manifest['tasks_completed']) == 12
    assert len(pd.read_csv(output / 'final_table.csv')) == 72
    assert len(pd.read_csv(output / 'final_compare.csv')) == 60
    assert (output / 'final_calendar.csv').is_file()
    assert len(list(output.glob('*_per_person.csv.gz'))) == 12
    assert json.loads((output / 'manifest.json').read_text())['input_sha256'] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(FileExistsError):
        run(output, source, n_reps=1)
