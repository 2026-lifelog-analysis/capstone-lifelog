"""기존 다섯 모델의 수식 보존 및 메인 전처리 연결을 검증합니다."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.seoyeon import comparison_models as cm


@pytest.fixture
def daily():
    rng = np.random.default_rng(20261016)
    frames = []
    for participant in ["synthetic_a", "synthetic_b"]:
        n = 95
        frame = pd.DataFrame({
            "participant": participant,
            "date": pd.date_range("2020-01-01", periods=n),
            "steps": rng.integers(2000, 10000, n).astype(float),
            "sleep_minutes": rng.integers(240, 500, n).astype(float),
            "bedtime_minutes": rng.integers(-100, 150, n).astype(float) % 1440,
            "complete": True,
        })
        frame.loc[37, "steps"] = np.nan
        frame.loc[37, "complete"] = False
        frames.append(frame.drop(index=[45]))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def original():
    source = Path(__file__).resolve().parents[2] / "모듈1_NetHealth" / "02_코드" / "models.py"
    if not source.exists():
        pytest.skip("원본 모듈이 없는 환경에서는 로컬 원본 대조만 생략")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cm.REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("seoyeon_original_models_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_converts_noon_clock_without_dropping_missing_rows():
    main = pd.DataFrame({
        "pid": ["a", "a", "b", "b"],
        "date": pd.date_range("2020-01-01", periods=4),
        "steps": [100., np.nan, 200., np.inf],
        "sleep_min": [330., 330., 300., 400.],
        "bed_min": [720., 960., np.nan, 700.],
    })
    before = main.copy(deep=True)
    got = cm.adapt_main(main)
    assert list(got) == ["participant", "date", *cm.FEATURES, "complete"]
    assert len(got) == len(main)
    np.testing.assert_allclose(got.bedtime_minutes, [0., 240., np.nan, 1420.], equal_nan=True)
    assert got.complete.tolist() == [True, False, False, False]
    assert_frame_equal(main, before)


def test_adapter_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing main input columns"):
        cm.adapt_main(pd.DataFrame({"pid": ["a"]}))


def test_clock_uses_shortest_distance_and_deterministic_tie():
    np.testing.assert_array_equal(cm.angular_difference(np.array([10., 1430., 720.]), 0.),
                                  [10., -10., -720.])
    assert cm.circular_median(np.array([1430., 10.])) == 10.
    assert np.isnan(cm.circular_median(np.array([np.nan])))


@pytest.mark.parametrize("config", [None, {"mad_scale": 1.0}, {
    "fixed14_days": 12, "fixed14_min_valid": 8,
    "fixed28_days": 24, "fixed28_min_valid": 18,
    "rolling_days": 21, "rolling_min_valid": 14, "cusum_k": .25,
}])
def test_port_exactly_matches_original_on_synthetic_data(daily, original, config):
    before = daily.copy(deep=True)
    expected = original.score_models(daily, "2020-02-07", "2020-02-08", config or {})
    actual = cm.score_models(daily, "2020-02-07", "2020-02-08", config)
    for a, b in zip(actual, expected):
        assert_frame_equal(a, b, check_exact=True)
    assert_frame_equal(daily, before)


def test_rolling_does_not_use_current_or_future_values(daily):
    scores, _ = cm.score_models(daily, "2020-02-07", "2020-02-08")
    changed = daily.copy()
    date = pd.Timestamp("2020-03-01")
    changed.loc[changed.date.ge(date), list(cm.FEATURES)] = [99999., 999., 720.]
    changed_scores, _ = cm.score_models(changed, "2020-02-07", "2020-02-08")
    before = scores.date.lt(date)
    assert_frame_equal(scores.loc[before], changed_scores.loc[before], check_exact=True)
    row = scores.date.eq(date) & scores.model.eq("rolling28")
    baseline_columns = ["baseline_n", *[f"{p}_{f}" for p in ["center", "scale"] for f in cm.FEATURES]]
    assert_frame_equal(scores.loc[row, baseline_columns], changed_scores.loc[row, baseline_columns],
                       check_exact=True)


def test_cusum_resets_after_missing_and_calendar_gap(daily):
    scores, _ = cm.score_models(daily, "2020-02-07", "2020-02-08")
    cusum = scores[scores.model.eq("cusum28")]
    for date in [pd.Timestamp("2020-02-08"), pd.Timestamp("2020-02-16")]:
        rows = cusum[cusum.date.eq(date)]
        assert rows.valid.all()
        for feature in cm.FEATURES:
            np.testing.assert_array_equal(rows[f"cplus_{feature}"],
                                          np.maximum(0, rows[f"z_{feature}"] - .5))
            np.testing.assert_array_equal(rows[f"cminus_{feature}"],
                                          np.maximum(0, -rows[f"z_{feature}"] - .5))


def test_zero_scale_is_invalid_not_epsilon_imputed(daily):
    constant = daily.assign(bedtime_minutes=0.)
    scores, _ = cm.score_models(constant, "2020-02-07", "2020-02-08")
    assert not scores.valid.any()
    assert scores.score.isna().all()
    assert "zero_mad" in scores.reason.values


@pytest.mark.parametrize("changes,message", [
    ({"development_end": "2020-02-08"}, "must precede"),
    ({"config": {"mad_scale": 0.}}, "MAD multiplier"),
    ({"config": {"rolling_min_valid": 29}}, "Invalid rolling"),
])
def test_invalid_settings_fail_explicitly(daily, changes, message):
    arguments = dict(development_end="2020-02-07", calibration_start="2020-02-08")
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        cm.score_models(daily, **arguments)


def test_duplicate_person_date_fails(daily):
    duplicate = pd.concat([daily, daily.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        cm.score_models(duplicate, "2020-02-07", "2020-02-08")


def test_empty_input_preserves_output_schema(daily):
    scores, baselines = cm.score_models(daily.iloc[:0], "2020-02-07", "2020-02-08")
    assert scores.empty and baselines.empty
    assert list(scores) == cm.SCORE_COLUMNS
    assert list(baselines) == cm.AUDIT_COLUMNS


def test_alarm_rules_match_original(daily, original):
    scores, _ = cm.score_models(daily, "2020-02-07", "2020-02-08")
    limits = dict.fromkeys(cm.MODELS, 1.)
    for inclusive in [True, False]:
        expected = original.apply_alarm_rules(scores, limits, inclusive=inclusive)
        actual = cm.apply_alarm_rules(scores, limits, inclusive=inclusive)
        assert_frame_equal(actual, expected, check_exact=True)


def test_port_structure_validator_ignores_documentation_only(tmp_path):
    from experiments.seoyeon.validate_model_port import compare_structure
    reference, copied = tmp_path / "reference.py", tmp_path / "copied.py"
    reference.write_text('def helper(value):\n    """Original documentation."""\n    return value + 1\n')
    copied.write_text('def helper(value):\n    """Updated documentation."""\n    return value + 1\n'
                      'def adapt_main(frame):\n    return frame\n')
    assert compare_structure(reference, copied) == ["helper"]


def test_port_structure_validator_rejects_algorithm_changes(tmp_path):
    from experiments.seoyeon.validate_model_port import compare_structure
    reference, copied = tmp_path / "reference.py", tmp_path / "copied.py"
    reference.write_text('def helper(value):\n    return value + 1\n')
    copied.write_text('def helper(value):\n    return value + 2\n'
                      'def adapt_main(frame):\n    return frame\n')
    with pytest.raises(AssertionError, match="Algorithm AST mismatch"):
        compare_structure(reference, copied)
