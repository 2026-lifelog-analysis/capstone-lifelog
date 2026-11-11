"""Behavioral tests for paired, participant-clustered experiment analysis."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from statistics_helpers import (
    aggregate_errors, bootstrap_contrasts, holm_adjust, validate_group,
)
from analyze_results import (CONDITIONS, _adjust_all, expected_nights, paired_wide,
                             read_group, run, subgroup_metrics, summarize_group)


def example_predictions():
    expected = pd.DataFrame({"pid": ["a", "b"], "date": pd.to_datetime(["2020-01-01"] * 2),
                             "fold": [1, 2], "y_true": [10., 20.]})
    rows = []
    for seed in (0, 1):
        part = expected.assign(seed=seed, y_pred=[8. + 4 * seed, 20.])
        rows.append(part)
    return pd.concat(rows, ignore_index=True), expected


def test_seed_errors_are_averaged_not_predictions():
    pred, expected = example_predictions()
    assert validate_group(pred, expected, seeds=(0, 1), folds=(1, 2))["status"] == "complete"
    nights = aggregate_errors(pred)
    assert nights.loc[nights.pid == "a", "abs_error"].item() == 2.
    assert nights.loc[nights.pid == "a", "squared_error"].item() == 4.


def test_missing_seed_or_night_is_partial():
    pred, expected = example_predictions()
    assert validate_group(pred.iloc[:-1], expected, (0, 1), (1, 2))["status"] == "partial"
    assert validate_group(pred[pred.seed == 0], expected, (0, 1), (1, 2))["status"] == "partial"


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_true", "wrong_fold", "extra_person"])
def test_corrupted_evaluation_identity_is_rejected(mutation):
    pred, expected = example_predictions()
    if mutation == "duplicate":
        pred = pd.concat([pred, pred.iloc[[0]]], ignore_index=True)
    elif mutation == "wrong_true":
        pred.loc[0, "y_true"] = 999.
    elif mutation == "wrong_fold":
        pred.loc[0, "fold"] = 2
    else:
        pred.loc[0, "pid"] = "holdout_person"
    with pytest.raises(ValueError):
        validate_group(pred, expected, (0, 1), (1, 2))


def test_nonfinite_prediction_is_incomplete():
    pred, expected = example_predictions()
    pred.loc[0, "y_pred"] = np.nan
    assert validate_group(pred, expected, (0, 1), (1, 2))["status"] == "incomplete"


def test_cluster_bootstrap_preserves_night_weighted_estimand():
    # One person has three nights. A mean of person means would give 5, not 2.5.
    nights = pd.DataFrame({"pid": ["a", "a", "a", "b"],
                           "base": [0., 0., 0., 10.], "new": [0.] * 4})
    result = bootstrap_contrasts(nights, {"gain": {"base": 1., "new": -1.}}, 2000, 7)
    assert result.loc[0, "estimate"] == 2.5
    assert result.loc[0, "ci_low"] == 0.
    assert result.loc[0, "ci_high"] == 10.


def test_interaction_detects_nonadditive_effect_without_new_resampling():
    nights = pd.DataFrame({"pid": ["a", "b", "c"], "base": [10.] * 3,
                           "A": [9.] * 3, "B": [8.] * 3, "C": [9.] * 3, "ABC": [7.] * 3})
    contrast = {"interaction": {"base": -2., "A": 1., "B": 1., "C": 1., "ABC": -1.}}
    result = bootstrap_contrasts(nights, contrast, 200, 3)
    assert result.loc[0, "estimate"] == -1.  # Actual gain 3 minus additive prediction 4.
    assert result.loc[0, "ci_low"] == result.loc[0, "ci_high"] == -1.


def test_holm_is_monotonic_in_sorted_raw_p_and_bounded():
    got = holm_adjust([0.01, 0.04, 0.03, np.nan])
    np.testing.assert_allclose(got[:3], [0.03, 0.06, 0.06])
    assert np.isnan(got[3])


def five_fold_predictions():
    expected = pd.DataFrame({"pid": list("abcde"), "date": pd.to_datetime(["2020-01-01"] * 5),
                             "fold": range(1, 6), "y_true": [10.] * 5})
    p = pd.concat([expected.assign(seed=s, y_pred=10. + s) for s in (0, 1, 2)], ignore_index=True)
    return p, expected


def test_complete_summary_and_partial_summary_have_distinct_contracts():
    p, expected = five_fold_predictions()
    row, nights, detail = summarize_group(p, expected, "catboost", "base", "sleep")
    assert row["mae"] == 1.
    assert row["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert row["n_nights"] == 5
    assert row["seed_mae_std"] == 1.
    assert len(detail) == 15
    partial, empty_nights, _ = summarize_group(p.iloc[:-1], expected, "catboost", "base", "sleep")
    assert partial["status"] == "partial"
    assert "mae" not in partial
    assert empty_nights.empty


def test_different_paired_nights_are_not_silently_intersected():
    p, _ = example_predictions()
    errors = aggregate_errors(p)
    with pytest.raises(ValueError):
        paired_wide({"base": errors, "new": errors.iloc[:-1]})


def test_final_participants_are_excluded_from_authoritative_truth():
    features = pd.DataFrame({"pid": ["a", "b"], "date": pd.to_datetime(["2020-01-01"] * 2),
                             "target_sleep_min": [100., 999.]})
    split = pd.DataFrame({"pid": ["a", "b"], "role": ["dev", "final"], "fold": [1, 0]})
    assert expected_nights(features, split, "sleep", "target_sleep_min")["pid"].tolist() == ["a"]


def test_gap_bins_count_missing_nights_not_elapsed_days():
    p, _ = five_fold_predictions()
    nights = aggregate_errors(p)
    features = nights[["pid", "date"]].assign(sleep_n28=[0, 1, 4, 8, 28],
                                                days_since_last=[np.nan, 1, 2, 7, 8])
    rows = subgroup_metrics(nights, features, "sleep", "catboost", "base")
    gaps = rows[rows.dimension == "gap"].set_index("bin")
    assert gaps.loc["missing_1-6", "n_nights"] == 2
    assert gaps.loc["missing_7+", "n_nights"] == 1
    features.loc[1, "days_since_last"] = 0
    with pytest.raises(ValueError):
        subgroup_metrics(nights, features, "sleep", "catboost", "base")


def test_partial_test_family_does_not_reduce_multiplicity_correction():
    observed = pd.DataFrame({"p_bootstrap_sign_approx": [0.01]})
    result = _adjust_all([observed], "fixed_family", 3)
    assert result.p_holm.item() == 0.03
    assert result.multiplicity_n_tests.item() == 3


def checkpoint_metadata(group, identity, fingerprints):
    from run_factorial import signature
    from src.forecast_eval import keys_hash
    features, settings = ["toy_lag1"], {"toy": True}
    return {"identity": list(identity), "input_fingerprints": fingerprints,
            "features": features, "settings": settings,
            "signature": signature(fingerprints, identity, features, settings),
            "final_test_scored": False, "n_rows": len(group), "n_people": int(group.pid.nunique()),
            "eval_keys_sha256": keys_hash(group), "mae": float((group.y_pred - group.y_true).abs().mean())}


@pytest.mark.parametrize("corruption", ["missing_meta", "identity", "inputs", "signature", "row_count", "mae"])
def test_checkpoint_provenance_corruption_is_rejected(tmp_path, monkeypatch, corruption):
    path = tmp_path / "catboost_base_sleep_f1_s0.parquet"
    path.touch()
    group = pd.DataFrame({"pid": ["a"], "date": pd.to_datetime(["2020-01-01"]),
                          "family": ["catboost"], "condition": ["base"], "target": ["sleep"],
                          "fold": [1], "seed": [0], "y_true": [10.], "y_pred": [8.]})
    fingerprints = {"features.parquet": "fixed", "run_factorial.py": "fixed-code"}
    metadata = checkpoint_metadata(group, ("catboost", "base", "sleep", 1, 0), fingerprints)
    if corruption == "identity":
        metadata["identity"][-1] = 1
    elif corruption == "inputs":
        metadata["input_fingerprints"] = {"features.parquet": "old"}
    elif corruption == "signature":
        metadata["signature"] = "wrong"
    elif corruption == "row_count":
        metadata["n_rows"] = 999
    elif corruption == "mae":
        metadata["mae"] = 999.
    if corruption != "missing_meta":
        path.with_suffix(".json").write_text(json.dumps(metadata))
    monkeypatch.setattr(pd, "read_parquet", lambda path: group.copy())
    with pytest.raises(ValueError):
        read_group(tmp_path, "catboost", "base", "sleep", fingerprints)


def test_partial_checkpoint_filename_is_never_consumed(tmp_path):
    (tmp_path / "catboost_base_sleep_f1_s0.partial.parquet").touch()
    assert read_group(tmp_path, "catboost", "base", "sleep", {}).empty


def test_analysis_outputs_complete_then_partial_run_without_scoring_partial(tmp_path, monkeypatch):
    import run_factorial
    artifacts = tmp_path / "artifacts"
    predictions = artifacts / "predictions"
    predictions.mkdir(parents=True)
    p, expected = five_fold_predictions()
    features = expected.rename(columns={"y_true": "target_sleep_min"}).drop(columns="fold")
    features = features.assign(sleep_n28=28, days_since_last=1)
    split = expected[["pid", "fold"]].assign(role="dev")
    split.to_csv(artifacts / "split.csv", index=False)
    (artifacts / "feature_spec.json").write_text(json.dumps({"targets": {"sleep": "target_sleep_min"}}))
    contents = {str(artifacts / "features.parquet"): features}
    fingerprints = {"features.parquet": "fixed-fixture"}
    monkeypatch.setattr(run_factorial, "input_fingerprints", lambda root: fingerprints)
    errors = dict(zip(CONDITIONS, [10, 9, 8, 9, 8, 8, 7, 7]))
    for condition in CONDITIONS:
        for (fold, seed), group in p.groupby(["fold", "seed"]):
            path = predictions / f"catboost_{condition}_sleep_f{fold}_s{seed}.parquet"
            path.touch()
            contents[str(path)] = group.assign(y_pred=group.y_true + errors[condition], family="catboost",
                                                condition=condition, target="sleep")
            identity = ("catboost", condition, "sleep", int(fold), int(seed))
            metadata = checkpoint_metadata(contents[str(path)], identity, fingerprints)
            path.with_suffix(".json").write_text(json.dumps(metadata))
    monkeypatch.setattr(pd, "read_parquet", lambda path: contents[str(path)].copy())
    metadata = run(tmp_path, ("catboost",), 100)
    assert metadata["complete"]
    summary = pd.read_csv(artifacts / "analysis/summary.csv")
    assert len(summary) == 8
    assert summary.status.eq("complete").all()
    addition = pd.read_csv(artifacts / "analysis/additivity.csv")
    assert addition.interaction.item() == -1.
    assert len(pd.read_csv(artifacts / "analysis/comparisons.csv")) == 13
    (predictions / "catboost_ABC_sleep_f5_s2.parquet").unlink()
    metadata = run(tmp_path, ("catboost",), 100)
    assert not metadata["complete"]
    summary = pd.read_csv(artifacts / "analysis/summary.csv")
    assert summary.loc[summary.condition == "ABC", "mae"].isna().all()
    assert (artifacts / "analysis/comparisons.csv").read_text().strip() == ""
