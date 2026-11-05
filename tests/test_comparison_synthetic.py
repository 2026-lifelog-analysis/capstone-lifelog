"""시험 구간 합성 변화의 배치·분모·결측 보존을 검증합니다."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.seoyeon.comparison_synthetic import injection, schedule, segment_keys


@pytest.fixture
def main():
    return pd.DataFrame({
        "pid": np.repeat(["a", "b"], 90),
        "date": list(pd.date_range("2020-01-01", periods=90)) * 2,
        "steps": 1000., "sleep_min": 400., "bed_min": 1400.,
    })


@pytest.fixture
def keys(main):
    return main[["pid", "date"]].rename(columns={"pid": "participant"})


@pytest.fixture
def segments():
    return pd.DataFrame({"pid": ["a"], "start": [pd.Timestamp("2020-03-01")],
                         "end": [pd.Timestamp("2020-03-14")], "seed": [0]})


def test_schedule_is_deterministic_independent_of_input_order(main, keys):
    split = {"test_start": "2020-03-01", "end": "2020-03-30"}
    actual = schedule(main, keys, split, seeds=range(20))
    shuffled = schedule(main.sample(frac=1, random_state=2),
                        keys.sample(frac=1, random_state=3), split, seeds=range(20))
    assert_frame_equal(actual, shuffled)
    assert len(actual) == 40
    assert not actual.duplicated(["seed", "pid"]).any()
    assert actual.start.ge(pd.Timestamp(split["test_start"])).all()
    assert actual.end.le(pd.Timestamp(split["end"])).all()
    assert actual.end.sub(actual.start).dt.days.eq(13).all()


def test_schedule_requires_prior_complete_days_and_common_days(main, keys):
    main.loc[main.pid.eq("b"), "steps"] = np.nan
    sparse = keys[keys.participant.eq("a") & keys.date.isin(pd.date_range("2020-03-01", periods=6))]
    assert schedule(main, sparse, {"test_start": "2020-03-01", "end": "2020-03-14"}).empty
    boundary = keys[keys.participant.eq("a") & keys.date.isin(pd.date_range("2020-03-01", periods=7))]
    actual = schedule(main, boundary, {"test_start": "2020-03-01", "end": "2020-03-14"}, seeds=[4])
    assert actual.pid.tolist() == ["a"]
    assert actual.start.tolist() == [pd.Timestamp("2020-03-01")]


def test_schedule_history_is_calendar_not_row_count(main, keys):
    sparse = main[main.date.lt("2020-02-01") | main.date.ge("2020-03-01")]
    assert schedule(sparse, keys, {"test_start": "2020-03-01", "end": "2020-03-14"}).empty


@pytest.mark.parametrize("scenario,first,last", [
    ("S1", (700., 400., 1400.), (700., 400., 1400.)),
    ("S2", (1000. * (1-.3/14), 400., 1400.), (700., 400., 1400.)),
    ("S3", (1000. * (1-.15/14), 400., 1400.), (850., 400., 1400.)),
    ("S4", (1000., 400., 1520.), (1000., 400., 1520.)),
    ("S5", (1000. * (1-.2/14), 400. * (1-.1/14), 1400.+60/14), (800., 360., 1460.)),
])
def test_injection_definitions_and_source_preservation(main, segments, scenario, first, last):
    before = main.copy(deep=True)
    got = injection(main, segments, scenario)
    features = ["steps", "sleep_min", "bed_min"]
    a = got.pid.eq("a")
    np.testing.assert_allclose(got.loc[a & got.date.eq("2020-03-01"), features].iloc[0], first)
    np.testing.assert_allclose(got.loc[a & got.date.eq("2020-03-14"), features].iloc[0], last)
    selected = a & got.date.between("2020-03-01", "2020-03-14")
    assert_frame_equal(got.loc[~selected], main.loc[~selected])
    assert_frame_equal(main, before)


def test_injection_preserves_missing_and_nonconsecutive_day_offsets(main, segments):
    main.loc[main.pid.eq("a") & main.date.eq("2020-03-01"), "steps"] = np.nan
    main = main.loc[~(main.pid.eq("a") & main.date.eq("2020-03-02"))].copy()
    got = injection(main, segments, "S2")
    selected = got.pid.eq("a")
    assert np.isnan(got.loc[selected & got.date.eq("2020-03-01"), "steps"].iloc[0])
    assert got.loc[selected & got.date.eq("2020-03-03"), "steps"].iloc[0] == 1000 * (1-.3*3/14)


def test_segment_keys_use_fixed_common_denominator(main, keys, segments):
    common = keys[~keys.date.eq("2020-03-03")]
    got = segment_keys(main, segments, common)
    assert len(got) == 13
    assert got.participant.eq("a").all()
    assert got.day_offset.tolist() == [0, 1, *range(3, 14)]
    assert not got.duplicated(["participant", "date"]).any()
    assert got.seed.eq(0).all()


def test_multiple_seeds_cannot_be_injected_together(main, segments):
    two = pd.concat([segments, segments.assign(seed=1)], ignore_index=True)
    with pytest.raises(ValueError, match="single seed"):
        injection(main, two, "S1")


def test_unknown_scenario_is_explicit_failure(main, segments):
    with pytest.raises(ValueError, match="Unknown scenario"):
        injection(main, segments, "S6")


def test_duplicate_common_key_rejected(main, keys):
    duplicate = pd.concat([keys, keys.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate common"):
        schedule(main, duplicate, {"test_start": "2020-03-01", "end": "2020-03-30"})


def test_selected_participant_recomputation_matches_full_group_reference(main, segments):
    from experiments.seoyeon import comparison_models as cm
    from experiments.seoyeon import comparison_synthetic as cs
    offset = np.tile(np.arange(90), 2)
    main = main.assign(steps=1000.+offset * 11., sleep_min=300.+offset % 29,
                       bed_min=1400.+offset % 19)
    daily = cm.adapt_main(main)
    split = {"development_end": "2020-02-05", "calibration_start": "2020-02-06"}
    _, baselines = cm.score_models(daily, **split)
    changed = injection(main, segments, "S5")
    full, _ = cm.score_models(cm.adapt_main(changed), **split)
    expected = full[full.participant.eq("a")].pivot(
        index=["participant", "date"], columns="model", values="score").reset_index()
    actual = cs._statistical_scores(changed, ["a"], split, baselines)
    assert_frame_equal(actual, expected, check_exact=True)


def test_run_jobs_preserves_input_and_uses_fixed_thresholds(tmp_path, main, keys, monkeypatch):
    import json
    from experiments.seoyeon import comparison_synthetic as cs
    from experiments.seoyeon import comparison_models as cm
    from experiments.seoyeon.comparison_seungho import K_VALUES
    main.to_csv(tmp_path / "main_daily.csv.gz", index=False)
    keys.assign(split="test").to_csv(tmp_path / "common_candidate_keys.csv.gz", index=False)
    (tmp_path / "split.json").write_text(json.dumps({"test_start": "2020-03-01", "end": "2020-03-14"}))
    pd.DataFrame({"model": ["group"]}).to_csv(tmp_path / "seoyeon_baselines.csv", index=False)
    thresholds = [{"id": label, "rate": rate, "thresholds": dict.fromkeys([*cm.MODELS, *K_VALUES], 3.)}
                  for label, rate in [("r05", .05), ("r033", 1/30)]]
    (tmp_path / "clean_thresholds_without_moment.json").write_text(json.dumps(thresholds))
    def fake_statistical(data, participants, split, baselines):
        return data[["pid", "date"]].rename(columns={"pid": "participant"}).assign(**dict.fromkeys(cm.MODELS, 1.))
    monkeypatch.setattr(cs, "_statistical_scores", fake_statistical)
    monkeypatch.setattr(cs, "directional_scores", lambda data: pd.DataFrame(1., index=data.index, columns=K_VALUES))
    seen = []
    def fake_sh(data, directional, h):
        assert len(data) == len(main)
        seen.append(h)
        return pd.DataFrame(2., index=data.index, columns=K_VALUES)
    monkeypatch.setattr(cs, "score_with_h", fake_sh)
    input_bytes = (tmp_path / "main_daily.csv.gz").read_bytes()
    result = cs.run_jobs(tmp_path, seeds=[0], scenarios=["S1"])
    assert result[0]["status"] == "passed"
    assert seen == [dict.fromkeys(K_VALUES, 3.), dict.fromkeys(K_VALUES, 3.)]
    assert (tmp_path / "main_daily.csv.gz").read_bytes() == input_bytes
    job = tmp_path / "synthetic" / "S1_seed00"
    assert (job / "canonical_daily.csv.gz").exists()
    for label in ["r05", "r033"]:
        scores = pd.read_csv(job / f"scores_{label}.csv.gz")
        assert len(scores) == 28
        assert scores[[*cm.MODELS, *K_VALUES]].notna().all().all()
    with pytest.raises(FileExistsError):
        cs.run_jobs(tmp_path, seeds=[0], scenarios=["S1"])
