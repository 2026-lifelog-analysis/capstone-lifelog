import numpy as np
import pandas as pd
import pytest

from coldstart import make_mixed_predictor
from sensor_model import (assign_temporal_blocks, make_sensor_predictor, select_l2,
                          zscore_within_person)

L2_GRID = (0.01, 1.0, 1e6)


def make_split(effect: float, n_people: int = 8, days: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(n_people):
        base = rng.uniform(-0.5, 0.5)
        for day in range(days):
            x = rng.normal()
            prob = 1 / (1 + np.exp(-(base + effect * x)))
            rows.append({"subject_id": f"id{person:02d}", "part": "train" if day < 60 else "test",
                         "feat": 5.0 * person + x, "Q1": int(rng.random() < prob)})
    return pd.DataFrame(rows)


def test_zscore_removes_person_level_and_scale():
    table = make_split(0.0)

    result = zscore_within_person(table, ["feat"])

    stats = result.groupby("subject_id")["feat"].agg(["mean", "std"])
    assert stats["mean"].abs().max() < 1e-9
    assert stats["std"].round(6).eq(1.0).all()
    assert table["feat"].mean() > 1


def test_zscore_of_constant_feature_is_zero():
    table = make_split(0.0).assign(feat=3.0)

    assert (zscore_within_person(table, ["feat"])["feat"] == 0).all()


def test_temporal_blocks_are_contiguous_and_balanced_per_person():
    train = make_split(0.0).query("part == 'train'")

    blocks = assign_temporal_blocks(train, n_blocks=3)

    first = blocks[train["subject_id"] == "id00"].tolist()
    assert first == sorted(first)
    assert pd.Series(first).value_counts().tolist() == [20, 20, 20]


def test_select_l2_prefers_weak_penalty_when_feature_is_informative():
    split = zscore_within_person(make_split(2.0), ["feat"])

    chosen = select_l2(split, "Q1", ["feat"], L2_GRID, n_blocks=3)

    assert chosen < 1e6


def test_sensor_predictor_beats_base_when_feature_is_informative():
    split = zscore_within_person(make_split(2.0), ["feat"])

    base = make_mixed_predictor(None)(split, "Q1")
    model = make_sensor_predictor(["feat"], L2_GRID, n_blocks=3)(split, "Q1")

    def loss(pred: pd.DataFrame) -> float:
        y, p = pred["y_true"], pred["prob"]
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    assert loss(model) < loss(base) - 0.05
    assert model.index.equals(base.index)


def test_sensor_predictor_stays_close_to_base_when_feature_is_noise():
    split = zscore_within_person(make_split(0.0), ["feat"])

    base = make_mixed_predictor(None)(split, "Q1")
    model = make_sensor_predictor(["feat"], L2_GRID, n_blocks=3)(split, "Q1")

    assert (model["prob"] - base["prob"]).abs().max() < 0.15


def test_empty_feature_list_is_rejected():
    with pytest.raises(ValueError):
        make_sensor_predictor([], L2_GRID, n_blocks=3)
