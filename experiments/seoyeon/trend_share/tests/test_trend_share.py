import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trend_share import (MIN_ROWS, block_change, compare, half_change, person_arrays,  # noqa: E402
                         slope_summary)


def test_block_change_is_mean_absolute_difference_of_block_means():
    values = np.array([0.0] * 10 + [1.0] * 10 + [0.0] * 10)
    block = np.repeat([0, 1, 2], 10)

    assert block_change(values, block) == pytest.approx(1.0)


def test_block_change_drops_blocks_with_too_few_rows():
    values = np.array([0.0] * 10 + [1.0] * (MIN_ROWS - 1))
    block = np.repeat([0, 1], [10, MIN_ROWS - 1])

    assert np.isnan(block_change(values, block))


def test_half_change_compares_first_and_second_half():
    assert half_change(np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert half_change(np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)


def simulate(kind: str, n_people: int = 30, n_days: int = 168, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    day = np.arange(n_days, dtype=float)
    people = []
    for _ in range(n_people):
        if kind == "trend":
            level = 3 + rng.normal(0, 0.3) - 0.01 * day
        else:  # 28일마다 수준이 바뀌는 요동, 추세 없음
            level = 3 + np.repeat(rng.normal(0, 0.8, n_days // 28 + 1), 28)[:n_days]
        x = np.clip(np.round(level + rng.normal(0, 0.7, n_days)), 1, 5)
        people.append(person_arrays(x, day))
    return people


def test_pure_trend_is_explained_by_the_trend_only_baseline():
    rows = compare(simulate("trend"), np.random.default_rng(0))

    block = next(r for r in rows if r["scale"] == "라벨" and r["measure"] == "28일 블록 변화")
    assert not block["beyond_trend"]
    assert block["trend_share"] > 0.6


def test_block_level_swings_are_not_explained_by_trend():
    rows = compare(simulate("swing"), np.random.default_rng(0))

    block = next(r for r in rows if r["scale"] == "라벨" and r["measure"] == "28일 블록 변화")
    assert block["beyond_trend"]
    assert block["trend_share"] < 0.3


def test_slope_summary_detects_downward_trend():
    result = slope_summary(simulate("trend"), np.random.default_rng(0))

    assert result["slope_28d"] == pytest.approx(-0.28, abs=0.08)
    assert result["hi"] < 0
    assert result["share_down"] > 0.9
