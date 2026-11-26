"""바꾼 판(sparse_onboarding.py)이 기본값에서 승호님 원본과 같은 결과를 내는지, 조건을 바꾸면 의도대로 바뀌는지.

실제 자료 없이 가짜 사람-일 표로 확인한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
ORIGINAL = HERE / "seungho_original" / "sparse_survey.py"
MODIFIED = HERE / "sparse_onboarding.py"
ITEMS = ("sleep_quality", "physical_energy")


def fake_daily(path: Path, n_people: int = 8, n_days: int = 120, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_people):
        start = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        level = rng.normal(0, 0.5)
        for d in range(n_days):
            answered = rng.random() < 0.8
            row = {"pid": f"p{i:02d}", "date": start + pd.Timedelta(days=d), "dow": (start + pd.Timedelta(days=d)).day_of_week,
                   "a_steps": rng.normal(8000, 2000), "s_tst_min": rng.normal(420, 40), "sp_tst_min": rng.normal(420, 40),
                   "z_steps": rng.normal(), "z_tst": rng.normal()}
            for item in ITEMS:
                row[item] = float(np.clip(np.round(3 + level - 0.004 * d + rng.normal(0, 1)), 1, 5)) if answered else np.nan
            row["tst_label"] = float(rng.random() < 0.4) if answered else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    for item in ITEMS:
        mean = df.groupby("pid")[item].transform("mean")
        df[f"{item}_label"] = np.where(df[item].notna(), (df[item] > mean).astype(float), np.nan)
    df.to_pickle(path)


try:  # macOS에서 LightGBM이 찾는 libomp (scikit-learn에 들어 있는 것)
    import sklearn
    OMP_DIR = Path(sklearn.__file__).parent / ".dylibs"
except ImportError:
    OMP_DIR = Path("/nonexistent")


def run(script: Path, daily: Path, out: Path, **env: str) -> Path:
    full_env = {**os.environ, "RQ_TARGETS": "", **env}
    if OMP_DIR.is_dir():
        full_env["DYLD_FALLBACK_LIBRARY_PATH"] = str(OMP_DIR)
    subprocess.run([sys.executable, str(script), str(daily), str(out)], check=True, env=full_env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def daily(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "daily.pkl"
    fake_daily(path)
    return path


@pytest.fixture(scope="module")
def original(daily, tmp_path_factory) -> Path:
    return run(ORIGINAL, daily, tmp_path_factory.mktemp("orig"))


def table(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "rq3_table.csv")


def test_default_settings_reproduce_original_exactly(daily, original, tmp_path):
    modified = run(MODIFIED, daily, tmp_path / "mod")

    pd.testing.assert_frame_equal(table(original), table(modified))
    pd.testing.assert_frame_equal(pd.read_csv(original / "rq3_diffs.csv"), pd.read_csv(modified / "rq3_diffs.csv"))


def test_short_onboarding_keeps_the_same_scored_days(daily, original, tmp_path):
    short = run(MODIFIED, daily, tmp_path / "short", ONBOARD="14", LABEL="half")

    a = pd.read_pickle(original / "rq3_preds.pkl").query("scored")
    b = pd.read_pickle(short / "rq3_preds.pkl").query("scored")
    key = ["pid", "date", "target", "budget", "method"]
    assert set(map(tuple, a[key].values)) == set(map(tuple, b[key].values))


def test_fixed_label_keeps_scored_labels_identical(daily, original, tmp_path):
    short = run(MODIFIED, daily, tmp_path / "fixed", ONBOARD="14", LABEL="half")

    key = ["pid", "date", "target", "budget", "method"]
    a = pd.read_pickle(original / "rq3_preds.pkl").query("scored").set_index(key)["y"].sort_index()
    b = pd.read_pickle(short / "rq3_preds.pkl").query("scored").set_index(key)["y"].sort_index()
    pd.testing.assert_series_equal(a, b)


def test_short_onboarding_changes_b1_scores(daily, original, tmp_path):
    short = run(MODIFIED, daily, tmp_path / "s14", ONBOARD="14", LABEL="half")

    assert not np.allclose(table(original)["B1"], table(short)["B1"])


def test_short_onboarding_reveals_fewer_labels(daily, original, tmp_path):
    short = run(MODIFIED, daily, tmp_path / "vis", ONBOARD="14", LABEL="half")

    def visible(out: Path) -> int:
        p = pd.read_pickle(out / "rq3_preds.pkl")
        return int(p.query("budget == 'k7' and method == 'P1'")["visible"].sum())

    assert visible(short) < visible(original)


def test_conditions_are_recorded(daily, tmp_path):
    out = run(MODIFIED, daily, tmp_path / "rec", ONBOARD="28", LABEL="onboard", SKIP="7", SHRINK="1")

    cond = json.loads((out / "rq3_summary.json").read_text())["conditions"]
    assert cond == {"ONBOARD": "28", "LABEL": "onboard", "SKIP": 7, "SHRINK": True}


def test_invalid_condition_is_rejected(daily, tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run(MODIFIED, daily, tmp_path / "bad", ONBOARD="half", LABEL="whatever")
