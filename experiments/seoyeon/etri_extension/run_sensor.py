"""
run_sensor.py — 방향 2의 2단계: 기록률 피처가 섞은 비율 위에 예측을 더하는가

실행: python run_sensor.py  (작업 위치는 이 폴더)
결과: results/sensor.csv (비교 요약), results/sensor_l2.csv (라벨별로 고른 ℓ2 세기)

비교: 섞은 비율(a) 대 섞은 비율 + 피처(b). diff가 양수면 피처를 더한 쪽이 낫다.
피처 묶음은 둘이다.
- coverage: 1분 간격 센서 7종의 하루 기록률
- coverage_start: 위에 기록 시작일 표시(전날 폰 기록이 없는 날)를 더한 것
밤 구간 기록률은 쓰지 않는다. 낮은 값 71일 중 67일이 다음 날 자료가 없는 날이어서
행동이 아니라 수집 구조를 재기 때문이다(results/coverage.csv로 2026-11-20 확인).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from coldstart import make_mixed_predictor
from compare import compare_predictors
from record_coverage import coverage_for_days
from labels import TARGETS, load_labels
from run_coverage import DATA_DIR, LABEL_PATH, MINUTE_SENSORS, RESULT_DIR, SEED, ZERO_COVERAGE
from scoring import DEFAULT_N_BOOT
from sensor_model import make_sensor_predictor, select_l2, zscore_within_person
from split import assign_split

N_BLOCKS = 3
# ℓ2 후보값에는 근거가 없다. 마지막 값은 "피처를 쓰지 않음"에 해당하도록 크게 두었다.
L2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 1e6)
PHONE_SENSOR = "mActivity"
START_DAY = "start_day"
COVERAGE_FEATURES = [f"{sensor}_day" for sensor in MINUTE_SENSORS]
FEATURE_SETS = {"coverage": COVERAGE_FEATURES, "coverage_start": COVERAGE_FEATURES + [START_DAY]}

logger = logging.getLogger(__name__)


def build_features(split: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for sensor in MINUTE_SENSORS:
        records = pd.read_parquet(Path(DATA_DIR) / "data_items" / f"ch2025_{sensor}.parquet",
                                  columns=["subject_id", "timestamp"])
        columns[f"{sensor}_day"] = coverage_for_days(records, split, "day")
        if sensor == PHONE_SENSOR:
            previous = split.assign(lifelog_date=split["lifelog_date"] - pd.Timedelta(days=1))
            before = coverage_for_days(records, previous, "day")
            columns[START_DAY] = (before == ZERO_COVERAGE).astype(float)
    return split.assign(**columns)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = build_features(assign_split(load_labels(LABEL_PATH)))
    table = zscore_within_person(raw, FEATURE_SETS["coverage_start"])
    base = make_mixed_predictor(None)
    frames, chosen = [], []
    for name, features in FEATURE_SETS.items():
        model = make_sensor_predictor(features, L2_GRID, N_BLOCKS)
        frames.append(compare_predictors(table, base, model, n_boot=DEFAULT_N_BOOT, seed=SEED)
                      .assign(feature_set=name))
        chosen += [{"feature_set": name, "target": target,
                    "l2": select_l2(table, target, features, L2_GRID, N_BLOCKS)}
                   for target in TARGETS]
    result = pd.concat(frames, ignore_index=True)
    l2_table = pd.DataFrame(chosen)
    RESULT_DIR.mkdir(exist_ok=True)
    result.to_csv(RESULT_DIR / "sensor.csv", index=False)
    l2_table.to_csv(RESULT_DIR / "sensor_l2.csv", index=False)
    columns = ["feature_set", "target", "logloss_a", "logloss_b", "diff", "low", "high",
               "auroc_a", "auroc_b", "b_wins"]
    logger.info("%s", result[columns].round(4).to_string(index=False))
    logger.info("%s", l2_table.pivot(index="target", columns="feature_set", values="l2").to_string())
    logger.info("평균 log loss: %s",
                result.groupby("feature_set")[["logloss_a", "logloss_b"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
