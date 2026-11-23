"""
run_rhythm.py — 방향 3: 화면 사용 리듬 지표가 섞은 비율 위에 예측을 더하는가

실행: python run_rhythm.py  (작업 위치는 이 폴더)
결과: results/rhythm.csv (비교 요약), results/rhythm_l2.csv (라벨별로 고른 ℓ2 세기),
      results/rhythm_features.csv (라벨 있는 날의 지표 원값), results/rhythm_missing.csv

비교: 섞은 비율(a) 대 섞은 비율 + 리듬 지표(b). diff가 양수면 지표를 더한 쪽이 낫다.
지표 묶음은 둘이다.
- timing: onset_min, offset_min (그날의 시각)
- timing_sd: 위에 onset_sd7, offset_sd7 (최근 7일 흔들림)을 더한 것
빈칸은 사람별 표준화 뒤 0(그 사람의 평균)으로 채운다.
근거: LOV3 III절 "a missing minute is then set to zero, i.e., to the channel mean".
모델·ℓ2 선택·블록 수는 방향 2(run_sensor.py)와 같다.
"""
from __future__ import annotations

import logging

import pandas as pd

from coldstart import make_mixed_predictor
from compare import compare_predictors
from labels import TARGETS, load_labels
from rhythm import FEATURE_COLUMNS, TIMING_COLUMNS, features_for_days, missing_share, rhythm_features
from run_coverage import DATA_DIR, LABEL_PATH, RESULT_DIR, SEED
from run_sensor import L2_GRID, N_BLOCKS
from scoring import DEFAULT_N_BOOT
from sensor_model import make_sensor_predictor, select_l2, zscore_within_person
from split import assign_split

SCREEN_SENSOR = "mScreenStatus"
FEATURE_SETS = {"timing": TIMING_COLUMNS, "timing_sd": FEATURE_COLUMNS}

logger = logging.getLogger(__name__)


def build_features(split: pd.DataFrame) -> pd.DataFrame:
    records = pd.read_parquet(DATA_DIR / "data_items" / f"ch2025_{SCREEN_SENSOR}.parquet")
    aligned = features_for_days(rhythm_features(records), split)
    return split.assign(**{column: aligned[column] for column in FEATURE_COLUMNS})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = build_features(assign_split(load_labels(LABEL_PATH)))
    missing = (raw.groupby("part")[FEATURE_COLUMNS].apply(lambda rows: rows.isna().mean())
               .T.assign(all=missing_share(raw)))
    table = zscore_within_person(raw, FEATURE_COLUMNS)
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
    result.to_csv(RESULT_DIR / "rhythm.csv", index=False)
    l2_table.to_csv(RESULT_DIR / "rhythm_l2.csv", index=False)
    raw.to_csv(RESULT_DIR / "rhythm_features.csv", index=False)
    missing.to_csv(RESULT_DIR / "rhythm_missing.csv")
    columns = ["feature_set", "target", "logloss_a", "logloss_b", "diff", "low", "high",
               "auroc_a", "auroc_b", "b_wins"]
    logger.info("빈칸 비율\n%s", missing.round(3).to_string())
    logger.info("지표 원값 요약\n%s", raw[FEATURE_COLUMNS].describe().round(1).to_string())
    logger.info("%s", result[columns].round(4).to_string(index=False))
    logger.info("%s", l2_table.pivot(index="target", columns="feature_set", values="l2").to_string())
    logger.info("평균 log loss: %s",
                result.groupby("feature_set")[["logloss_a", "logloss_b"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
