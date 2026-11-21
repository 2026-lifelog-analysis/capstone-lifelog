"""
run_coverage.py — 방향 2의 1단계: 센서별 하루 기록률과, 기록률·라벨의 사람 내 연관

실행: python run_coverage.py  (작업 위치는 이 폴더)
결과: results/coverage.csv (라벨 있는 날 × 센서·구간 기록률)
      results/coverage_summary.csv (센서·구간별 기록률 분포)
      results/coverage_association.csv (학습 구간에서의 사람 내 상관)

연관은 학습 구간에서만 계산한다. 평가 구간은 뒤의 예측 실험을 위해 남겨 둔다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from association import within_person_correlation
from record_coverage import WINDOWS, coverage_for_days
from labels import TARGETS, load_labels
from scoring import DEFAULT_N_BOOT, holm_adjust
from split import assign_split

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[2] / "data" / "etri2024"
LABEL_PATH = DATA_DIR / "ch2025_metrics_train.csv"
RESULT_DIR = HERE / "results"
SEED = 0
# 기록 간격 중앙값이 60초인 센서 7종 (2026-11-20 확인).
MINUTE_SENSORS = ("mACStatus", "mActivity", "mScreenStatus", "mGps", "wHr", "wLight", "wPedo")
ZERO_COVERAGE = 0.0

logger = logging.getLogger(__name__)


def build_coverage(split: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for sensor in MINUTE_SENSORS:
        records = pd.read_parquet(DATA_DIR / "data_items" / f"ch2025_{sensor}.parquet",
                                  columns=["subject_id", "timestamp"])
        for window in WINDOWS:
            columns[f"{sensor}_{window}"] = coverage_for_days(records, split, window)
    return split.assign(**columns)


def summarize(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "mean": table[f].mean(), "median": table[f].median(),
             "p10": table[f].quantile(0.1), "zero_days": int((table[f] == ZERO_COVERAGE).sum()),
             "zero_share": float((table[f] == ZERO_COVERAGE).mean())} for f in features]
    return pd.DataFrame(rows)


def associate(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "target": t,
             **within_person_correlation(train, f, t, n_boot=DEFAULT_N_BOOT, seed=SEED)}
            for f in features for t in TARGETS]
    result = pd.DataFrame(rows)
    tested = result.dropna(subset=["p_value"])
    adjusted = holm_adjust({(row.feature, row.target): row.p_value for row in tested.itertuples()})
    keys = list(zip(result["feature"], result["target"]))
    return result.assign(p_holm=[adjusted.get(key, float("nan")) for key in keys],
                         interval_excludes_zero=(result["low"] > 0) | (result["high"] < 0))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    split = assign_split(load_labels(LABEL_PATH))
    table = build_coverage(split)
    features = [f"{sensor}_{window}" for sensor in MINUTE_SENSORS for window in WINDOWS]
    summary = summarize(table, features)
    association = associate(table[table["part"] == "train"], features)
    RESULT_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULT_DIR / "coverage.csv", index=False)
    summary.to_csv(RESULT_DIR / "coverage_summary.csv", index=False)
    association.to_csv(RESULT_DIR / "coverage_association.csv", index=False)
    logger.info("%s", summary.round(3).to_string(index=False))
    hits = association[association["interval_excludes_zero"]]
    logger.info("%s", hits.round(3).to_string(index=False))
    logger.info("비교 %d개 중 구간이 0을 벗어난 것 %d개, Holm 보정 후 0.05 미만 %d개",
                len(association), len(hits), int((association["p_holm"] < 0.05).sum()))


if __name__ == "__main__":
    main()
