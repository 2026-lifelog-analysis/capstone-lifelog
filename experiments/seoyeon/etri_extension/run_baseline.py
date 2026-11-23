"""
run_baseline.py — 공통 기반 실행: 분할을 만들고 B0(전체 비율)과 B1(고정 개인 비율)을 비교한다

실행: python run_baseline.py  (작업 위치는 이 폴더)
결과: results/split.csv, results/split_summary.csv, results/b0_vs_b1.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from baselines import global_rate_prediction, personal_rate_prediction
from compare import compare_predictors
from labels import TARGETS, load_labels
from scoring import DEFAULT_N_BOOT
from split import assign_split

HERE = Path(__file__).resolve().parent
LABEL_PATH = HERE.parents[2] / "data" / "etri2024" / "ch2025_metrics_train.csv"
RESULT_DIR = HERE / "results"
SEED = 0

logger = logging.getLogger(__name__)


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    """사람별 학습·간격·평가 날 수와 학습 구간 양성 비율."""
    counts = split.groupby(["subject_id", "part"]).size().unstack(fill_value=0)
    train_rates = split[split["part"] == "train"].groupby("subject_id")[list(TARGETS)].mean()
    return counts.join(train_rates.add_prefix("train_rate_")).reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    split = assign_split(load_labels(LABEL_PATH))
    summary = compare_predictors(split, global_rate_prediction, personal_rate_prediction,
                                 n_boot=DEFAULT_N_BOOT, seed=SEED)
    RESULT_DIR.mkdir(exist_ok=True)
    split.to_csv(RESULT_DIR / "split.csv", index=False)
    split_summary(split).to_csv(RESULT_DIR / "split_summary.csv", index=False)
    summary.to_csv(RESULT_DIR / "b0_vs_b1.csv", index=False)
    logger.info("%s", split_summary(split).round(2).to_string(index=False))
    logger.info("%s", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
