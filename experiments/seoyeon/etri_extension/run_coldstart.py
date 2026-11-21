"""
run_coldstart.py — 방향 1 실행: 개인 기록 일수별로 전체 비율·개인 비율·섞는 방식을 비교한다

실행: python run_coldstart.py  (작업 위치는 이 폴더)
결과: results/coldstart.csv (기록 일수 × 비교 쌍 × 라벨), results/coldstart_strength.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from coldstart import (_prior_strength_without, leave_one_out_global_prediction,
                       make_mixed_predictor, make_personal_predictor)
from compare import compare_predictors
from labels import TARGETS, load_labels
from scoring import DEFAULT_N_BOOT
from shrinkage import strength_from_icc
from split import assign_split

HERE = Path(__file__).resolve().parent
LABEL_PATH = HERE.parents[2] / "data" / "etri2024" / "ch2025_metrics_train.csv"
RESULT_DIR = HERE / "results"
SEED = 0
REVEALED_DAYS: tuple[int | None, ...] = (3, 7, 14, None)
ALL_DAYS_NAME = "all"

logger = logging.getLogger(__name__)


# 춘자 논문(State, not Trait, ICTC 2026 C6-1) 표 1의 ICC ρ_t 열.
# 일원 변량효과 분산분석으로 학습 450밤 전체에서 추정한 값이다(논문 IV절).
# 우리 평가 날의 라벨도 그 450밤에 들어 있으므로, 이 값에는 평가 구간 정보가 요약된 형태로 섞여 있다.
PAPER_ICC = {"Q1": 0.092, "Q2": 0.065, "Q3": 0.047, "S1": 0.131, "S2": 0.179, "S3": 0.245}
PAPER_STRENGTHS = {target: strength_from_icc(icc) for target, icc in PAPER_ICC.items()}


def _comparisons(k: int | None) -> dict[str, tuple]:
    """비교 쌍 이름 → (a, b). 결과의 diff가 양수면 b가 낫다.

    mixed는 m을 나머지 사람들로 추정한 것, mixed_icc는 m을 논문 ICC로 고정한 것이다.
    """
    global_rate = leave_one_out_global_prediction
    personal = make_personal_predictor(k)
    mixed = make_mixed_predictor(k)
    mixed_icc = make_mixed_predictor(k, strengths=PAPER_STRENGTHS)
    return {"global_vs_personal": (global_rate, personal),
            "global_vs_mixed": (global_rate, mixed),
            "personal_vs_mixed": (personal, mixed),
            "global_vs_mixed_icc": (global_rate, mixed_icc),
            "personal_vs_mixed_icc": (personal, mixed_icc)}


def run_all(split: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    frames = [compare_predictors(split, a, b, n_boot=n_boot, seed=SEED)
              .assign(revealed_days=ALL_DAYS_NAME if k is None else str(k), pair=name)
              for k in REVEALED_DAYS for name, (a, b) in _comparisons(k).items()]
    return pd.concat(frames, ignore_index=True)


def strength_table(split: pd.DataFrame) -> pd.DataFrame:
    """라벨별·사람별로 추정된 사전 강도 m (그 사람을 뺀 나머지로 추정)."""
    train = split[split["part"] == "train"]
    rows = [{"target": target, "subject_id": person,
             "strength": _prior_strength_without(train, target, person)}
            for target in TARGETS for person in sorted(train["subject_id"].unique())]
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    split = assign_split(load_labels(LABEL_PATH))
    result = run_all(split, DEFAULT_N_BOOT)
    strength = strength_table(split)
    RESULT_DIR.mkdir(exist_ok=True)
    result.to_csv(RESULT_DIR / "coldstart.csv", index=False)
    strength.to_csv(RESULT_DIR / "coldstart_strength.csv", index=False)
    columns = ["revealed_days", "pair", "target", "logloss_a", "logloss_b", "diff", "low", "high",
               "clipped_days_a", "clipped_days_b", "b_wins"]
    logger.info("%s", result[columns].round(4).to_string(index=False))
    logger.info("%s", strength.groupby("target")["strength"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
