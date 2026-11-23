"""
scoring.py — UCLA 외로움 척도(3판, 20문항) 채점

StudentLife의 survey/LonelinessScale.csv 형식(uid, type, 문항 20개)을 받아 총점(20~80)을 낸다.
응답은 Never=1, Rarely=2, Sometimes=3, Often=4이고, 긍정 문항 9개는 역채점한다.
문항이 하나라도 비면 총점은 NaN이다(일부만 더하지 않는다).
"""
from __future__ import annotations

import pandas as pd

ANSWER_POINTS = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
REVERSE_ITEMS = (1, 5, 6, 9, 10, 15, 16, 19, 20)   # 1부터 센 문항 번호
N_ITEMS = 20
MAX_POINT = 4


def ucla_scores(survey: pd.DataFrame) -> pd.DataFrame:
    """설문 표에서 사람·시점별 UCLA 총점을 계산한다. 반환 열: uid, type, score."""
    items = survey.iloc[:, 2:2 + N_ITEMS]
    if items.shape[1] != N_ITEMS:
        raise ValueError(f"문항 열이 {N_ITEMS}개여야 하는데 {items.shape[1]}개임")
    answers = pd.Series(items.to_numpy().ravel()).dropna().unique()
    unknown = sorted(set(answers) - set(ANSWER_POINTS))
    if unknown:
        raise ValueError(f"알 수 없는 응답 값: {unknown}")

    points = items.apply(lambda col: col.map(ANSWER_POINTS).astype("float64"))
    reverse_cols = [items.columns[i - 1] for i in REVERSE_ITEMS]
    reversed_points = (MAX_POINT + 1) - points[reverse_cols]
    scored = pd.concat([points.drop(columns=reverse_cols), reversed_points], axis=1)
    total = scored.sum(axis=1, min_count=N_ITEMS)
    return pd.DataFrame({"uid": survey["uid"].to_numpy(), "type": survey["type"].to_numpy(),
                         "score": total.to_numpy()})
