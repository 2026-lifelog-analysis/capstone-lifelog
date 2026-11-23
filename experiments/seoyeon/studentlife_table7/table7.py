"""
table7.py — 사람 단위 요약, 사후 UCLA와의 상관, 논문 표 7과의 비교

논문: Wang 외 2014 (UbiComp), Table 7 "Correlations between automatic sensor data and loneliness scale"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# 특징 이름: (논문 r, 논문 p). 모두 사후(post) UCLA와의 상관
PAPER_TABLE7 = {
    "act_all": (-0.388, 0.018),
    "act_day": (-0.326, 0.049),
    "act_evening": (-0.464, 0.004),
    "dist_all": (-0.338, 0.044),
    "dist_day": (-0.336, 0.042),
    "indoor_day": (-0.332, 0.045),
}
FEATURE_LABELS = {
    "act_all": "활동 시간 (하루 전체)", "act_day": "활동 시간 (낮)", "act_evening": "활동 시간 (저녁)",
    "dist_all": "이동 거리 (하루 전체)", "dist_day": "이동 거리 (낮)", "indoor_day": "실내 이동 (낮)",
}
AGGREGATIONS = ("mean", "median")
MIN_PEOPLE = 3


def person_summary(daily: pd.DataFrame, cols: list[str], agg: str = "mean", min_bins: int = 0,
                   bins_col: str = "n_bins") -> pd.DataFrame:
    """일별 표(uid, date, 특징들)를 사람 단위로 요약한다. min_bins보다 적게 기록된 날은 뺀다."""
    if agg not in AGGREGATIONS:
        raise ValueError(f"agg는 {AGGREGATIONS} 중 하나여야 함: {agg}")
    kept = daily[daily[bins_col] >= min_bins] if min_bins > 0 and bins_col in daily else daily
    return kept.groupby("uid")[cols].agg(agg)


def correlate(features: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """사람 단위 특징(index=uid)과 점수(index=uid)의 Pearson 상관. 반환 열: feature, n, r, p"""
    rows = []
    for col in features.columns:
        joined = pd.concat([features[col], scores.rename("score")], axis=1, join="inner").dropna()
        n = len(joined)
        if n < MIN_PEOPLE or joined[col].nunique() < 2 or joined["score"].nunique() < 2:
            r, p = np.nan, np.nan
        else:
            result = stats.pearsonr(joined[col], joined["score"])
            r, p = float(result.statistic), float(result.pvalue)
        rows.append({"feature": col, "n": n, "r": r, "p": p})
    return pd.DataFrame(rows, columns=["feature", "n", "r", "p"])


def compare_with_paper(corr: pd.DataFrame, tolerance: float = 0.10) -> pd.DataFrame:
    """상관표에 논문 값, 차이(우리 − 논문), 부호 일치, 허용 범위 안 여부를 붙인다."""
    paper_r = corr["feature"].map(lambda f: PAPER_TABLE7[f][0])
    paper_p = corr["feature"].map(lambda f: PAPER_TABLE7[f][1])
    diff = corr["r"] - paper_r
    return corr.assign(paper_r=paper_r, paper_p=paper_p, diff=diff,
                       same_sign=np.sign(corr["r"]) == np.sign(paper_r),
                       within_tol=diff.abs() <= tolerance)


def sensitivity_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """설정 조합별 결과(feature, r)를 특징마다 범위로 줄인다. r을 못 구한 조합은 세지 않는다."""
    grouped = runs.dropna(subset=["r"]).groupby("feature", sort=False)["r"]
    return pd.DataFrame({"r_min": grouped.min(), "r_median": grouped.median(), "r_max": grouped.max(),
                         "n_runs": grouped.size(),
                         "share_negative": grouped.apply(lambda s: float((s < 0).mean()))}).reset_index()
