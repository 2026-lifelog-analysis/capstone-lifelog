"""승호의 네 CUSUM 단계를 시간순 보정·시험에 연결하는 비교용 변형입니다.

원래 점수식, 28일 창, 최소 14일, ±8 절단, 결측·달력 간격 처리와 k를
유지합니다. 차이는 h의 경보율을 보정 기간에서만 측정한다는 점입니다.
같은 날 집단 중앙값은 당일 전체 표본을 사용하므로 사전 예측이 아닙니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import baseline as bl

K_VALUES = {'sh_base': .5, 'sh_dow': .5, 'sh_common': .5, 'sh_final': .25}


def directional_scores(main_df: pd.DataFrame) -> pd.DataFrame:
    """원 승호 구현과 동등한, 걸음 감소 방향 CUSUM 전 점수 네 열."""
    grouped = main_df.groupby('date')['steps']
    median = grouped.transform('median').where(grouped.transform('count') >= bl.GROUP_MIN_N)
    anchored = main_df.assign(_anchored=main_df['steps'].astype(float) - median)
    base = -bl.personal_z(main_df, 'steps')
    dow = -bl.personal_z(main_df, 'steps', dow_adjust=True)
    common = -bl.personal_z(anchored, '_anchored', dow_adjust=True)
    return pd.DataFrame({'sh_base': base, 'sh_dow': dow,
                         'sh_common': common, 'sh_final': common}, index=main_df.index)


def _validate(df, scores):
    if not df.index.is_unique or not df.index.equals(scores.index):
        raise ValueError('Scores must have the same unique index as input data.')
    if df.duplicated(['pid', 'date']).any():
        raise ValueError('Duplicate person dates are not supported.')
    if set(K_VALUES) - set(scores):
        raise ValueError('All four directional score columns are required.')
    if np.isinf(scores[list(K_VALUES)].to_numpy(float)).any():
        raise ValueError('Infinite directional scores are not supported.')


def fit_h(df: pd.DataFrame, badscores: pd.DataFrame, calibration_mask,
          rate: float, iters: int = 30) -> dict[str, float]:
    """보정 인일 경보율만 사용해 h를 원 방식의 30회 이분 탐색으로 결정합니다.

    보정 시작 전 누적 상태는 보존합니다. 시험값이 탐색 상한에 들어가는 것도
    막기 위해 마지막 보정일 뒤의 행은 계산에서 제외합니다. 모든 행을 보정에
    포함하면 원 bl.cusum_threshold와 동일합니다. 목표율의 정확 달성을 보장하는
    최적화는 아니며, 원 코드와 같은 이분 탐색 규칙을 유지한 설계입니다.
    """
    _validate(df, badscores)
    if not 0 < rate < 1 or iters < 1:
        raise ValueError('rate must be between zero and one; iters must be positive.')
    if isinstance(calibration_mask, pd.Series) and not calibration_mask.index.equals(df.index):
        raise ValueError('calibration mask index must match the input.')
    mask = np.asarray(calibration_mask, dtype=bool)
    if mask.shape != (len(df),) or not mask.any():
        raise ValueError('Nonempty calibration mask required.')
    until = df.loc[mask, 'date'].max()
    history = df['date'] <= until
    selected = pd.Series(mask, index=df.index)
    thresholds = {}
    for name, k in K_VALUES.items():
        data = df.loc[history, ['pid', 'date']].assign(_score=badscores.loc[history, name])
        idx, parts = bl._cusum_prepare(data, '_score')
        chosen = selected.loc[idx].to_numpy(bool)
        initial = bl._cusum_run(parts, k, None)
        valid = chosen & np.isfinite(initial)
        if not valid.any():
            raise ValueError(f'No finite calibration scores for {name}.')
        lo, hi = 0., float(np.nanmax(initial))
        for _ in range(iters):
            mid = (lo + hi) / 2
            result = bl._cusum_run(parts, k, mid)
            if (result[valid] > mid).mean() > rate:
                lo = mid
            else:
                hi = mid
        thresholds[name] = hi
    return thresholds


def score_with_h(df: pd.DataFrame, badscores: pd.DataFrame,
                 thresholds: dict[str, float]) -> pd.DataFrame:
    """전체 이력의 상태를 이어 계산하며, 외부 경보도 동일한 h의 엄격 초과입니다."""
    _validate(df, badscores)
    if set(thresholds) != set(K_VALUES) or any(
            not np.isfinite(h) or h < 0 for h in thresholds.values()):
        raise ValueError('Four finite nonnegative thresholds are required.')
    return pd.DataFrame({name: bl.cusum(df.assign(_score=badscores[name]), '_score',
                                      k=k, h=thresholds[name])
                         for name, k in K_VALUES.items()}, index=df.index)
