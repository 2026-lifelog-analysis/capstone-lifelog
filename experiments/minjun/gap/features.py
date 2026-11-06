"""
④ 기록이 끊겼을 때 — 특징 정의 (fd.add_features에 넘기는 함수들)

'유효 기록 밤' = 공통 코드의 과거 입력 규칙과 같음:
  data.nights 중 overlap == False, asleep_invalid == False, known_at <= 그 줄의 forecast_at.
  밤 D 자신은 쓰지 않는다 (밤 D-1까지만).

  feat_gap    : n_valid7 (최근 달력 7밤 중 유효 기록 수), days_since_last (마지막 유효 밤 이후 일수, 상한 없음),
                last_sleep / last_bed / last_wake (마지막 유효 밤의 값)
  feat_recent : rec1_<목표> .. rec7_<목표> (유효 밤 중 k번째로 최근 것의 값. 날짜 간격 정보 없음)
  feat_long   : med28_<목표>, sd28_<목표> (달력 [D-28, D-1] 창의 유효 밤 값 중앙값·표준편차. sd는 2밤 이상)
"""
import numpy as np
import pandas as pd

COLS = {'sleep': 'sleep_min', 'bed': 'bedtime', 'wake': 'waketime'}
LAST = {'sleep': 'last_sleep', 'bed': 'last_bed', 'wake': 'last_wake'}
N_REC = 7


def _valid_nights(data):
    n = data.nights
    v = n[~n['overlap'].astype(bool) & ~n['asleep_invalid'].astype(bool)]
    return v[['pid', 'date', 'known_at', 'sleep_min', 'bedtime', 'waketime']].sort_values(['pid', 'date'])


def _per_person(valid):
    """사람별 (일 번호 배열, known_at 배열, {목표: 값 배열})."""
    return {p: (g['date'].values.astype('datetime64[D]').astype(np.int64), g['known_at'].to_numpy(),
                {k: g[c].to_numpy(dtype=float) for k, c in COLS.items()})
            for p, g in valid.groupby('pid')}


def _last_valid(table, valid, before):
    """각 줄에 대해 date < before 인 마지막 유효 밤 (merge_asof, 사람별)."""
    left = table[['pid']].assign(key=before.values, _i=np.arange(len(table))).sort_values('key')
    right = valid.rename(columns={'date': 'key'}).assign(night=lambda d: d['key']).sort_values('key')
    m = pd.merge_asof(left, right, on='key', by='pid', allow_exact_matches=False, direction='backward')
    return m.sort_values('_i').set_index(table.index)


def feat_gap(data):
    t = data.table
    valid = _valid_nights(data)
    out = pd.DataFrame(index=t.index)
    out['n_valid7'] = t[[f'sleep_lag{k}' for k in range(1, 8)]].notna().sum(axis=1).astype(float)
    # 마지막 유효 밤: date < D. 그 밤이 아직 확정 안 됐으면(known_at > forecast_at) 그 이전 밤으로
    last = _last_valid(t, valid, t['date'])
    late = last['known_at'].notna() & (last['known_at'] > t['forecast_at'])
    if late.any():
        prev = _last_valid(t.loc[late], valid, last.loc[late, 'night'])
        last.loc[late, :] = prev[last.columns]
    assert not (last['known_at'] > t['forecast_at']).any()
    out['days_since_last'] = (t['date'] - last['night']).dt.days.astype(float)     # 기록 없음 = NaN
    for k, c in COLS.items():
        out[LAST[k]] = last[c].astype(float)
    return out


def feat_recent(data):
    t = data.table
    arrs = _per_person(_valid_nights(data))
    res = {f'rec{j}_{k}': np.full(len(t), np.nan) for k in COLS for j in range(1, N_REC + 1)}
    d0 = t['date'].values.astype('datetime64[D]').astype(np.int64)
    for i, (p, d, f) in enumerate(zip(t['pid'].to_numpy(), d0, t['forecast_at'].to_numpy())):
        if p not in arrs:
            continue
        day, known, vals = arrs[p]
        hi = np.searchsorted(day, d, 'left')            # 밤 D 이전
        lo = max(0, hi - N_REC - 1)                     # 확정 안 된 밤 1개를 빼도 7개 남게 여유 1
        idx = np.arange(lo, hi)[known[lo:hi] <= f][::-1][:N_REC]
        for j, ix in enumerate(idx, start=1):
            for k, v in vals.items():
                res[f'rec{j}_{k}'][i] = v[ix]
    return pd.DataFrame(res, index=t.index)


def feat_long(data):
    t = data.table
    arrs = _per_person(_valid_nights(data))
    res = {f'{s}28_{k}': np.full(len(t), np.nan) for k in COLS for s in ('med', 'sd')}
    d0 = t['date'].values.astype('datetime64[D]').astype(np.int64)
    for i, (p, d, f) in enumerate(zip(t['pid'].to_numpy(), d0, t['forecast_at'].to_numpy())):
        if p not in arrs:
            continue
        day, known, vals = arrs[p]
        lo, hi = np.searchsorted(day, d - 28, 'left'), np.searchsorted(day, d, 'left')
        ok = known[lo:hi] <= f
        if not ok.any():
            continue
        for k, v in vals.items():
            x = v[lo:hi][ok]
            res[f'med28_{k}'][i] = np.median(x)
            if len(x) >= 2:
                res[f'sd28_{k}'][i] = np.std(x, ddof=1)
    return pd.DataFrame(res, index=t.index)


def rec(target):
    return [f'rec{j}_{target}' for j in range(1, N_REC + 1)]
