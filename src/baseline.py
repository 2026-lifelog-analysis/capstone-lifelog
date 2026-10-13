"""
baseline.py — 기준선('평소')과 이탈 점수

    from src import baseline as bl
    nh['steps_z']   = bl.personal_z(nh, 'steps')            # 개인 기준선 z (직전 달력 28일)
    nh['steps_bad'] = bl.bad_score(nh['steps_z'], 'steps')  # '클수록 나쁨'으로 방향 통일
    nh['steps_cs']  = bl.cusum(nh.assign(b=nh['steps_bad']), 'b')  # 나빠지는 쪽 누적합

z점수 = (오늘 값 − 기준선 중앙값) / 기준선 MAD×1.4826
  중앙값·MAD는 평균·표준편차 대신 쓰는 '튀는 값에 덜 흔들리는' 값이에요.
"""
import warnings

import numpy as np
import pandas as pd

from .preprocessing import dow_offsets

Z_CLIP = 8        # z를 ±8로 자름 (이상치는 지우지 않고 영향만 줄임)
WINDOW = 28       # 개인 기준선 길이 (달력 28일)
MIN_PERIODS = 14  # 기준선 28일 안에 기록이 14일 이상 있어야 계산
GROUP_MIN_N = 30  # 집단 기준선: 그날 기록 있는 사람이 30명 이상일 때만 계산

# 지표별 '나빠지는 방향' (+1: 클수록 나쁨, -1: 작을수록 나쁨)
BAD_DIRECTION = {
    'steps': -1, 'steps_per_hour': -1, 'active_min': -1,
    'sleep_min': -1,
    'bed_min': +1,     # 늦게 잘수록 나쁨
    'meanrate': +1,    # 심박이 높을수록 나쁨
}


def _clip(z, clip):
    return z.clip(-clip, clip) if clip else z


def _per_person_daily(df, col):
    """
    사람마다 값을 '하루 한 칸' 달력 배열로 펼쳐요 (기록 없는 날은 빈칸 그대로, 채우지 않음).
    돌려주는 것: (사람 ID, 원래 줄 번호들, 원래 날짜들, 달력 Series)
    """
    for pid, g in df.groupby('pid', sort=False):
        s = g.set_index('date')[col].sort_index()
        s = s[~s.index.duplicated()]
        full = s.asfreq('D') if len(s) else s
        yield pid, g.index, g['date'], full


def personal_z(df, col, window=WINDOW, gap=0, min_periods=MIN_PERIODS, clip=Z_CLIP, dow_adjust=False):
    """
    개인 기준선 z점수: '나의 평소'와 비교해서 오늘이 얼마나 다른가.
      기준선 = 오늘 기준 (gap+window)일 전 ~ (gap+1)일 전의 **달력** 날짜 (기록 줄 수가 아님)
               → 오늘과 미래 값은 절대 쓰지 않아요
      기록 없는 날은 빈칸으로 두고 계산 (채우지 않음)

    인자
      window      : 기준선 길이(일). 기본 28
      gap         : 최근 며칠을 건너뛸지. 0이면 '직전 28일', 14면 '15~42일 전'
      min_periods : 기준선 안에 기록이 이만큼 이상 있어야 계산 (기본 14)
      clip        : z를 ±clip으로 자름 (기본 8, None이면 안 자름)
      dow_adjust  : True면 요일 보정. 기준선 창 안의 과거 값으로 '요일 평균 − 전체 평균'을 구해서
                    오늘 값과 창 값에서 빼요 (예: 주말마다 걸음이 적은 사람의 주말을 덜 이상하게 봄)
    반환: df와 같은 줄 순서의 Series
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for pid, idx, dates, full in _per_person_daily(df, col):
        x = full.to_numpy(float)
        n = len(x)
        pad = np.concatenate([np.full(window + gap, np.nan), x])
        win = np.lib.stride_tricks.sliding_window_view(pad, window)[:n]   # win[i] = x[i-gap-window : i-gap]
        today = x
        if dow_adjust:
            dow = full.index.dayofweek.to_numpy()
            dpad = np.concatenate([np.zeros(window + gap, int), dow])
            dwin = np.lib.stride_tricks.sliding_window_view(dpad, window)[:n]
            off = dow_offsets(win, dwin)                                  # (n, 7), 창 안 과거 값만 사용
            win = win - np.take_along_axis(off, dwin, axis=1)
            today = x - off[np.arange(n), dow]
        cnt = np.sum(~np.isnan(win), axis=1)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            mad = np.nanmedian(np.abs(win - med[:, None]), axis=1) * 1.4826
            z = (today - med) / mad
        z[(cnt < min_periods) | ~(mad > 0)] = np.nan
        out.loc[idx] = pd.Series(z, index=full.index).reindex(dates.values).to_numpy()
    return _clip(out, clip)


def group_z(df, col, by='date', min_n=GROUP_MIN_N, clip=Z_CLIP):
    """
    집단 기준선 z점수: '같은 날 다른 학생들'과 비교해서 오늘 내가 얼마나 다른가.
      그날(by) 모든 사람의 중앙값·MAD와 비교. 그날 기록 있는 사람이 min_n명 미만이면 빈칸.
      clip : z를 ±clip으로 자름 (기본 8, None이면 안 자름)
    """
    g = df.groupby(by)[col]
    med = g.transform('median')
    mad = (df[col] - med).abs().groupby(df[by]).transform('median') * 1.4826
    z = (df[col] - med) / mad.replace(0, np.nan)
    z[g.transform('count') < min_n] = np.nan
    return _clip(z, clip)


def bad_score(z, col):
    """z점수를 '클수록 나쁨'으로 방향을 맞춰요. col은 BAD_DIRECTION에 있는 지표 이름."""
    return z * BAD_DIRECTION.get(col, 1)


def _cusum_prepare(df, col):
    """CUSUM 계산용: 날짜순으로 정렬한 줄 번호와 사람별로 나눈 값 배열."""
    d = df[['pid', 'date', col]].sort_values(['pid', 'date'], kind='stable')
    cut = np.flatnonzero(d['pid'].to_numpy()[1:] != d['pid'].to_numpy()[:-1]) + 1
    return d.index, np.split(d[col].to_numpy(float), cut)


def _cusum_run(parts, k, h):
    """사람별 값 배열에 CUSUM을 돌려 하나로 이어 붙인 S 배열을 돌려줘요."""
    res = []
    for vals in parts:
        s, r = 0.0, np.full(len(vals), np.nan)
        for i, v in enumerate(vals.tolist()):
            if v != v:          # 빈칸(NaN)인 날: S는 그대로, 출력은 빈칸
                continue
            s = max(0.0, s + v - k)
            r[i] = s
            if h is not None and s > h:
                s = 0.0
        res.append(r)
    return np.concatenate(res) if res else np.array([])


def cusum(df, col, k=0.5, h=None):
    """
    CUSUM(누적합): 하루하루 조금씩 나빠지는 것을 쌓아서 잡아내는 방법.
      S(오늘) = max(0, S(어제) + 오늘 점수 − k)
      col : '클수록 나쁨' 점수 열 (예: bad_score(personal_z(...)))
      k   : 허용치. 하루 점수가 k보다 작으면 쌓이지 않고 줄어들어요 (기본 0.5)
      h   : 경보선. S > h 인 날 경보를 울리고 다음 날부터 S를 0에서 다시 쌓아요.
            None이면 되돌리지 않고 S만 계속 쌓아요 (오래 나쁜 사람은 S가 끝없이 커짐에 주의)
            모든 사람에게 같은 h를 쓰려면 cusum_threshold로 정하세요.
    - 사람마다 날짜 순서로 계산. 값이 빈 날은 S를 그대로 두고 그날 출력은 빈칸
    반환: S 값 Series (df와 같은 줄 순서). 경보 = S > h
    """
    idx, parts = _cusum_prepare(df, col)
    return pd.Series(_cusum_run(parts, k, h), index=idx).reindex(df.index)


def cusum_threshold(df, col, k=0.5, rate=1 / 30, iters=30):
    """
    CUSUM 경보선 h를 정해요: 모든 사람·모든 날을 합쳐서 경보 비율이 rate(기본 3.3%)가 되는 h 하나.
    (S > h 이면 경보 후 0으로 되돌리는 방식이라, h를 바꿔 가며 반으로 나눠 찾아요)
    """
    _, parts = _cusum_prepare(df, col)
    s0 = _cusum_run(parts, k, None)
    lo, hi = 0.0, float(np.nanmax(s0)) if np.isfinite(np.nanmax(s0)) else 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = _cusum_run(parts, k, mid)
        ok = ~np.isnan(s)
        if (s[ok] > mid).mean() > rate:
            lo = mid
        else:
            hi = mid
    return hi


def _calendar_rolling(df, col, k, func, min_periods):
    """사람별 달력 k일 창에 func를 적용 (빈 날은 빈칸 그대로)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for pid, idx, dates, full in _per_person_daily(df, col):
        r = full.rolling(k, min_periods=min_periods).apply(func, raw=True)
        out.loc[idx] = r.reindex(dates.values).to_numpy()
    return out


def rolling_mean(df, col, k=3, min_periods=None):
    """사람별 최근 달력 k일(오늘 포함) 평균. 예: z점수의 3일 평균. 빈 날은 빼고 평균."""
    mp = min_periods or max(1, k // 2 + 1)
    return _calendar_rolling(df, col, k, np.nanmean, mp)


def rolling_slope(df, col, k=14, min_periods=7):
    """사람별 최근 달력 k일(오늘 포함)의 기울기(하루당 변화량). 서서히 줄어드는지 보는 용도."""
    def f(a):
        m = ~np.isnan(a)
        return np.polyfit(np.arange(len(a))[m], a[m], 1)[0]
    return _calendar_rolling(df, col, k, f, min_periods)
