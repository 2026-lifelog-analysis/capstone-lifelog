# 작성 2026-10-15 01:28
"""
experiments/common.py — 튜닝 실험 공용 도구 (docs/EXPERIMENT_PLAN.md)

src/는 고치지 않고, 변형 기준선은 ev.evaluate_injection(score_fn=...)으로 넣는다.
  load(wear)            : NetHealth 하루 표 (캐시)
  split_pids(df)        : 사람 단위 개발용 70% / 확인용 30% (시드 0)
  personal_z_grouped    : 기준선 창 안에서 '같은 그룹 날'의 평균을 빼는 개인 z (요일·평일/주말 보정)
  make_score_fn         : 변형 z로 방법 5개 점수를 만듦. CUSUM 경보선 h는 기본 경로와 같은 방식으로 맞춤
  run / compare / calendar_alarm_rates : 평가 실행, 실험 간 사람 단위 비교, 달력 상태별 대조군 경보율
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline as bl          # noqa: E402
from src import data_loader as dl       # noqa: E402
from src import evaluation as ev        # noqa: E402
from src import preprocessing as pp     # noqa: E402

OUT = ROOT / 'results' / 'tuning'
CS = '개인 기준선 + CUSUM'

# 주입 시나리오 (계획서 2장): 이름 → (지표, 크기, mode, shape)
SCENARIOS = {
    'S1': ('steps', -0.30, 'relative', 'sudden'),
    'S2': ('steps', -0.30, 'relative', 'gradual'),
    'S3': ('steps', -0.15, 'relative', 'gradual'),
    'S4': ('bed_min', 120, 'absolute', 'sudden'),
}

_CACHE = {}


def load(wear=pp.WEAR_MIN):
    if wear not in _CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _CACHE[wear] = dl.load_nethealth(wear=wear)
    return _CACHE[wear]


def split_pids(df, frac=0.7, seed=0):
    """사람 단위로 섞어 (개발용 pid 집합, 확인용 pid 집합). 같은 시드면 항상 같음."""
    pids = np.array(sorted(df['pid'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = int(round(len(pids) * frac))
    return set(pids[:n]), set(pids[n:])


def dev_set(df, seed=0):
    dev, _ = split_pids(load(), seed=seed)       # 분할은 항상 기본(착용 80%) 표 기준
    return df[df['pid'].isin(dev)].reset_index(drop=True)


def holdout_set(df, seed=0):
    _, hold = split_pids(load(), seed=seed)
    return df[df['pid'].isin(hold)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 여러 지표 동시 주입: ev.inject_drift를 감싸서 같은 구간에 INJECT_EXTRA의 지표도 넣음
#   INJECT_EXTRA[:] = [('sleep_min', -0.10, 'relative'), ('bed_min', 60, 'absolute')]
#   비어 있으면 레포 inject_drift와 완전히 같음
# ---------------------------------------------------------------------------
INJECT_EXTRA = []
_orig_inject = ev.inject_drift


def _inject_multi(df, col, effect, mode='relative', length=14, shape='sudden', seed=ev.SEED, **kw):
    out, seg = _orig_inject(df, col, effect, mode=mode, length=length, shape=shape, seed=seed, **kw)
    if INJECT_EXTRA and len(seg):
        rows = ev._segment_rows(out, seg)
        day = (rows['date'] - rows['start']).dt.days.to_numpy()
        frac = np.ones(len(rows)) if shape == 'sudden' else (day + 1) / length
        for c2, e2, m2 in INJECT_EXTRA:
            v = out.loc[rows.index, c2].to_numpy(float)
            out.loc[rows.index, c2] = v * (1 + e2 * frac) if m2 == 'relative' else v + e2 * frac
    return out, seg


ev.inject_drift = _inject_multi


# ---------------------------------------------------------------------------
# 변형 개인 기준선
# ---------------------------------------------------------------------------
WEEKEND_GROUP = np.array([0, 0, 0, 0, 0, 1, 1])     # 월~금 = 0, 토·일 = 1


def personal_z_grouped(df, col, group_of_dow=None, window=bl.WINDOW, min_periods=bl.MIN_PERIODS,
                       clip=bl.Z_CLIP):
    """
    bl.personal_z(dow_adjust=True)와 같은 계산인데, 요일 7개 대신 group_of_dow로 묶은 그룹을 씀.
      group_of_dow=None        → 보정 없음 (bl.personal_z와 같음)
      group_of_dow=arange(7)   → 요일 7개 (bl.personal_z(dow_adjust=True)와 같음)
      group_of_dow=WEEKEND_GROUP → 평일/주말 2그룹
    보정값은 기준선 창 안의 과거 값으로만 계산 (pp.dow_offsets).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for pid, idx, dates, full in bl._per_person_daily(df, col):
        x = full.to_numpy(float)
        n = len(x)
        pad = np.concatenate([np.full(window, np.nan), x])
        win = np.lib.stride_tricks.sliding_window_view(pad, window)[:n]
        today = x
        if group_of_dow is not None:
            grp = np.asarray(group_of_dow)[full.index.dayofweek.to_numpy()]
            gpad = np.concatenate([np.zeros(window, int), grp])
            gwin = np.lib.stride_tricks.sliding_window_view(gpad, window)[:n]
            off = pp.dow_offsets(win, gwin)
            win = win - np.take_along_axis(off, gwin, axis=1)
            today = x - off[np.arange(n), grp]
        cnt = np.sum(~np.isnan(win), axis=1)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            mad = np.nanmedian(np.abs(win - med[:, None]), axis=1) * 1.4826
            z = (today - med) / mad
        z[(cnt < min_periods) | ~(mad > 0)] = np.nan
        out.loc[idx] = pd.Series(z, index=full.index).reindex(dates.values).to_numpy()
    return out.clip(-clip, clip) if clip else out


def make_score_fn(df_clean, col, z_fn, k=0.5, rate=ev.ALARM_RATE, seed=ev.SEED):
    """
    ev.method_scores와 같은 5개 방법을 만들되 개인 z는 z_fn(df, col)로 계산.
    CUSUM 경보선 h는 기본 경로처럼 주입 없는 데이터에서 bl.cusum_threshold로 정하고, 경보 후 0으로 되돌림.
    """
    pb0 = bl.bad_score(z_fn(df_clean, col), col)
    h = bl.cusum_threshold(df_clean.assign(_b=pb0), '_b', k=k, rate=rate)

    def score_fn(d):
        pz = z_fn(d, col)
        pb = bl.bad_score(pz, col)
        return pd.DataFrame({
            '무작위': np.random.default_rng(seed).random(len(d)),
            '단일 z(|z|)': pz.abs(),
            '개인 기준선': pb,
            CS: bl.cusum(d.assign(_b=pb), '_b', k=k, h=h),
            '집단 기준선': bl.bad_score(bl.group_z(d, col), col),
        }, index=d.index)
    score_fn.h = h
    return score_fn


# ---------------------------------------------------------------------------
# 실행 · 비교
# ---------------------------------------------------------------------------
def run(df, scenario, n_reps=20, seed=ev.SEED, score_fn=None, col=None, **kw):
    """시나리오 하나 평가. col을 주면 시나리오 지표 대신 사용 (예: steps → steps_per_hour)."""
    c, eff, mode, shape = SCENARIOS[scenario]
    c = col or c
    t = time.time()
    res = ev.evaluate_injection(df, c, eff, mode=mode, shape=shape, n_reps=n_reps, seed=seed,
                                score_fn=score_fn, **kw)
    if score_fn is not None and hasattr(score_fn, 'h'):
        # score_fn 경로는 기준값을 분위수로 잡으므로 CUSUM만 h로 다시 판정한 값과 같은지 확인용으로 남김
        res['setting']['cusum_h'] = score_fn.h
    res['sec'] = round(time.time() - t, 1)
    return res


def summary(res, name):
    """results_table + 사람별 평균(중앙값은 반복 수 때문에 계단식이라 해상도 보완용)."""
    t = ev.results_table(res)
    m = res['per_person'].groupby('method')[['탐지율', '오경보율', '차이']].mean().add_suffix('(평균)')
    t = t.merge(m.round(3), left_on='방법', right_index=True)
    t.insert(0, '실험', name)
    return t


def compare(res_a, res_b, method=CS, metric='차이'):
    """같은 사람끼리 짝지어 실험 B − 실험 A (method 하나). Wilcoxon 부호순위."""
    from scipy.stats import wilcoxon
    pa = res_a['per_person'].query('method == @method').set_index('pid')[metric]
    pb = res_b['per_person'].query('method == @method').set_index('pid')[metric]
    x = pd.concat([pa, pb], axis=1, keys=['a', 'b']).dropna()
    d = x['b'] - x['a']
    p = float(wilcoxon(x['b'], x['a']).pvalue) if (d != 0).any() else np.nan
    return {'사람 수': len(x), '평균 차이': round(float(d.mean()), 4),
            '중앙값 차이': round(float(d.median()), 4),
            '좋아진 사람': int((d > 0).sum()), '나빠진 사람': int((d < 0).sum()), 'p값': p}


def compare_methods(res, method_a, method_b, metric='차이'):
    """한 평가 안의 두 방법을 같은 사람끼리 짝지어 B − A. Wilcoxon 부호순위."""
    return _paired(res['per_person'], method_a, method_b, metric)


def _paired(pp_, a, b, metric):
    from scipy.stats import wilcoxon
    w = pp_.pivot(index='pid', columns='method', values=metric)[[a, b]].dropna()
    d = w[b] - w[a]
    p = float(wilcoxon(w[b], w[a]).pvalue) if (d != 0).any() else np.nan
    return {'사람 수': len(w), '평균 차이': round(float(d.mean()), 4), '중앙값 차이': round(float(d.median()), 4),
            '좋아진 사람': int((d > 0).sum()), '나빠진 사람': int((d < 0).sum()), 'p값': p}


def calendar_alarm_rates(df, scores, method=CS, rate=ev.ALARM_RATE, h=None):
    """
    주입 없는 데이터에서 달력 상태별 하루 경보율. 이상적이면 모두 rate 근처로 평평.
    scores : 방법별 점수 DataFrame (df와 같은 index). CUSUM은 h를 주면 S > h로 판정.
    """
    s = scores[method]
    thr = h if (h is not None and method == CS) else s.quantile(1 - rate)
    a = (s > thr).astype(float).where(s.notna())
    d = df.assign(_a=a, 주말=np.where(df['dow'] >= 5, '주말', '평일'))
    rows = []
    for col in ['event_tag', 'term_tag', '주말']:
        g = d.groupby(col)['_a'].agg(['mean', 'count'])
        for k, r in g.iterrows():
            rows.append({'구분': col, '값': k, '경보율': round(r['mean'], 4), '날 수': int(r['count'])})
    m = d.assign(월=d['date'].dt.month).groupby('월')['_a'].mean()
    rows += [{'구분': '월', '값': int(k), '경보율': round(v, 4), '날 수': None} for k, v in m.items()]
    return pd.DataFrame(rows)
