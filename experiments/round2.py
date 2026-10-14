# 작성 2026-10-15 02:16
"""
2차 튜닝 — 오경보 줄이기 + 같은 오경보에서 더 잡기 (개발용 70% 사람만).

출발점 (조합 실험 1위): 요일 7개 보정 + 같은 날 공통효과 제거
  A = 걸음만,  B = 걸음 + 수면 + 취침

바꾸는 것 (3 × 2 × 4 = 24개 설정 × 주입 3개 = 72판)
  k      : CUSUM 허용치 0.25 / 0.5 / 1.0
  scale  : off = 변동폭(MAD) 하나 / wk = 평일·주말 변동폭을 따로 (주말이 원래 더 들쭉날쭉한 사람)
  rate   : 하루 경보율 1/14 · 1/30(레포 기준) · 1/60 · 1/90  → 오경보-탐지율 곡선

한 판 안에서 같이 비교하는 탐지기 (모두 같은 하루 경보율로 기준값을 맞춤)
  A CUSUM      : 레포와 같은 CUSUM (경보 후 0으로)
  A 2일 연속   : CUSUM이 기록 기준 이틀 연속 경보선을 넘어야 경보, 그 뒤 0으로 되돌림
                 (경보일 점수 = 1+S, 아니면 0 → 분위수 기준값이 0이면 정확히 경보일만 넘음)
                 ※ 처음엔 '되돌리지 않는 CUSUM의 min(오늘, 어제)'로 했으나 긴 고점 구간이 경보 예산을
                   다 써서 탐지율 26%로 무너짐 → 이 방식으로 교체
  B CUSUM / B 2일 연속
  A 또는 B     : 두 CUSUM을 각자 주입 없는 분포에서의 백분위로 바꾼 뒤 큰 쪽 → 경보 예산을 나눠 씀
  집단 기준선  : 걸음, 비교용

사용법: python3 -I experiments/round2.py [반복 수=20] [프로세스 수=10]
출력:   results/tuning/round2_table.csv, runs_round2/*.pkl
"""
import itertools
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C          # noqa: E402
import combo as K           # noqa: E402  (불러오면 여러 지표 주입이 켜짐)

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

ev, bl, pp = C.ev, C.bl, C.pp

FEATS = {'A': ['steps'], 'B': ['steps', 'sleep_min', 'bed_min']}
RATES = {'1/14': 1 / 14, '1/30': 1 / 30, '1/60': 1 / 60, '1/90': 1 / 90}
CONFIGS = [dict(k=k, scale=s, rate=r) for k, s, r in itertools.product([0.25, 0.5, 1.0], ['off', 'wk'], RATES)]
SCEN = ['S1', 'S3', 'S5']
MIN_SCALE_N = 6     # 그룹별 변동폭은 창 안에 그 그룹 기록이 6일 이상일 때만, 아니면 전체 변동폭


def personal_z_v2(df, col, scale_by_weekend=False, window=bl.WINDOW, min_periods=bl.MIN_PERIODS,
                  clip=bl.Z_CLIP):
    """요일 7개 평균 보정 개인 z. scale_by_weekend=True면 변동폭(MAD)을 평일·주말 따로 계산."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for pid, idx, dates, full in bl._per_person_daily(df, col):
        x = full.to_numpy(float)
        n = len(x)
        pad = np.concatenate([np.full(window, np.nan), x])
        win = np.lib.stride_tricks.sliding_window_view(pad, window)[:n]
        dow = full.index.dayofweek.to_numpy()
        dpad = np.concatenate([np.zeros(window, int), dow])
        dwin = np.lib.stride_tricks.sliding_window_view(dpad, window)[:n]
        off = pp.dow_offsets(win, dwin)
        win = win - np.take_along_axis(off, dwin, axis=1)
        today = x - off[np.arange(n), dow]
        cnt = np.sum(~np.isnan(win), axis=1)
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(win, axis=1)
            dev = np.abs(win - med[:, None])
            mad = np.nanmedian(dev, axis=1) * 1.4826
            if scale_by_weekend:
                same = (dwin >= 5) == (dow >= 5)[:, None]
                dg = np.where(same, dev, np.nan)
                ng = np.sum(~np.isnan(dg), axis=1)
                mad_g = np.nanmedian(dg, axis=1) * 1.4826
                mad = np.where((ng >= MIN_SCALE_N) & (mad_g > 0), mad_g, mad)
            z = (today - med) / mad
        z[(cnt < min_periods) | ~(mad > 0)] = np.nan
        out.loc[idx] = pd.Series(z, index=full.index).reindex(dates.values).to_numpy()
    return out.clip(-clip, clip) if clip else out


def bad_combined(d, feats, scale):
    parts = []
    for f in feats:
        x = d[f].astype(float)
        g = x.groupby(d['date'])
        x = (x - g.transform('median')).where(g.transform('count') >= bl.GROUP_MIN_N)
        parts.append(personal_z_v2(d.assign(_x=x), '_x', scale == 'wk') * bl.BAD_DIRECTION[f])
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _consec_run(parts, k, h, m=2):
    """CUSUM이 기록 기준 m일 연속 h를 넘으면 경보, 그 뒤 0으로 되돌림. 반환: 경보일은 1+S, 아니면 0, 빈 날 NaN."""
    res = []
    for vals in parts:
        s, run, r = 0.0, 0, np.full(len(vals), np.nan)
        for i, v in enumerate(vals.tolist()):
            if v != v:
                continue
            s = max(0.0, s + v - k)
            run = run + 1 if s > h else 0
            r[i] = 0.0
            if run >= m:
                r[i] = 1.0 + s
                s, run = 0.0, 0
        res.append(r)
    return np.concatenate(res) if res else np.array([])


def consec_cusum(d, col, k, h):
    idx, parts = bl._cusum_prepare(d, col)
    return pd.Series(_consec_run(parts, k, h), index=idx).reindex(d.index)


def consec_threshold(d, col, k, rate, iters=30):
    """2일 연속 규칙에서 경보일 비율이 rate 이하가 되는 가장 작은 h."""
    _, parts = bl._cusum_prepare(d, col)
    lo, hi = 0.0, float(np.nanmax(bl._cusum_run(parts, k, None)))
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = _consec_run(parts, k, mid)
        ok = ~np.isnan(r)
        if (r[ok] > 0).mean() > rate:
            lo = mid
        else:
            hi = mid
    return hi


def make_score_fn(df_clean, cfg):
    k, rate = cfg['k'], RATES[cfg['rate']]
    hs, hc, ref = {}, {}, {}
    for name, feats in FEATS.items():
        pb = bad_combined(df_clean, feats, cfg['scale'])
        hs[name] = bl.cusum_threshold(df_clean.assign(_b=pb), '_b', k=k, rate=rate)
        hc[name] = consec_threshold(df_clean.assign(_b=pb), '_b', k=k, rate=rate)
        cs = bl.cusum(df_clean.assign(_b=pb), '_b', k=k, h=hs[name])
        ref[name] = np.sort(cs.dropna().to_numpy())

    def pct(name, s):
        r = ref[name]
        return pd.Series(np.searchsorted(r, s.to_numpy(), side='right') / len(r), index=s.index).where(s.notna())

    def score_fn(d):
        cols = {'무작위': np.random.default_rng(ev.SEED).random(len(d))}
        pcs = {}
        for name, feats in FEATS.items():
            pb = bad_combined(d, feats, cfg['scale'])
            dd = d.assign(_b=pb)
            cs = bl.cusum(dd, '_b', k=k, h=hs[name])
            cols[f'{name} CUSUM'] = cs
            cols[f'{name} 2일 연속'] = consec_cusum(dd, '_b', k=k, h=hc[name])
            pcs[name] = pct(name, cs)
        cols['A 또는 B'] = pd.concat(pcs.values(), axis=1).max(axis=1, skipna=True)
        cols['집단 기준선'] = bl.bad_score(bl.group_z(d, 'steps'), 'steps')
        return pd.DataFrame(cols, index=d.index)
    return score_fn


def one(task, n_reps):
    ci, sc = task
    cfg = CONFIGS[ci]
    df = C.dev_set(C.load())
    (col, eff, mode), *extra = K.SCEN[sc]
    K._EXTRA[:] = extra
    res = ev.evaluate_injection(df, col, eff, mode=mode, shape=K.SHAPE[sc], n_reps=n_reps,
                                rate=RATES[cfg['rate']], score_fn=make_score_fn(df, cfg))
    return ci, sc, res


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = C.OUT
    (out / 'runs_round2').mkdir(parents=True, exist_ok=True)
    tasks = [(i, s) for s in SCEN for i in range(len(CONFIGS))]
    print(f'설정 {len(CONFIGS)}개 × 주입 {len(SCEN)}개 = {len(tasks)}판 / 반복 {n_reps}회', flush=True)

    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (ci, sc, r) in enumerate(ex.map(partial(one, n_reps=n_reps), tasks), 1):
            with open(out / 'runs_round2' / f'c{ci:02d}_{sc}.pkl', 'wb') as fh:
                pickle.dump(r, fh)
            rows.append(C.summary(r, '').assign(시나리오=sc, **CONFIGS[ci]))
            if i % 12 == 0:
                print(f'  {i}/{len(tasks)} 끝', flush=True)
    tab = pd.concat(rows, ignore_index=True).drop(columns='실험')
    tab.to_csv(out / 'round2_table.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    x = tab[tab['rate'] == '1/30']
    p = x.pivot_table(index=['k', 'scale', '방법'], columns='시나리오', values='차이(평균)')
    print('\n=== 하루 경보 1/30에서 차이(평균) ===')
    print(p.round(3).to_string())


if __name__ == '__main__':
    main()
