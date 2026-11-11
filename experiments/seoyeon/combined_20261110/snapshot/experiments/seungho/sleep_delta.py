# 작성 2026-11-06 07:00 (KST)
"""
sleep_delta.py — ① 입력 특징 개선: 평소 대비 최근 수면 변화량 (오늘 밤 수면 예측, 공통 프로토콜 v3 위에서)

질문: CatBoost를 고정하고, 기본 입력(최근 7밤 + 요일)에 "평소 대비 최근 변화량"을 넣으면 MAE가 몇 분 달라지는가.
함정: 변화량 Δ = 최근 − 평소를 주면 모델은 최근 7밤을 이미 알기 때문에 평소(L) 자체도 얻는다.
      그래서 "평소 수준 L만" 넣은 대조군(B1)을 같이 돌려, 효과가 변화량 표현 때문인지 긴 과거 때문인지 가른다.

    from src import forecast_dataset as fd, forecast_run as fr
    from experiments.seungho import sleep_delta as sd
    data, split, man, out = fr.prepare()
    table = fd.add_features(data, sd.feat_delta)
    fr.run(data, split, man, out, experiments=sd.stage('S1'), table=table, baseline='cb_base')

정의 (모두 목표와 같은 양: sleep → 총수면, bed → 취침 시각, wake → 기상 시각. 단위 분)
  lag_k      : 밤 D−k의 값. fd.lag_features와 같은 가림 규칙(달력 기준, D 18:00까지 안 끝난 밤·겹침·무효 → NaN)
  R{r}       : 최근 r밤 평균 (lag1..r 중 유효한 밤). r=1이면 어제 값
  L          : 평소 = 과거 w밤의 중앙값(또는 평균). in = lag1..w, ex = lag8..(7+w) (최근 7밤을 뺀 평소)
               유효 밤이 m개 미만이면 NaN
  Δ          : R − L (분)
  z          : Δ / (1.4826 × MAD(lag1..28)). 유효 7밤 미만이거나 MAD = 0이면 NaN
  dowgap     : 같은 요일 평소(lag7,14,21,28 중앙값, 유효 2개 이상) − L28  (이 사람의 이 요일이 평소와 얼마나 다른가)
  ewd        : 가중 변화량 = Σ w_k lag_k / Σ w_k − L28, w_k = 0.5^(k/2) (k=1..7, 유효한 밤만). 어제일수록 무겁다
  slope      : lag1..7의 시간 기울기 (분/밤, 최근 쪽으로 증가하면 양수). 유효 4밤 이상
  madratio   : MAD(lag1..7) / MAD(lag1..28). 유효 4밤·7밤 이상, 분모 0이면 NaN
  nextreme   : 최근 7밤 중 평소보다 60분 넘게 나쁜 쪽인 밤 수 (sleep: L−60 미만, bed·wake: L+60 초과)

규칙
  - 줄마다 그 줄의 과거만으로 계산한다. 데이터 전체에서 맞추는 값(집단 통계, 스케일러)은 만들지 않는다
  - ffill / bfill / 보간 없음. NaN은 그대로 CatBoost(nan_mode='Min')에 넘긴다
  - 공통 예측 표의 열은 바꾸지 않고 열만 더한다 (fd.add_features). 공통 파일 src/forecast_*.py는 수정하지 않음
  - 유효 기록 수·마지막 기록 이후 일수는 ④(민준) 담당이라 입력으로 넣지 않는다
"""
import warnings
from itertools import product

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe

MAX_LAG = 63                     # 56밤 평소를 최근 7밤 뒤(lag8..63)로 잡는 경우까지
RECENTS = (1, 3, 7)
WINDOWS = (14, 28, 56)
MIN_VALID = (3, 7, 14)
DEFAULT = dict(r=7, w=28, stat='med', span='in', m=7)   # 주 비교 정의 (결과 보기 전에 고정)
EW_HALF = 2.0                    # 가중 변화량 반감기 (밤)
EXTREME_MIN = 60                 # nextreme 기준 (분)
MAD_K = 1.4826
SEEDS = (0, 1, 2, 3, 4)

# 실험에서 쓰는 평소 정의만 계산 (전부 교차하면 열이 770개라 메모리 낭비)
LEVEL_KEYS = ([(w, 'med', span, 7) for w in WINDOWS for span in ('in', 'ex')]
              + [(28, 'mean', 'in', 7)] + [(28, 'med', 'in', m) for m in MIN_VALID if m != 7])

QTY = {t: fd.LAG_PREFIX[t] for t in fd.TARGETS}       # sleep→sleep, bed→bedtime, wake→waketime
WORSE = {'sleep': -1, 'bed': +1, 'wake': +1}            # nextreme에서 "나쁜 쪽" 방향


# ---------- 이름 ----------

def lname(w=28, stat='med', span='in', m=7):
    return f'L{w}{stat}{span}m{m}'


def dname(r=7, form='diff', **lkw):
    k = {**{x: DEFAULT[x] for x in ('w', 'stat', 'span', 'm')}, **lkw}
    return f'd{form}R{r}_{lname(**k)}'


def col(target, name):
    return f'sd_{QTY[target]}_{name}'


# ---------- 계산 ----------

def _nanstat(M, how):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(M, axis=1) if how == 'med' else np.nanmean(M, axis=1)


def _mad(M):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(M, axis=1, keepdims=True)
        return np.nanmedian(np.abs(M - med), axis=1)


def _cnt(M):
    return np.sum(~np.isnan(M), axis=1)


def level(M, w=28, stat='med', span='in', m=7):
    """M: (줄, MAX_LAG) lag 행렬 (열 j = lag j+1). 평소 L."""
    lo = 0 if span == 'in' else 7
    S = M[:, lo:lo + w]
    out = _nanstat(S, stat)
    out[_cnt(S) < m] = np.nan
    return out


def recent(M, r=7):
    return _nanstat(M[:, :r], 'mean')


def scale28(M):
    S = M[:, :28]
    s = MAD_K * _mad(S)
    s[(_cnt(S) < 7) | (s == 0)] = np.nan
    return s


def quantity_features(M, target):
    """한 양(총수면/취침/기상)의 모든 후보 특징. 반환: {이름: 배열}."""
    f = {}
    Ls = {}
    for key in LEVEL_KEYS:
        Ls[key] = level(M, *key)
    Rs = {r: recent(M, r) for r in RECENTS}
    sc = scale28(M)
    for (w, stat, span, m), L in Ls.items():
        f[lname(w, stat, span, m)] = L
        for r, R in Rs.items():
            d = R - L
            f[dname(r, 'diff', w=w, stat=stat, span=span, m=m)] = d
            f[dname(r, 'z', w=w, stat=stat, span=span, m=m)] = d / sc
    L28 = Ls[(28, 'med', 'in', 7)]
    # 같은 요일 평소 − 평소
    D = M[:, [6, 13, 20, 27]]
    dl = _nanstat(D, 'med')
    dl[_cnt(D) < 2] = np.nan
    f['dowgap'] = dl - L28
    # 가중 변화량
    R7 = M[:, :7]
    wk = 0.5 ** (np.arange(1, 8) / EW_HALF)
    W = np.where(np.isnan(R7), 0.0, wk)
    with np.errstate(invalid='ignore', divide='ignore'):
        ew = np.nansum(np.nan_to_num(R7) * W, axis=1) / W.sum(axis=1)
    ew[W.sum(axis=1) == 0] = np.nan
    f['ewd'] = ew - L28
    # 기울기: x = −k (최근일수록 큼)
    x = -np.arange(1, 8, dtype=float)
    ok = ~np.isnan(R7)
    n = ok.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        xm = (ok * x).sum(axis=1) / n
        ym = np.nansum(R7, axis=1) / n
        dx = np.where(ok, x - xm[:, None], 0.0)
        dy = np.where(ok, R7 - ym[:, None], 0.0)
        sl = (dx * dy).sum(axis=1) / (dx ** 2).sum(axis=1)
    sl[n < 4] = np.nan
    f['slope'] = sl
    # 변동성 비
    m7, m28 = _mad(R7), _mad(M[:, :28])
    with np.errstate(invalid='ignore', divide='ignore'):
        mr = m7 / m28
    mr[(n < 4) | (_cnt(M[:, :28]) < 7) | (m28 == 0)] = np.nan
    f['madratio'] = mr
    # 나쁜 쪽 극단 밤 수
    dev = (R7 - L28[:, None]) * WORSE[target]
    ne = np.sum(dev > EXTREME_MIN, axis=1).astype(float)
    ne[np.isnan(L28) | (n == 0)] = np.nan
    f['nextreme'] = ne
    return f


def lag_matrices(table, nights, max_lag=MAX_LAG):
    """목표별 (줄, max_lag) lag 행렬. 가림 규칙은 fd.lag_features 그대로."""
    lags, _ = fd.lag_features(table[['pid', 'date', 'forecast_at']], nights, n_lags=max_lag)
    return {t: lags[[f'{QTY[t]}_lag{k}' for k in range(1, max_lag + 1)]].to_numpy(dtype=float)
            for t in fd.TARGETS}


def delta_features(table, nights, max_lag=MAX_LAG):
    """table과 같은 index의 새 열 표 (모든 목표 × 모든 후보 특징)."""
    mats = lag_matrices(table, nights, max_lag)
    cols = {}
    for t, M in mats.items():
        for name, v in quantity_features(M, t).items():
            cols[col(t, name)] = v
    return pd.DataFrame(cols, index=table.index)


def feat_delta(data):
    """fd.add_features에 넘기는 특징 함수."""
    return delta_features(data.table, data.nights)


# ---------- 실험 ----------

def _cb(seed=0):
    return (lambda: fe.CatBoostModel(random_seed=seed)) if seed else fe.CatBoostModel


def exp(name, names, seed=0, cross=False):
    """기본 입력 + 목표별 특징 names. cross=True면 names를 세 양 모두에서 붙임."""
    def feats(t):
        qs = list(fd.TARGETS) if cross else [t]
        return fd.base_features(t) + [col(q, n) for q in qs for n in names]
    return fe.Experiment(name + (f'_s{seed}' if seed else ''), _cb(seed), feats)


L0 = lname()            # L28medinm7
D0 = dname()            # ddiffR7_L28medinm7


def main_exps(seed=0):
    """주 비교 B0~B3."""
    return [exp('cb_base', [], seed), exp('cb_level', [L0], seed), exp('cb_delta', [D0], seed),
            exp('cb_level_delta', [L0, D0], seed)]


def sensitivity_exps():
    """한 축씩 바꾼 변화량 (다른 축은 DEFAULT) + 창별 대조군."""
    e = []
    for r in (1, 3):
        e.append(exp(f'A1_R{r}', [dname(r=r)]))
    for w in (14, 56):
        e.append(exp(f'A2_w{w}', [dname(w=w)]))
        e.append(exp(f'A2_level_w{w}', [lname(w=w)]))
    e.append(exp('A3_mean', [dname(stat='mean')]))
    e.append(exp('A4_ex', [dname(span='ex')]))
    e.append(exp('A4_level_ex', [lname(span='ex')]))
    e.append(exp('A5_z', [dname(form='z')]))
    e.append(exp('A6_dowgap', [D0, 'dowgap']))
    for m in (3, 14):
        e.append(exp(f'A7_m{m}', [dname(m=m)]))
    return e


def grid_exps():
    """교차 그리드: 최근 창 × 평소 창 × 최근 포함 여부 × 표현 (주 비교 정의는 cb_delta와 같아 뺌)."""
    e = []
    for r, w, span, form in product(RECENTS, WINDOWS, ('in', 'ex'), ('diff', 'z')):
        n = dname(r=r, form=form, w=w, span=span)
        if n != D0:
            e.append(exp(f'G_R{r}_w{w}_{span}_{form}', [n]))
    return e


def form_exps():
    """변화량의 다른 형태 (주 비교 B2에 하나씩 더함) + 다중 창."""
    e = [exp(f'C_{n}', [D0, n]) for n in ('ewd', 'slope', 'madratio', 'nextreme')]
    e.append(exp('X_cross', [D0], cross=True))
    e.append(exp('X_cross_level', [L0, D0], cross=True))
    e.append(exp('M_multiR', [dname(r=r) for r in RECENTS]))
    e.append(exp('M_multiW', [dname(w=w) for w in WINDOWS]))
    return e


SHAPES = ['ewd', 'slope', 'madratio', 'nextreme']


def combo_exps():
    """미리 정한 조합 (결과를 보고 고르지 않음)."""
    multi = [dname(r=r) for r in RECENTS] + [dname(w=w) for w in (14, 56)]
    allf = [L0, D0, dname(form='z'), 'dowgap'] + multi[:2] + multi[3:] + SHAPES
    return [exp('K_delta_ewd_slope', [D0, 'ewd', 'slope']),
            exp('K_delta_mad_ext', [D0, 'madratio', 'nextreme']),
            exp('K_level_delta_shapes', [L0, D0] + SHAPES),
            exp('K_level_multi', [L0] + multi),
            exp('K_all', allf),
            exp('K_all_cross', allf, cross=True)]


def seed_exps(seeds=SEEDS[1:]):
    """확정 단계: 주 비교 B0~B3를 시드 1~4로 (시드 0은 S1). 상위 후보는 run_delta가 붙인다."""
    return [x for s in seeds for x in main_exps(s)]


STAGES = {'S1': lambda: main_exps(0), 'S2': sensitivity_exps, 'S2b': grid_exps,
          'S3': form_exps, 'S4': combo_exps, 'S0S5': seed_exps}


def stage(name):
    return STAGES[name]()


def all_phase1():
    """1차(시드 0) 전체 실험 이름 → Experiment."""
    out = {}
    for s in ('S1', 'S2', 'S2b', 'S3', 'S4'):
        for e in stage(s):
            if e.name in out:
                raise ValueError(f'실험 이름 중복: {e.name}')
            out[e.name] = e
    return out


def with_seed(e, seed):
    """같은 입력, 다른 CatBoost 시드."""
    return fe.Experiment(f'{e.name}_s{seed}', _cb(seed), e.features)

