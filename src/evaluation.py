"""
evaluation.py — 방법 비교와 평가 (모듈 1)

1) 라벨 없이 비교 : compare_alarms — 개인 vs 집단 기준선이 '같은 날, 같은 사람'에게 울리는지
2) 가짜 이탈 주입 : inject_drift → evaluate_injection → results_table / wilcoxon_table / plot_person_box
   - 정상 구간 14일에 일부러 변화를 넣고(예: 걸음 −30%) 잡아내는지 봄
   - 대조군: 같은 구간을 주입 없이 한 번 더 돌려서 '원래도 울렸을' 경보를 뺌
   - 경보 기준값: 대조군 전체에서 하루 오경보율이 3.3%(ALARM_RATE)가 되도록 **하나만** 정함
   - 사람마다 계산한 뒤 분포(중앙값, 박스플롯, Wilcoxon)로 보고
   - 정확도·F1·AUROC만 단독으로 쓰지 않음
3) 결과 기록 : log_result → results/results.csv
"""
import os

import numpy as np
import pandas as pd

from . import baseline as bl
from .data_loader import PATHS

ALARM_RATE = 1 / 30   # 경보 기준: 오경보가 사람당 30일에 1번 = 전체 날의 3.3%
SEED = 0              # 랜덤이 들어가는 곳의 기본 시드


# ---------------------------------------------------------------------------
# 1. 라벨 없이 두 방법 비교
# ---------------------------------------------------------------------------
def _person_rate_summary(alarm, pid):
    """사람별 경보율(경보 날 ÷ 기록 날)의 (중앙값, 1사분위, 3사분위)."""
    r = alarm.groupby(pid).mean()
    return (float(r.median()), float(r.quantile(0.25)), float(r.quantile(0.75)))


def compare_alarms(df, score_a, score_b, rate=ALARM_RATE):
    """
    라벨이 없을 때 두 방법(A, B)을 비교함.
    둘 다 '두 점수가 모두 있는 날' 전체에서 rate(3.3%)만 경보가 울리도록 기준값을 **하나씩** 정한 뒤:
      같은날비율     : A가 울린 날 중 B도 울린 비율
      0번인사람      : (A, B) 경보가 한 번도 안 울린 사람 수
      상위10%몫      : (A, B) 경보를 가장 많이 받은 10% 사람이 전체 경보 중 차지하는 비율
                       (높을수록 경보가 몇몇 사람에게 몰림)
      사람별오경보율 : (A, B) 각각 사람별 경보율의 (중앙값, 1사분위, 3사분위)
                       라벨이 없으니 모든 경보를 오경보로 봤을 때의 사람별 분포
    """
    x = df[['pid', score_a, score_b]].dropna()
    a = x[score_a] > x[score_a].quantile(1 - rate)
    b = x[score_b] > x[score_b].quantile(1 - rate)
    ca = a.groupby(x['pid']).sum()
    cb = b.groupby(x['pid']).sum()
    n10 = max(1, int(len(ca) * 0.1))
    return {'같은날비율': float((a & b).sum() / a.sum()),
            '0번인사람': (int((ca == 0).sum()), int((cb == 0).sum())),
            '상위10%몫': (float(ca.nlargest(n10).sum() / ca.sum()), float(cb.nlargest(n10).sum() / cb.sum())),
            '사람별오경보율': (_person_rate_summary(a, x['pid']), _person_rate_summary(b, x['pid'])),
            '사람수': len(ca), '날수': len(x)}


# ---------------------------------------------------------------------------
# 2. 가짜 이탈 주입
# ---------------------------------------------------------------------------
def inject_drift(df, col, effect, mode='relative', length=14, shape='sudden', seed=SEED,
                 window=bl.WINDOW, min_periods=bl.MIN_PERIODS, min_days_in=7):
    """
    사람마다 '정상 구간' 하나를 골라 length일 동안 col 값을 일부러 바꿈 (모듈 1 평가용).

    구간 고르는 규칙 (사람마다 무작위, seed로 고정)
      - 시작일 직전 달력 window(28)일에 기록이 min_periods(14)일 이상 → 기준선을 만들 수 있음
      - 주입 구간 length(14)일 안에 기록이 min_days_in(7)일 이상
      - 조건을 만족하는 시작일이 없는 사람은 건너뜀
    바꾸는 방법
      mode='relative' : 값 × (1 + effect)   예) effect=-0.3 → 걸음 30% 감소
      mode='absolute' : 값 + effect         예) effect=120  → 잠든 시각 2시간 늦어짐
      shape='sudden'  : 첫날부터 effect 전부 (갑자기)
      shape='gradual' : 1/length → 1까지 매일 조금씩 커짐 (서서히)
    빈 날은 빈 채로 둠 (채우지 않음).

    반환: (바뀐 df 복사본, 구간표 pid | start | end)
    """
    if shape not in ('sudden', 'gradual'):
        raise ValueError("shape는 'sudden' 또는 'gradual'")
    if mode not in ('relative', 'absolute'):
        raise ValueError("mode는 'relative' 또는 'absolute'")
    rng = np.random.default_rng(seed)
    segs = []
    for pid, idx, dates, full in bl._per_person_daily(df.sort_values(['pid', 'date']), col):
        ok = (~full.isna()).to_numpy(int)
        cs = np.concatenate([[0], np.cumsum(ok)])
        n = len(ok)
        starts = np.arange(window, n - length + 1)
        if not len(starts):
            continue
        before = cs[starts] - cs[starts - window]
        inside = cs[starts + length] - cs[starts]
        cand = starts[(before >= min_periods) & (inside >= min_days_in)]
        if not len(cand):
            continue
        s = full.index[rng.choice(cand)]
        segs.append({'pid': pid, 'start': s, 'end': s + pd.Timedelta(days=length - 1)})
    seg = pd.DataFrame(segs, columns=['pid', 'start', 'end'])

    out = df.copy()
    rows = _segment_rows(out, seg)
    day = (rows['date'] - rows['start']).dt.days.to_numpy()
    frac = np.ones(len(rows)) if shape == 'sudden' else (day + 1) / length
    v = out.loc[rows.index, col].to_numpy(float)
    out.loc[rows.index, col] = v * (1 + effect * frac) if mode == 'relative' else v + effect * frac
    return out, seg


def _segment_rows(df, seg):
    """df에서 구간표(seg) 안에 드는 줄만 골라 start를 붙여 돌려줌 (index는 df 그대로)."""
    m = df[['pid', 'date']].reset_index().merge(seg, on='pid')
    m = m[(m['date'] >= m['start']) & (m['date'] <= m['end'])]
    return m.set_index('index')[['pid', 'date', 'start']]


def method_scores(df, col, dow_adjust=False, k=0.5, h=None, min_n=bl.GROUP_MIN_N, seed=SEED):
    """
    비교할 방법 5개의 하루 점수를 한 표로 만듦 (모두 '클수록 이상').
      무작위             : 데이터와 상관없는 난수 (seed 고정). 우연 수준 비교용
      단일 z(|z|)        : 개인 기준선 z의 절댓값. 방향 구분 없이 '평소와 다르면' 경보
      개인 기준선        : 개인 기준선 z를 '나빠지는 방향'으로 맞춘 점수
      개인 기준선 + CUSUM: 위 점수의 누적합 (허용치 k, 경보선 h를 넘으면 0으로 되돌림)
      집단 기준선        : 같은 날 다른 사람들과 비교한 z를 '나빠지는 방향'으로 맞춘 점수
    dow_adjust : 개인 기준선에 요일 보정을 할지
    h          : CUSUM 경보선. None이면 되돌리지 않음 (evaluate_injection이 알아서 정해 줌)
    반환: df와 같은 index의 DataFrame (열 = 방법 이름)
    """
    pz = bl.personal_z(df, col, dow_adjust=dow_adjust)
    pb = bl.bad_score(pz, col)
    return pd.DataFrame({
        '무작위': np.random.default_rng(seed).random(len(df)),
        '단일 z(|z|)': pz.abs(),
        '개인 기준선': pb,
        '개인 기준선 + CUSUM': bl.cusum(df.assign(_b=pb), '_b', k=k, h=h),
        '집단 기준선': bl.bad_score(bl.group_z(df, col, min_n=min_n), col),
    }, index=df.index)


def evaluate_injection(df, col, effect, mode='relative', shape='sudden', length=14, n_reps=10,
                       seed=SEED, rate=ALARM_RATE, dow_adjust=False, k=0.5, score_fn=None):
    """
    가짜 이탈 주입 평가 (모듈 1). n_reps번 반복(시드 seed, seed+1, ...)해서 사람마다 비율을 냄.

    절차
      1) 대조군: 주입 없는 원래 데이터로 방법별 점수 계산
      2) 기준값: 대조군 전체 날 중 rate(3.3%)만 넘도록 방법마다 **하나** (모든 사람 공통)
         CUSUM은 경보 후 0으로 되돌리므로 경보선 h 자체를 그렇게 맞춤 (baseline.cusum_threshold)
      3) 반복마다 inject_drift로 사람마다 14일 구간에 주입 → 점수 다시 계산
      4) 구간 14일 안에 하루라도 기준값을 넘으면 '탐지'
         같은 구간을 대조군에서 봤을 때 하루라도 넘으면 '오경보'

    dow_adjust, k : 기본 방법(method_scores)의 요일 보정 여부, CUSUM 허용치
    score_fn : df → 방법별 점수 DataFrame. 주면 이걸 쓰고, 기준값은 모두 상위 rate 분위로 정함
    반환: dict
      per_person : 사람×방법 표 — 탐지율, 오경보율, 차이(=탐지율−오경보율, Youden's J),
                   탐지 지연(첫 경보가 주입 시작 며칠 뒤인지, 0=첫날, 탐지된 반복의 중앙값),
                   하루 오경보율(대조군 전체 기간)
      threshold  : 방법별 기준값
      pr_auc     : 방법별 PR-AUC (보조). 주입 구간 날 vs 같은 구간 대조군 날
      day_fa_all : 방법별 대조군 전체 하루 오경보율 (기준값이 맞게 정해졌는지 확인용, ≈ rate)
      segments   : 반복별 구간표
    """
    from sklearn.metrics import average_precision_score

    cs_name, h = '개인 기준선 + CUSUM', None
    if score_fn is None:
        clean = method_scores(df, col, dow_adjust=dow_adjust, k=k)
        h = bl.cusum_threshold(df.assign(_b=clean['개인 기준선']), '_b', k=k, rate=rate)
        clean[cs_name] = bl.cusum(df.assign(_b=clean['개인 기준선']), '_b', k=k, h=h)
        score_fn = lambda d: method_scores(d, col, dow_adjust=dow_adjust, k=k, h=h)  # noqa: E731
    else:
        clean = score_fn(df)
    methods = list(clean.columns)
    thr = clean.quantile(1 - rate)
    if h is not None:
        thr[cs_name] = h
    alarm_clean = clean.gt(thr) & clean.notna()

    day_alarm = alarm_clean.astype(float).where(clean.notna())
    day_fa = day_alarm.groupby(df['pid']).mean()

    recs, pos, neg, seg_all = [], {m: [] for m in methods}, {m: [] for m in methods}, []
    for r in range(n_reps):
        inj, seg = inject_drift(df, col, effect, mode=mode, length=length, shape=shape, seed=seed + r)
        seg_all.append(seg.assign(rep=r))
        sc = score_fn(inj)
        rows = _segment_rows(df, seg)
        day = (rows['date'] - rows['start']).dt.days
        for m in methods:
            a_inj = (sc.loc[rows.index, m] > thr[m])
            a_ctl = alarm_clean.loc[rows.index, m]
            first = day.where(a_inj).groupby(rows['pid']).min()
            g = pd.DataFrame({'detected': a_inj.groupby(rows['pid']).any(),
                              'false_alarm': a_ctl.groupby(rows['pid']).any(),
                              'delay': first})
            recs.append(g.assign(method=m, rep=r).reset_index())
            pos[m].append(sc.loc[rows.index, m])
            neg[m].append(clean.loc[rows.index, m])

    R = pd.concat(recs, ignore_index=True)
    pp_ = R.groupby(['method', 'pid']).agg(탐지율=('detected', 'mean'), 오경보율=('false_alarm', 'mean'),
                                           탐지지연=('delay', 'median')).reset_index()
    pp_['차이'] = pp_['탐지율'] - pp_['오경보율']
    pp_['하루오경보율'] = [day_fa.loc[p, m] for p, m in zip(pp_['pid'], pp_['method'])]

    pr = {}
    for m in methods:
        p, n = pd.concat(pos[m]), pd.concat(neg[m])
        s = pd.concat([p, n]).to_numpy(float)
        lo = np.nanmin(s) - 1 if np.isfinite(np.nanmin(s)) else -1
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        pr[m] = float(average_precision_score(y, np.nan_to_num(s, nan=lo)))

    return {'per_person': pp_, 'threshold': thr.to_dict(), 'pr_auc': pr,
            'day_fa_all': day_alarm.mean().to_dict(),
            'segments': pd.concat(seg_all, ignore_index=True),
            'setting': dict(col=col, effect=effect, mode=mode, shape=shape, length=length,
                            n_reps=n_reps, seed=seed, rate=rate, dow_adjust=dow_adjust, k=k, cusum_h=h)}


def results_table(res, order=None):
    """
    방법들을 한 표로 정리함. 숫자는 모두 '사람별 값의 중앙값'.
      | 방법 | 탐지율 | 오경보율 | 차이 | 탐지 지연 | 하루 오경보율 | 전체 하루 오경보율 | PR-AUC(보조) | 사람 수 |
      탐지율   : 주입 구간 14일 안에 경보가 하루라도 울린 비율 (주 지표)
      오경보율 : 같은 구간을 주입 없이 돌렸을 때 경보가 울린 비율
      차이     : 탐지율 − 오경보율 (Youden's J). 0이면 우연 수준
      탐지 지연: 주입 시작 후 첫 경보까지 며칠 (0 = 첫날)
      하루 오경보율      : 대조군 전체 기간에서 사람별 경보 날 비율의 중앙값
      전체 하루 오경보율 : 모든 사람·날을 합친 비율 (기준값 정의상 ≈ 3.3%)
    """
    pp_ = res['per_person']
    t = pp_.groupby('method').agg(탐지율=('탐지율', 'median'), 오경보율=('오경보율', 'median'),
                                  차이=('차이', 'median'), 탐지_지연=('탐지지연', 'median'),
                                  하루_오경보율=('하루오경보율', 'median'), 사람_수=('pid', 'nunique'))
    t['전체_하루_오경보율'] = pd.Series(res['day_fa_all'])
    t['PR-AUC(보조)'] = pd.Series(res['pr_auc'])
    t = t.reindex(order or list(res['threshold'].keys()))
    t.columns = [c.replace('_', ' ') for c in t.columns]
    t = t[['탐지율', '오경보율', '차이', '탐지 지연', '하루 오경보율', '전체 하루 오경보율', 'PR-AUC(보조)', '사람 수']]
    return t.rename_axis('방법').reset_index().round(3)


def wilcoxon_table(res, metric='차이', ref='개인 기준선'):
    """
    방법 간 차이가 우연인지 사람 단위 짝지은 검정(Wilcoxon 부호순위)으로 확인함.
    ref 방법과 나머지 방법을 같은 사람끼리 짝지어 metric을 비교.
    반환: | 방법 | 비교 대상 | 사람 수 | 중앙값 차이(방법−ref) | p값 |
    """
    from scipy.stats import wilcoxon
    w = res['per_person'].pivot(index='pid', columns='method', values=metric)
    rows = []
    for m in w.columns:
        if m == ref:
            continue
        x = w[[m, ref]].dropna()
        d = x[m] - x[ref]
        p = float(wilcoxon(x[m], x[ref]).pvalue) if (d != 0).any() else np.nan
        rows.append({'방법': m, '비교 대상': ref, '사람 수': len(x),
                     '중앙값 차이': round(float(d.median()), 3), 'p값': p})
    return pd.DataFrame(rows)


def plot_person_box(res, metric='차이', ax=None, order=None):
    """사람별 metric 분포를 방법마다 박스플롯으로 그림."""
    import matplotlib.pyplot as plt
    pp_ = res['per_person']
    order = order or list(res['threshold'].keys())
    data = [pp_.loc[pp_['method'] == m, metric].dropna() for m in order]
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=15)
    ax.set_ylabel(metric)
    s = res['setting']
    ax.set_title(f"{s['col']} {s['mode']} {s['effect']:+g} ({s['shape']}) — 사람별 {metric}")
    return ax


# ---------------------------------------------------------------------------
# 3. 결과 기록 — 모두 같은 표에 쌓기
# ---------------------------------------------------------------------------
def log_result(who, dataset, method, setting, metrics, path=None):
    """
    결과 한 줄을 results/results.csv에 추가함 (results/는 커밋하지 않음).
      who     : 이름
      dataset : 'NetHealth' 등
      method  : '개인 기준선', '집단 기준선' 등
      setting : 'window=28, steps -30% sudden' 등 설정 설명
      metrics : 숫자들이 든 dict (compare_alarms 결과, results_table의 한 줄 등)
    """
    p = path or PATHS['results']
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    row = {'시각': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), '담당': who,
           '데이터': dataset, '방법': method, '설정': setting}
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            v = round(float(v), 4)
        elif isinstance(v, (tuple, list, dict)):
            v = str(v)
        row[k] = v
    new = pd.DataFrame([row])
    if os.path.exists(p):      # 열이 다른 결과도 한 표에 쌓이도록 기존 표와 합쳐서 다시 저장
        new = pd.concat([pd.read_csv(p, encoding='utf-8-sig'), new], ignore_index=True)
    new.to_csv(p, index=False, encoding='utf-8-sig')
    return row


def log_table(table, who, dataset, setting, path=None):
    """results_table 결과를 방법마다 한 줄씩 log_result로 기록함."""
    for _, r in table.iterrows():
        m = r.drop('방법').to_dict()
        log_result(who, dataset, r['방법'], setting, m, path=path)
