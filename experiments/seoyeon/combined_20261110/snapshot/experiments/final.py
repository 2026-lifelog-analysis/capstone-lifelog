# 작성 2026-10-15 03:30
"""
최종 평가 — 확인용 30% 사람에서 **한 번만** 돌리는 발표용 결과.

최종 방법 (개발용 탐색으로 확정, 이 파일에서는 바꾸지 않음)
  걸음(steps) → 같은 날 전체 사람 중앙값을 뺌 → 요일 7개 보정 개인 기준선(직전 달력 28일)
  → '나빠지는 방향' z → CUSUM(k=0.25, 경보 후 0으로) → 하루 경보율 1/30 (ev.ALARM_RATE)

같은 판에서 비교하는 방법 (같은 주입 구간, 같은 하루 경보율)
  무작위 / 집단 기준선 / 레포 기본(개인+CUSUM k=0.5)
  개선 사다리: 레포 기본 → +요일 보정 → +같은 날 공통효과 제거 → 최종(k=0.25)

같은 날 기준(공통효과 중앙값, 집단 기준선)은 확인용 사람만이 아니라 **전체 사람**으로 계산한다.
확인용 174명만으로는 같은 날 30명 미만인 날이 5%라 점수가 빠지기 때문이다. 실제 서비스에서도
비교 집단은 전체 사용자다. 개발용 사람은 주입 없는 원래 값으로 들어간다.

주입: S1 걸음 −30% 갑자기 / S3 걸음 −15% 서서히 / S5 걸음 −20%·수면 −10%·취침 +60분 서서히
곡선: 하루 경보율 1/14 · 1/30 · 1/60 · 1/90 (주 결과는 1/30)

사용법: python3 -I experiments/final.py [반복 수=20] [프로세스 수=8] [기록자=진승호]
출력:   results/final/*.csv, results/results.csv (ev.log_table)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C          # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

ev, bl = C.ev, C.bl

SCEN = {
    'S1': ([('steps', -0.30, 'relative')], 'sudden', '걸음 −30% 갑자기'),
    'S3': ([('steps', -0.15, 'relative')], 'gradual', '걸음 −15% 서서히'),
    'S5': ([('steps', -0.20, 'relative'), ('sleep_min', -0.10, 'relative'), ('bed_min', 60, 'absolute')],
           'gradual', '걸음 −20%·수면 −10%·취침 +60분 서서히'),
}
RATES = {'1/14': 1 / 14, '1/30': 1 / 30, '1/60': 1 / 60, '1/90': 1 / 90}
FINAL = '최종'
BASE = '레포 기본'
LADDER = [BASE, '+요일 보정', '+공통효과 제거', FINAL]
ORDER = ['무작위', '집단 기준선'] + LADDER
# 사다리 단계별 설정: (요일 보정, 공통효과 제거, k)
STEPS = {BASE: (False, False, 0.5), '+요일 보정': (True, False, 0.5),
         '+공통효과 제거': (True, True, 0.5), FINAL: (True, True, 0.25)}


def _with_ref(d, ref):
    """평가 대상 d에 참고 집단 ref(주입 없음)를 붙인 표와, d 줄을 가리키는 표시."""
    both = pd.concat([d.assign(_own=True), ref.assign(_own=False)], ignore_index=True)
    return both


def same_day_median(d, ref, col='steps'):
    """d의 각 줄에 대해 그날 (d ∪ ref) 사람들의 중앙값. 30명 미만이면 빈칸."""
    both = _with_ref(d[['date', col]], ref[['date', col]])
    g = both.groupby('date')[col]
    med = g.median().where(g.count() >= bl.GROUP_MIN_N)
    return d['date'].map(med)


def group_bad(d, ref, col='steps'):
    """집단 기준선 (같은 날 전체 사람 대비 z, ±8, 나빠지는 방향). bl.group_z와 같은 계산, 참고 집단만 넓힘."""
    both = _with_ref(d[['date', col]], ref[['date', col]])
    g = both.groupby('date')[col]
    med = g.median()
    mad = (both[col] - both['date'].map(med)).abs().groupby(both['date']).median() * 1.4826
    cnt = g.count()
    z = (d[col] - d['date'].map(med)) / d['date'].map(mad.replace(0, np.nan))
    z = z.where(d['date'].map(cnt) >= bl.GROUP_MIN_N)
    return bl.bad_score(z.clip(-bl.Z_CLIP, bl.Z_CLIP), col)


def personal_bad(d, ref, dow, anchor, col='steps'):
    x = d[col].astype(float)
    if anchor:
        x = x - same_day_median(d, ref, col)
    z = C.personal_z_grouped(d.assign(_x=x), '_x', np.arange(7) if dow else None)
    return z * bl.BAD_DIRECTION[col]


def make_score_fn(df_clean, ref, rate):
    hs = {}
    for m, (dow, anc, k) in STEPS.items():
        pb = personal_bad(df_clean, ref, dow, anc)
        hs[m] = bl.cusum_threshold(df_clean.assign(_b=pb), '_b', k=k, rate=rate)

    def score_fn(d):
        cols = {'무작위': np.random.default_rng(ev.SEED).random(len(d)),
                '집단 기준선': group_bad(d, ref)}
        for m, (dow, anc, k) in STEPS.items():
            pb = personal_bad(d, ref, dow, anc)
            cols[m] = bl.cusum(d.assign(_b=pb), '_b', k=k, h=hs[m])
        return pd.DataFrame(cols, index=d.index)[ORDER]
    score_fn.h = hs
    return score_fn


def one(task, n_reps):
    sc, rate_name = task
    effects, shape, _ = SCEN[sc]
    df = C.holdout_set(C.load())
    ref = C.dev_set(C.load())
    (col, eff, mode), *extra = effects
    C.INJECT_EXTRA[:] = extra
    res = ev.evaluate_injection(df, col, eff, mode=mode, shape=shape, n_reps=n_reps,
                                rate=RATES[rate_name], score_fn=make_score_fn(df, ref, RATES[rate_name]))
    return sc, rate_name, res


def calendar(rate=ev.ALARM_RATE):
    """주입 없는 확인용 데이터에서 달력 상태별 하루 경보율 (레포 기본 vs 최종)."""
    df = C.holdout_set(C.load())
    ref = C.dev_set(C.load())
    f = make_score_fn(df, ref, rate)
    sc = f(df)
    rows = []
    for m in [BASE, FINAL]:
        a = (sc[m] > f.h[m]).astype(float).where(sc[m].notna())
        d = df.assign(_a=a, 주말=np.where(df['dow'] >= 5, '주말', '평일'))
        for col in ['event_tag', 'term_tag', '주말']:
            for k, r in d.groupby(col)['_a'].agg(['mean', 'count']).iterrows():
                rows.append({'방법': m, '구분': col, '값': k, '경보율': round(r['mean'], 4), '날 수': int(r['count'])})
        cnt = (sc[m] > f.h[m]).groupby(df['pid']).sum()
        n10 = max(1, int(len(cnt) * 0.1))
        rows.append({'방법': m, '구분': '몰림', '값': '상위10%몫', '경보율': round(cnt.nlargest(n10).sum() / cnt.sum(), 4)})
        rows.append({'방법': m, '구분': '몰림', '값': '0번인 사람', '경보율': int((cnt == 0).sum()), '날 수': len(cnt)})
    return pd.DataFrame(rows)


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    who = sys.argv[3] if len(sys.argv) > 3 else '진승호'
    out = C.OUT.parent / 'final'
    out.mkdir(parents=True, exist_ok=True)

    hold = C.holdout_set(C.load())
    print(f"확인용: {hold['pid'].nunique()}명, {len(hold):,}일 / 반복 {n_reps}회", flush=True)

    tasks = [(s, r) for s in SCEN for r in RATES]
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for sc, rn, r in ex.map(partial(one, n_reps=n_reps), tasks):
            res[(sc, rn)] = r
            print(f'  끝: {sc} {rn}', flush=True)

    rows, cmp = [], []
    for (sc, rn), r in res.items():
        rows.append(C.summary(r, '').drop(columns='실험').assign(시나리오=sc, 하루경보=rn))
        for m in ORDER:
            if m != BASE:
                cmp.append({'시나리오': sc, '하루경보': rn, '방법': m, '비교 대상': BASE,
                            **C.compare_methods(r, BASE, m)})
        if rn == '1/30':
            ev.log_table(ev.results_table(r, order=ORDER), who, 'NetHealth',
                         f"최종 평가(확인용 30%) {sc} {SCEN[sc][2]}, reps={n_reps}, rate=1/30, "
                         f"최종=요일7+같은날공통효과제거+steps+CUSUM k=0.25")
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(out / 'final_table.csv', index=False, encoding='utf-8-sig')
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(out / 'final_compare.csv', index=False, encoding='utf-8-sig')
    cal = calendar()
    cal.to_csv(out / 'final_calendar.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    show = ['방법', '탐지율(평균)', '오경보율(평균)', '차이(평균)', '탐지 지연', '하루 오경보율', 'PR-AUC(보조)', '사람 수']
    for sc in SCEN:
        t = tab[(tab['시나리오'] == sc) & (tab['하루경보'] == '1/30')].set_index('방법').reindex(ORDER).reset_index()
        print(f'\n=== {sc} {SCEN[sc][2]} · 하루 경보 1/30 · 확인용 ===')
        print(t[show].to_string(index=False))
    print('\n=== 레포 기본 대비 (같은 사람끼리, 차이) ===')
    print(cmp[cmp['하루경보'] == '1/30'].to_string(index=False))
    print('\n=== 곡선: 최종 / 집단 기준선 ===')
    c = tab[tab['방법'].isin([FINAL, '집단 기준선'])]
    print(c.pivot_table(index=['방법', '하루경보'], columns='시나리오',
                        values=['탐지율(평균)', '오경보율(평균)']).round(3).to_string())
    print('\n=== 달력 상태별 경보율 (주입 없음, 1/30) ===')
    print(cal.pivot_table(index=['구분', '값'], columns='방법', values='경보율').to_string())


if __name__ == '__main__':
    main()
