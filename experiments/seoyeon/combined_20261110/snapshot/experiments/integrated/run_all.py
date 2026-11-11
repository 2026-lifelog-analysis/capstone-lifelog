"""
run_all.py — PR 1·3·4가 각자 조건으로 낸 점수들을 공통 평가 한 번에 같은 조건으로 다시 잰다.

무엇을 하나
  공통 evaluate_injection 은 score_fn 하나를 받아 "구간 고르기 → 가짜 변화 심기 → 기준값 긋기 → 세기"를 전부 한다.
  그 score_fn 자리에 열 11개짜리 점수 함수를 넣는다. 공통 코드(src/)는 고치지 않는다.
    공통 5개 : 무작위, 단일 z(|z|), 개인 기준선, 개인 기준선 + CUSUM, 집단 기준선     (ev.method_scores 그대로)
    PR 1  3개: +요일 보정, +공통효과 제거, 최종(k=0.25)                               (experiments/final.py 함수 호출)
    PR 3  2개: 수면 중앙 SD 개인 z, 수면 중앙 SD 개인+CUSUM                          (experiments/minjun/regularity 함수 호출)
  PR 1의 '레포 기본'은 공통 '개인 기준선 + CUSUM'과 같은 계산이라 따로 넣지 않고 같은지만 확인한다.

두 가지 기준값 규칙
  방식 A : 공통 노트북 02 그대로. 전원, 하루 경보율 1/30, evaluate_injection 호출.
  방식 C : PR 3 방식 B. 14일 구간 오경보율 0.20 에 기준값을 맞춘 뒤 같은 11개 점수를 센다. 구간·심기는 A와 같은 함수.

흔들림 시나리오 (PR 3)
  공통 inject_drift 는 곱하기·더하기만 만들 수 있어 평균 0 난수 흔들림을 못 넣는다. 그래서 이 시나리오는 evaluate_injection 을
  거치지 않고, 구간은 공통 inject_drift(효과 0)로 받고 PR 3 의 regularity.jitter_sleep 으로 흔든 뒤 같은 세는 법(14일 안 경보
  한 번이면 탐지, 심기 전 같은 구간이면 오경보)으로 센다. 방식 A 기준값도 공통과 같은 식(심기 전 전체 날의 상위 1/30 분위수).
  공통 4개 시나리오를 이 루프로도 세어 evaluate_injection 결과와 같은지 확인한다 (--loop-check).

넣지 못한 것
  PR 2 모델 6개 (공통 점수 함수 모양이 아니고 가짜 변화 평가가 없음).

조건 차이 (RESULTS.md 에도 적음)
  - PR 1 본문은 사람 70/30 분할, 확인용 30%에서 평가. 여기서는 전원. 그래서 수치가 다르다.
  - PR 1의 '같은 날 집단 중앙값'은 원래 주입 없는 70%를 참고 집단으로 썼다. 여기서는 전원이 평가 대상이라
    참고 집단이 없고, 그날 중앙값에 심은 변화가 조금 섞인다 (사람마다 구간이 달라 같은 날 겹치는 사람은 적다).
  - PR 1 변형은 본문에서 걸음만 평가했다. 여기서는 같은 변형을 잠든 시각 시나리오에도 적용해 본다.
  - evaluate_injection 은 score_fn 경로에서 CUSUM 열도 분위수로 기준값을 긋는다. CUSUM 안의 초기화선 h와
    그 분위수는 조금 다를 수 있다 (PR 2가 실측: 방법당 1인일). 방식 C는 h 자체를 맞춘다.

사용 (레포 맨 위에서):
  python experiments/integrated/run_all.py [--data data/nethealth] [--reps 10] [--scenarios 이름들] [--out results/integrated]
                                           [--append] [--loop-check 시나리오 이름]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments' / 'minjun' / 'regularity'))

from src import baseline as bl, data_loader as dl, evaluation as ev  # noqa: E402
import final as F1          # noqa: E402  PR 1 (experiments/final.py; 같은 폴더의 common.py 를 끌어온다)
import regularity as R3     # noqa: E402  PR 3

assert not F1.C.INJECT_EXTRA, 'PR 1 common.py 의 다중 주입 목록이 비어 있어야 공통 inject_drift 와 같다'

RATE = ev.ALARM_RATE
WIN_FA = 0.20
CS = '개인 기준선 + CUSUM'
COMMON = ['무작위', '단일 z(|z|)', '개인 기준선', CS, '집단 기준선']
PR1 = {'PR1 +요일 보정': (True, False, 0.5), 'PR1 +공통효과 제거': (True, True, 0.5), 'PR1 최종': (True, True, 0.25)}
PR3_Z, PR3_CS = 'PR3 수면 중앙 SD 개인 z', 'PR3 수면 중앙 SD 개인+CUSUM'
ORDER = COMMON + list(PR1) + [PR3_Z, PR3_CS]
SCEN = {                       # 공통 노트북 02 셀 9 와 같음
    '걸음 -30% 갑자기': ('steps', -0.3, 'relative', 'sudden'),
    '걸음 -30% 서서히': ('steps', -0.3, 'relative', 'gradual'),
    '잠든 시각 +120분 갑자기': ('bed_min', 120, 'absolute', 'sudden'),
    '잠든 시각 +120분 서서히': ('bed_min', 120, 'absolute', 'gradual'),
    '수면 시각 흔들림 SD 90분': ('bed_min', 90.0, 'jitter', 'jitter'),     # PR 3 시나리오. 공통 함수 밖 (아래 inject)
}
JITTER = 'jitter'
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:5.0f}s]', *a, flush=True)


# ---------------------------------------------------------------------------
# 점수 함수: 기반 점수(z 계열)와 CUSUM 을 나눠 둔다. 방식 C 가 기반 점수와 k 를 따로 쓰기 때문.
# ---------------------------------------------------------------------------
def pr3_mid_sd(d):
    d2 = d if 'mid' in d.columns else R3.add_sleep_mid(d)      # 흔들림 경로는 jitter_sleep 이 mid 를 이미 갱신함
    return R3.sleep_mid_sd(d2)


def inject(nh, col, eff, mode, shape, seed):
    """공통 inject_drift. 흔들림만 구간을 공통 함수로 받고 PR 3 jitter_sleep 으로 흔든다."""
    if shape == JITTER:
        _, seg = ev.inject_drift(nh, col, 0.0, mode='absolute', length=14, shape='sudden', seed=seed)
        return R3.jitter_sleep(nh, seg, sigma=eff, seed=seed - ev.SEED), seg
    return ev.inject_drift(nh, col, eff, mode=mode, shape=shape, seed=seed)


def base_scores(d, col, ref):
    """CUSUM 전 단계의 점수들. 반환: (기반 점수 표, 각 CUSUM 열의 (기반 열 이름, k))"""
    pz = bl.personal_z(d, col)
    pb = bl.bad_score(pz, col)
    out = {'무작위': np.random.default_rng(ev.SEED).random(len(d)), '단일 z(|z|)': pz.abs(), '개인 기준선': pb,
           '집단 기준선': bl.bad_score(bl.group_z(d, col), col)}
    cus = {CS: ('개인 기준선', 0.5)}
    for m, (dow, anc, k) in PR1.items():
        out[f'{m}|base'] = F1.personal_bad(d, ref, dow, anc, col=col)
        cus[m] = (f'{m}|base', k)
    z3 = bl.personal_z(d.assign(mid_sd=pr3_mid_sd(d)), 'mid_sd')
    out[PR3_Z] = z3                                   # 커질수록 나쁨 (방향 +1)
    cus[PR3_CS] = (PR3_Z, 0.5)
    return pd.DataFrame(out, index=d.index), cus


def make_score_fn(clean, col, ref):
    B0, cus = base_scores(clean, col, ref)
    hs = {m: bl.cusum_threshold(clean.assign(_b=B0[b]), '_b', k=k, rate=RATE) for m, (b, k) in cus.items()}

    def score_fn(d):
        B, _ = base_scores(d, col, ref)
        S = {c: B[c] for c in COMMON if c != CS}
        for m, (b, k) in cus.items():
            S[m] = bl.cusum(d.assign(_b=B[b]), '_b', k=k, h=hs[m])
        S[PR3_Z] = B[PR3_Z]
        return pd.DataFrame(S, index=d.index)[ORDER]
    score_fn.h = hs
    score_fn.cus = cus
    return score_fn


# ---------------------------------------------------------------------------
# 방식 C: 구간 오경보율 0.20 맞춤. 구간·심기는 evaluate_injection 과 같은 함수를 같은 시드로 부른다.
# ---------------------------------------------------------------------------
def collect(nh, col, eff, mode, shape, n_reps, ref):
    """반복별 구간 날의 대조군/주입 기반 점수. 열: rep, pid, date, start, day, idx, c|<열>, i|<열>"""
    B0, cus = base_scores(nh, col, ref)
    parts = []
    for r in range(n_reps):
        inj, seg = inject(nh, col, eff, mode, shape, ev.SEED + r)
        Bi, _ = base_scores(inj, col, ref)
        rows = ev._segment_rows(nh, seg)
        t = rows.reset_index().rename(columns={'index': 'idx'})
        t['rep'] = r
        t['day'] = (t['date'] - t['start']).dt.days
        for c in B0.columns:
            t[f'c|{c}'] = B0.loc[rows.index, c].to_numpy()
            t[f'i|{c}'] = Bi.loc[rows.index, c].to_numpy()
        parts.append(t)
        log('  방식 C 수집', col, eff, shape, 'rep', r)
    return pd.concat(parts, ignore_index=True), B0, cus


def alarms_C(nh, B0, seg, name, cus):
    if name in cus:
        base, k = cus[name]
        idx = seg['idx'].to_numpy()

        def fa(hh):
            S = R3.cusum_full(nh, B0[base], hh, k=k)
            return R3.window_fa(seg, S.loc[idx].to_numpy() > hh)
        lo, hi = 0.0, 60.0
        for _ in range(22):
            mid = (lo + hi) / 2
            if fa(mid) > WIN_FA:
                lo = mid
            else:
                hi = mid
        h = hi
        S = R3.cusum_full(nh, B0[base], h, k=k)
        return h, S.loc[idx].to_numpy() > h, R3.cusum_segment(nh, S, seg, seg[f'i|{base}'], h, k=k) > h
    sid = seg['rep'].astype(str) + '_' + seg['pid'].astype(str)
    Mc = pd.Series(seg[f'c|{name}'].to_numpy()).groupby(sid.to_numpy()).max().fillna(-np.inf)
    t = float(np.quantile(Mc.to_numpy(), 1 - WIN_FA))
    return t, seg[f'c|{name}'].to_numpy() > t, seg[f'i|{name}'].to_numpy() > t


def evaluate_C(nh, B0, seg, cus, scen):
    rows, pps = [], []
    for name in ORDER:
        thr, a_ctl, a_inj = alarms_C(nh, B0, seg, name, cus)
        pp, w = R3.seg_metrics(seg, a_ctl, a_inj)
        cd = w[w['detected'] & ~w['fa']]
        rows.append({'시나리오': scen, '방식': 'C', '방법': name, '기준값': thr, '탐지율': w['detected'].mean(),
                     '구간 오경보율': w['fa'].mean(), '차이(평균)': pp['차이'].mean(), '차이(중앙값)': pp['차이'].median(),
                     '탐지 지연': cd['delay'].median(), '사람 수': len(pp)})
        pps.append(pp.assign(시나리오=scen, 방식='C', method=name))
    return pd.DataFrame(rows), pd.concat(pps, ignore_index=True)


def alarms_A_loop(nh, B0, seg, name, cus, hs):
    """evaluate_injection 의 score_fn 경로와 같은 기준값: 심기 전 전체 날 점수의 상위 1/30 분위수. CUSUM 은 h 로 초기화한 S 의 분위수."""
    if name in cus:
        base, k = cus[name]
        S = bl.cusum(nh.assign(_b=B0[base]), '_b', k=k, h=hs[name])
        t = float(S.quantile(1 - RATE))
        idx = seg['idx'].to_numpy()
        return t, S.loc[idx].to_numpy() > t, R3.cusum_segment(nh, S, seg, seg[f'i|{base}'], hs[name], k=k) > t
    t = float(B0[name].quantile(1 - RATE))
    return t, seg[f'c|{name}'].to_numpy() > t, seg[f'i|{name}'].to_numpy() > t


def evaluate_A_loop(nh, B0, seg, cus, hs, scen):
    rows, pps = [], []
    for name in ORDER:
        thr, a_ctl, a_inj = alarms_A_loop(nh, B0, seg, name, cus, hs)
        pp, w = R3.seg_metrics(seg, a_ctl, a_inj)
        cd = w[w['detected'] & ~w['fa']]
        rows.append({'시나리오': scen, '방식': 'A(루프)', '방법': name, '기준값': thr,
                     '탐지율': pp['탐지율'].median(), '오경보율': pp['오경보율'].median(), '차이': pp['차이'].median(),
                     '탐지 지연': cd['delay'].median(), '탐지율(평균)': pp['탐지율'].mean(), '오경보율(평균)': pp['오경보율'].mean(),
                     '차이(평균)': pp['차이'].mean(), '사람 수': len(pp)})
        pps.append(pp.assign(시나리오=scen, 방식='A(루프)', method=name))
    return pd.DataFrame(rows), pd.concat(pps, ignore_index=True)


def _merge_csv(path, new, key='시나리오'):
    kw = {'compression': 'gzip'} if path.suffix == '.gz' else {}
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old[key].isin(new[key].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False, encoding='utf-8-sig', **kw)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--scenarios', nargs='+', default=list(SCEN))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'integrated'))
    ap.add_argument('--skip-c', action='store_true')
    ap.add_argument('--append', action='store_true', help='기존 table_*.csv 에서 같은 시나리오 행만 바꿔 넣음')
    ap.add_argument('--loop-check', nargs='*', default=[], help='이 공통 시나리오를 루프 방식 A 로도 세어 evaluate_injection 과 대조')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nh = dl.load_nethealth(path=a.data)
    log('하루 표', nh.shape, '사람', nh['pid'].nunique())

    rows_A, pp_A, rows_C, pp_C, checks = [], [], [], [], []
    for scen in a.scenarios:
        col, eff, mode, shape = SCEN[scen]
        ref = nh.iloc[0:0][['date', col]]                     # 참고 집단 없음 (전원이 평가 대상)
        fn = make_score_fn(nh, col, ref)
        log(scen, 'CUSUM 경보선', {m: round(h, 3) for m, h in fn.h.items()})

        # 확인: PR 1 '레포 기본' 기반 점수 == 공통 '개인 기준선'
        B0, _ = base_scores(nh, col, ref)
        pb_pr1 = F1.personal_bad(nh, ref, False, False, col=col)
        d = (pb_pr1 - B0['개인 기준선']).abs()
        checks.append({'시나리오': scen, '검사': "PR 1 '레포 기본' 기반 점수 = 공통 '개인 기준선'",
                       '최대 차': float(np.nanmax(d)), '빈칸 불일치': int((pb_pr1.isna() != B0['개인 기준선'].isna()).sum())})

        is_jitter = shape == JITTER
        if not is_jitter:
            # 방식 A: 공통 함수 그대로
            res = ev.evaluate_injection(nh, col, eff, mode=mode, shape=shape, n_reps=a.reps, seed=ev.SEED, score_fn=fn)
            tab = ev.results_table(res, order=ORDER)
            m = res['per_person'].groupby('method')[['탐지율', '오경보율', '차이']].mean().add_suffix('(평균)')
            tab = tab.merge(m.round(4), left_on='방법', right_index=True)
            tab.insert(0, '방식', 'A')
            tab.insert(0, '시나리오', scen)
            rows_A.append(tab)
            pp_A.append(res['per_person'].assign(시나리오=scen, 방식='A'))
            log(scen, '방식 A 완료')

        need_seg = is_jitter or (not a.skip_c) or (scen in a.loop_check)
        if need_seg:
            base_nh = R3.add_sleep_mid(nh) if is_jitter else nh
            seg, B0, cus = collect(base_nh, col, eff, mode, shape, a.reps, ref)
            if is_jitter or scen in a.loop_check:
                r, p = evaluate_A_loop(base_nh, B0, seg, cus, fn.h, scen)
                if is_jitter:
                    r['방식'] = 'A'
                    p['방식'] = 'A'
                    rows_A.append(r)
                    pp_A.append(p)
                else:
                    r.to_csv(out / f'loop_check_{scen}.csv', index=False, encoding='utf-8-sig')
                    p.to_csv(out / f'loop_check_{scen}_per_person.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
                log(scen, '루프 방식 A 완료')
            if not a.skip_c:
                r, p = evaluate_C(base_nh, B0, seg, cus, scen)
                rows_C.append(r)
                pp_C.append(p)
                log(scen, '방식 C 완료')

    A = pd.concat(rows_A, ignore_index=True)
    PA = pd.concat(pp_A, ignore_index=True)
    CK = pd.DataFrame(checks)
    if a.append:
        _merge_csv(out / 'table_A.csv', A)
        _merge_csv(out / 'per_person_A.csv.gz', PA)
        _merge_csv(out / 'check_pr1_base.csv', CK)
        A = pd.read_csv(out / 'table_A.csv')
    else:
        A.to_csv(out / 'table_A.csv', index=False, encoding='utf-8-sig')
        PA.to_csv(out / 'per_person_A.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
        CK.to_csv(out / 'check_pr1_base.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n=== 방식 A (전원, 하루 경보율 1/30): 차이 = 탐지율 - 오경보율, 사람별 중앙값 ===')
    shown = list(A['시나리오'].unique())
    print(A.pivot(index='방법', columns='시나리오', values='차이(평균)').reindex(ORDER)[shown].round(3).to_string())
    print('\n=== 방식 A: 탐지율 중앙값 ===')
    print(A.pivot(index='방법', columns='시나리오', values='탐지율').reindex(ORDER)[shown].round(2).to_string())
    if rows_C:
        Cc = pd.concat(rows_C, ignore_index=True)
        PCc = pd.concat(pp_C, ignore_index=True)
        if a.append:
            _merge_csv(out / 'table_C.csv', Cc)
            _merge_csv(out / 'per_person_C.csv.gz', PCc)
            Cc = pd.read_csv(out / 'table_C.csv')
        else:
            Cc.to_csv(out / 'table_C.csv', index=False, encoding='utf-8-sig')
            PCc.to_csv(out / 'per_person_C.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')
        print('\n=== 방식 C (구간 오경보율 0.20): 차이(평균) ===')
        print(Cc.pivot(index='방법', columns='시나리오', values='차이(평균)').reindex(ORDER)[list(Cc['시나리오'].unique())].round(3).to_string())
    print('\n=== 검사 ===')
    print(pd.DataFrame(checks).to_string(index=False))
    log('저장', out)


if __name__ == '__main__':
    main()
