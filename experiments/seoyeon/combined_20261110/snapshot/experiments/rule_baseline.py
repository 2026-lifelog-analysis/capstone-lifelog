# 작성 2026-10-15 04:05
"""
단순 규칙 기준선 — "기기 기본 기능보다 나은가"에 답하기 위한 비교 (확인용 30% 사람).

규칙: 최근 7일(오늘 포함, 달력) 걸음 평균이 그 앞 28일(달력) 평균보다 30% 이상 낮으면 경보
  점수 = 1 − 최근7일 평균 / 앞28일 평균   (최근 7일 중 기록 4일 이상, 앞 28일 중 14일 이상일 때만)
  튜닝할 값이 없는 고정 규칙이라 확인용 사람에서 바로 잰다 (최종 방법 선택과 무관).

두 가지로 비교
  ① 하루 경보 1/30 맞춤 : 규칙의 기준을 30% 대신 하루 경보율 1/30이 되는 값으로 → 최종 방법과 같은 예산
  ② 감소율 고정 그대로  : 규칙이 원래 울리는 만큼(하루 경보율 p0) 울림. 최종 방법도 같은 p0로 맞춰 비교
                         20·30·40·50% 네 기준으로 곡선을 그려, 14일 오경보율이 같은 지점끼리 비교
같은 시드라 주입 구간은 final.py와 같다.

사용법: python3 -I experiments/rule_baseline.py [반복 수=20] [프로세스 수=6]
출력:   results/final/rule_table.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C          # noqa: E402
import final as F           # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

ev, bl = C.ev, C.bl
RULE = '단순 규칙 (7일 vs 앞 28일)'
DROPS = [0.20, 0.30, 0.40, 0.50]   # 규칙 기준(감소율). 0.30이 기본 규칙, 나머지는 곡선용


def rule_score(d, col='steps', recent=7, before=28):
    """1 − 최근 recent일 평균 / 그 앞 before일 평균. 클수록 많이 줄어듦."""
    out = pd.Series(np.nan, index=d.index, dtype=float)
    for pid, idx, dates, full in bl._per_person_daily(d, col):
        r = full.rolling(recent, min_periods=4).mean()
        b = full.shift(recent).rolling(before, min_periods=14).mean()
        s = 1 - r / b.where(b > 0)
        out.loc[idx] = s.reindex(dates.values).to_numpy()
    return out


def make_score_fn(df, ref, rate):
    f = F.make_score_fn(df, ref, rate)

    def score_fn(d):
        sc = f(d)[['무작위', F.BASE, F.FINAL]]
        sc[RULE] = rule_score(d)
        return sc
    return score_fn


def one(task, n_reps, rates):
    sc, rn = task
    effects, shape, _ = F.SCEN[sc]
    df = C.holdout_set(C.load())
    ref = C.dev_set(C.load())
    (col, eff, mode), *extra = effects
    C.INJECT_EXTRA[:] = extra
    res = ev.evaluate_injection(df, col, eff, mode=mode, shape=shape, n_reps=n_reps,
                                rate=rates[rn], score_fn=make_score_fn(df, ref, rates[rn]))
    return sc, rn, res


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = C.OUT.parent / 'final'
    out.mkdir(parents=True, exist_ok=True)

    hold = C.holdout_set(C.load())
    s = rule_score(hold)
    rates = {'1/30': 1 / 30}
    for dr in DROPS:
        p0 = float((s[s.notna()] >= dr).mean())
        rates[f'{int(dr * 100)}%고정'] = p0
        print(f'단순 규칙({int(dr * 100)}% 감소)이 원래 울리는 하루 경보율: {p0:.4f} (≈ {1 / p0:.0f}일에 1번)', flush=True)

    tasks = [(sc, rn) for sc in F.SCEN for rn in rates]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for sc, rn, r in ex.map(partial(one, n_reps=n_reps, rates=rates), tasks):
            t = C.summary(r, '').drop(columns='실험').assign(시나리오=sc, 하루경보=rn)
            if rn != '1/30':
                t['규칙 기준값'] = r['threshold'][RULE]
            rows.append(t)
            print(f'  끝: {sc} {rn}', flush=True)
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(out / 'rule_table.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    show = ['시나리오', '하루경보', '방법', '탐지율(평균)', '오경보율(평균)', '차이(평균)', '탐지 지연', '하루 오경보율']
    print(tab[tab['방법'] != '무작위'][show].to_string(index=False))
    if '규칙 기준값' in tab:
        print('\n고정 판에서 분위수로 잡힌 규칙 기준값:', tab['규칙 기준값'].dropna().round(3).unique())


if __name__ == '__main__':
    main()
