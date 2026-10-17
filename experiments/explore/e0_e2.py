# 작성 2026-10-15 01:30
"""
E0~E2 (docs/EXPERIMENT_PLAN.md 4장) — 개발용 70% 사람에서만.

  E0        기본값 (레포 그대로)                         S1~S4
  E0-seed   기본값, 시드만 1000번대로 → 시드 흔들림 측정   S1~S4
  E0-check  score_fn 경로로 E0 재현 → 실험 도구 검증       S1
  E1-b      steps_per_hour (착용시간 정규화)               S1~S3
  E1-c      착용 90% 기준                                  S1~S3
  E2-b      요일 7개 보정 (레포 dow_adjust)                S1~S4
  E2-c      평일/주말 2그룹 보정                           S1~S4

사용법: python3 -I experiments/explore/e0_e2.py [반복 수=20] [프로세스 수=6]
출력:   results/tuning/e0_e2_table.csv, e0_e2_compare.csv, e0_e2_calendar.csv, runs/*.pkl
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C          # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

TASKS = ([('E0', s) for s in ['S1', 'S2', 'S3', 'S4']]
         + [('E0-seed', s) for s in ['S1', 'S2', 'S3', 'S4']]
         + [('E0-check', 'S1')]
         + [('E1-b', s) for s in ['S1', 'S2', 'S3']]
         + [('E1-c', s) for s in ['S1', 'S2', 'S3']]
         + [('E2-b', s) for s in ['S1', 'S2', 'S3', 'S4']]
         + [('E2-c', s) for s in ['S1', 'S2', 'S3', 'S4']])


def z_none(d, c):
    return C.personal_z_grouped(d, c, None)


def z_weekend(d, c):
    return C.personal_z_grouped(d, c, C.WEEKEND_GROUP)


def one(task, n_reps):
    exp, sc = task
    col = C.SCENARIOS[sc][0]
    if exp == 'E1-c':
        df = C.dev_set(C.load(wear=90))
    else:
        df = C.dev_set(C.load())
    if exp == 'E0':
        res = C.run(df, sc, n_reps=n_reps)
    elif exp == 'E0-seed':
        res = C.run(df, sc, n_reps=n_reps, seed=1000)
    elif exp == 'E0-check':
        res = C.run(df, sc, n_reps=n_reps, score_fn=C.make_score_fn(df, col, z_none))
    elif exp == 'E1-b':
        res = C.run(df, sc, n_reps=n_reps, col='steps_per_hour')
    elif exp == 'E1-c':
        res = C.run(df, sc, n_reps=n_reps)
    elif exp == 'E2-b':
        res = C.run(df, sc, n_reps=n_reps, dow_adjust=True)
    elif exp == 'E2-c':
        res = C.run(df, sc, n_reps=n_reps, score_fn=C.make_score_fn(df, col, z_weekend))
    return exp, sc, res


def calendar_table():
    """주입 없는 개발용 데이터에서 달력 상태별 경보율 (E0 / E2-b / E2-c, 걸음·취침)."""
    df = C.dev_set(C.load())
    rows = []
    for col in ['steps', 'bed_min']:
        for name, kw in [('E0', {}), ('E2-b', {'dow_adjust': True})]:
            sc = C.ev.method_scores(df, col, **kw)
            pb = sc['개인 기준선']
            h = C.bl.cusum_threshold(df.assign(_b=pb), '_b')
            sc[C.CS] = C.bl.cusum(df.assign(_b=pb), '_b', h=h)
            for m in ['개인 기준선', C.CS, '집단 기준선']:
                rows.append(C.calendar_alarm_rates(df, sc, m, h=h).assign(실험=name, 지표=col, 방법=m))
        f = C.make_score_fn(df, col, z_weekend)
        sc = f(df)
        for m in ['개인 기준선', C.CS]:
            rows.append(C.calendar_alarm_rates(df, sc, m, h=f.h).assign(실험='E2-c', 지표=col, 방법=m))
    return pd.concat(rows, ignore_index=True)


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = C.OUT
    (out / 'runs').mkdir(parents=True, exist_ok=True)

    dev = C.dev_set(C.load())
    print(f"개발용: {dev['pid'].nunique()}명, {len(dev):,}일 / 반복 {n_reps}회 / 작업 {len(TASKS)}개")

    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for exp, sc, r in ex.map(partial(one, n_reps=n_reps), TASKS):
            res[(exp, sc)] = r
            with open(out / 'runs' / f'{exp}_{sc}.pkl', 'wb') as fh:
                pickle.dump(r, fh)
            print(f'  끝: {exp} {sc} ({r["sec"]}초)', flush=True)

    tab = pd.concat([C.summary(r, e).assign(시나리오=s) for (e, s), r in res.items()], ignore_index=True)
    tab.to_csv(out / 'e0_e2_table.csv', index=False, encoding='utf-8-sig')

    cmp_rows = []
    for (e, s), r in res.items():
        if e == 'E0':
            continue
        for m in [C.CS, '개인 기준선']:
            cmp_rows.append({'실험': e, '시나리오': s, '방법': m, **C.compare(res[('E0', s)], r, method=m)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out / 'e0_e2_compare.csv', index=False, encoding='utf-8-sig')

    cal = calendar_table()
    cal.to_csv(out / 'e0_e2_calendar.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    show = ['실험', '시나리오', '방법', '탐지율', '오경보율', '차이', '탐지 지연', '차이(평균)', 'PR-AUC(보조)', '사람 수']
    print('\n=== 결과표 (사람별 중앙값 + 평균) ===')
    print(tab[tab['방법'].isin([C.CS, '개인 기준선', '집단 기준선', '무작위'])][show].to_string(index=False))
    print('\n=== E0 대비 (같은 사람끼리, 차이 기준) ===')
    print(cmp.to_string(index=False))
    print('\n=== 달력 상태별 대조군 경보율 (CUSUM, 걸음) ===')
    c = cal[(cal['방법'] == C.CS) & (cal['지표'] == 'steps')]
    print(c.pivot_table(index=['구분', '값'], columns='실험', values='경보율').round(4).to_string())


if __name__ == '__main__':
    main()
