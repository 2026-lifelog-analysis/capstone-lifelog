# 작성 2026-11-06 08:20 (KST)
"""
run_extended.py — ① 확장 계열 실행 (공통 5-fold, 최종 평가 참가자는 쓰지 않음)

    python -m experiments.seungho.run_extended phase3 --chunk 0 --nchunks 3     # 계열별·양 조합·큰 조합 (시드 0)
    python -m experiments.seungho.run_extended greedy --target sleep             # 블록 단위 전진 선택 (시드 0)
    python -m experiments.seungho.run_extended phase5 --seed 1 --phase3 <폴더들> --greedy <폴더>   # 상위 확인 (시드 1~4)

전진 선택 규칙 (결과 보기 전에 고정)
  - 출발: cb_base (목표별 기본 입력). 단계마다 남은 블록을 하나씩 더해 oof MAE가 가장 작은 것을 채택
  - 멈춤: 가장 좋은 블록의 개선이 MIN_GAIN분 미만이거나 MAX_STEPS 단계
  - 개발 fold에서 고른 것이라 숫자는 낙관적이다. 확정은 시드 1~4 재확인 + 나중에 팀 최종 평가 93명
"""
import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from src import forecast_dataset as fd
from src import forecast_run as fr
from experiments.seungho import sleep_delta as sd
from experiments.seungho import extended as ext

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE = 'cb_base'
MIN_GAIN = 0.05
MAX_STEPS = 10
TOP_K = 5


def _prepare(synthetic=False):
    data, split, man, out = fr.prepare(synthetic=synthetic)
    if synthetic:
        table = fd.add_features(data, sd.feat_delta, _synthetic_ext)
    else:
        table = fd.add_features(data, sd.feat_delta, ext.feat_extended)
    return data, split, man, out, table


def _synthetic_ext(data):
    """가짜 데이터에는 설문·달력·활동 원본 열이 없어 같은 이름의 가짜 표를 만든다 (흐름 확인용)."""
    import numpy as np
    rng = np.random.default_rng(0)
    pids = data.table['pid'].unique()
    sv = pd.DataFrame({'pid': pids, **{c: rng.normal(size=len(pids)) for g in ext.P_GROUPS.values() for c in g}})
    days = pd.DataFrame({'date': pd.date_range('2015-01-01', '2017-12-31')})
    for c in ['h_class', 'h_finals', 'h_break', 'h_winter', 'h_summer', 'h_term_week', 'h_days_to_finals',
              'h_fed']:
        days[c] = rng.integers(0, 2, len(days)).astype(float)
    days['off'] = (days['date'].dt.dayofweek >= 5).astype(float)
    a = data.activity[['pid', 'date']].copy()
    for c in ext.ACT:
        a[c] = rng.normal(size=len(a))
    return ext.extended_features(data, survey=sv, cal=days, act=a)


def _summary(rd):
    return pd.read_csv(Path(rd) / 'summary.csv')


def phase3(a):
    data, split, man, out, table = _prepare(a.synthetic)
    exps = ext.phase3_exps()
    names = [n for n in exps if not a.only or n in a.only][a.chunk::a.nchunks]
    todo = [sd.exp(BASE, [])] + [exps[n] for n in names]
    print(f'[3차 묶음 {a.chunk}/{a.nchunks}] {len(todo)}개', flush=True)
    _, summ, comp, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=a.targets,
                               name=f'seungho-ext-p3c{a.chunk}', baseline=BASE, synthetic=a.synthetic)
    print(summ[['target', 'experiment', 'oof_mae']].round(2).to_string(index=False))
    print(f'저장: {rd}', flush=True)


def greedy(a):
    data, split, man, out, table = _prepare(a.synthetic)
    t = a.target
    chosen, path = [], []
    _, summ, _, rd = fr.run(data, split, man, out, experiments=[sd.exp(BASE, [])], table=table, targets=[t],
                            name=f'seungho-ext-gr-{t}-k0', synthetic=a.synthetic)
    cur = float(summ['oof_mae'].iloc[0])
    path.append({'step': 0, 'added': '(기본)', 'oof_mae': cur, 'gain': 0.0, 'blocks': '', 'run': rd.name})
    print(f'[전진 선택 {t}] 시작 {cur:.3f}', flush=True)
    blocks = list(a.blocks or ext.GREEDY_BLOCKS)
    for step in range(1, MAX_STEPS + 1):
        left = [b for b in blocks if b not in chosen]
        if not left:
            break
        todo = [ext.exp(f'GR_{t}_k{step}_{b}', chosen + [b]) for b in left]
        _, summ, _, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=[t],
                                name=f'seungho-ext-gr-{t}-k{step}', synthetic=a.synthetic)
        s = summ.sort_values('oof_mae')
        best = s.iloc[0]
        b = best['experiment'].split(f'_k{step}_', 1)[1]
        gain = cur - float(best['oof_mae'])
        cand = '; '.join(f"{r.experiment.split(f'_k{step}_', 1)[1]} {cur - r.oof_mae:+.3f}"
                         for r in s.head(6).itertuples())
        print(f'  단계 {step}: 최고 {b} 개선 {gain:+.3f}분 | 상위: {cand}', flush=True)
        path.append({'step': step, 'added': b, 'oof_mae': float(best['oof_mae']), 'gain': gain,
                     'blocks': ','.join(chosen + [b]), 'run': rd.name, 'top6': cand})
        if gain < MIN_GAIN:
            path[-1]['accepted'] = False
            break
        path[-1]['accepted'] = True
        chosen.append(b)
        cur = float(best['oof_mae'])
    p = pd.DataFrame(path)
    gdir = Path(out) / f'seungho-ext-greedy'
    gdir.mkdir(parents=True, exist_ok=True)
    p.to_csv(gdir / f'path_{t}.csv', index=False)
    (gdir / f'chosen_{t}.json').write_text(json.dumps(chosen, ensure_ascii=False))
    print(f'[전진 선택 {t}] 고른 블록 {chosen} → {cur:.3f}', flush=True)
    print(f'저장: {gdir}', flush=True)


def phase5(a):
    """시드 a.seed로: 목표별 전진 선택 결과 + 3차 상위 TOP_K + K2_everything."""
    data, split, man, out, table = _prepare(a.synthetic)
    exps = ext.phase3_exps()
    s3 = pd.concat([_summary(d) for d in a.phase3]).drop_duplicates(['target', 'experiment'])
    s3 = s3[(s3['status'] == 'complete') & (s3['experiment'] != BASE)]
    gdir = Path(a.greedy)
    for t in fd.TARGETS:
        top = s3[s3['target'] == t].nsmallest(TOP_K, 'oof_mae')['experiment'].tolist()
        names = list(dict.fromkeys(top + ['K2_everything']))
        todo = [sd.exp(BASE, [], a.seed)] + [sd.with_seed(exps[n], a.seed) for n in names]
        cf = gdir / f'chosen_{t}.json'
        if cf.exists():
            chosen = json.loads(cf.read_text())
            if chosen:
                todo.append(ext.exp(f'GRfinal_{t}', chosen, a.seed))
        print(f'[확인 시드 {a.seed} {t}] {[e.name for e in todo]}', flush=True)
        _, summ, _, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=[t],
                                name=f'seungho-ext-p5s{a.seed}-{t}', baseline=todo[0].name,
                                synthetic=a.synthetic)
        print(summ[['target', 'experiment', 'oof_mae']].round(3).to_string(index=False))
        print(f'저장: {rd}', flush=True)


def greedy_final0(a):
    """전진 선택 결과를 시드 0 이름(GRfinal_t)으로 한 번 더 저장 (판정에서 시드 0~4 묶음을 만들기 위해)."""
    data, split, man, out, table = _prepare(a.synthetic)
    gdir = Path(a.greedy)
    for t in fd.TARGETS:
        cf = gdir / f'chosen_{t}.json'
        chosen = json.loads(cf.read_text()) if cf.exists() else []
        if not chosen:
            continue
        todo = [sd.exp(BASE, []), ext.exp(f'GRfinal_{t}', chosen)]
        _, summ, _, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=[t],
                                name=f'seungho-ext-grfinal-{t}', baseline=BASE, synthetic=a.synthetic)
        print(f'저장: {rd}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='① 확장 계열 (공통 5-fold)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p3 = sub.add_parser('phase3')
    p3.add_argument('--chunk', type=int, default=0)
    p3.add_argument('--nchunks', type=int, default=1)
    p3.add_argument('--only', nargs='+')
    p3.add_argument('--targets', nargs='+', choices=list(fd.TARGETS))
    p3.add_argument('--synthetic', action='store_true')
    g = sub.add_parser('greedy')
    g.add_argument('--target', required=True, choices=list(fd.TARGETS))
    g.add_argument('--blocks', nargs='+')
    g.add_argument('--synthetic', action='store_true')
    p5 = sub.add_parser('phase5')
    p5.add_argument('--seed', type=int, required=True)
    p5.add_argument('--phase3', nargs='+', required=True)
    p5.add_argument('--greedy', required=True)
    p5.add_argument('--synthetic', action='store_true')
    gf = sub.add_parser('grfinal')
    gf.add_argument('--greedy', required=True)
    gf.add_argument('--synthetic', action='store_true')
    a = ap.parse_args(argv)
    {'phase3': phase3, 'greedy': greedy, 'phase5': phase5, 'grfinal': greedy_final0}[a.cmd](a)


if __name__ == '__main__':
    main()
