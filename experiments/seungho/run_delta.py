# 작성 2026-11-06 07:15 (KST)
"""
run_delta.py — ① 평소 대비 최근 수면 변화량을 공통 5-fold로 실행 (최종 평가 참가자는 쓰지 않음)

    # 1차: 시드 0으로 S1~S4 전체. 여러 프로세스로 나눠 돌림 (각 묶음에 cb_base가 같이 들어감 = 재현 확인)
    python -m experiments.seungho.run_delta phase1 --chunk 0 --nchunks 3
    # 2차: 주 비교 B0~B3 + 목표별 1차 상위 5개를 시드 1~4로 다시 (시드 하나 = 묶음 하나)
    python -m experiments.seungho.run_delta phase2 --seed 1 --phase1 <1차 run 폴더들>
    # 분석: 합치기, 시드 잡음, 사람 단위 부트스트랩, 판정, 구간별 분석
    python -m experiments.seungho.run_delta analyze --runs <run 폴더들> --out <폴더>
    # 가짜 데이터로 흐름만 확인
    python -m experiments.seungho.run_delta phase1 --synthetic --only cb_base cb_delta

판정 기준 (설계서 4.2, 결과 보기 전에 고정). 셋 다 만족해야 "후보"
  1. 다섯 fold 모두 개선 (시드 평균 fold MAE 기준)
  2. 사람 단위 부트스트랩(2,000회) 95% 구간이 0을 포함하지 않음
  3. 개선폭 > 2 × cb_base 시드 간 oof MAE 표준편차
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from src import preprocessing as pp
from experiments.seungho import sleep_delta as sd

BASE = 'cb_base'
TOP_K = 5
N_BOOT = 2000
BOOT_SEED = 0
DELTA_BINS = [0, 15, 30, 60, np.inf]
DELTA_LABELS = ['0~15', '15~30', '30~60', '60+']
HIST_EDGES = [-0.5, 0.5, 3.5, 7.5, 14.5, 21.5, 28.5]          # ③(서연)과 같은 구간
HIST_LABELS = ['0', '1~3', '4~7', '8~14', '15~21', '22~28']
BREAKDOWN = ['cb_level', 'cb_delta', 'cb_level_delta']        # + 목표별 최고 변형

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def _prepare(synthetic):
    data, split, man, out = fr.prepare(synthetic=synthetic)
    table = fd.add_features(data, sd.feat_delta)
    return data, split, man, out, table


def _print(summary, comp, rd, synthetic):
    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.max_rows', 500):
        if synthetic:
            print('[가짜 데이터] 아래 숫자는 실제 성능이 아님')
        print(summary[['target', 'experiment', 'status', 'fold_mae_mean', 'oof_mae']].round(2).to_string(index=False))
        if len(comp):
            m = comp[comp['fold'] == 'mean']
            print(m[['target', 'new', 'status', 'base_mae', 'new_mae', 'diff', 'n_improved']].round(2)
                  .to_string(index=False))
    print(f'저장: {rd}')


def phase1(a):
    data, split, man, out, table = _prepare(a.synthetic)
    exps = sd.all_phase1()
    names = [n for n in exps if n != BASE]
    if a.only:
        names = [n for n in names if n in a.only]
    mine = names[a.chunk::a.nchunks]
    todo = [exps[BASE]] + [exps[n] for n in mine]
    print(f'[1차 묶음 {a.chunk}/{a.nchunks}] {len(todo)}개 실험: {[e.name for e in todo]}', flush=True)
    s, summ, comp, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=a.targets,
                               name=f'seungho-delta-p1c{a.chunk}', baseline=BASE, synthetic=a.synthetic)
    _print(summ, comp, rd, a.synthetic)


def _phase1_summary(dirs):
    s = pd.concat([pd.read_csv(Path(d) / 'summary.csv') for d in dirs], ignore_index=True)
    return s.drop_duplicates(['target', 'experiment'])


def top_by_target(dirs, k=TOP_K):
    """1차에서 주 비교(B0~B3)를 뺀 실험 중 목표별 oof MAE가 가장 작은 k개."""
    s = _phase1_summary(dirs)
    main = {e.name for e in sd.main_exps(0)}
    s = s[(s['status'] == 'complete') & ~s['experiment'].isin(main)]
    return {t: g.nsmallest(k, 'oof_mae')['experiment'].tolist() for t, g in s.groupby('target')}


def phase2(a):
    data, split, man, out, table = _prepare(a.synthetic)
    exps = sd.all_phase1()
    tops = top_by_target(a.phase1)
    print(f'[2차 시드 {a.seed}] 목표별 상위 {TOP_K}: {tops}', flush=True)
    (Path(out) / 'runs').mkdir(parents=True, exist_ok=True)
    main = sd.main_exps(a.seed)
    s, summ, comp, rd = fr.run(data, split, man, out, experiments=main, table=table,
                               name=f'seungho-delta-p2s{a.seed}-main', baseline=main[0].name,
                               synthetic=a.synthetic)
    _print(summ, comp, rd, a.synthetic)
    for t, names in tops.items():
        todo = [main[0]] + [sd.with_seed(exps[n], a.seed) for n in names]
        s, summ, comp, rd = fr.run(data, split, man, out, experiments=todo, table=table, targets=[t],
                                   name=f'seungho-delta-p2s{a.seed}-top-{t}', baseline=main[0].name,
                                   synthetic=a.synthetic)
        _print(summ, comp, rd, a.synthetic)
    (rd.parent / f'seungho-delta-p2s{a.seed}-tops.json').write_text(json.dumps(tops, ensure_ascii=False))


# ---------- 분석 ----------

SEED_RE = re.compile(r'^(.*?)(?:_s(\d+))?$')


def family_seed(name):
    m = SEED_RE.match(name)
    return m.group(1), int(m.group(2) or 0)


def load_runs(dirs):
    """
    run 폴더들을 읽어 (목표, 실험, fold, 사람)별 |오차| 합·제곱 합·밤 수로 줄인다 (예측 파일이 커서 한 번에 안 읽음).
    같은 실험이 여러 폴더에 있으면(묶음마다 넣은 cb_base) 예측이 같은지 확인하고 하나만 쓴다.
    """
    agg, seen = [], {}
    for d in dirs:
        d = Path(d)
        for ch in pd.read_csv(d / 'predictions.csv', dtype={'pid': str}, chunksize=2_000_000):
            ch['err'] = ch['y_pred'] - ch['y_true']
            for (t, e), g in ch.groupby(['target', 'experiment']):
                key = (t, e)
                if key in seen and seen[key] != d:
                    continue
                seen.setdefault(key, d)
                agg.append(g.assign(ae=g['err'].abs(), se=g['err'] ** 2)
                           .groupby(['target', 'experiment', 'fold', 'pid'])
                           .agg(sae=('ae', 'sum'), sse=('se', 'sum'), n=('ae', 'size')).reset_index())
    a = pd.concat(agg).groupby(['target', 'experiment', 'fold', 'pid'], as_index=False).sum()
    return a, seen


def check_duplicates(dirs, name=BASE):
    """여러 묶음에 들어간 cb_base 예측이 서로 같은지 (결정성 확인)."""
    ref, res = None, []
    for d in dirs:
        p = pd.read_csv(Path(d) / 'predictions.csv', dtype={'pid': str}, usecols=['pid', 'date', 'target',
                                                                                  'experiment', 'y_pred'])
        p = p[p['experiment'] == name].sort_values(['target', 'pid', 'date']).reset_index(drop=True)
        if not len(p):
            continue
        if ref is None:
            ref = p
        else:
            res.append(float(np.abs(p['y_pred'].to_numpy() - ref['y_pred'].to_numpy()).max()))
    return res


def summarize(a):
    """실험별 oof MAE·RMSE·fold MAE."""
    g = a.groupby(['target', 'experiment'])
    s = pd.DataFrame({'oof_mae': g['sae'].sum() / g['n'].sum(),
                      'oof_rmse': np.sqrt(g['sse'].sum() / g['n'].sum()),
                      'n_nights': g['n'].sum(), 'n_people': g['pid'].nunique()}).reset_index()
    f = a.groupby(['target', 'experiment', 'fold']).apply(lambda x: x['sae'].sum() / x['n'].sum(),
                                                          include_groups=False).unstack('fold')
    f.columns = [f'fold{c}' for c in f.columns]
    s = s.merge(f.reset_index(), on=['target', 'experiment'])
    s[['family', 'seed']] = s['experiment'].apply(lambda n: pd.Series(family_seed(n)))
    return s


def boot_diff(base, new, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    사람별 (|오차| 합, 밤 수)로 사람을 재표집해 밤 단위 MAE 차이(기본 − 개선)의 분포를 낸다.
    base, new: index = pid, 열 sae, n (시드 평균 sae)
    """
    j = base.join(new, lsuffix='_b', rsuffix='_n', how='inner')
    rng = np.random.default_rng(seed)
    P = len(j)
    W = rng.multinomial(P, np.full(P, 1 / P), size=n_boot).astype(float)
    num = W @ (j['sae_b'] - j['sae_n']).to_numpy()
    den = W @ j['n_b'].to_numpy()
    d = num / den
    point = float((j['sae_b'] - j['sae_n']).sum() / j['n_b'].sum())
    per = (j['sae_b'] / j['n_b'] - j['sae_n'] / j['n_n'])
    return {'diff': point, 'ci_lo': float(np.quantile(d, 0.025)), 'ci_hi': float(np.quantile(d, 0.975)),
            'person_improved_frac': float((per > 0).mean()), 'n_people': int(P)}


def judge(a, s):
    """
    실험 가족(시드 묶음)마다 cb_base 같은 시드들과 짝지어 비교.
    시드가 여러 개면 사람별 |오차| 합을 시드 평균 → 부트스트랩. fold 일관성도 시드 평균 fold MAE로.
    """
    noise = (s[s['family'] == BASE].groupby('target')['oof_mae']
             .agg(seed_sd='std', seed_n='size', seed_range=lambda x: x.max() - x.min()))
    rows = []
    for (t, fam), g in s.groupby(['target', 'family']):
        if fam == BASE:
            continue
        seeds = sorted(g['seed'])
        base_names = [BASE if k == 0 else f'{BASE}_s{k}' for k in seeds]
        new_names = g.set_index('seed').loc[seeds, 'experiment'].tolist()
        bb = a[(a['target'] == t) & a['experiment'].isin(base_names)]
        nn = a[(a['target'] == t) & a['experiment'].isin(new_names)]
        if bb['experiment'].nunique() != len(seeds):
            continue
        ns = len(seeds)
        bp = bb.groupby('pid')[['sae', 'n']].sum() / ns
        npp = nn.groupby('pid')[['sae', 'n']].sum() / ns
        r = boot_diff(bp, npp)
        bf = bb.groupby('fold')[['sae', 'n']].sum()
        nf = nn.groupby('fold')[['sae', 'n']].sum()
        fold_diff = bf['sae'] / bf['n'] - nf['sae'] / nf['n']
        sdv = noise['seed_sd'].get(t, np.nan)
        r.update({'target': t, 'family': fam, 'n_seeds': ns, 'seeds': ','.join(map(str, seeds)),
                  'base_mae': float(bp['sae'].sum() / bp['n'].sum()),
                  'new_mae': float(npp['sae'].sum() / npp['n'].sum()),
                  'n_improved': int((fold_diff > 0).sum()),
                  'fold_diffs': ' '.join(f'{x:.2f}' for x in fold_diff),
                  'seed_sd': sdv})
        r['pass_folds'] = r['n_improved'] == 5
        r['pass_ci'] = r['ci_lo'] > 0
        r['pass_noise'] = bool(ns > 1 and np.isfinite(sdv) and r['diff'] > 2 * sdv) if ns > 1 else None
        r['candidate'] = bool(r['pass_folds'] and r['pass_ci'] and r['pass_noise'])
        rows.append(r)
    cols = ['target', 'family', 'n_seeds', 'seeds', 'base_mae', 'new_mae', 'diff', 'ci_lo', 'ci_hi',
            'n_improved', 'fold_diffs', 'person_improved_frac', 'n_people', 'seed_sd',
            'pass_folds', 'pass_ci', 'pass_noise', 'candidate']
    return pd.DataFrame(rows)[cols], noise.reset_index()


def breakdown(dirs, table, families_by_target, out):
    """
    구간별 MAE: |Δ| 크기, 개인 과거 28밤 유효 기록 수, 평일/주말 밤, 연구 기간 전/후. 시드 0 예측으로.
    """
    want = {(t, f) for t, fs in families_by_target.items() for f in fs + [BASE]}
    parts = []
    for d in dirs:
        for ch in pd.read_csv(Path(d) / 'predictions.csv', dtype={'pid': str}, chunksize=2_000_000,
                              parse_dates=['date']):
            m = ch[[(t, e) in want for t, e in zip(ch['target'], ch['experiment'])]]
            if len(m):
                parts.append(m)
    p = pd.concat(parts).drop_duplicates(['target', 'experiment', 'pid', 'date'])
    p['ae'] = (p['y_pred'] - p['y_true']).abs()
    info = table[['pid', 'date']].copy()
    for t in fd.TARGETS:
        info[f'absd_{t}'] = table[sd.col(t, sd.D0)].abs()
        info[f'hist28_{t}'] = table[f'hist28_{t}']
    p = p.merge(info, on=['pid', 'date'], how='left')
    rows = []
    for t, g in p.groupby('target'):
        g = g.copy()
        g['absd_bin'] = pd.cut(g[f'absd_{t}'], DELTA_BINS, labels=DELTA_LABELS, right=False)
        g['absd_bin'] = g['absd_bin'].cat.add_categories('NaN').fillna('NaN')
        g['hist_bin'] = pd.cut(g[f'hist28_{t}'], HIST_EDGES, labels=HIST_LABELS)
        g['weekend'] = np.where(g['date'].dt.dayofweek.isin([4, 5]), '금·토 밤', '일~목 밤')
        g['period'] = np.where(g['date'] < pd.Timestamp(pp.PHASE_CUTOFF), f'~{pp.PHASE_CUTOFF}', f'{pp.PHASE_CUTOFF}~')
        for by in ('absd_bin', 'hist_bin', 'weekend', 'period'):
            x = g.groupby([by, 'experiment'], observed=True)['ae'].agg(['mean', 'size']).reset_index()
            x = x.rename(columns={by: 'bin', 'mean': 'mae', 'size': 'n_nights'})
            b = x[x['experiment'] == BASE][['bin', 'mae']].rename(columns={'mae': 'base_mae'})
            x = x.merge(b, on='bin')
            x['diff'] = x['base_mae'] - x['mae']
            x.insert(0, 'by', by)
            x.insert(0, 'target', t)
            rows.append(x)
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(out / 'breakdown.csv', index=False)
    return r


def analyze(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in a.runs]
    dup = check_duplicates([d for d in dirs if 'p1c' in d.name])
    print(f'[재현 확인] 묶음 간 cb_base 예측 최대 차이: {dup}')
    agg, _ = load_runs(dirs)
    s = summarize(agg)
    s.to_csv(out / 'all_experiments.csv', index=False)
    j, noise = judge(agg, s)
    j = j.sort_values(['target', 'diff'], ascending=[True, False])
    j.to_csv(out / 'judgement.csv', index=False)
    noise.to_csv(out / 'seed_noise.csv', index=False)

    data, split, man, _, table = _prepare(a.synthetic)
    lags, _ = fd.lag_features(table[['pid', 'date', 'forecast_at']], data.nights, n_lags=28)
    for t in fd.TARGETS:
        table[f'hist28_{t}'] = lags[[f'{sd.QTY[t]}_lag{k}' for k in range(1, 29)]].notna().sum(axis=1)
    best = {t: [g.sort_values('diff', ascending=False)['family'].iloc[0]] for t, g in
            j[~j['family'].isin(BREAKDOWN)].groupby('target')}
    fams = {t: BREAKDOWN + best.get(t, []) for t in fd.TARGETS}
    bd = breakdown(dirs, table, fams, out)
    pd.Series({**{f'dup_maxdiff_{i}': v for i, v in enumerate(dup)}, 'runs': [str(d) for d in dirs]}) \
        .to_json(out / 'meta.json', force_ascii=False)

    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print('\n[시드 잡음] cb_base 시드별 oof MAE의 표준편차')
        print(noise.round(3).to_string(index=False))
        print('\n[판정] diff = cb_base − 실험 (양수 = 개선, 분). 시드가 여러 개면 시드 평균')
        print(j.drop(columns=['fold_diffs', 'n_people']).round(3).to_string(index=False))
        print('\n[구간별] 주 비교 + 목표별 최고')
        print(bd.round(2).to_string(index=False))
    print(f'\n저장: {out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='① 평소 대비 최근 수면 변화량 (공통 5-fold)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('phase1')
    p1.add_argument('--chunk', type=int, default=0)
    p1.add_argument('--nchunks', type=int, default=1)
    p1.add_argument('--only', nargs='+', default=None)
    p1.add_argument('--targets', nargs='+', default=None, choices=list(fd.TARGETS))
    p1.add_argument('--synthetic', action='store_true')
    p2 = sub.add_parser('phase2')
    p2.add_argument('--seed', type=int, required=True)
    p2.add_argument('--phase1', nargs='+', required=True)
    p2.add_argument('--synthetic', action='store_true')
    an = sub.add_parser('analyze')
    an.add_argument('--runs', nargs='+', required=True)
    an.add_argument('--out', required=True)
    an.add_argument('--synthetic', action='store_true')
    a = ap.parse_args(argv)
    {'phase1': phase1, 'phase2': phase2, 'analyze': analyze}[a.cmd](a)


if __name__ == '__main__':
    main()
