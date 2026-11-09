"""
② 모델 구조 비교 — 분석 (모델 재학습 없음)

    python -m experiments.minjun.models.analyze <run.py 결과 폴더>

지표: 밤 단위 MAE(분)와 ±30분 적중률. 시드별 값을 평균한다(예측을 평균한 앙상블이 아님).

비교 두 가지 (모두 같은 입력에서 CatBoost 대비, Δ = CatBoost MAE − 비교 모델 MAE, 양수 = 비교 모델이 나음)
  raw : 모델 예측 그대로
  cal : 중심 보정. 모델마다 손실·조기 종료 기준이 달라 예측의 중심(평균 쪽인지 중앙값 쪽인지)이 다르다.
        MAE 채점에서는 중앙값 쪽이 유리하므로, 모든 모델에 똑같이 '다른 fold 잔차(정답 − 예측)의 중앙값'만큼
        예측을 옮긴다. 채점하는 fold의 정답은 쓰지 않는다(교차 보정). 모델 구조의 차이만 보려는 것.

신뢰구간·p값: 사람과 시드를 함께 다시 뽑는 bootstrap (B = 5000).
  사람 = 개발 참가자 복원 추출(밤 단위 MAE 차이 재계산), 시드 = 시드 3개 복원 추출 → 재학습 흔들림을 포함.
  묶음마다 Holm 보정.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_run as fr

B = 5000
MARGIN = 1.0
REF = 'catboost'
BK = ['어젯밤 기록 있음', '1~6밤 끊김', '7밤 이상 끊김', '참가 첫날(과거 기록 없음)']


def holm(p):
    p = np.asarray(p, float)
    idx = np.argsort(p)
    q = np.minimum(np.maximum.accumulate(p[idx] * (len(p) - np.arange(len(p)))), 1)
    out = np.empty(len(p))
    out[idx] = q
    return out


def load(rd):
    """팔(모델__입력)마다 시드별 |오차| (raw, cal), 적중, 키."""
    arms = {}
    for f in sorted(rd.glob('preds_*__*_s*.parquet')):
        arm, s = f.stem[len('preds_'):].rsplit('_s', 1)
        p = pd.read_parquet(f).sort_values(['target', 'pid', 'date']).reset_index(drop=True)
        yt, yp = p['y_true'].to_numpy(float), p['y_pred'].to_numpy(float)
        res = yt - yp
        shift = np.zeros(len(p))
        for (t, fo), idx in p.groupby(['target', 'fold']).groups.items():
            other = ((p['target'] == t) & (p['fold'] != fo)).to_numpy()
            shift[np.asarray(idx)] = np.median(res[other])
        a = arms.setdefault(arm, {'keys': p[['pid', 'date', 'target']].copy(), 'raw': [], 'cal': [], 'sec': []})
        assert p[['pid', 'date', 'target']].equals(a['keys']), f'{arm} seed {s} 키 다름'
        a['raw'].append(np.abs(res))
        a['cal'].append(np.abs(yt - (yp + shift)))
        a['sec'].append(float(p.drop_duplicates(['target', 'fold'])['fit_sec'].sum()))
        a.setdefault('seeds', []).append(int(s))
    return arms


class Boot:
    def __init__(self, pids, n_seeds, seed=0):
        self.pids = np.array(sorted(set(pids)))
        rng = np.random.default_rng(seed)
        P = len(self.pids)
        self.W = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(np.float64)
        self.S_idx = rng.integers(0, n_seeds, (B, n_seeds))

    def diff(self, pid, D):
        """D: (시드, 밤) 차이 행렬. 사람·시드를 함께 다시 뽑음."""
        pos = pd.Index(self.pids).get_indexer(pid)
        S = np.stack([np.bincount(pos, weights=d, minlength=len(self.pids)) for d in D], axis=1)   # P × 시드
        N = np.bincount(pos, minlength=len(self.pids)).astype(np.float64)
        est = S.mean(1).sum() / N.sum()
        Sb = S[:, self.S_idx].mean(2)                                                         # P × B
        bs = np.einsum('bp,pb->b', self.W, Sb) / np.maximum(self.W @ N, 1)
        p = float(np.mean(np.abs(bs - est) >= abs(est))) if est != 0 else 1.0
        per_seed = (S.sum(0) / N.sum()).round(3).tolist()
        return {'delta': est, 'ci95_lo': np.percentile(bs, 2.5), 'ci95_hi': np.percentile(bs, 97.5),
                'ci90_lo': np.percentile(bs, 5), 'ci90_hi': np.percentile(bs, 95), 'p': max(p, 1 / B),
                'delta_by_seed': per_seed, 'n_nights': int(N.sum())}


def main(rd):
    rd = Path(rd)
    meta = json.loads((rd / 'meta.json').read_text(encoding='utf-8'))
    data, split, man, _ = fr.prepare()
    assert fd.table_fingerprint(data.table) == meta['data_fingerprint']
    arms = load(rd)
    ns = {k: len(v['raw']) for k, v in arms.items()}
    n_seeds = min(ns.values())
    keys = next(iter(arms.values()))['keys']
    for a in arms.values():
        assert a['keys'].equals(keys)
        assert sorted(a['seeds'])[:n_seeds] == list(range(n_seeds))
    feats = pd.read_parquet(rd / 'features.parquet')[['pid', 'date', 'days_since_last']]
    feats['bucket'] = pd.cut(feats['days_since_last'], [0, 1, 7, np.inf], labels=BK[:3]) \
        .cat.add_categories(BK[3]).fillna(BK[3]).astype(str)
    bucket = keys.merge(feats[['pid', 'date', 'bucket']], on=['pid', 'date'], how='left')['bucket'].to_numpy()
    tgt = keys['target'].to_numpy()
    pid = keys['pid'].to_numpy()
    boot = Boot(pid, n_seeds)
    T = list(fd.TARGETS)

    def stack(arm, kind):
        a = arms[arm]
        order = np.argsort(a['seeds'])[:n_seeds]
        return np.stack([a[kind][i] for i in order])

    def compare(base, new, t, kind, sub=None):
        m = tgt == t
        if sub:
            m = m & (bucket == sub)
        A, Bm = stack(base, kind)[:, m], stack(new, kind)[:, m]
        r = boot.diff(pid[m], A - Bm)
        r.update({'kind': kind, 'target': t, 'subset': sub or '전체', 'base': base, 'new': new,
                  'base_mae': A.mean(), 'new_mae': Bm.mean(),
                  'equiv_1min': bool(r['ci90_lo'] > -MARGIN and r['ci90_hi'] < MARGIN)})
        return r

    models = [m for m in ['ridge', 'catboost', 'catboost_wide', 'mlp', 'ftt'] if f'{m}__base' in arms]
    fams = {}
    for kind in ('raw', 'cal'):
        for inp in ('base', 'ext'):
            fams[f'vs_catboost_{inp}_{kind}'] = pd.DataFrame(
                [compare(f'{REF}__{inp}', f'{m}__{inp}', t, kind) for m in models if m != REF for t in T])
    if 'catboost_wide__base' in arms:
        for inp in ('base', 'ext'):
            fams[f'vs_catboost_wide_{inp}_cal'] = pd.DataFrame(
                [compare(f'catboost_wide__{inp}', f'{m}__{inp}', t, 'cal') for m in ('mlp', 'ftt') for t in T])
    fams['base_to_ext_cal'] = pd.DataFrame([compare(f'{m}__base', f'{m}__ext', t, 'cal') for m in models for t in T])
    fams['vs_catboost_by_bucket_cal'] = pd.DataFrame(
        [compare(f'{REF}__{inp}', f'{m}__{inp}', t, 'cal', b) for inp in ('base', 'ext') for m in models
         if m != REF for t in T for b in BK])
    for k, df in fams.items():
        df['p_holm'] = holm(df['p'])
        df.to_csv(rd / f'analysis_{k}.csv', index=False)

    rows = []
    for arm, a in arms.items():
        for kind in ('raw', 'cal'):
            X = stack(arm, kind)
            for t in T:
                m = tgt == t
                per_seed = X[:, m].mean(1)
                rows.append({'arm': arm, 'kind': kind, 'target': t, 'mae': per_seed.mean(),
                             'mae_seed_min': per_seed.min(), 'mae_seed_max': per_seed.max(),
                             'hit30': float((X[:, m] <= 30).mean())})
    summ = pd.DataFrame(rows)
    tsum = pd.DataFrame([{'arm': k, 'fit_sec_per_seed': float(np.mean(v['sec']))} for k, v in arms.items()])
    bucket_tab = pd.DataFrame([{'arm': arm, 'kind': kind, 'target': t, 'bucket': b,
                                'mae': stack(arm, kind)[:, (tgt == t) & (bucket == b)].mean()}
                               for arm in arms for kind in ('raw', 'cal') for t in T for b in BK])
    choices = json.loads((rd / 'choices.json').read_text(encoding='utf-8'))
    ch = pd.DataFrame([{'model': k.split('|')[0], 'input': k.split('|')[1], 'config': json.dumps(v['config'])}
                       for k, v in choices.items()])
    ch_tab = ch.groupby(['model', 'input', 'config']).size().rename('n_target_folds').reset_index()
    for name, df in [('summary', summ), ('fit_time', tsum), ('bucket_mae', bucket_tab), ('choices', ch_tab)]:
        df.to_csv(rd / f'analysis_{name}.csv', index=False)

    cols = ['kind', 'target', 'subset', 'base', 'new', 'base_mae', 'new_mae', 'delta', 'ci95_lo', 'ci95_hi',
            'p_holm', 'equiv_1min', 'delta_by_seed']
    with pd.option_context('display.width', 260, 'display.max_columns', 30, 'display.max_rows', 500):
        print('시드 수:', ns, '| bootstrap B =', B, '(사람·시드 함께 재표집)')
        for kind in ('raw', 'cal'):
            print(f'\n[MAE(분) {kind}: 시드 평균]')
            print(summ[summ['kind'] == kind].pivot_table(index='target', columns='arm', values='mae').round(2).to_string())
        print('\n[시드별 MAE 범위 (raw)]')
        s = summ[summ['kind'] == 'raw'].assign(range=lambda d: d['mae_seed_max'] - d['mae_seed_min'])
        print(s.pivot_table(index='target', columns='arm', values='range').round(3).to_string())
        print('\n[±30분 적중률 (raw)]')
        print(summ[summ['kind'] == 'raw'].pivot_table(index='target', columns='arm', values='hit30').round(3).to_string())
        print('\n[학습 시간(초, 시드 1회)]')
        print(tsum.round(0).to_string(index=False))
        for k, df in fams.items():
            if 'bucket' in k:
                continue
            print(f'\n[{k}] Δ = base MAE − new MAE (양수 = new가 나음)')
            print(df[cols].round(3).to_string(index=False))
        print('\n[공백 구간별 CatBoost 대비 (cal)]')
        print(fams['vs_catboost_by_bucket_cal'][cols].round(3).to_string(index=False))
        print('\n[고른 설정]')
        print(ch_tab.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
