"""
② 모델 구조 비교 — 분석 (모델 재학습 없음)

    python -m experiments.minjun.models.analyze <run.py 결과 폴더>

  - 지표: 밤 단위 MAE(분)와 ±30분 적중률. 시드별 값을 평균 (예측을 평균한 앙상블이 아님)
  - 비교: 같은 입력에서 CatBoost 대비 Ridge·MLP·FT-Transformer. Δ = CatBoost MAE − 비교 모델 MAE (양수 = 비교 모델이 나음)
          주 묶음 = 공통 기본 입력(base) 9개, 보조 묶음 = 확장 입력(ext) 9개 + 모델별 base→ext 12개. 묶음마다 Holm 보정
  - 신뢰구간: 사람 클러스터 bootstrap (개발 참가자 복원 추출, 밤 단위 MAE 차이 재계산), B = 5000
  - 시드 흔들림: 모델·입력·목표마다 시드별 MAE의 표준편차
  - 공백 구간별 MAE: 마지막 유효 기록 이후 일수(④ feat_gap)로 나눔
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
LABEL = {'ridge': 'Ridge', 'catboost': 'CatBoost', 'mlp': 'MLP', 'ftt': 'FT-Transformer'}


def holm(p):
    p = np.asarray(p, float)
    idx = np.argsort(p)
    q = p[idx] * (len(p) - np.arange(len(p)))
    q = np.minimum(np.maximum.accumulate(q), 1)
    out = np.empty(len(p))
    out[idx] = q
    return out


class Boot:
    def __init__(self, pids, seed=0):
        self.pids = np.array(sorted(set(pids)))
        rng = np.random.default_rng(seed)
        P = len(self.pids)
        self.W = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(np.float32)

    def diff(self, pid, d):
        g = pd.DataFrame({'pid': pid, 'd': d}).groupby('pid')['d'].agg(['sum', 'size']).reindex(self.pids, fill_value=0)
        S, N = g['sum'].to_numpy(np.float64), g['size'].to_numpy(np.float64)
        est = S.sum() / N.sum()
        bs = (self.W @ S) / np.maximum(self.W @ N, 1)
        p = float(np.mean(np.abs(bs - est) >= abs(est))) if est != 0 else 1.0
        return {'delta': est, 'ci95_lo': np.percentile(bs, 2.5), 'ci95_hi': np.percentile(bs, 97.5),
                'ci90_lo': np.percentile(bs, 5), 'ci90_hi': np.percentile(bs, 95), 'p': max(p, 1 / B),
                'n_people': int((N > 0).sum()), 'n_nights': int(N.sum())}


def load(rd):
    acc, rows, secs = {}, [], []
    for f in sorted(rd.glob('preds_*__*_s*.parquet')):
        arm, s = f.stem[len('preds_'):].rsplit('_s', 1)
        p = pd.read_parquet(f).sort_values(['target', 'pid', 'date']).reset_index(drop=True)
        ae = np.abs(p['y_true'].to_numpy(float) - p['y_pred'].to_numpy(float))
        hit = (ae <= 30).astype(float)
        for t in p['target'].unique():
            m = (p['target'] == t).to_numpy()
            rows.append({'arm': arm, 'seed': int(s), 'target': t, 'mae': ae[m].mean(), 'hit30': hit[m].mean()})
        sec = p.drop_duplicates(['target', 'fold'])['fit_sec'].sum()
        secs.append({'arm': arm, 'seed': int(s), 'fit_sec_total': float(sec)})
        if arm not in acc:
            acc[arm] = {'keys': p[['pid', 'date', 'target']].copy(), 'ae': np.zeros(len(p)), 'hit': np.zeros(len(p)), 'n': 0}
        a = acc[arm]
        assert p[['pid', 'date', 'target']].equals(a['keys']), f'{arm} seed {s} 키 다름'
        a['ae'] += ae
        a['hit'] += hit
        a['n'] += 1
    avg = {}
    for arm, a in acc.items():
        avg[arm] = a['keys'].assign(ae=a['ae'] / a['n'], hit30=a['hit'] / a['n'])
    return avg, pd.DataFrame(rows), pd.DataFrame(secs), {k: v['n'] for k, v in acc.items()}


def main(rd):
    rd = Path(rd)
    meta = json.loads((rd / 'meta.json').read_text(encoding='utf-8'))
    data, split, man, _ = fr.prepare()
    assert fd.table_fingerprint(data.table) == meta['data_fingerprint']
    avg, seed_rows, secs, nseeds = load(rd)
    feats = pd.read_parquet(rd / 'features.parquet')[['pid', 'date', 'days_since_last']]
    feats['bucket'] = pd.cut(feats['days_since_last'], [0, 1, 7, np.inf], labels=BK[:3]) \
        .cat.add_categories(BK[3]).fillna(BK[3]).astype(str)
    any_arm = next(iter(avg.values()))
    keys = any_arm[['pid', 'date', 'target']].merge(feats[['pid', 'date', 'bucket']], on=['pid', 'date'], how='left')
    assert keys['bucket'].notna().all()
    for arm in avg:
        assert avg[arm][['pid', 'date', 'target']].equals(any_arm[['pid', 'date', 'target']])
        avg[arm]['bucket'] = keys['bucket'].to_numpy()
    boot = Boot(any_arm['pid'])
    T = list(fd.TARGETS)

    def compare(base, new, t, bucket=None):
        a, b = avg[base], avg[new]
        m = (a['target'] == t).to_numpy().copy()
        if bucket:
            m = m & (a['bucket'] == bucket).to_numpy()
        r = boot.diff(a.loc[m, 'pid'].to_numpy(), a.loc[m, 'ae'].to_numpy() - b.loc[m, 'ae'].to_numpy())
        r.update({'target': t, 'base': base, 'new': new, 'subset': bucket or '전체',
                  'base_mae': a.loc[m, 'ae'].mean(), 'new_mae': b.loc[m, 'ae'].mean(),
                  'equiv_1min': bool(r['ci90_lo'] > -MARGIN and r['ci90_hi'] < MARGIN)})
        return r

    models = [m for m in ['ridge', 'catboost', 'mlp', 'ftt'] if any(k.startswith(m + '__') for k in avg)]
    fams = {}
    for inp in ('base', 'ext'):
        rows = [compare(f'{REF}__{inp}', f'{m}__{inp}', t) for m in models if m != REF
                for t in T if f'{m}__{inp}' in avg and f'{REF}__{inp}' in avg]
        if rows:
            fams[f'vs_catboost_{inp}'] = pd.DataFrame(rows)
    rows = [compare(f'{m}__base', f'{m}__ext', t) for m in models for t in T
            if f'{m}__base' in avg and f'{m}__ext' in avg]
    if rows:
        fams['base_to_ext'] = pd.DataFrame(rows)
    rows = [compare(f'{REF}__{inp}', f'{m}__{inp}', t, b) for inp in ('base', 'ext') for m in models if m != REF
            for t in T for b in BK if f'{m}__{inp}' in avg and f'{REF}__{inp}' in avg]
    if rows:
        fams['vs_catboost_by_bucket'] = pd.DataFrame(rows)
    for k, df in fams.items():
        df['p_holm'] = holm(df['p'])
        df.to_csv(rd / f'analysis_{k}.csv', index=False)

    summ = (seed_rows.groupby(['target', 'arm']).agg(mae=('mae', 'mean'), mae_sd=('mae', 'std'),
                                                     hit30=('hit30', 'mean'), n_seeds=('seed', 'size'))
            .reset_index())
    tsum = secs.groupby('arm')['fit_sec_total'].mean().rename('fit_sec_per_seed').reset_index()
    bucket_tab = pd.concat([a.assign(arm=k) for k, a in avg.items()]).groupby(['target', 'bucket', 'arm']) \
        .agg(mae=('ae', 'mean'), n=('ae', 'size')).reset_index()
    choices = json.loads((rd / 'choices.json').read_text(encoding='utf-8'))
    ch = pd.DataFrame([{'model': k.split('|')[0], 'input': k.split('|')[1], 'target': k.split('|')[2],
                        'fold': k.split('|')[3], 'config': json.dumps(v['config'])} for k, v in choices.items()])
    ch_tab = ch.groupby(['model', 'input', 'config']).size().rename('n_target_folds').reset_index()
    for name, df in [('summary', summ), ('fit_time', tsum), ('bucket_mae', bucket_tab), ('choices', ch_tab)]:
        df.to_csv(rd / f'analysis_{name}.csv', index=False)

    cols = ['target', 'subset', 'base', 'new', 'base_mae', 'new_mae', 'delta', 'ci95_lo', 'ci95_hi', 'p_holm', 'n_nights']
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500):
        print('시드 수:', nseeds)
        print('\n[모델·입력별 MAE(분): 시드 평균 ± 표준편차, ±30분 적중률]')
        print(summ.pivot_table(index=['target'], columns='arm', values='mae').round(2).to_string())
        print(summ.round(3).to_string(index=False))
        print('\n[학습 시간(초, 시드 1회 = 목표 3 × fold 5)]')
        print(tsum.round(0).to_string(index=False))
        for k, df in fams.items():
            print(f'\n[{k}] Δ = base MAE − new MAE (양수 = new가 나음), Holm은 이 묶음 안에서')
            print(df[cols].round(3).to_string(index=False))
        print('\n[공백 구간별 MAE]')
        print(bucket_tab.pivot_table(index=['target', 'bucket'], columns='arm', values='mae').round(2).to_string())
        print('\n[고른 설정: (모델, 입력)별로 목표·fold 15칸 중 몇 번 골랐나]')
        print(ch_tab.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
