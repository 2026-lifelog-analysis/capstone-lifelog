"""
④ 분석 (모델 재학습 없음): run.py 결과 폴더를 읽어 계산

    python -m experiments.minjun.gap.analyze <results/forecast/v3/gap3/시각>

통계 방법 
  - 지표: 밤 단위 MAE(분). 시드별 MAE를 시드 평균 (밤마다 |오차|를 시드 평균한 값의 평균과 같음)
  - 두 팔 비교: Δ = base MAE − new MAE (양수 = new가 나음). 같은 평가 밤끼리 짝.
  - 불확실성: 사람 클러스터 bootstrap (개발 참가자를 복원 추출, 밤 단위 MAE 차이를 다시 계산), B=5000
      95% CI, 양측 p = 중심화한 bootstrap 분포에서 |Δ* − Δ| ≥ |Δ|의 비율
  - 시드 잡음 기준: (1) 팔마다 시드별 MAE의 표준편차, (2) A의 시드 0~4 평균 vs 5~9 평균을 같은 방법으로 비교
    (같은 모델끼리라 '효과 0'이어야 하는 비교. 여기서 유의가 나오면 방법이 잡음을 효과로 잡는다는 뜻)
  - 동등성: 90% CI가 (−1, +1)분 안이면 '±1분 안에서 차이 없음'
  - 다중 비교: 주 비교 묶음(P1·P2 × 목표 3 = 6개), 보충 묶음, 탐색 묶음 각각 Holm 보정
  - 한계: bootstrap은 학습된 모델을 고정하고 평가 대상 사람만 다시 뽑는다. 학습 데이터 변동과
    '입력 열이 바뀔 때의 트리 구조 변동'은 반영하지 않으므로 작은 구간의 p값은 낙관적일 수 있다.
    p 하한은 1/B.
  - '시드 평균' = 시드별 MAE의 평균 (예측을 평균한 앙상블이 아님)
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
BASE = 'gap_A_base'
ARMS = ['gap_A_base', 'gap_R0_recent', 'gap_R1_count', 'gap_R2_since', 'gap_E_last', 'gap_F_lastsin',
        'gap_H_long28']
BK = ['어젯밤 기록 있음', '1~6밤 빠짐', '7밤 이상 빠짐', '참가 첫날(과거 기록 없음)']


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    q = p[ok][idx] * (ok.sum() - np.arange(ok.sum()))
    q = np.minimum(np.maximum.accumulate(q), 1)
    tmp = np.empty(ok.sum()); tmp[idx] = q
    out[ok] = tmp
    return out


class Boot:
    """사람 클러스터 bootstrap. 같은 재표집 가중치를 모든 비교에 공유 (재현·일관성)."""
    def __init__(self, pids, seed=0):
        self.pids = np.array(sorted(set(pids)))
        rng = np.random.default_rng(seed)
        P = len(self.pids)
        self.W = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(np.float32)

    def diff(self, df):
        """df: pid, d(밤별 base|err| − new|err|) → (Δ, ci95, ci90, p, n_people, n_nights)"""
        g = df.groupby('pid')['d'].agg(['sum', 'size']).reindex(self.pids, fill_value=0)
        S, N = g['sum'].to_numpy(np.float64), g['size'].to_numpy(np.float64)
        est = S.sum() / N.sum()
        bs = (self.W @ S) / np.maximum(self.W @ N, 1)
        p = float(np.mean(np.abs(bs - est) >= abs(est))) if est != 0 else 1.0
        return {'delta': est, 'ci95_lo': np.percentile(bs, 2.5), 'ci95_hi': np.percentile(bs, 97.5),
                'ci90_lo': np.percentile(bs, 5), 'ci90_hi': np.percentile(bs, 95), 'p': max(p, 1 / B),
                'n_people': int((N > 0).sum()), 'n_nights': int(N.sum())}


def load(rd):
    """메모리 절약: 팔마다 키는 한 번만, |오차|·적중은 시드 합으로 누적. A는 시드 0~4 / 5~9 합도 따로."""
    meta = json.loads((rd / 'meta.json').read_text(encoding='utf-8'))
    feats = pd.read_parquet(rd / 'features.parquet')
    acc, seed_mae = {}, []
    for f in sorted(rd.glob('preds_*_s*.parquet')):
        arm, s = f.stem[len('preds_'):].rsplit('_s', 1)
        s = int(s)
        p = pd.read_parquet(f)
        ae = (p['y_true'].to_numpy(float) - p['y_pred'].to_numpy(float))
        ae = np.abs(ae)
        hit = (ae <= 30).astype(float)
        for t in p['target'].unique():
            m = (p['target'] == t).to_numpy()
            seed_mae.append({'arm': arm, 'seed': s, 'target': t, 'mae': ae[m].mean(), 'hit30': hit[m].mean()})
        if arm not in acc:
            acc[arm] = {'keys': p[['pid', 'date', 'target']].copy(), 'ae': np.zeros(len(p)), 'hit': np.zeros(len(p)),
                        'n': 0, 'h1': np.zeros(len(p)), 'n1': 0, 'h2': np.zeros(len(p)), 'n2': 0}
        a = acc[arm]
        assert p[['pid', 'date', 'target']].equals(a['keys']), f'{arm} seed {s} 행 순서 다름'
        a['ae'] += ae; a['hit'] += hit; a['n'] += 1
        if s < 5:
            a['h1'] += ae; a['n1'] += 1
        else:
            a['h2'] += ae; a['n2'] += 1
    return meta, feats, acc, pd.DataFrame(seed_mae)


def seed_avg(acc, arm, half=None):
    a = acc[arm]
    out = a['keys'].copy()
    if half is None:
        out['ae'] = a['ae'] / a['n']; out['hit30'] = a['hit'] / a['n']
        return out, a['n']
    k = 'h1' if half == 1 else 'h2'
    out['ae'] = a[k] / a['n' + k[1]]
    return out, a['n' + k[1]]


def main(rd):
    rd = Path(rd)
    meta, feats, errs, seed_mae = load(rd)  # errs = 팔별 누적
    data, split, man, _ = fr.prepare()
    assert fd.table_fingerprint(data.table) == meta['data_fingerprint'], '데이터 지문이 run 당시와 다름'
    # 구간
    #   경과일 1 = 어젯밤 기록 있음 / 2~7 = 1~6밤 빠짐 / 8 이상 = 7밤 이상 빠짐 / 기록 없음 = 참가 첫날
    feats['bucket'] = pd.cut(feats['days_since_last'], [0, 1, 7, np.inf],
                             labels=[BK[0], BK[1], BK[2]]).cat.add_categories(BK[3]).fillna(BK[3]).astype(str)
    feats['zero7'] = feats['n_valid7'] == 0
    feats['gap2'] = feats['days_since_last'] >= 2          # 기록이 끊긴 밤 (참가 첫날 제외)
    lab = feats[['pid', 'date', 'bucket', 'zero7', 'gap2', 'days_since_last']]
    avg = {}
    errs = {k: v for k, v in errs.items() if k in ARMS}
    seed_mae = seed_mae[seed_mae['arm'].isin(ARMS)]
    for arm in errs:
        a, n = seed_avg(errs, arm)
        a = a.merge(lab, on=['pid', 'date'], how='left')
        assert a['bucket'].notna().all()
        avg[arm] = a
    nseeds = {arm: errs[arm]['n'] for arm in errs}
    boot = Boot(avg[BASE]['pid'])

    def compare(base, new, t, mask_name='all'):
        a, b = avg[base], avg[new]
        assert a[['pid', 'date', 'target']].equals(b[['pid', 'date', 'target']])
        m = (a['target'] == t)
        if mask_name == 'n_valid7==0':
            m &= a['zero7']
        elif mask_name == '경과일>=2':
            m &= a['gap2']
        elif mask_name != 'all':
            m &= a['bucket'] == mask_name
        df = pd.DataFrame({'pid': a.loc[m, 'pid'], 'd': a.loc[m, 'ae'].to_numpy() - b.loc[m, 'ae'].to_numpy()})
        r = boot.diff(df)
        r.update({'target': t, 'base': base, 'new': new, 'subset': mask_name,
                  'base_mae': a.loc[m, 'ae'].mean(), 'new_mae': b.loc[m, 'ae'].mean(),
                  'equiv_1min': bool(r['ci90_lo'] > -MARGIN and r['ci90_hi'] < MARGIN)})
        return r

    T = list(fd.TARGETS)
    # 주 비교 (3차 실행 전 meta.json에 정한 것. 그대로 유지)
    prim = [compare('gap_R1_count', 'gap_R2_since', t, 'all') | {'id': 'P1'} for t in T] + \
           [compare('gap_E_last', 'gap_F_lastsin', t, 'n_valid7==0') | {'id': 'P2'} for t in T]
    prim = pd.DataFrame(prim)
    prim['p_holm'] = holm(prim['p'])
    # 보충 비교: 주 비교를 실제 대상 밤으로 좁힘 + 입력 정보가 같은 대조(참가 첫날 E→F)
    sup_pairs = [('gap_R1_count', 'gap_R2_since', '경과일>=2', 'P1 공백 밤'),
                 ('gap_R1_count', 'gap_R2_since', BK[1], 'P1 1~6밤 빠짐'),
                 ('gap_R1_count', 'gap_R2_since', BK[2], 'P1 7밤 이상 빠짐'),
                 ('gap_E_last', 'gap_F_lastsin', BK[2], 'P2 7밤 이상 빠짐'),
                 ('gap_E_last', 'gap_F_lastsin', BK[3], '정보 같음 대조(참가 첫날)')]
    sup = pd.DataFrame([compare(b_, n_, t, s) | {'id': i} for b_, n_, s, i in sup_pairs for t in T])
    sup['p_holm'] = holm(sup['p'])
    # 탐색 비교
    exp_pairs = [(BASE, 'gap_R2_since', 'all'), ('gap_R0_recent', 'gap_R1_count', 'all'),
                 ('gap_R0_recent', 'gap_R1_count', '경과일>=2'), (BASE, 'gap_R0_recent', 'all'),
                 (BASE, 'gap_H_long28', 'all'), (BASE, 'gap_E_last', 'n_valid7==0')]
    for arm in ('gap_E_last', 'gap_R0_recent', 'gap_H_long28'):
        exp_pairs += [(BASE, arm, b) for b in BK]
    expl = pd.DataFrame([compare(b_, n_, t, s) for b_, n_, s in exp_pairs for t in T if b_ in avg and n_ in avg])
    expl['p_holm'] = holm(expl['p'])
    # 시드 잡음 기준: A 시드 0~4 vs 5~9
    null = []
    has_null = errs[BASE]['n1'] > 0 and errs[BASE]['n2'] > 0
    if has_null:
        h1, _ = seed_avg(errs, BASE, half=1); h2, _ = seed_avg(errs, BASE, half=2)
    for t in (T if has_null else []):
        m = h1['target'] == t
        r = boot.diff(pd.DataFrame({'pid': h1.loc[m, 'pid'], 'd': h1.loc[m, 'ae'].to_numpy() - h2.loc[m, 'ae'].to_numpy()}))
        null.append(r | {'target': t, 'comparison': 'A seeds0-4 vs A seeds5-9'})
    null = pd.DataFrame(null)
    summ = (seed_mae.groupby(['target', 'arm']).agg(mae_mean=('mae', 'mean'), mae_sd=('mae', 'std'),
                                                    hit30_mean=('hit30', 'mean'), n_seeds=('seed', 'size'))
            .reset_index())
    bk = pd.concat([a.assign(arm=arm) for arm, a in avg.items()])
    bucket_tab = (bk.groupby(['target', 'bucket', 'arm']).agg(mae=('ae', 'mean'), hit30=('hit30', 'mean'),
                                                              n_nights=('ae', 'size'), n_people=('pid', 'nunique'))
                  .reset_index())
    # 같은 사람 안에서 공백에 따른 오차 증가 (모델 A와 H, 사람마다 구간 MAE 차이 → 사람 평균, 사람 bootstrap)
    wp = []
    rng = np.random.default_rng(1)
    for arm in (BASE, 'gap_H_long28'):
        a = avg[arm]
        for t in T:
            g = a[a['target'] == t].groupby(['pid', 'bucket'])['ae'].mean().unstack()
            for other in (BK[1], BK[2]):
                x = (g[other] - g[BK[0]]).dropna().to_numpy()
                bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
                wp.append({'arm': arm, 'target': t, 'compare': f'{other} − {BK[0]}', 'n_people': len(x),
                           'mean_increase': x.mean(), 'ci95_lo': np.percentile(bs, 2.5),
                           'ci95_hi': np.percentile(bs, 97.5), 'share_people_worse': float((x > 0).mean())})
    wp = pd.DataFrame(wp)
    # 구간 구성(설명용): 짧은 주수면 비율, '어젯밤 하나만 빠짐' 비율, 1월·8월 비율, 참가 시작 후 일수
    comp = data.table[['pid', 'date', 'target_sleep_min']].merge(lab, on=['pid', 'date'])
    dev = set(split.loc[split['role'] == 'dev', 'pid'])
    comp = comp[comp['pid'].isin(dev)]
    first = data.table.groupby('pid')['date'].transform('min')
    comp = comp.merge(data.table[['pid', 'date']].assign(day_in=(data.table['date'] - first).dt.days),
                      on=['pid', 'date'])
    comp['month'] = comp['date'].dt.month
    comp_tab = comp.groupby('bucket').agg(
        n_nights=('date', 'size'), n_people=('pid', 'nunique'),
        short120_share=('target_sleep_min', lambda s: float((s < 120).mean())),
        since2_share=('days_since_last', lambda s: float((s == 2).mean())),
        jan_share=('month', lambda s: float((s == 1).mean())), aug_share=('month', lambda s: float((s == 8).mean())),
        day_in_max=('day_in', 'max'))

    for name, df in [('primary', prim), ('supplementary', sup), ('exploratory', expl), ('seed_null', null),
                     ('arm_summary', summ), ('bucket_table', bucket_tab), ('within_person_gap', wp)]:
        df.to_csv(rd / f'analysis_{name}.csv', index=False)
    comp_tab.to_csv(rd / 'analysis_bucket_composition.csv')
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 500):
        print('시드 수:', nseeds, '| bootstrap B =', B, '(p 하한 1/B) | 탐색 비교 수 =', len(expl),
              '| 보충 비교 수 =', len(sup))
        print('\n[팔별 MAE: 시드 평균 ± 표준편차, ±30분 적중률]')
        print(summ.round(3).to_string(index=False))
        print('\n[시드 잡음 기준: 같은 A끼리 (시드 0~4 평균 vs 5~9 평균)]')
        print(null[['target', 'delta', 'ci95_lo', 'ci95_hi', 'p']].round(4).to_string(index=False) if len(null) else '(시드 10개가 있어야 계산)')
        cols = ['id', 'target', 'subset', 'base', 'new', 'base_mae', 'new_mae', 'delta', 'ci95_lo', 'ci95_hi',
                'ci90_lo', 'ci90_hi', 'p', 'p_holm', 'equiv_1min', 'n_people', 'n_nights']
        print('\n[주 비교] Δ = base MAE − new MAE (양수 = new가 나음)')
        print(prim[cols].round(4).to_string(index=False))
        print('\n[보충 비교 (사후)]')
        print(sup[cols].round(4).to_string(index=False))
        print('\n[탐색 비교]')
        print(expl[cols[1:]].round(4).to_string(index=False))
        print('\n[공백 구간별 MAE (시드 평균)]')
        print(bucket_tab.pivot_table(index=['target', 'bucket'], columns='arm', values='mae').round(2).to_string())
        print('\n[같은 사람 안에서 공백에 따른 오차 증가]')
        print(wp.round(3).to_string(index=False))
        print('\n[구간 구성]')
        print(comp_tab.round(4).to_string())

if __name__ == '__main__':
    main(sys.argv[1])
