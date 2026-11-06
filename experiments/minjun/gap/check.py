"""
④ 보조 점검 (모델 재학습 없음)

    python -m experiments.minjun.gap.check <results/forecast/v3/gap3/시각>

1. 누수 대조: features.parquet의 특징을 data.nights / data.activity에서 행 단위로 직접 다시 계산해 비교
   (무작위 3,000행 + 경과일 2 이상 행 3,000행). 기준: 밤 D보다 앞이고 known_at <= forecast_at,
   overlap·asleep_invalid 아님. (features.parquet에 있는 열만 대조)
2. 큰 오차(시드 평균 |오차| > 120분) 위험과 분포: 구간별 비율, 큰 오차 중 각 구간이 차지하는 몫 (모델 A)
3. 공백-오차 증가의 다른 집계: 사람 구분 없이 구간 MAE 차이, 사람 안 차이를 공백 밤 수로 가중 (모델 A)
4. 월별 구성: 구간별 1·6·7·8월 비율
5. 7밤이 모두 빈 밤에서 A의 예측이 (목표, fold, 요일)마다 상수인지
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_run as fr

BK = ['어젯밤 기록 있음', '1~6밤 빠짐', '7밤 이상 빠짐', '참가 첫날(과거 기록 없음)']
COLS = {'sleep': 'sleep_min', 'bed': 'bedtime', 'wake': 'waketime'}


def bucket(s):
    return pd.cut(s, [0, 1, 7, np.inf], labels=BK[:3]).cat.add_categories(BK[3]).fillna(BK[3]).astype(str)


def leak_check(data, feats):
    t = data.table.reset_index(drop=True)
    f = feats.reset_index(drop=True)
    assert t[['pid', 'date']].equals(f[['pid', 'date']])
    rng = np.random.default_rng(0)
    gap_idx = np.flatnonzero(f['days_since_last'].fillna(99).to_numpy() >= 2)
    idx = np.unique(np.concatenate([rng.choice(len(t), 3000, replace=False),
                                    rng.choice(gap_idx, min(3000, len(gap_idx)), replace=False)]))
    n = data.nights
    v = n[~n['overlap'].astype(bool) & ~n['asleep_invalid'].astype(bool)]
    vg = {p: g.sort_values('date') for p, g in v.groupby('pid')}
    act = data.activity[['pid', 'date', 'complypercent']].set_index(['pid', 'date'])['complypercent']
    bad = {}
    for i in idx:
        p, D, fa = t.at[i, 'pid'], t.at[i, 'date'], t.at[i, 'forecast_at']
        g = vg.get(p, v.iloc[:0])
        ok = g[(g['date'] < D) & (g['known_at'] <= fa)]
        exp = {'n_valid7': float(((ok['date'] >= D - pd.Timedelta(days=7))).sum()),
               'days_since_last': float((D - ok['date'].iloc[-1]).days) if len(ok) else np.nan,
               'n_valid28': float((ok['date'] >= D - pd.Timedelta(days=28)).sum())}
        w28 = ok[ok['date'] >= D - pd.Timedelta(days=28)]
        rec = ok.iloc[::-1].head(7)
        for k, c in COLS.items():
            exp[f'last_{k}'] = float(ok[c].iloc[-1]) if len(ok) else np.nan
            exp[f'med28_{k}'] = float(w28[c].median()) if len(w28) else np.nan
            exp[f'sd28_{k}'] = float(w28[c].std(ddof=1)) if len(w28) >= 2 else np.nan
            for j in range(1, 8):
                exp[f'rec{j}_{k}'] = float(rec[c].iloc[j - 1]) if len(rec) >= j else np.nan
        if 'wear_lag1' in f.columns:      # 이전 실행 결과(착용 특징 포함)를 점검할 때만
            wear = np.array([act.get((p, D - pd.Timedelta(days=k)), np.nan) for k in range(1, 8)], dtype=float)
            exp['wear_lag1'] = wear[0]
            exp['n_wear7'] = float(np.nansum(wear >= 80))
            exp['wear_mean7'] = float(np.nanmean(wear)) if (~np.isnan(wear)).any() else np.nan
        exp = {c: e for c, e in exp.items() if c in f.columns}
        for c, e in exp.items():
            got = f.at[i, c]
            if not ((np.isnan(e) and np.isnan(got)) or (not np.isnan(e) and not np.isnan(got) and abs(e - got) < 1e-6)):
                bad[c] = bad.get(c, 0) + 1
    return len(idx), len(exp), bad


def main(rd):
    rd = Path(rd)
    meta = json.loads((rd / 'meta.json').read_text(encoding='utf-8'))
    data, split, man, _ = fr.prepare()
    assert fd.table_fingerprint(data.table) == meta['data_fingerprint']
    feats = pd.read_parquet(rd / 'features.parquet')
    n_rows, n_feat, bad = leak_check(data, feats)
    print(f'[1] 누수 대조: {n_rows}행 × 특징 {n_feat}개, 불일치 {sum(bad.values())}건 {bad}')

    lab = feats[['pid', 'date', 'days_since_last']].assign(bucket=lambda d: bucket(d['days_since_last']))
    errs = []
    for s in range(10):
        p = pd.read_parquet(rd / f'preds_gap_A_base_s{s}.parquet')
        errs.append((p['y_true'].astype(float) - p['y_pred'].astype(float)).abs().to_numpy())
    p = p[['pid', 'date', 'target', 'fold']].assign(ae=np.mean(errs, axis=0)).merge(lab, on=['pid', 'date'])
    p['big'] = p['ae'] > 120
    rows = []
    for t, g in p.groupby('target'):
        tot_big = g['big'].sum()
        for b, gb in g.groupby('bucket'):
            rows.append({'target': t, 'bucket': b, 'big_rate': gb['big'].mean(), 'share_of_big': gb['big'].sum() / tot_big})
        cont = g[g['bucket'] == BK[0]]
        brk = g[g['bucket'] != BK[0]]
        rows.append({'target': t, 'bucket': '공백·첫날 전체', 'big_rate': brk['big'].mean(),
                     'share_of_big': brk['big'].sum() / tot_big, 'risk_ratio': brk['big'].mean() / cont['big'].mean()})
    big = pd.DataFrame(rows)
    print('\n[2] 큰 오차(>120분) 위험 (모델 A, 시드 평균 |오차|)')
    print(big.round(3).to_string(index=False))

    rows = []
    for t, g in p.groupby('target'):
        bm = g.groupby('bucket')['ae'].mean()
        pm = g.groupby(['pid', 'bucket'])['ae'].agg(['mean', 'size']).unstack()
        for b in BK[1:3]:
            d = (pm['mean'][b] - pm['mean'][BK[0]])
            w = pm['size'][b]
            ok = d.notna()
            rows.append({'target': t, 'compare': f'{b} − {BK[0]}', 'pooled_diff': bm[b] - bm[BK[0]],
                         'within_person_mean': d[ok].mean(),
                         'within_person_weighted_by_gap_nights': np.average(d[ok], weights=w[ok])})
    agg = pd.DataFrame(rows)
    print('\n[3] 공백-오차 증가: 집계 방식별 (모델 A)')
    print(agg.round(2).to_string(index=False))

    dev = set(split.loc[split['role'] == 'dev', 'pid'])
    m = lab[lab['pid'].isin(dev)].assign(month=lambda d: d['date'].dt.month)
    mon = m.groupby('bucket')['month'].apply(lambda s: pd.Series({f'm{k}': (s == k).mean() for k in (1, 6, 7, 8)})).unstack()
    print('\n[4] 구간별 월 비율')
    print(mon.round(3).to_string())

    z = p[p['bucket'].isin(BK[2:])].merge(data.table[['pid', 'date', 'dow']], on=['pid', 'date'])
    p0 = pd.read_parquet(rd / 'preds_gap_A_base_s0.parquet').merge(
        z[['pid', 'date', 'target', 'dow']], on=['pid', 'date', 'target'])
    nun = p0.groupby(['target', 'fold', 'dow'])['y_pred'].nunique()
    print(f'\n[5] 7밤이 모두 빈 밤에서 A(시드 0) 예측의 (목표, fold, 요일)별 고유값 수: 최대 {nun.max()}')

    big.to_csv(rd / 'check_big_error.csv', index=False)
    agg.to_csv(rd / 'check_gap_aggregation.csv', index=False)
    mon.to_csv(rd / 'check_month_share.csv')
    (rd / 'check_leak.json').write_text(json.dumps({'rows': n_rows, 'features': n_feat, 'mismatch': bad},
                                                   ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
