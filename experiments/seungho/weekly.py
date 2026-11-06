# 작성 2026-11-06 12:20 (KST)
"""
weekly.py — 목표를 "오늘 밤"에서 "여러 밤 평균"으로 바꾸면 예측이 쓸 만해지는가 (공통 분할·입력 재사용)

    python -m experiments.seungho.weekly                 # 실제 데이터
    python -m experiments.seungho.weekly --synthetic     # 흐름 확인

배경: 오늘 밤 하나는 같은 사람도 밤마다 1시간 넘게 흔들려(연속 두 밤 차이 중앙값 총수면 73분),
      미래까지 아는 개인 중앙값조차 MAE 70분 언저리다. 여러 밤 평균은 그 잡음이 상쇄된다.

목표 (공통 프로토콜의 정답 정의를 그대로 쓰고 평균만 냄. 공통 파일은 수정하지 않음)
  next7 : 예측 시각 D 18:00, 정답 = 밤 D ~ D+6 중 정답 있는 밤의 평균. 유효 밤 MIN_VALID개 미만이면 정답 없음
  week  : 달력 주. 예측 시각 = 월요일 밤 D 18:00, 정답 = 그 주 월~일 밤 평균 (같은 최소 유효 수)
          월요일 밤 줄이 예측 표에 있어야 한다 (입력은 그 줄의 것)
  tonight : 같은 줄들의 오늘 밤 정답 (비교용, 공통 프로토콜과 같은 값)

입력: 모두 D 18:00까지 확정된 정보 (sleep_delta + extended 특징 그대로) + 다음 7일 중 쉬는 날 수(미리 정해진 일정)
평가: 공통 분할표 그대로, 개발 528명 5-fold, 최종 평가 93명은 쓰지 않음. 모든 실험이 같은 평가 줄을 쓴다.
      연속된 줄의 정답이 겹치므로(next7) 신뢰구간은 사람 단위 재표집으로만 낸다.
기준점: 개인 7밤 평균, 개인 28밤 중앙값, 오라클(그 사람의 정답 중앙값, 미래 포함 = 누출 허용 참고선)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from experiments.seungho import sleep_delta as sd
from experiments.seungho import extended as ext
from experiments.seungho import run_delta as rdl
from experiments.seungho import run_extended as rex

MIN_VALID = 5
HORIZON = 7
FOLDS = (1, 2, 3, 4, 5)


def multi_night_targets(data, horizon=HORIZON, min_valid=MIN_VALID):
    """table 줄마다 next7 정답(목표별). 정답이 있는 밤만 평균 (없는 밤을 채우지 않음)."""
    t = data.table[['pid', 'date'] + list(fd.TARGETS.values())]
    out = pd.DataFrame(index=data.table.index)
    for tg, y in fd.TARGETS.items():
        vals, cnt = 0.0, 0
        for k in range(horizon):
            s = t[['pid', 'date', y]].copy()
            s['date'] = s['date'] - pd.Timedelta(days=k)
            m = data.table[['pid', 'date']].merge(s, on=['pid', 'date'], how='left')[y].to_numpy()
            vals = vals + np.nan_to_num(m)
            cnt = cnt + ~np.isnan(m)
        v = vals / np.where(cnt > 0, cnt, np.nan)
        v[cnt < min_valid] = np.nan
        out[f'y_next7_{tg}'] = v
    return out


def week_targets(data, min_valid=MIN_VALID):
    """월요일 밤 줄에만 그 주(월~일 밤) 평균. 다른 줄은 NaN."""
    nx = multi_night_targets(data, 7, min_valid)
    mon = (data.table['dow'] == 0).to_numpy()
    return pd.DataFrame({c.replace('next7', 'week'): np.where(mon, nx[c], np.nan) for c in nx}, index=nx.index)


def off_next7(data):
    cal = ext.calendar_days().set_index('date')['off']
    return np.sum([cal.reindex(data.table['date'] + pd.Timedelta(days=k)).to_numpy(dtype=float)
                   for k in range(1, HORIZON + 1)], axis=0)


def experiments_for(tg):
    """(이름, 모델 만들기, 입력 열)"""
    p = fd.LAG_PREFIX[tg]
    lag7 = [f'{p}_lag{k}' for k in range(1, 8)]
    off7 = ['x_off_next7']
    base = fd.base_features(tg)
    b = lambda blocks: ext.feats_of(blocks)(tg)
    return [
        ('personal_mean7', 'mean7', lag7),
        ('personal_median28', 'med28', [sd.col(tg, sd.L0)]),
        ('cb_base', 'cb', base),
        ('cb_level', 'cb', b(['level'])),
        ('cb_level_off7', 'cb', b(['level']) + off7),
        ('T_ALL_off7', 'cb', b(ext.FAMILY['T']) + off7),
        ('K2_core_wide_off7', 'cb', b(['delta', 'off', 'fed', 'term', 'lag8_14', 'lag15_28', 'other_lags14',
                                       'dR14_28']) + off7),
        ('K2_everything_off7', 'cb', b(ext.ALL_EXT + ['other_lags14', 'steps']) + off7),
    ]


def _predict(kind, train, test, ycol, feats):
    if kind == 'mean7':
        g = float(train[ycol].median())
        return test[feats].mean(axis=1).fillna(g).to_numpy()
    if kind == 'med28':
        g = float(train[ycol].median())
        return test[feats[0]].fillna(g).to_numpy()
    m = fe.CatBoostModel().fit(train, ycol, feats)
    return m.predict(test)


def run_target(table, split, ycol, tg, horizon_name):
    dev = split[split['role'] == 'dev'][['pid', 'fold']]
    d = table.merge(dev, on='pid', how='inner')
    d = d[d[ycol].notna()].reset_index(drop=True)
    rows, preds = [], []
    for name, kind, feats in experiments_for(tg):
        fd.check_features(feats)
        yp = np.full(len(d), np.nan)
        for f in FOLDS:
            tr, te = d['fold'] != f, d['fold'] == f
            yp[te.to_numpy()] = _predict(kind, d[tr], d[te], ycol, feats)
        preds.append(pd.DataFrame({'pid': d['pid'], 'date': d['date'], 'fold': d['fold'], 'horizon': horizon_name,
                                   'target': tg, 'experiment': name, 'y_true': d[ycol], 'y_pred': yp}))
        print(f'  {horizon_name} {tg} {name}: MAE {np.mean(np.abs(yp - d[ycol])):.2f}', flush=True)
    # 오라클 (누출 허용 참고선): 그 사람의 정답 중앙값, 미래 포함
    orc = d.groupby('pid')[ycol].transform('median').to_numpy()
    preds.append(pd.DataFrame({'pid': d['pid'], 'date': d['date'], 'fold': d['fold'], 'horizon': horizon_name,
                               'target': tg, 'experiment': 'oracle_person_median(leak)', 'y_true': d[ycol],
                               'y_pred': orc}))
    return pd.concat(preds, ignore_index=True)


def summarize(P):
    P = P.assign(ae=(P['y_pred'] - P['y_true']).abs())
    rows = []
    for (h, tg, e), g in P.groupby(['horizon', 'target', 'experiment'], sort=False):
        fm = g.groupby('fold')['ae'].mean()
        rows.append({'horizon': h, 'target': tg, 'experiment': e, 'mae': g['ae'].mean(),
                     'rmse': float(np.sqrt(((g['y_pred'] - g['y_true']) ** 2).mean())),
                     'within30': float((g['ae'] <= 30).mean() * 100), 'within15': float((g['ae'] <= 15).mean() * 100),
                     'fold_mae': ' '.join(f'{x:.2f}' for x in fm), 'n_rows': len(g), 'n_people': g['pid'].nunique(),
                     'y_sd': float(g['y_true'].std())})
    s = pd.DataFrame(rows)
    # 사람 단위 부트스트랩: cb_base 대비
    bs = []
    for (h, tg), g in P.groupby(['horizon', 'target'], sort=False):
        per = g.groupby(['experiment', 'pid'])['ae'].agg(sae='sum', n='size')
        base = per.loc['cb_base']
        for e in g['experiment'].unique():
            if e == 'cb_base':
                continue
            r = rdl.boot_diff(base, per.loc[e])
            fb = g[g.experiment == 'cb_base'].groupby('fold')['ae'].mean()
            fn = g[g.experiment == e].groupby('fold')['ae'].mean()
            bs.append({'horizon': h, 'target': tg, 'experiment': e, 'diff_vs_cb_base': r['diff'],
                       'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'], 'n_improved': int((fb - fn > 0).sum())})
    return s.merge(pd.DataFrame(bs), on=['horizon', 'target', 'experiment'], how='left')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    a = ap.parse_args(argv)
    data, split, man, out, table = rex._prepare(a.synthetic)
    ys = pd.concat([multi_night_targets(data), week_targets(data)], axis=1)
    table = pd.concat([table, ys], axis=1)
    table['x_off_next7'] = np.zeros(len(table)) if a.synthetic else off_next7(data)
    for tg, y in fd.TARGETS.items():
        table[f'y_tonight_{tg}'] = table[y]
    P = []
    for h in ('tonight', 'next7', 'week'):
        for tg in fd.TARGETS:
            col = f'y_{h}_{tg}'
            if h == 'tonight':
                # 비교용: week 줄(월요일)과 같은 줄이 아니라 next7과 같은 줄에서 오늘 밤 정답
                tmp = table.copy()
                tmp.loc[tmp[f'y_next7_{tg}'].isna(), col] = np.nan
                P.append(run_target(tmp, split, col, tg, h))
            else:
                P.append(run_target(table, split, col, tg, h))
    P = pd.concat(P, ignore_index=True)
    s = summarize(P)
    rd = Path(out) / 'seungho-weekly' / (datetime.now().strftime('%Y%m%d-%H%M%S') + ('-synthetic' if a.synthetic else ''))
    rd.mkdir(parents=True, exist_ok=True)
    s.to_csv(rd / 'summary.csv', index=False)
    P.assign(date=P['date'].dt.strftime('%Y-%m-%d')).to_csv(rd / 'predictions.csv', index=False)
    (rd / 'meta.json').write_text(json.dumps({'min_valid': MIN_VALID, 'horizon': HORIZON,
                                              'split_sha256': man['split_sha256'],
                                              'data_fingerprint': man['data_fingerprint'],
                                              'final_test_scored': False}, indent=2))
    with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.max_rows', 200):
        print(s.drop(columns=['fold_mae']).round(2).to_string(index=False))
    print(f'저장: {rd}')


if __name__ == '__main__':
    main()
