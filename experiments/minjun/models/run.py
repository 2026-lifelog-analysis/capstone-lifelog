"""
② 모델 구조 비교 — 실행 (공통 프로토콜 v3, 개발 참가자 5-fold. 최종 평가 참가자는 쓰지 않음)

    python -m experiments.minjun.models.run --bench                 # 모델·설정별 학습 1회 시간 측정 (fold 1, sleep, base)
    python -m experiments.minjun.models.run [--models ...] [--inputs base ext] [--seeds 5] [--out 폴더]

입력
  base : 공통 기본 입력 fd.base_features (목표별 과거 7밤 + 요일)            ← 주 비교
  ext  : base + 28일 창 중앙값·표준편차 + 최근 실제 기록 7개 (④ features)      ← 보조 비교

공정성 규칙
  - fold마다 학습 참가자의 10%(사람 단위, 시드별 고정)를 '학습 안 검증'으로 뗀다. 설정 선택·조기 종료에만 쓴다.
  - 설정 선택: 시드 0에서 모델마다 GRID(3개)를 모두 학습해 학습 안 검증 MAE가 가장 작은 설정을
    (모델, 입력, 목표, fold)마다 고른다. 다른 시드는 그 설정을 그대로 쓴다.
  - 모든 모델은 같은 90% 학습 참가자로 학습한다. 검증 fold 참가자의 정답은 어디에도 쓰지 않는다.

저장 (커밋하지 않음): <out>/
  features.parquet, meta.json, choices.json (고른 설정·학습 안 검증 MAE·학습 시간),
  preds_<모델>__<입력>_s<시드>.parquet (pid, date, target, fold, y_true, y_pred)
이미 있는 preds 파일은 건너뛴다(중단 후 이어서 실행).
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
from src import forecast_split as fs
from experiments.minjun.gap.features import feat_gap, feat_long, feat_recent, rec
from experiments.minjun.models.models import MODELS, timed_fit

INNER_FRAC = 0.10
INPUTS = {
    'base': fd.base_features,
    'ext': lambda t: fd.base_features(t) + [f'med28_{t}', f'sd28_{t}'] + rec(t),
}


def inner_split(train_pids, seed, fold):
    pids = np.array(sorted(train_pids))
    rng = np.random.default_rng(1000 * seed + fold)
    n_val = int(round(len(pids) * INNER_FRAC))
    val = set(rng.choice(pids, n_val, replace=False))
    return [p for p in pids if p not in val], sorted(val)


def fold_data(table, split, target, fold, seed):
    y = fd.TARGETS[target]
    tr_p, va_p = fs.fold_pids(split, fold)
    in_tr, in_va = inner_split(tr_p, seed, fold)
    train = table[table['pid'].isin(in_tr) & table[y].notna()]
    inval = table[table['pid'].isin(in_va) & table[y].notna()]
    ev = fe.eval_nights(table, split, target)
    ev = ev[ev['fold'] == fold]
    valid = ev[fe.KEY].merge(table, on=fe.KEY, how='left')
    return train, inval, valid, ev, y


def load_choices(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def bench(table, split):
    t, fold, inp = 'sleep', 1, 'base'
    train, inval, valid, ev, y = fold_data(table, split, t, fold, 0)
    feats = INPUTS[inp](t)
    print(f'학습 {len(train):,}밤, 학습 안 검증 {len(inval):,}밤, 평가 {len(valid):,}밤')
    for name, cls in MODELS.items():
        for cfg in cls.GRID:
            m, sec = timed_fit(cls, cfg, 0, train, train[y], inval, inval[y], feats)
            mae = float(np.mean(np.abs(m.predict(inval) - inval[y].to_numpy())))
            print(f'{name:9s} {json.dumps(cfg):55s} {sec:7.1f}s  학습안검증 MAE {mae:.2f}  {m.info}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--inputs', nargs='+', default=list(INPUTS))
    ap.add_argument('--targets', nargs='+', default=list(fd.TARGETS))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--grid-limit', type=int, default=None, help='설정 후보 수 제한 (앞에서부터)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    data, split, man, out = fr.prepare()
    table = fd.add_features(data, feat_gap, feat_long, feat_recent)
    if a.bench:
        bench(table, split)
        return
    rd = Path(a.out) if a.out else out / 'models2' / datetime.now().strftime('%Y%m%d-%H%M%S')
    rd.mkdir(parents=True, exist_ok=True)
    fp = fd.table_fingerprint(data.table)
    assert fp == man['data_fingerprint']
    if not (rd / 'features.parquet').exists():
        extra = [c for c in table.columns if c not in data.table.columns]
        table[['pid', 'date'] + extra].to_parquet(rd / 'features.parquet', index=False)
    meta = {'data_fingerprint': fp, 'split_sha256': man['split_sha256'], 'protocol': fd.PROTOCOL,
            'code': fr.code_version(), 'inner_frac': INNER_FRAC, 'seeds': list(range(a.seeds)),
            'inputs': {k: {t: f(t) for t in fd.TARGETS} for k, f in INPUTS.items()},
            'grids': {k: v.GRID for k, v in MODELS.items()}, 'grid_limit': a.grid_limit,
            'final_test_scored': False, 'written_at': datetime.now().isoformat(timespec='seconds')}
    (rd / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    cpath = rd / 'choices.json'
    choices = load_choices(cpath)

    for seed in range(a.seeds):
        for name in a.models:
            cls = MODELS[name]
            grid = cls.GRID[:a.grid_limit] if a.grid_limit else cls.GRID
            for inp in a.inputs:
                pfile = rd / f'preds_{name}__{inp}_s{seed}.parquet'
                if pfile.exists():
                    continue
                parts, t0 = [], datetime.now()
                for t in a.targets:
                    feats = INPUTS[inp](t)
                    for fold in fs.FOLDS:
                        train, inval, valid, ev, y = fold_data(table, split, t, fold, seed)
                        key = f'{name}|{inp}|{t}|{fold}'
                        if key not in choices:
                            assert seed == 0, f'{key}: 설정 선택은 시드 0에서 해야 함'
                            tried, best = [], None
                            for cfg in grid:
                                m, sec = timed_fit(cls, cfg, seed, train, train[y], inval, inval[y], feats)
                                mae = float(np.mean(np.abs(m.predict(inval) - inval[y].to_numpy())))
                                tried.append({'config': cfg, 'inner_mae': mae, 'sec': sec, 'info': m.info})
                                if best is None or mae < best[0]:
                                    best = (mae, cfg, m, sec)
                            choices[key] = {'config': best[1], 'tried': tried}
                            cpath.write_text(json.dumps(choices, indent=1, ensure_ascii=False), encoding='utf-8')
                            model, sec = best[2], best[3]
                        else:
                            model, sec = timed_fit(cls, choices[key]['config'], seed, train, train[y],
                                                   inval, inval[y], feats)
                        p = model.predict(valid[fe.KEY + fd.META_COLS + feats])
                        assert p.shape == (len(valid),) and np.isfinite(p).all(), key
                        parts.append(ev[['pid', 'date', 'fold', 'y_true']].assign(
                            target=t, y_pred=p, fit_sec=sec, info=json.dumps(model.info)))
                df = pd.concat(parts, ignore_index=True).astype({'y_true': 'float32', 'y_pred': 'float32'})
                df.to_parquet(pfile, index=False)
                mae = df.assign(e=(df['y_true'] - df['y_pred']).abs()).groupby('target')['e'].mean().round(2)
                print(f'{name}__{inp} seed {seed} 완료 ({(datetime.now() - t0).seconds}s) MAE {mae.to_dict()}',
                      flush=True)
    print(f'저장: {rd}')


if __name__ == '__main__':
    main()
