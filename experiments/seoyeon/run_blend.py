"""
run_blend.py — ③ 개인·집단 기준선 결합을 공통 5-fold로 실행 (최종 평가 참가자는 쓰지 않음)

    python -m experiments.seoyeon.run_blend               # 실제 NetHealth (data/nethealth 필요)
    python -m experiments.seoyeon.run_blend --synthetic   # 가짜 데이터로 흐름만 확인

공통 fr.run이 저장하는 파일(scores, summary, compare, predictions, run_meta)에 더해 같은 run 폴더에 저장:
  blend_params.csv     : fold·목표마다 학습 참가자 밤으로 고른 집단 값·w·k
  mae_by_history.csv   : 개인 과거 28밤 유효 기록 수 구간별 MAE (bb.BIN_LABELS)
  compare_extra.csv    : 결합 기준선끼리·개인/집단 단독과의 비교 (기본 − 개선, 양수 = 개선)
"""
import argparse

import pandas as pd

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from experiments.seoyeon import baseline_blend as bb

BASELINE = 'personal_mean7'
# (기본, 개선) 쌍. fr.run의 compare.csv는 BASELINE 대비만 있으므로 결합끼리의 비교를 따로 냄
EXTRA_PAIRS = [('group_median', 'shrink_blend28'), ('personal_median28', 'shrink_blend28'),
               ('personal_median28', 'fixed_blend28'), ('fixed_blend28', 'shrink_blend28')]


def blend_params(log, targets, folds):
    """PARAM_LOG(run_cv 순서: 목표 → fold → 실험)에 fold 번호를 붙임."""
    p = pd.DataFrame(log)
    if p.empty:
        return p
    p['fold'] = p.groupby(['target', 'model']).cumcount().map(lambda i: list(folds)[i])
    return p[['target', 'model', 'fold', 'n_train_nights'] + [c for c in ('group', 'w', 'k') if c in p]]


def main(argv=None):
    ap = argparse.ArgumentParser(description='③ 개인·집단 기준선 결합 (공통 5-fold)')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--targets', nargs='+', default=None, choices=list(fd.TARGETS))
    a = ap.parse_args(argv)

    data, split, man, out = fr.prepare(synthetic=a.synthetic)
    table = fd.add_features(data, bb.feat_personal_history)
    bb.PARAM_LOG.clear()
    scores, summary, comp, rd = fr.run(data, split, man, out, experiments=bb.experiments(), table=table,
                                       targets=a.targets, name='seoyeon-blend', baseline=BASELINE,
                                       synthetic=a.synthetic)

    params = blend_params(bb.PARAM_LOG, a.targets or list(fd.TARGETS), fe.fs.FOLDS)
    params.to_csv(rd / 'blend_params.csv', index=False)
    preds = pd.read_csv(rd / 'predictions.csv', dtype={'pid': str}, parse_dates=['date'])
    hist = bb.mae_by_history(preds, table)
    hist.to_csv(rd / 'mae_by_history.csv', index=False)
    extra = pd.concat([fe.compare(scores, b, n).assign(base=b, new=n) for b, n in EXTRA_PAIRS],
                      ignore_index=True)
    extra.to_csv(rd / 'compare_extra.csv', index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        if a.synthetic:
            print('[가짜 데이터] 아래 숫자는 실제 성능이 아님')
        print('\n[요약] status가 complete인 줄만 결론에 씀')
        print(summary.drop(columns=['note', 'complete']).round(2).to_string(index=False))
        print(f'\n[비교: {BASELINE} 대비] diff = 기본 − 개선 (양수 = 개선)')
        print(comp[comp['fold'] == 'mean'][['target', 'base', 'new', 'status', 'base_mae', 'new_mae', 'diff',
                                            'n_improved']].round(2).to_string(index=False))
        print('\n[비교: 결합끼리]')
        print(extra[extra['fold'] == 'mean'][['target', 'base', 'new', 'status', 'base_mae', 'new_mae', 'diff',
                                              'n_improved']].round(2).to_string(index=False))
        print('\n[고른 값] fold별')
        print(params.round(2).to_string(index=False))
        print('\n[개인 과거 28밤 유효 기록 수 구간별 MAE(분)]')
        print(hist.pivot_table(index=['target', 'bin'], columns='experiment', values='mae', sort=False)
                  .round(1).to_string())
        print(hist.groupby(['target', 'bin'], sort=False)['n_nights'].first().to_string())
    print(f'\n저장: {rd}')


if __name__ == '__main__':
    main()
