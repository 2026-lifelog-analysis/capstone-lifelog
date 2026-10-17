"""
run.py — 수면 리듬 탐지기 평가를 돌리고 집계 표를 저장한다.

사용 (레포 맨 위 폴더에서):
  python experiments/minjun/regularity/run.py [--data data/nethealth] [--scenarios S14 BED-S14 JB90] [--reps 10]
                                              [--out results/minjun_regularity]
출력 (--out 폴더, results/ 는 .gitignore 대상):
  results_main.csv : 시나리오 × 방식 × 탐지기의 탐지율, 구간 오경보율, J, 지연
  compare.csv      : 시나리오 × 방식마다 '수면 중앙 시각 SD 개인+CUSUM' 과 수준 탐지기 중 J 최고와의 차이 (ΔJ, 95% CI, p)
  per_person.csv.gz: 사람별 표 (pid 포함. 올리지 않음)
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_loader as dl  # noqa: E402
import regularity as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=None, help='NetHealth 원자료 폴더 (기본 data/nethealth)')
ap.add_argument('--scenarios', nargs='+', default=['S14', 'BED-S14', 'JB90'])
ap.add_argument('--reps', type=int, default=R.N_REPS)
ap.add_argument('--out', default=str(ROOT / 'results' / 'minjun_regularity'))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(*s):
    print(f'[{time.time() - t0:5.0f}s]', *s, flush=True)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nh = dl.load_nethealth(path=a.data)
log('하루 표', nh.shape, '사람', nh['pid'].nunique())

nh, S, h_A = R.clean_scores(nh)
log('대조군 점수 완료. 방식 A CUSUM 경보선', {m: round(h, 3) for m, h in h_A.items()})

rows, pps = [], []
for scen in a.scenarios:
    seg = R.collect(nh, S, scen, n_reps=a.reps)
    log(scen, '구간 날', len(seg), '사람', seg['pid'].nunique())
    r, pp = R.evaluate(nh, S, h_A, seg, scen)
    rows.append(r)
    pps.append(pp)
    log(scen, '평가 완료')

RES = pd.concat(rows, ignore_index=True)
PP = pd.concat(pps, ignore_index=True)
RES.to_csv(out / 'results_main.csv', index=False, encoding='utf-8-sig')
PP.to_csv(out / 'per_person.csv.gz', index=False, encoding='utf-8-sig', compression='gzip')

cmp_rows = []
target = R.name(R.REG, 'cs')
for (scen, sch), g in RES.groupby(['시나리오', '방식'], sort=False):
    lv = g[g['종류'] == '수준']
    best = lv.loc[lv['J평균'].idxmax(), '탐지기']
    pa = PP[(PP['시나리오'] == scen) & (PP['방식'] == sch) & (PP['탐지기'] == target)]
    pb = PP[(PP['시나리오'] == scen) & (PP['방식'] == sch) & (PP['탐지기'] == best)]
    dj, lo, hi, n = R.paired_boot_ci(pa, pb)
    cmp_rows.append({'시나리오': scen, '방식': sch, '규칙성 탐지기': R.label(target),
                     '규칙성 J': float(g.loc[g['탐지기'] == target, 'J평균'].iloc[0]),
                     '수준 탐지기 중 최고': R.label(best), '최고 J': float(lv['J평균'].max()),
                     'ΔJ(규칙성 − 최고)': dj, 'ΔJ 95% 하한': lo, 'ΔJ 95% 상한': hi,
                     'Wilcoxon p': R.paired_wilcoxon(pa, pb), '사람 수': n})
CMP = pd.DataFrame(cmp_rows)
CMP.to_csv(out / 'compare.csv', index=False, encoding='utf-8-sig')

pd.set_option('display.width', 220)
for sch in ['A', 'B']:
    print(f'\n방식 {sch}: J 평균')
    print(RES[RES['방식'] == sch].pivot(index='이름', columns='시나리오', values='J평균')
          .reindex(columns=a.scenarios).round(3).to_string())
print('\n규칙성 탐지기 대 수준 탐지기 최고')
print(CMP.round(4).to_string())
log('저장', out)
