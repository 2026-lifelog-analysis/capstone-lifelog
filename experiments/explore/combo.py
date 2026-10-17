# 작성 2026-10-15 01:47
"""
조합 실험 — 여러 튜닝 요소를 동시에 적용 (개발용 70% 사람만).

요소 (전부 곱해서 3 × 2 × 2 × 3 = 36개 설정)
  dow      : none | dow7 (요일 7개) | custom (지표별 그룹: 걸음 {월~목, 금, 토, 일}, 수면·취침 {일~목, 금·토})
  holiday  : 0 | 1  (1이면 event_tag=='휴일'인 날을 기준선 창과 경보 대상에서 모두 제외)
  anchor   : 0 | 1  (1이면 같은 날 전체 학생 중앙값을 뺀 잔차에 개인 기준선 = E4-b)
  feats    : steps | steps+sleep | steps+sleep+bed  (방향 맞춘 z의 평균 → CUSUM)

주입 시나리오
  S1  걸음 −30% 갑자기          (지표 하나만 변함)
  S3  걸음 −15% 서서히          (약한 쇠퇴)
  S5  걸음 −20% · 수면 −10% · 취침 +60분, 서서히  (생활 전반이 같이 무너지는 경우)

평가는 레포 ev.evaluate_injection 그대로. 여러 지표 주입은 ev.inject_drift를 감싸서 같은 구간에 같이 넣음.

사용법: python3 -I experiments/explore/combo.py [반복 수=20] [프로세스 수=10]
출력:   results/tuning/combo_table.csv, combo_rank.csv, runs_combo/*.pkl
"""
import itertools
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C          # noqa: E402

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

ev, bl = C.ev, C.bl

FEATS = {'steps': ['steps'], 'steps+sleep': ['steps', 'sleep_min'],
         'steps+sleep+bed': ['steps', 'sleep_min', 'bed_min']}
DOW_GROUPS = {
    'none': None,
    'dow7': {f: np.arange(7) for f in ['steps', 'sleep_min', 'bed_min']},
    'custom': {'steps': np.array([0, 0, 0, 0, 1, 2, 3]),          # 월~목 / 금 / 토 / 일
               'sleep_min': np.array([0, 0, 0, 0, 1, 1, 0]),      # 금·토 밤 / 나머지
               'bed_min': np.array([0, 0, 0, 0, 1, 1, 0])},
}
# 시나리오: 주 지표(구간 고르기 기준) + 같은 구간에 같이 넣을 지표들
SCEN = {
    'S1': [('steps', -0.30, 'relative')],
    'S3': [('steps', -0.15, 'relative')],
    'S5': [('steps', -0.20, 'relative'), ('sleep_min', -0.10, 'relative'), ('bed_min', 60, 'absolute')],
}
SHAPE = {'S1': 'sudden', 'S3': 'gradual', 'S5': 'gradual'}

CONFIGS = [dict(dow=d, holiday=h, anchor=a, feats=f)
           for d, h, a, f in itertools.product(DOW_GROUPS, [0, 1], [0, 1], FEATS)]


def cfg_name(c):
    return f"dow={c['dow']} hol={c['holiday']} anc={c['anchor']} f={c['feats']}"


# ---------------------------------------------------------------------------
# 점수
# ---------------------------------------------------------------------------
def feature_bad(d, feat, cfg):
    """지표 하나의 '클수록 나쁨' 개인 z."""
    x = d[feat].astype(float)
    if cfg['anchor']:
        g = x.groupby(d['date'])
        x = (x - g.transform('median')).where(g.transform('count') >= bl.GROUP_MIN_N)
    if cfg['holiday']:
        x = x.where(d['event_tag'] != '휴일')
    groups = DOW_GROUPS[cfg['dow']]
    z = C.personal_z_grouped(d.assign(_x=x), '_x', None if groups is None else groups[feat])
    return z * bl.BAD_DIRECTION[feat]


def combined(d, cfg):
    parts = [feature_bad(d, f, cfg) for f in FEATS[cfg['feats']]]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def group_combined(d, cfg):
    parts = [bl.bad_score(bl.group_z(d, f), f) for f in FEATS[cfg['feats']]]
    s = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return s.where(d['event_tag'] != '휴일') if cfg['holiday'] else s


def make_score_fn(df_clean, cfg, k=0.5):
    pb0 = combined(df_clean, cfg)
    h = bl.cusum_threshold(df_clean.assign(_b=pb0), '_b', k=k)

    def score_fn(d):
        pb = combined(d, cfg)
        return pd.DataFrame({
            '무작위': np.random.default_rng(ev.SEED).random(len(d)),
            '개인 기준선': pb,
            C.CS: bl.cusum(d.assign(_b=pb), '_b', k=k, h=h),
            '집단 기준선': group_combined(d, cfg),
        }, index=d.index)
    score_fn.h = h
    return score_fn


# ---------------------------------------------------------------------------
# 여러 지표 주입: ev.inject_drift를 감싸서 같은 구간에 나머지 지표도 넣음
# ---------------------------------------------------------------------------
_EXTRA = []
_orig_inject = ev.inject_drift


def _inject_multi(df, col, effect, mode='relative', length=14, shape='sudden', seed=ev.SEED, **kw):
    out, seg = _orig_inject(df, col, effect, mode=mode, length=length, shape=shape, seed=seed, **kw)
    if _EXTRA and len(seg):
        rows = ev._segment_rows(out, seg)
        day = (rows['date'] - rows['start']).dt.days.to_numpy()
        frac = np.ones(len(rows)) if shape == 'sudden' else (day + 1) / length
        for c2, e2, m2 in _EXTRA:
            v = out.loc[rows.index, c2].to_numpy(float)
            out.loc[rows.index, c2] = v * (1 + e2 * frac) if m2 == 'relative' else v + e2 * frac
    return out, seg


ev.inject_drift = _inject_multi


def one(task, n_reps):
    ci, sc = task
    cfg = CONFIGS[ci]
    df = C.dev_set(C.load())
    (col, eff, mode), *extra = SCEN[sc]
    _EXTRA[:] = extra
    res = ev.evaluate_injection(df, col, eff, mode=mode, shape=SHAPE[sc], n_reps=n_reps,
                                score_fn=make_score_fn(df, cfg))
    return ci, sc, res


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = C.OUT
    (out / 'runs_combo').mkdir(parents=True, exist_ok=True)
    tasks = [(i, s) for s in SCEN for i in range(len(CONFIGS))]
    print(f'설정 {len(CONFIGS)}개 × 시나리오 {len(SCEN)}개 = {len(tasks)}개 / 반복 {n_reps}회', flush=True)

    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (ci, sc, r) in enumerate(ex.map(partial(one, n_reps=n_reps), tasks), 1):
            res[(ci, sc)] = r
            with open(out / 'runs_combo' / f'c{ci:02d}_{sc}.pkl', 'wb') as fh:
                pickle.dump(r, fh)
            if i % 12 == 0:
                print(f'  {i}/{len(tasks)} 끝', flush=True)

    rows, cmp = [], []
    base = CONFIGS.index(dict(dow='none', holiday=0, anchor=0, feats='steps'))
    for (ci, sc), r in res.items():
        t = C.summary(r, cfg_name(CONFIGS[ci])).assign(시나리오=sc, **CONFIGS[ci])
        rows.append(t)
        for m in [C.CS, '집단 기준선']:
            cmp.append({'시나리오': sc, '방법': m, **CONFIGS[ci], **C.compare(res[(base, sc)], r, method=m)})
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(out / 'combo_table.csv', index=False, encoding='utf-8-sig')
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(out / 'combo_rank.csv', index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    for sc in SCEN:
        c = cmp[(cmp['시나리오'] == sc) & (cmp['방법'] == C.CS)].sort_values('평균 차이', ascending=False)
        print(f'\n=== {sc}: 개인+CUSUM, 기본(dow=none hol=0 anc=0 f=steps) 대비 평균 차이 상위 10 ===')
        print(c.head(10)[['dow', 'holiday', 'anchor', 'feats', '평균 차이', '좋아진 사람', '나빠진 사람', 'p값']].to_string(index=False))


if __name__ == '__main__':
    main()
