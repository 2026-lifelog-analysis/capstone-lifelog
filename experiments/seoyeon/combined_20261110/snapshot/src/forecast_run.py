"""
forecast_run.py — 공통 예측 실험 실행 (개발 참가자 5-fold만. 최종 평가는 돌리지 않음)

    python -m src.forecast_run --check                     # 실제 데이터: 예측 표 요약 + 분할표 생성·검사
    python -m src.forecast_run --diagnose                  # dataDate 해석 진단 (종료일 vs 시작일 겹침 비율)
    python -m src.forecast_run --name base                 # 기본 실험 3개 (group_median, personal_mean7, ridge_base)
    python -m src.forecast_run --experiments catboost_base --baseline catboost_base --name cb
    python -m src.forecast_run --synthetic                 # 가짜 원본 → results/forecast_synthetic/<protocol>/

저장 (모두 results/ 아래라 .gitignore로 커밋 안 됨). <out> = results/forecast/<protocol>/ (지금 fd.PROTOCOL)
  <out>/split.csv, split_manifest.json        : 참가자 분할표 (원본 ID 포함) + 해시
  <out>/runs/<run_id>/scores.csv              : 목표 × 실험 × fold MAE, 참가자·밤 수, 완전 여부, 평가 키 해시
  <out>/runs/<run_id>/summary.csv             : status, fold 평균·표준편차, OOF MAE
  <out>/runs/<run_id>/compare.csv             : 기준 실험 대비 차이 (기본 − 개선)
  <out>/runs/<run_id>/predictions.csv         : pid | date | target | experiment | fold | y_true | y_pred
  <out>/runs/<run_id>/run_meta.json           : 코드 버전, 시드, 설정, 데이터 지문, 분할 해시

팀원 실험은 이 파일을 복사하지 말고 prepare() / run(experiments=[...], table=...)을 불러 쓰면 된다.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import forecast_dataset as fd
from . import forecast_eval as fe
from . import forecast_split as fs

ROOT = Path(__file__).resolve().parents[1]
OUT_REAL = ROOT / 'results' / 'forecast' / fd.PROTOCOL
OUT_SYNTH = ROOT / 'results' / 'forecast_synthetic' / fd.PROTOCOL

DEFAULT_EXPERIMENTS = ['group_median', 'personal_mean7', 'ridge_base']
BASELINE = 'ridge_base'


def code_version():
    """git 커밋과 src 작업 트리 변경 여부. git이 없으면 None."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src'], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {'commit': commit, 'src_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'src_dirty': None}


def prepare(synthetic=False, out_dir=None, rebuild_split=False, data_kw=None):
    """공통 데이터 + 분할표. 반환: (data, split, manifest, out_dir)"""
    data_kw = data_kw or {}
    if synthetic:
        from . import forecast_synthetic as fsyn
        A, S = fsyn.make_raw()
        data = fd.build_forecast_table(S, A, **data_kw)
        out = Path(out_dir or OUT_SYNTH)
    else:
        data = fd.load_forecast_data(**data_kw)
        out = Path(out_dir or OUT_REAL)
    fp = fd.table_fingerprint(data.table)
    prev = out.parent / fd.PREVIOUS_PROTOCOL if out_dir is None and fd.PREVIOUS_PROTOCOL else None
    split, man = fs.load_or_make_split(data.table, out, data_fingerprint=fp, protocol=fd.PROTOCOL,
                                       rebuild=rebuild_split, previous_dir=prev)
    return data, split, man, out


def run(data, split, manifest, out_dir, experiments=None, table=None, targets=None, name=None,
        baseline=None, synthetic=False):
    """
    5-fold 실행 + 저장. experiments는 Experiment 목록 또는 fe.EXPERIMENTS 이름 목록.
    table을 주면(fd.add_features 결과 등) 그 표로 돌림. table은 data.table의 모든 열을 같은 순서·같은 값으로
    가져야 하고 추가 열만 허용 (fd.check_common_columns, 학습 전에 검사). baseline 기본 = 첫 번째 실험.
    반환: (scores, summary, compare, run_dir)
    """
    experiments = [fe.EXPERIMENTS[e] if isinstance(e, str) else e
                   for e in (experiments or DEFAULT_EXPERIMENTS)]
    names = [e.name for e in experiments]
    baseline = baseline or (BASELINE if BASELINE in names else names[0])
    if baseline not in names:
        raise ValueError(f'baseline {baseline!r}이 실험 목록 {names}에 없음')
    # 학습·저장 전에 검사: 공통 데이터가 분할표를 만든 데이터와 같고, table은 열만 더했는지
    fp = fd.table_fingerprint(data.table)
    if manifest.get('data_fingerprint') and fp != manifest['data_fingerprint']:
        raise ValueError('data.table이 분할표를 만든 공통 데이터와 다름 (지문 불일치). '
                         'data.table을 고치지 말고 fd.add_features로 열만 더하세요')
    table = data.table if table is None else table
    extra_cols = fd.check_common_columns(data.table, table, 'table')
    tg = list(targets or fd.TARGETS)
    scores, preds = fe.run_cv(table, split, experiments, targets=tg)
    summary = fe.summarize(scores, preds)
    comp = pd.concat([fe.compare(scores, baseline, n).assign(base=baseline, new=n)
                      for n in names if n != baseline], ignore_index=True) if len(names) > 1 else pd.DataFrame()

    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + (f'-{name}' if name else '')
    rd, i = Path(out_dir) / 'runs' / run_id, 1
    while rd.exists():                                   # 같은 초에 두 번 돌린 경우
        rd, i = Path(out_dir) / 'runs' / f'{run_id}-{i}', i + 1
    rd.mkdir(parents=True)
    scores.to_csv(rd / 'scores.csv', index=False)
    summary.to_csv(rd / 'summary.csv', index=False)
    comp.to_csv(rd / 'compare.csv', index=False)
    preds.assign(date=preds['date'].dt.strftime('%Y-%m-%d')).to_csv(rd / 'predictions.csv', index=False)
    meta = {'run_id': rd.name, 'synthetic': synthetic, 'protocol': fd.PROTOCOL, 'code': code_version(),
            'python': sys.version.split()[0], 'pandas': pd.__version__, 'packages': _versions(),
            'seed': manifest['seed'], 'model_seed': fe.SEED, 'data_settings': data.settings,
            'data_fingerprint': fp, 'extra_columns': extra_cols,
            'split_sha256': manifest['split_sha256'], 'split_settings':
                {k: manifest[k] for k in ('final_frac', 'n_folds', 'rounding', 'fold_rule')},
            'targets': tg, 'baseline': baseline, 'experiments':
                {e.name: {'model': getattr(e.make_model, '__name__', repr(e.make_model)),
                          'params': getattr(_safe_make(e), 'params', None),
                          'features': {t: e.features(t) for t in tg}}
                 for e in experiments},
            'final_test_scored': False,
            'all_complete': bool((summary['status'] == 'complete').all())}
    (rd / 'run_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                      encoding='utf-8')
    return scores, summary, comp, rd


def _safe_make(e):
    try:
        return e.make_model()
    except Exception:            # 기록용이라 만들기 실패는 무시
        return None


def _versions():
    out = {}
    for m in ('numpy', 'sklearn', 'catboost'):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    return out


def print_split(man):
    print(f"분할 [{man.get('protocol')}]: 참가자 {man['n_participants']}명 = 최종 평가 {man['n_final']} + "
          f"개발 {man['n_dev']} (fold 크기 {man['fold_sizes']}) | seed {man['seed']}")
    print(f"  split_sha256 {man['split_sha256']}")
    print(f"  data_fingerprint {man.get('data_fingerprint')}")
    if 'previous_protocol' in man:
        print(f"  이전 프로토콜 {man['previous_protocol']} 분할과 같음: {man['same_split_as_previous']} "
              f"(최종 평가 참가자 같음: {man['same_final_as_previous']})")


def print_diagnose(S):
    d = fd.date_assumption_check(S)
    print('dataDate 해석 진단 (분모 = 시각 검사 통과 조각, 분자 = 같은 사람의 앞 조각들 중 가장 늦은 종료보다 먼저 시작)')
    for k, lab in [('end_date', '종료일 가정'), ('start_date', '시작일 가정')]:
        r = d[k]
        print(f"  {lab}: {r['overlapping']:,} / {r['pieces']:,} = {r['rate']:.4%}")
    print(f"  |시계 차이 − (bedtimedur − 1)| ≤ {fd.CLOCK_TOL}분: {d['clock_matches_bedtimedur']:.5%} "
          f"(분모 {d['pieces_with_both_times']:,}조각)")


def main(argv=None):
    ap = argparse.ArgumentParser(description='오늘 밤 수면 예측 공통 5-fold 실행 (최종 평가 제외)')
    ap.add_argument('--synthetic', action='store_true', help='가짜 원본으로 흐름만 확인')
    ap.add_argument('--check', action='store_true', help='데이터·분할만 만들고 확인')
    ap.add_argument('--diagnose', action='store_true', help='dataDate 해석 진단만 출력')
    ap.add_argument('--experiments', nargs='+', default=DEFAULT_EXPERIMENTS, choices=list(fe.EXPERIMENTS),
                    help=f'돌릴 기본 실험 (기본: {" ".join(DEFAULT_EXPERIMENTS)})')
    ap.add_argument('--baseline', default=None, help='비교 기준 실험 (기본: ridge_base가 있으면 그것, 없으면 첫 실험)')
    ap.add_argument('--targets', nargs='+', default=None, choices=list(fd.TARGETS), help='목표 (기본: 셋 다)')
    ap.add_argument('--rebuild-split', action='store_true', help='저장된 분할표와 다를 때 다시 만듦')
    ap.add_argument('--out', default=None, help='저장 폴더 (기본 results/forecast[_synthetic]/<protocol>)')
    ap.add_argument('--name', default=None, help='run 폴더 이름 끝에 붙일 말')
    a = ap.parse_args(argv)

    if a.diagnose:
        if a.synthetic:
            from . import forecast_synthetic as fsyn
            S = fsyn.make_raw()[1]
        else:
            from . import data_loader as dl
            S = dl.read_raw()[1]
        print_diagnose(S)
        return
    data, split, man, out = prepare(a.synthetic, a.out, a.rebuild_split)
    if a.synthetic:
        print('[가짜 데이터] 아래 숫자는 실제 성능이 아님')
    fd.print_summary(data.info)
    print_split(man)
    if a.check:
        return
    scores, summary, comp, rd = run(data, split, man, out, experiments=a.experiments, targets=a.targets,
                                    name=a.name, baseline=a.baseline, synthetic=a.synthetic)
    base = json.loads((rd / 'run_meta.json').read_text(encoding='utf-8'))['baseline']
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print('\n[fold별 MAE(분)]')
        print(scores.pivot_table(index=['target', 'experiment'], columns='fold', values='mae').round(1))
        print('\n[요약] fold_mae_std는 다섯 fold MAE의 표본 표준편차(ddof=1). status가 complete가 아니면 숫자 없음')
        print(summary.drop(columns=['note', 'complete']).round(1).to_string(index=False))
        if not comp.empty:
            print(f'\n[비교] diff = {base} MAE − 다른 실험 MAE (양수 = 개선)')
            print(comp[comp['fold'] == 'mean'][['target', 'base', 'new', 'status', 'base_mae', 'new_mae', 'diff',
                                                'n_improved']].round(1).to_string(index=False))
    if not (summary['status'] == 'complete').all():
        print('\n경고: 완전한 5-fold가 아닌 결과가 있음. summary.csv의 status/note 확인')
    print(f'\n저장: {rd}')


if __name__ == '__main__':
    main()
