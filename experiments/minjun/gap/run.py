"""
④ 기록이 끊겼을 때 — 실행 (공통 코드 PR #6, 프로토콜 v3 위)

    python -m experiments.minjun.gap.run [--seeds 10]

질문 (11/04 합의): 유효 기록 일수만 넣은 경우와 마지막 기록 이후 지난 일수까지 넣은 경우를 비교한다.
공통 기본 입력(날짜 기준 과거 7밤)은 빈칸 위치로 기록 수·경과일을 이미 담고 있어서, 날짜 정보가 없는
'최근 유효 기록 7개'에서 출발해 두 값을 하나씩 더한다.

  A  gap_A_base    : 공통 기본 입력 (= catboost_base)
  R0 gap_R0_recent : 최근 유효 기록 7개 + 요일
  R1 gap_R1_count  : R0 + n_valid7                 ← 유효 기록 일수만
  R2 gap_R2_since  : R1 + days_since_last          ← + 마지막 기록 이후 일수
  E  gap_E_last    : A + 마지막 유효 기록값         (7밤이 모두 빈 밤용)
  F  gap_F_lastsin : E + days_since_last
  H  gap_H_long28  : A + 28일 창 중앙값·표준편차   (참고. ③ 개인 기준선과 겹침)
주 비교: P1 = R1 → R2 (전체), P2 = E → F (n_valid7 == 0). 결과를 보기 전에 meta.json에 기록한다.
CatBoost 설정·분할·평가 밤은 공통 그대로이고, 학습 시드(random_seed)만 0..S-1로 반복한다.

저장: results/forecast/v3/gap3/<시각>/  (커밋하지 않음)
  features.parquet, preds_<조건>_s<시드>.parquet (pid, date, target, fold, y_true, y_pred), meta.json
최종 평가 참가자(15%)는 쓰지 않는다 (fe.run_cv).
"""
import argparse
import json
from datetime import datetime

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_run as fr
from experiments.minjun.gap.features import feat_gap, feat_long, feat_recent, LAST, rec

ARMS = {
    'gap_A_base': fd.base_features,
    'gap_R0_recent': lambda t: rec(t) + ['dow'],
    'gap_R1_count': lambda t: rec(t) + ['n_valid7', 'dow'],
    'gap_R2_since': lambda t: rec(t) + ['n_valid7', 'days_since_last', 'dow'],
    'gap_E_last': lambda t: fd.base_features(t) + [LAST[t]],
    'gap_F_lastsin': lambda t: fd.base_features(t) + [LAST[t], 'days_since_last'],
    'gap_H_long28': lambda t: fd.base_features(t) + [f'med28_{t}', f'sd28_{t}'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    a = ap.parse_args()

    data, split, man, out = fr.prepare()
    table = fd.add_features(data, feat_gap, feat_long, feat_recent)
    fp = fd.table_fingerprint(data.table)
    assert fp == man['data_fingerprint']
    rd = out / 'gap3' / datetime.now().strftime('%Y%m%d-%H%M%S')
    rd.mkdir(parents=True)
    extra = [c for c in table.columns if c not in data.table.columns]
    table[['pid', 'date'] + extra].to_parquet(rd / 'features.parquet', index=False)
    meta = {'data_fingerprint': fp, 'split_sha256': man['split_sha256'], 'protocol': fd.PROTOCOL,
            'code': fr.code_version(), 'catboost_params': fe.CATBOOST_PARAMS, 'seeds': list(range(a.seeds)),
            'arms': {n: {t: ARMS[n](t) for t in fd.TARGETS} for n in a.arms},
            'primary': {'P1': ['gap_R1_count', 'gap_R2_since', 'all'],
                        'P2': ['gap_E_last', 'gap_F_lastsin', 'n_valid7==0']},
            'final_test_scored': False}
    (rd / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

    for s in range(a.seeds):
        exps = [fe.Experiment(n, lambda s=s: fe.CatBoostModel(random_seed=s), ARMS[n]) for n in a.arms]
        t0 = datetime.now()
        scores, preds = fe.run_cv(table, split, exps)
        st = fe.summarize(scores, preds)
        assert (st['status'] == 'complete').all(), st
        for n, g in preds.groupby('experiment'):
            g = g.drop(columns='experiment').astype({'y_true': 'float32', 'y_pred': 'float32'})
            g.to_parquet(rd / f'preds_{n}_s{s}.parquet', index=False)
        print(f'seed {s} 완료 ({(datetime.now() - t0).seconds}s)', flush=True)
        print(st[['target', 'experiment', 'oof_mae']].round(3).to_string(index=False), flush=True)
    print(f'저장: {rd}')


if __name__ == '__main__':
    main()
