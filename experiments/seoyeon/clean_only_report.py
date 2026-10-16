"""Publish only completed clean-data comparisons, with explicit recovery limits."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon.comparison_report import NAMES, alarm_table, table
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources, verified_outputs

LABELS = ['SY fixed14', 'SY fixed28', 'SY rolling28', 'SY CUSUM28', 'SY group',
          'SY MOMENT', 'SH base', 'SH weekday', 'SH common', 'SH final']
RATES = [('r05', 5.), ('r033', 100 / 30)]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _evidence(view: Path) -> tuple:
    files = ['alarm_summary.csv', 'coverage.csv', 'pairwise.csv', 'clean_thresholds.json']
    evaluation = verified_outputs(view, 'clean_evaluation.json', files)
    paths = [view / name for name in files + ['clean_evaluation.json', 'split.json',
        'protocol.json', 'moment_clean/execution.json', 'model_port_validation.json',
        'replay_verification/verification.json']]
    paths.append(view.parent / 'cache_provenance.json')
    hashes = {str(path): io.sha256(path) for path in paths}
    protocol, moment = _load(view / 'protocol.json'), _load(view / 'moment_clean/execution.json')
    expected = io.protocol()
    if any(protocol.get(key) != expected[key] for key in
           ['main_commit', 'reference_commit', 'preprocessing', 'fractions', 'rates']):
        raise ValueError('Unexpected clean-only protocol')
    if (moment.get('status') != 'complete' or moment.get('requested_persondays') != 87886
            or moment.get('completed_persondays') != 87886
            or moment.get('new_inferred_persondays') != 9550
            or moment.get('exact_six_field_cache_persondays') != 78336):
        raise ValueError('Incomplete or inconsistent clean inference')
    proof = _load(paths[-1])
    if (proof.get('verified_persondays') != 78336 or proof.get('fresh_replay_persondays') != 8
            or proof.get('model_runtime_identical') is not True
            or proof.get('all_six_inputs_rebuilt') is not True
            or proof.get('all_three_endpoint_errors_and_mse_recomputed') is not True):
        raise ValueError('Missing verified recovery provenance')
    if (evaluation.get('threshold_test_access') is not False
            or evaluation.get('all_requested_common_moment_finite') is not True
            or evaluation.get('calibration_rows') != 36260 or evaluation.get('test_rows') != 23721):
        raise ValueError('Unexpected common evaluation or leakage guard')
    port, replay = (_load(view / p) for p in
                    ['model_port_validation.json', 'replay_verification/verification.json'])
    if port.get('status') != 'passed' or replay.get('status') != 'passed':
        raise ValueError('Unverified model port or statistical replay')
    frames = [pd.read_csv(view / name) for name in files[:3]]
    _validate_tables(*frames)
    return (*frames, moment, proof, port, replay, _load(view / 'split.json'), hashes)


def _validate_tables(alarm: pd.DataFrame, coverage: pd.DataFrame, pairs: pd.DataFrame) -> None:
    common = alarm.loc[alarm.scope.eq('common')]
    if len(common) != 40 or common.duplicated(['rate_id', 'model', 'split']).any():
        raise ValueError('Missing or duplicate common alarm metrics')
    for rate_id, _ in RATES:
        for split, days, people in [('calibration', 36260, 181), ('test', 23721, 135)]:
            rows = common.loc[common.rate_id.eq(rate_id) & common.split.eq(split)]
            cov = coverage.loc[coverage.rate_id.eq(rate_id) & coverage.split.eq(split)]
            if (set(rows.model) != set(MODELS) or not rows.days.eq(days).all()
                    or not rows.people.eq(people).all() or not rows.top_people.eq(int(people*.1)).all()
                    or not np.allclose(rows.alarm_rate, rows.alarm_days / days)
                    or set(cov.model) != set(MODELS) or len(cov) != 10
                    or not cov.common_days.eq(days).all() or not cov.finite_days.ge(days).all()):
                raise ValueError('Inconsistent common cohort or alarm arithmetic')
        part = pairs.loc[pairs.rate_id.eq(rate_id)]
        if len(part) != 45 or part.duplicated(['a', 'b']).any() or not part.common_days.eq(23721).all():
            raise ValueError('Incomplete pairwise comparisons')


def _figures(alarm: pd.DataFrame) -> dict:
    figures = {}
    for name in ['daily_alarm_rates', 'alarm_concentration']:
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
        for axis, (rate_id, target) in zip(axes, RATES):
            part = alarm.loc[alarm.scope.eq('common') & alarm.rate_id.eq(rate_id)]
            y = np.arange(len(MODELS))
            if name == 'daily_alarm_rates':
                for split, offset, color, hatch in [('calibration', -.19, '#4477AA', ''),
                                                    ('test', .19, '#EE7733', '//')]:
                    values = part.loc[part.split.eq(split)].set_index('model').loc[MODELS, 'alarm_rate']*100
                    axis.barh(y + offset, values, height=.36, color=color, hatch=hatch, label=split)
                axis.axvline(target, color='#444444', linestyle='--', label='calibration target')
                axis.set_xlim(left=0)
                axis.set_xlabel('Observed daily alarm rate (%)')
            else:
                test = part.loc[part.split.eq('test')].set_index('model').loc[MODELS]
                axis.barh(y-.19, test.top10_alarm_share*100, height=.36, color='#4477AA',
                          label='Share of alarms held by top 13 people')
                axis.barh(y+.19, test.zero_people/test.people*100, height=.36,
                          color='#EE7733', hatch='//', label='Share of people with zero alarms')
                axis.set_xlim(0, 100)
                axis.set_xlabel('Share (%) | denominators differ; see legend')
            axis.set(yticks=y, yticklabels=LABELS, title=f'Calibration target {target:.2f}%')
            axis.invert_yaxis()
            axis.grid(axis='x', alpha=.18)
            axis.legend(fontsize=8, loc='lower right')
        if name == 'daily_alarm_rates':
            upper = max(axis.get_xlim()[1] for axis in axes)
            for axis in axes:
                axis.set_xlim(0, upper)
        fig.suptitle('NetHealth clean data | common test: 135 people / 23,721 person-days\n'
                     'Alarm frequency and concentration are not clinical accuracy')
        figures[name] = fig
    return figures


def _pair_table(pairs: pd.DataFrame, rate_id: str) -> str:
    part = pairs.loc[pairs.rate_id.eq(rate_id) & (pairs.a.eq('moment') | pairs.b.eq('moment'))]
    rows = []
    for row in part.itertuples():
        other = row.b if row.a == 'moment' else row.a
        rows.append([NAMES[other], int(row.overlap_days), f'{row.jaccard:.4%}',
                     int(row.disagreement_days), f'{row.disagreement_days/row.common_days:.4%}'])
    return table(['MOMENT와 비교한 방법', '공동 경보 인일', 'Jaccard', '불일치 인일', '불일치율'], rows)


def _text(alarm, coverage, pairs, moment, proof, port, replay, split) -> str:
    cov = coverage.loc[coverage.rate_id.eq('r05') & coverage.split.eq('test')].set_index('model')
    cover_table = table(['방법', '단독 계산 가능 시험 인일', '공통 시험 인일'],
        [[NAMES[m], f'{int(cov.loc[m, "finite_days"]):,}', f'{int(cov.loc[m, "common_days"]):,}'] for m in MODELS])
    return f'''# 서연 기존 모듈 1 × 공통 전처리 × 승호 튜닝: 미주입 자료 비교

작성 시각(UTC): {datetime.now(timezone.utc).isoformat()}. 실제 완료된 미주입 자료만 보고합니다.
**사용자 요청에 따라 합성 실험은 이번 공개 범위에서 제외했습니다. MOMENT 합성 추론·열 방법 합성 비교는 미실시이며 완료로 주장하지 않습니다.**
통계 아홉 방법의 과거 합성 결과는 로컬에 보존하지만 이 보고서의 표·그림·결론에는 포함하지 않습니다.

## 1. 동일 조건과 실제 평가 표본

main 전처리 `07d88da81819845e4f03e79d57f64ab416c366ed`, 승호 원형 `1806f0a5604ab08111a94d62c01ac874a2fcb91c`를 고정했습니다.
활동 착용률 80%, 수면 조각 병합 90분, 최소 관측 42일, 시간순 개발·보정·시험 40·30·30입니다.
80% 미만 활동·심박은 결측 처리하지만 수면은 유지합니다. 서연의 걸음·수면량·취침 시각 세 특징이 모두 유한한 날만 종합점수를 냅니다.
연결부는 열 이름·개인 식별자 형식을 맞추고 정오 기준 취침 시각을 `(bed_min + 720) mod 1440`으로 자정 기준 분으로 바꿉니다. 추가 70% 필터나 별도 수면 병합은 없습니다.

|구간|달력 경계|공통 평가 표본|
|---|---|---|
|개발|{split['start']}~{split['development_end']}|집단 기준선 구성|
|보정|{split['calibration_start']}~{split['calibration_end']}|181명·36,260인일|
|시험|{split['test_start']}~{split['end']}|135명·23,721인일|

전체 달력일의 누적 40%·70% 경계를 내림했습니다. 이 비율은 이전 설계를 유지한 값이지 이번에 최적성을 검증한 값이 아닙니다.
열 방법 모두 계산 가능한 동일 사람·날짜 교집합에서 비교했습니다. MOMENT 누락을 이유로 표본을 줄여 재보정하지 않았습니다.
보정 목표 5%와 1/30은 두 설계 기준을 비교하기 위한 값입니다. 경보 개수를 강제로 같게 만든 것은 아닙니다.
서연 여섯 방법은 보정점수의 `higher` 분위수와 엄격 초과 `>`를 사용합니다. 승호는 보정 구간만 세는 30회 이분 탐색으로 누적합 초기화선과 경보선 h를 동일하게 맞춥니다.
시험 점수로 임계값을 조정하지 않았으며 연속 2일 지속 규칙은 적용하지 않았습니다.

{cover_table}

## 2. 기존 서연 방식과 승호 방식의 차이

|항목|서연 다섯 통계 모델|승호 네 단계|
|---|---|---|
|입력과 방향|걸음·수면량·취침 시각, 양방향 이탈|걸음 감소 방향|
|개인 기준선|고정14/최소10일, 고정28/최소20일, 과거 이동28/최소20일|과거 이동28/최소14일|
|중심·척도|중앙값·1.4826 MAD, 취침 원형 잔차 [-720,720)분|중앙값·1.4826 MAD, 점수 ±8 절단|
|종합|세 특징 절댓값 표준화 점수의 최댓값|걸음 감소 점수|
|CUSUM|특징별 양방향 k=0.5|한 방향, 기본·요일·공통 k=0.5, 최종 k=0.25|
|결측·날짜 공백|누적 상태 초기화|누적 상태 유지|
|경보 후|누적 상태 유지|동일 경보선 h 초과 후 초기화|
|집단 정보|개발기간 전체 인일의 고정 중심·척도|당일 30명 이상 중앙값 제거|

서연 고정14·고정28·이동28·CUSUM28·개발기간 집단 기준선에 고정 MOMENT를 더했습니다. 승호는 기본→요일 보정→당일 공통효과 제거→k 변경 네 단계입니다.
MOMENT는 `AutonLab/MOMENT-1-large` 고정 가중치, 걸음·수면량·취침 시각 3채널×512달력일 입력의 당일 복원 오차입니다. 현재 관측이 입력에 있으므로 미래예측 모델 평가가 아닙니다.
이전 수식을 유지한 목적은 전처리를 통일한 상태에서 기존 방법의 실제 경보 행동을 비교하는 것입니다. 1.4826, 원형 MAD, 14·28일, 최댓값 종합이 최적이라는 새 증거는 아닙니다.
특히 원형 잔차의 1.4826 보정이 모든 원형분포에서 정규분포와 같은 일치성을 가진다고 검증하지 않았습니다.
원모델 {port['ast_function_count']}개 함수의 구문과 실제 {port['sample_people']}명·{port['sample_person_days']:,}인일·{port['score_rows']:,}모델행을 대조했습니다. 표본 내 값은 정확히 같았으나 모든 참가자 원함수 비교는 아닙니다.
당일 집단 중앙값은 당일 자료가 모인 뒤 이용합니다. 완전히 실시간·미래예측 상황으로 해석할 수 없습니다.

## 3. 실제 경보 결과

경보율의 분모는 위의 공통 평가 인일입니다. 경보 0회 사람의 분모는 시험 135명입니다.
상위 10% 경보 몫은 경보 수 상위 13명(135×0.1 내림)의 경보를 전체 경보로 나눈 값입니다. 개인별 관측일 차이의 영향도 포함합니다.

### 보정 목표 5%

{alarm_table(alarm, 'r05')}

### 보정 목표 1/30

{alarm_table(alarm, 'r033')}

보정 목표와 시험 실측의 차이는 새 기간의 경보 빈도 변화입니다. 시험 경보가 적거나 사람별로 고르다는 사실만으로 더 정확하거나 임상적으로 우수하다고 판정할 수 없습니다.

## 4. MOMENT와 다른 방법이 같은 날을 선택하는가

Jaccard는 공동 경보 인일 ÷ 둘 중 하나 이상 경보 인일입니다. 불일치율은 두 방법의 판정이 다른 인일 ÷ 공통 시험 23,721인일입니다.
이는 정답 일치율이 아니라 방법 간 일치도입니다. 동일 표본에서도 낮은 Jaccard는 서로 다른 날을 고른다는 뜻일 뿐 우열을 뜻하지 않습니다.

### 보정 목표 5%

{_pair_table(pairs, 'r05')}

### 보정 목표 1/30

{_pair_table(pairs, 'r033')}

나머지를 포함한 45쌍×2조건의 집계는 로컬 `pairwise.csv`에 보존했습니다.

## 5. 실제 실행·복구 검증의 범위

- 미주입 요청·완료: {moment['completed_persondays']:,}/{moment['requested_persondays']:,}인일. 새 복구 프로세스의 신규 추론 {moment['new_inferred_persondays']:,}인일, 같은 실험의 검증된 교차 프로세스 재사용 {moment['exact_six_field_cache_persondays']:,}인일입니다.
- 기존 78,336인일은 원프로세스 신규 78,333건과 선행 점검 추론 재사용 3건입니다. 원래 선행 점검은 신규 4행이며, 완료 범위 밖 1행은 가져오지 않았습니다. 진단 추론과 요청 인일을 더해 전체 자료량을 부풀리지 않습니다.
- 재사용 후보 전체의 여섯 입력과 감사값을 재구축하고 세 특징 끝점 잔차·MSE를 재계산했습니다. 실제 신경망 재실행 대조는 표본 {proof['fresh_replay_persondays']}행이며 최대 절대차 {proof['fresh_replay_max_abs']:g}였습니다. 전체 78,336행 신경망 재실행을 뜻하지 않습니다.
- 모델·체크포인트·실행환경을 대조하고 같은 실험 입력에 한해서만 재사용했습니다. 다른 과거 실험의 예측값을 가져오지 않았습니다.
- 부분 파일의 해시는 중단 후 처음 확보했습니다. 복구 검사 전후 보존은 확인할 수 있지만 중단 전 바이트의 무결성 증거는 없습니다.
- 초기 통계 실행에는 미추적 소스의 시작 해시가 빠져 있었습니다. 이를 소급 기록한 것으로 포장하지 않고 별도 재실행에서 {len(replay['tables'])}개 표의 전체 셀을 정확 대조했습니다.
- 합성 MOMENT·열 방법 합성 비교는 사용자 요청으로 미실시입니다. 이전 아홉 통계 방법의 로컬 합성 완료를 열 방법 완료로 바꾸지 않았습니다.

## 6. 그래프와 해석 방법

`daily_alarm_rates.png`·`.pdf`: 파랑(보정)과 주황 사선(시험)은 실측 경보율, 점선은 보정 목표입니다. 가로축은 0부터 시작하며 두 조건의 범위가 같습니다. 목표와 가깝다고 상태판단 정확도가 높은 것은 아닙니다.

`alarm_concentration.png`·`.pdf`: 파랑은 상위 13명의 전체 경보 몫(분모=경보 수), 주황 사선은 경보 0회 사람 비율(분모=135명)입니다. 서로 분모가 다른 두 지표를 직접 같은 확률로 해석하지 않습니다. 모든 수치는 위 표에서 확인할 수 있습니다.

## 7. 의의와 검증하지 못한 부분

동일 전처리·보정·시험 표본에서 여섯 서연 방법과 네 승호 단계를 실제 실행하여 경보 빈도·집중도·겹침·계산 가능량을 확인했습니다.
세 특징 양방향 변화와 걸음 감소 누적은 목표 자체가 다릅니다. 기준선 적응, 입력 수, 누적 초기화·결측 처리·절단이 함께 달라 결과 차이를 단일 튜닝 효과로 확정할 수 없습니다.
NetHealth 일별 정답 상태 라벨이 없어 정확도·민감도·특이도·임상 오탐률·최우수 모델은 검증하지 않았습니다. 합성 변화 반응·탐지 지연도 이번 공개 결과로 판단하지 않습니다.
최소 42일 참가자 선정은 전체 관측기간에 의존하므로 모집단 선정까지 완전히 전향적인 검증은 아닙니다. 공통 교집합에 포함되지 않은 사람으로의 일반화도 검증하지 않았습니다.
80% 전처리가 과거 70%보다 자료 품질이 우수하다는 비교도 아니며, 기기 측정오차·수면 조각 중복을 새로 교정한 연구도 아닙니다.
MOMENT의 사전학습, 512일 입력, 마스크·정규화 선택이 임상 타당성을 보증하지 않습니다. CPU와 MPS 결과의 동등성, 다른 환경·새 클론의 끝까지 실행은 별도 검증하지 않았습니다.

## 8. 재현 의존성과 공개 범위

새 클론과 NetHealth CSV 세 개만으로 전체 검증이 바로 재현되는 구조는 아닙니다.
`data/nethealth/`의 Activity·Sleep·Calendar 원자료 외에 상위 `모듈1_NetHealth/02_코드/models.py`, 보존 검사 대상 기존 `결과보고서.docx` 및 원코드 파일이 필요합니다.
MOMENT는 별도 torch·momentfm 실행환경, `모듈1_NetHealth/00_원자료/moment_checkpoint/`의 config·가중치와 `03_처리기록/moment_checkpoint.json`이 필요합니다.
선행 승호 원설정 재현은 무시 경로 `_archive/seungho_1806f0a/experiments/`의 common·final·rule_baseline 및 해당 Git 객체에 의존합니다. 그 재현의 사람 70·30 분할은 이 보고서 시간순 40·30·30 결과와 합치지 않았습니다.
현재 복구·보고서 재현은 로컬 실행 기록에도 의존합니다. 기록용 `03_서연_튜닝비교.ipynb`는 기본적으로 검증된 로컬 집계만 읽으며 추론·합성 작업을 자동 시작하지 않습니다.
원자료·개인별 행·전처리 CSV·가중치·환경·개인정보가 들어갈 수 있는 로컬 로그·노트북 실행 출력은 공개하지 않습니다. 코드·검증된 집계 보고서·개인정보 없는 집계 그림만 공개 대상입니다.
원자료·기존 공통 코드·공통 01·02 노트북·기존 DOCX와 과거 결과는 보존합니다. 보고서 생성 성공은 Git 푸시·PR 반영 성공을 의미하지 않습니다.
'''


def generate_clean_report(view: Path, report: Path, figures: Path) -> dict:
    """Write new aggregate-only artifacts after checking clean-stage evidence."""
    view, report, figures = Path(view), Path(report), Path(figures)
    manifest_path = view / 'clean_only_report_manifest.json'
    if report.exists() or figures.exists() or manifest_path.exists():
        raise FileExistsError('Clean-only publication output already exists')
    source = source_hashes(['clean_only_report.py', 'comparison_report.py', 'comparison_evaluate.py',
                            'comparison_io.py', 'comparison_safety.py'])
    *evidence, before = _evidence(view)
    content = _text(*evidence)
    made = _figures(evidence[0])
    figures.mkdir(parents=True, exist_ok=False)
    try:
        for name, figure in made.items():
            for extension in ['png', 'pdf']:
                figure.savefig(figures / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    finally:
        for figure in made.values():
            plt.close(figure)
    if before != {path: io.sha256(path) for path in before}:
        raise ValueError('Clean-only report evidence changed during generation')
    provenance = verify_sources(source)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        stream.write(content)
    outputs = {path.name: io.sha256(path) for path in sorted(figures.iterdir())}
    result = dict(status='passed', scope='clean_only', synthetic_claim=False,
        report_sha256=io.sha256(report), output_sha256=io.sha256(report),
        outputs_sha256=outputs, evidence_sha256=before, **provenance)
    io.write_json(figures / 'plot_manifest.json', result)
    io.write_json(manifest_path, result)
    return result
