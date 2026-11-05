"""Generate a comparison report only from completed, verified local executions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources, verified_outputs

NAMES = dict(zip(MODELS, ['서연 고정14', '서연 고정28', '서연 이동28', '서연 양방향 CUSUM',
    '서연 개발기간 집단', '서연 MOMENT', '승호 기본', '승호 요일', '승호 공통효과 제거', '승호 최종']))


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
                     + ['|' + '|'.join(map(str, row)) + '|' for row in rows])


def alarm_table(frame, rate_id, models=MODELS):
    part = frame.loc[frame.rate_id.eq(rate_id) & frame.scope.eq('common')].set_index(['model', 'split'])
    rows = []
    for model in models:
        cal, test = part.loc[(model, 'calibration')], part.loc[(model, 'test')]
        rows.append([NAMES[model], f'{cal.alarm_rate:.4%}', int(test.alarm_days),
            f'{test.alarm_rate:.4%}', int(test.zero_people), f'{test.top10_alarm_share:.4%}'])
    return table(['방법', '보정 실측 경보율', '시험 경보 인일', '시험 실측 경보율',
                  '시험 경보 0회 사람', '상위 10% 경보 몫'], rows)


def synthetic_table(frame, rate_id, models=MODELS):
    part = frame.loc[frame.rate_id.eq(rate_id)].set_index(['model', 'scenario'])
    rows = []
    for model in models:
        cells = [NAMES[model]]
        for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']:
            row = part.loc[(model, scenario)]
            cells.append(f'{row.detection_rate_mean:.2%} / {row.delta_mean_mean * 100:+.2f}%p')
        rows.append(cells)
    return table(['방법', 'S1', 'S2', 'S3', 'S4', 'S5'], rows)


def generate(run, aggregate, output):
    run, aggregate, output = Path(run), Path(aggregate), Path(output)
    if output.exists() or (run / 'report_manifest.json').exists():
        raise FileExistsError('Report output already exists')
    records = {
        'clean': run / 'clean_evaluation.json',
        'synthetic': aggregate / 'aggregate_manifest.json',
        'original': run / 'original_seungho/manifest.json',
        'replay': run / 'replay_verification/verification.json',
        'port': run / 'model_port_validation.json',
        'figures': run / 'figures/plot_manifest.json',
    }
    manifests = {name: json.loads(path.read_text()) for name, path in records.items()}
    if any(record['status'] != 'passed' for record in manifests.values()):
        raise ValueError('All actual runs and validations must pass before reporting completion')
    verified_outputs(run, 'clean_evaluation.json', ['alarm_summary.csv', 'coverage.csv'])
    verified_outputs(aggregate, 'aggregate_manifest.json', ['seed_distribution_summary.csv'])
    verified_outputs(run / 'figures', 'plot_manifest.json',
                     ['daily_alarm_rates.png', 'synthetic_detection.png', 'synthetic_paired_delta.png'])
    paths = [*records.values(), run / 'alarm_summary.csv', run / 'coverage.csv',
        aggregate / 'seed_distribution_summary.csv', run / 'moment_clean/execution.json',
        run / 'split.json', run / 'protocol.json', run / 'inputs_before.json', run / 'preprocessing_counts.csv']
    before = {str(path): io.sha256(path) for path in paths}
    sources = source_hashes(['comparison_report.py', 'comparison_evaluate.py', 'comparison_io.py'])
    initial = json.loads((run / 'inputs_before.json').read_text())
    if initial != io.preserved_inputs():
        raise ValueError('An original input or tracked file changed')
    alarm = pd.read_csv(run / 'alarm_summary.csv')
    coverage = pd.read_csv(run / 'coverage.csv')
    synthetic = pd.read_csv(aggregate / 'seed_distribution_summary.csv')
    split = json.loads((run / 'split.json').read_text())
    moment = json.loads((run / 'moment_clean/execution.json').read_text())
    counts = pd.read_csv(run / 'preprocessing_counts.csv').set_index('stage')
    counts_table = table(['전처리 단계', '사람 수', '행·인일 수'],
        [[label, int(counts.loc[key, 'people']), f"{int(counts.loc[key, 'rows']):,}"]
         for key, label in [('raw_activity','활동 원자료'),('raw_sleep','수면 원자료 조각'),
            ('activity_dedup','활동 중복 처리 후'),('sleep_days','수면 일별 처리 후'),
            ('outer_union','외부 결합 후'),('after_min42','관측 42일 기준 후')]])
    port, original = manifests['port'], manifests['original']
    common = alarm.query("rate_id == 'r05' and scope == 'common' and model == 'fixed28'").set_index('split')
    cal, test = common.loc['calibration'], common.loc['test']
    cover = coverage.query("rate_id == 'r05' and split == 'test'").set_index('model')
    coverage_table = table(['방법', '시험기간 단독 계산 가능 인일', '공통 시험 인일'],
        [[NAMES[m], int(cover.loc[m, 'finite_days']), int(cover.loc[m, 'common_days'])] for m in MODELS])
    text = f'''# 서연 기존 모듈 1을 공통 전처리에 적용한 튜닝 비교

실제 완료된 실행만으로 작성했습니다. 작성 시각(UTC): {datetime.now(timezone.utc).isoformat()}.
실행 폴더: `{run}`. 원래 공통 코드, 01·02 노트북, 과거 결과보고서와 결과는 변경하지 않았습니다.

## 1. 무엇을 같게 하고 무엇을 비교했는가

공통 main 전처리를 그대로 재실행한 다음 서연의 기존 다섯 통계 모델과 고정 MOMENT, 승호의 네 CUSUM 단계를 적용했습니다.
공통 전처리는 `07d88da81819845e4f03e79d57f64ab416c366ed`, 승호 원형은 `1806f0a5604ab08111a94d62c01ac874a2fcb91c`로 고정했습니다.
이번 결과는 과거 착용률 70% 결과의 숫자를 다시 사용한 것이 아닙니다. **전처리는 공통 80%, 모델 계산 방식은 서연의 기존 방식**입니다.

{counts_table}

|비교 표본|이번 실행|
|---|---|
|공통 보정 표본|{int(cal.people)}명·{int(cal.days):,}인일|
|공통 시험 표본|{int(test.people)}명·{int(test.days):,}인일|

공통 코드는 착용률 80% 미만의 활동·심박 값을 결측으로 만들고 수면은 유지합니다. 수면 조각 처리와 90분 병합 규칙도 그대로입니다.
세 특징을 동시에 쓰는 서연 모델은 걸음·수면량·취침 시각이 모두 유한해야 점수를 냅니다. 행 수와 유효 평가 인일은 다릅니다.
연결부는 `pid`와 열 이름을 바꾸고 정오 기준 취침 분을 `(bed_min + 720) mod 1440`으로 자정 기준 분으로 변환합니다. 추가 70% 필터나 수면 재전처리는 없습니다.

## 2. 방법의 차이와 선정 이유

|항목|서연 기존 방식의 적용|승호 방식의 적용|
|---|---|---|
|입력|걸음·수면량·취침 시각 3개|걸음 1개|
|변화 방향|평소보다 증가·감소 모두|걸음 감소 방향|
|개인 중심·척도|중앙값·1.4826 MAD, 취침은 원형 거리|28일 중앙값·1.4826 MAD, 점수 ±8 절단|
|기준선|고정14(최소10일), 고정28(최소20일), 이동28(최소20일)|이동28(최소14일)|
|취침 시각|최단 원형 차이 [-720,720)분|최종 모델 입력에 없음|
|특징 종합|세 절댓값 표준화 점수의 최댓값|걸음의 악화 방향 점수|
|CUSUM|세 특징별 양방향, k=0.5|걸음 한 방향, 기본·요일·공통 k=0.5, 최종 k=0.25|
|결측·날짜 공백|누적 상태 초기화|누적 상태 유지|
|경보 후|누적 상태 유지|동일 경보선 h 초과 후 초기화|
|집단의 역할|개발기간 전체 인일의 고정 중심·척도|같은 날 30명 이상 걸음 중앙값 제거|
|사전학습 모델|고정 MOMENT-1-large의 복원 오차|사용하지 않음|

기존 수식을 이식한 이유는 전처리를 공통으로 고정한 상태에서 **이미 사용한 모듈의 행동 차이**를 비교하기 위해서입니다.
이번 결과로 중앙값, 1.4826, 원형 MAD, 최댓값 종합, 14·28일이 최적이라고 새로 검증한 것은 아닙니다.
특히 취침 원형 잔차에 1.4826을 곱하는 것은 기존 설계의 유지이며 원형분포에 대한 일반적 정규 일치성 보증이 아닙니다.
서연의 이전 코드와 {port['ast_function_count']}개 함수 구문을 대조했고, 실제 {port['sample_people']}명·{port['sample_person_days']:,}일에서 {port['score_rows']:,}개 모델행을 원함수와 대조했습니다. 그 표본에서 모든 열이 정확히 같았으며, 전체 참가자를 원함수와 모두 비교했다는 뜻은 아닙니다. 근거는 `model_port_validation.json`입니다.

승호 네 단계는 기본 → 요일 보정 → 같은 날 공통효과 제거 → k=0.25입니다. 같은 날 집단값은 당일 관측이 모인 뒤 이용하는 정보이며 내일 예측 정보가 아닙니다.
MOMENT는 동일한 사전학습 가중치를 고정해 3채널×512달력일을 복원합니다. 현재 관측값도 입력에 있으므로 미래예측이 아니라 당일 복원 기반 점수입니다.
기존 실험의 신경망 출력을 가져오지 않았으며, 이번 실행에서 새로 계산한 입력과 정확히 같은 여섯 필드만 검증 후 재사용했습니다.

## 3. 시간순 분할과 공정 비교의 범위

- 개발: {split['start']}~{split['development_end']}.
- 보정: {split['calibration_start']}~{split['calibration_end']}.
- 시험: {split['test_start']}~{split['end']}.

전체 관측 달력의 누적 40%, 70% 경계를 내림해 40·30·30으로 나눴습니다. 이전 모듈의 선택을 유지한 설계값이지 이번 비교에서 재최적화한 비율이 아닙니다.
보정·시험 각각 열 방법이 모두 계산 가능한 사람·날짜를 고정했습니다. MOMENT 누락이 생겼다면 일부를 버리고 재보정하지 않고 검증 실패로 처리하도록 했습니다.
목표 경보율 5%는 이전 모듈 기준, 1/30은 팀 공통 기준을 함께 비교하기 위한 값입니다. 실제 경보 개수를 같게 강제하지 않았습니다.
서연 다섯 모델과 MOMENT는 보정 점수의 `higher` 분위수·엄격 초과(`>`)를 쓰고, 승호는 보정 구간만을 세는 원방식 30회 이분 탐색으로 h를 맞췄습니다.
이번 시간순 비교에서는 승호의 누적합 초기화선과 판정선을 같은 h로 통일했습니다. 시험 점수로 임계값을 선택하지 않았습니다.
경보는 하루 점수 초과입니다. 과거 보고서의 연속 2일 지속 규칙을 이번 결과에 섞지 않았습니다.

공통 main의 관측 42일 참가자 선정은 전체 관측기간을 사용합니다. 따라서 임계값 보정은 미래 시험값을 쓰지 않아도, 모집단 선정까지 완전히 미래 정보가 없는 전향적 실험은 아닙니다.
공통 교집합 비교는 조건을 충족한 사람에게 제한됩니다. 다음 단독 계산 가능량도 함께 읽어야 합니다.

{coverage_table}

## 4. 실제 미주입 자료의 결과

표의 모든 방법은 위의 동일 공통 보정·시험 표본을 씁니다. 상위 10% 인원 수는 공통 참가자의 10%를 내림합니다.
경보가 적거나 고르게 분포한다는 사실을 정확도 또는 임상 오탐률로 해석할 수 없습니다.

### 보정 목표 5%

{alarm_table(alarm, 'r05')}

### 보정 목표 1/30

{alarm_table(alarm, 'r033')}

시험 경보율이 보정 목표와 다른 것은 새로운 기간에서 관측된 경보 빈도의 차이입니다. 목표율을 달성했는지와 실제 상태를 맞혔는지는 별개입니다.
방법별 경보 겹침·Jaccard·불일치 인일은 `pairwise.csv`, 정확한 임계값은 `clean_thresholds.json`에 있습니다.

## 5. 실제 자료에 인공 변화를 넣은 동일 구간 비교

시험기간에만 사람마다 14일 구간을 선택하고 20회(seed 0~19) 반복했습니다. 직전 28일 완전 관측 14일 이상, 구간 내 공통 유효일 7일 이상을 요구했습니다.
이 최소량과 변화량은 구성적 검증을 위한 설계값입니다. 주입 가능한 사람과 시점을 미래 관측 가능성으로 고른 오프라인 평가입니다.
같은 seed의 모든 방법·시나리오에 사람·날짜를 같게 적용했습니다. 임계값은 미주입 보정값으로 고정하고, 이동 기준선과 CUSUM 상태는 변화 후 자료로 다시 계산했습니다.

|시나리오|14일에 주입한 변화|
|---|---|
|S1|걸음 −30%, 즉시|
|S2|걸음 −30%까지 점진 변화|
|S3|걸음 −15%까지 점진 변화|
|S4|취침 +120분, 즉시|
|S5|걸음 −20%·수면량 −10%·취침 +60분까지 점진 변화|

점진 변화는 달력 일차 `(0~13 + 1)/14`에 비례합니다. 결측값에 임의 값을 채우지 않았습니다.
아래 셀은 **주입 구간에서 한 번 이상 경보가 난 비율 / 같은 사람·날짜 미주입 대비 차이(%p)**의 20seed 평균입니다.
미주입 구간은 임상적으로 정상이라는 라벨이 아니므로 이를 특이도나 임상 오탐률로 쓰지 않습니다.

### 보정 목표 5%

{synthetic_table(synthetic, 'r05')}

### 보정 목표 1/30

{synthetic_table(synthetic, 'r033')}

사람별 20회 평균은 `participant_summary.csv`, 각 seed의 참가자 평균은 `seed_summary.csv`, 원구간별 판정은 `segment_results.csv.gz`에 있습니다.
같은 사람을 20번 주입한 것이므로 20배의 독립 참가자가 아닙니다. 여기서는 독립 행으로 간주한 유의성 검정을 하지 않았습니다.
탐지 지연은 주입 시작을 0일로 하여 **탐지된 구간만** 계산하며 미탐지는 결측입니다. 조건부 지연과 그 분모를 집계 CSV에 별도로 남겼습니다.

## 6. 승호 원설정 재현과 이번 비교는 다릅니다

원설정은 별도로 개발 {original['dev_people']}명·확인용 {original['holdout_people']}명·{original['holdout_rows']:,}인일, S1·S3·S5, 네 경보율(1/14·1/30·1/60·1/90), 각{original['n_reps']}회를 실제 실행했습니다.
원설정의 사람 70·30 분할, 확인용 전체 기간 보정, 내부 초기화선 h와 평가기 분위수 q 차이를 그대로 보존했습니다.
완료 기록은 `original_seungho/manifest.json`, 72행 요약은 `final_table.csv`, 짝지은 60행 비교는 `final_compare.csv`입니다.
따라서 원설정 표의 탐지율을 위의 40·30·30 미래 시험 결과와 같은 분모·같은 평가라고 묶지 않습니다.
승호의 과거 탐색 203개 과업 전체나 당시 ‘1위’ 선정 과정을 모두 재현한 것은 아닙니다.

## 7. 얻은 의의와 확인할 수 없는 부분

1. 같은 전처리·보정·시험 표본에서 기존 서연 방법과 승호 방법이 어떤 날 경보를 내는지, 계산 가능한 자료량이 얼마나 다른지 실제로 비교할 수 있습니다.
2. 걸음 감소 누적에 초점을 맞춘 승호 모델과 수면·시각을 포함한 양방향 서연 모델은 목표가 다릅니다. S4처럼 입력 걸음을 바꾸지 않는 변화에서 승호의 반응이 없다는 것은 센서 입력 범위의 차이이지 모든 상태판단 능력의 열등함을 뜻하지 않습니다.
3. 고정 기준선은 처음 상태를 유지하고 이동 기준선은 변화에 적응합니다. 누적합 초기화·결측 처리·점수 절단도 다르므로 결과 차이를 단일 요인 효과로 확정하지 않습니다.
4. 당일 공통효과 제거는 전체가 함께 바뀌는 효과도 제거할 수 있습니다. 이번 주입에서도 전체 당일 중앙값을 재계산했으며, 이것이 임상적으로 제거해야 할 효과인지는 라벨 없이 판단할 수 없습니다.
5. NetHealth 일별 정답 상태 라벨이 없어 정확도·민감도·특이도·임상 오탐률, 실제 건강상태 악화에 대한 최우수 방법은 검증하지 않았습니다. 인공 변화 탐지는 실제 상태 변화 검증을 대신하지 않습니다.
6. 이번 실험은 공통 80% 전처리 위에서의 비교이며, 과거 70% 전처리보다 품질이 좋아졌다는 근거가 아닙니다. 수면 조각의 중복·기기 측정오차도 공통 전처리를 유지한 범위 밖에서 새로 교정하지 않았습니다.
7. MOMENT 사전학습 가중치의 사용이 이 자료에서 임상적 유효성을 보증하지 않습니다. 입력 포함 당일 복원, 512일 창·마스크·정규화 설계의 영향을 분리한 추가 비교는 하지 않았습니다.

## 8. 코드·실행·그림 검증 기록

- 기존 원자료·추적 파일·과거 모델 코드·DOCX의 시작/완료 해시를 대조했습니다. 신경망 체크포인트도 해시로 고정했습니다.
- 최초 루트 전처리/점수/보정 단계에는 신규 미추적 소스의 시작 해시가 빠져 있었습니다. 이를 과거에 기록한 것처럼 쓰지 않고, 별도 `replay_verification`에서 소스 전후 해시를 남겨 전체 재실행하고 11개 표의 모든 셀과 세 설정 JSON의 정확한 일치를 확인했습니다.
- 최초 CSV 기본 파서 실행은 `.initial_parse` 파일로 보존했습니다. 비교 전 `round_trip` 파서로 통일해 재실행했으며 현재 표는 후자의 결과만 사용합니다.
- MOMENT 미주입 요청 {moment['requested_persondays']:,}인일, 완료 {moment['completed_persondays']:,}인일. 신규 추론 {moment['new_inferred_persondays']:,}, 같은 실행의 여섯 입력 정확 일치 재사용 {moment['exact_six_field_cache_persondays']:,}인일입니다. 진단용 반복 추론은 모델 기록에 별도입니다.
- 합성 입력이 바뀌면 새 추론합니다. 새 프로세스로 이어진 경우 이번 실행의 입력·모델·저장값 검증과 실제 재추론 대조를 통과한 이번 실행 캐시만 가져옵니다. 이를 과거 실험 결과 재사용이나 동일 프로세스라고 부르지 않습니다.
- 반복·마스크 점검은 지정 표본에 대한 수치 안정성 점검이지 신경망의 전체 정확성 증명이 아닙니다.

계산 소스는 `comparison_run.py`, `comparison_models.py`, `comparison_seungho.py`, `comparison_moment.py`, `comparison_moment_inputs.py`, `comparison_synthetic.py`입니다.
승호 원설정 재현은 `comparison_original.py`, 집계는 `comparison_evaluate.py`·`comparison_aggregate.py`, 재실행 대조는 `comparison_verify.py`입니다.
기록용 노트북은 `notebooks/03_서연_튜닝비교.ipynb`이며 저장된 소스에는 실행 출력과 원자료를 넣지 않습니다.

### 그래프 읽는 방법

- `figures/daily_alarm_rates.png`: 파랑은 보정, 주황은 시험의 실측 경보율, 점선은 보정 목표입니다. 점선과 가깝다고 정확도가 높은 것이 아닙니다.
- `figures/synthetic_detection.png`: 동일 14일 구간에서 경보가 한 번 이상 난 비율의 20seed 평균입니다. 미주입 경보가 많은 모델도 높게 나올 수 있으므로 다음 그림을 함께 읽습니다.
- `figures/synthetic_paired_delta.png`: 주입 탐지에서 같은 구간의 미주입 경보를 뺀 차이입니다. 양수는 구성한 변화에 경보가 더 반응했다는 뜻이며 실제 질병 탐지 향상은 아닙니다.

각 그림의 수치는 CSV에서 직접 계산했고 PDF도 함께 저장했습니다. 개인정보 행은 로컬 무시 경로에만 남겼으며 이번 작업에서는 푸시하지 않았습니다.
'''
    if before != {str(path): io.sha256(path) for path in paths}:
        raise ValueError('Report evidence changed during generation')
    provenance = verify_sources(sources)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(text)
    io.write_json(run / 'report_manifest.json', dict(status='passed', output=str(output),
        output_sha256=io.sha256(output), evidence=before, **provenance))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['run', 'aggregate', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    generate(args.run, args.aggregate, args.output)
