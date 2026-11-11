기준선·특징 결합 비교 — 2026-11-10 실험, 2026-11-11 공개

이 폴더는 완료된 실험의 코드와 집계 결과를 보존합니다.
원자료와 참가자별 특징표·분할표·예측은 저장소의 데이터 공개 규칙에 따라 포함하지 않습니다.
publication_manifest.json의 SHA-256으로 복사된 실험 파일의 원본 일치를 확인할 수 있습니다.

읽는 순서
1. artifacts/analysis/summary.csv: 60개 조건의 MAE, RMSE, 사람별 MAE, 30분 이내 비율.
2. artifacts/analysis/comparisons.csv, interactions.csv, additivity.csv: A/B/C의 단독·결합 효과.
3. artifacts/analysis/model_comparisons.csv: 네 모델의 대응 비교, 신뢰구간, Holm 보정.
4. artifacts/analysis/subgroups.csv: 과거 기록량·마지막 기록 이후 공백별 집계.
5. artifacts/resume_verification.json: FTT 84회 추가, 기존 6회 재사용, 기존 파일 보존 검증.
6. artifacts/four_model_independent_verification.json: 네 모델 예측 360개와 비교 30개 독립 재계산.
7. artifacts/feature_spec.json: 실제 입력 열, 출처 커밋, 분할·평가 밤의 지문.

설계와 해석
- 개발 528명, 247,475밤. 참가자 분리 5분할, 시드 0/1/2. 최종 93명은 학습·평가에서 제외.
- sleep: 주수면 minsasleep 합(분).
- bed: timetobed, 침대에 들어간 시각. 생리학적으로 잠든 시각이 아닙니다.
- wake: timeoutofbed, 침대에서 나온 시각.
- A: 학습 참가자로만 맞춘 ShrinkBlend를 기준으로 남은 오차를 학습하고 예측 때 기준선을 복원.
- B: 승호님의 L4w_lag7_offbase28_term 고정 구성. 개인 패턴·쉬는 날·학사 일정 입력.
- C: 민준님의 최근 실제 유효 기록 7개와 28일 통계 입력.
- B+C에서는 중복 기본 입력을 제거합니다.
- CatBoost 8조건 실험은 외부 학습 참가자 전체를 사용합니다.
- modelcheck_* 네 모델 비교는 동일한 내부 학습 참가자 약 90%를 사용합니다.
  나머지 약 10%는 신경망 조기 종료용입니다. 두 실험 표의 수치를 섞지 않습니다.
- 각 모델은 기존 구현의 첫 번째 고정 설정입니다. 설정 탐색을 거친 최적 성능 비교가 아닙니다.
- MLP와 FT-Transformer는 MPS, CatBoost와 Ridge는 CPU에서 실행했습니다.
- 오차는 시드별 오차를 평균한 값이며, 시드 예측을 평균한 앙상블 점수가 아닙니다.
- 사람 단위 부트스트랩 2,000회, 시드 20261110. 신뢰구간은 개별 95% 구간입니다.
- 근사 양측 부트스트랩 부호 꼬리 확률에 고정 비교군별 Holm 보정:
  주 비교 39개, 상호작용 12개, 네 모델 비교 30개.
- 신뢰구간은 학습 완료된 모델과 개발집단에 조건부입니다.
  재학습 불확실성이나 독립된 최종 93명 성능을 포함하지 않습니다.

주요 결과: 전체 결합 MAE(분)
모델             sleep       bed        wake
CatBoost         68.305175   61.460026  66.132288
Ridge            69.645505   63.004369  69.620331
MLP              68.416533   61.666008  66.261688
FT-Transformer   68.405047   61.530763  65.910340

FTT는 전체 결합 CatBoost보다 wake MAE가 약 13초 작지만 wake RMSE는 0.277분 큽니다.
sleep MAE는 약 6초 크고, bed의 약 4초 차이는 Holm 보정 후 유의하지 않습니다.
B+C에 A를 더한 이번 잔차학습 결합에서는 추가 개선이 확인되지 않았습니다.
혼합 기준선 단독은 개인 중앙값 단독보다 세 대상 모두 약 0.8~1.0분 개선했습니다.
단독 효과의 합과 실제 결합 효과는 다릅니다. 정확한 수치와 구간은 CSV를 사용합니다.

환경과 코드 검사
Python 3.12 환경에서 다음을 실행합니다. 아래 명령의 작업 폴더는 이 폴더입니다.
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r artifacts/requirements.lock.txt
  .venv/bin/python -B -m pytest tests -q

code/ 6개, tests/ 3개, snapshot/ 40개 Python 파일은 완료된 실험과 동일합니다.
snapshot의 Python 파일 전체가 입력 지문에 포함되므로 일부를 제거하지 않습니다.
새 위치에서 수행한 검사 기록은 artifacts/publication_test_verification.json에 있습니다.

완료 결과의 재집계: 학습을 다시 하지 않습니다
데이터 이용 권한이 있는 팀원이 비공개 원본 실험에서 다음을 이 폴더의 같은 위치로 복원합니다.
  artifacts/features.parquet
  artifacts/split.csv
  artifacts/predictions/  (JSON과 Parquet 쌍 전체)
원본 feature_spec.json과 일치하는 파일이어야 합니다. 참가자별 파일은 Git에서 제외됩니다.
  .venv/bin/python -B code/analyze_results.py --bootstrap 2000
입력 지문, 정답, 평가 밤, 참가자 분할, 누락과 중복을 검사한 뒤 집계합니다.
이 명령은 artifacts/analysis의 집계표를 다시 작성합니다.

원자료부터 새로 실행할 때
완료 원본과 분리된 작업복사본을 사용합니다. prepare_features.main()은 특징표와 명세를 다시 씁니다.
원자료는 저장소 루트의 data/nethealth에, 기존 v3 분할은 results/forecast/v3에 있어야 합니다.
prepare_features.py에는 원래 실행 경로를 보존했습니다. 다른 환경에서는 아래처럼 SOURCE만 지정합니다.
.venv/bin/python -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "code")
import prepare_features as preparation
preparation.SOURCE = Path.cwd().parents[2]
preparation.main()
PY
위 예시는 이 폴더에서 실행합니다. SOURCE는 capstone-lifelog 저장소 루트입니다.
기존 v3 분할이 없으면 스냅샷의 forecast_split.load_or_make_split으로 seed=0,
final_frac=0.15, n_folds=5, protocol='v3'를 사용해 만들고 EXPECTED_SPLIT과 대조합니다.
기존 분할 불일치를 rebuild=True로 우회하지 않습니다.
과거 예측 파일이 없으면 과거 정답 대조 여부와 feature_spec 해시가 달라질 수 있으므로
새 준비 자료와 완료 원본의 예측을 섞지 않습니다.

새 실험의 실행 명령
  .venv/bin/python -B code/run_factorial.py
  .venv/bin/python -B code/run_models.py --families catboost ridge --device cpu
  .venv/bin/python -B code/run_models.py --families mlp ftt --device mps
  .venv/bin/python -B code/analyze_results.py --bootstrap 2000
기존 체크포인트가 있으면 설정·입력·정답을 확인하고 건너뜁니다.
CPU 신경망 재실행 결과를 원래 MPS 실행과 동일한 결과로 취급하지 않습니다.
