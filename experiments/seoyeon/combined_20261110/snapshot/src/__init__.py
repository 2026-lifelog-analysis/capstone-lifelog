"""
NetHealth 공통 분석 코드.

    from src import data_loader as dl, preprocessing as pp, baseline as bl, evaluation as ev

  data_loader   : 원본 CSV → 하루 한 줄 표 (load_nethealth)
  preprocessing : 전처리 규칙 (중복 제거, 착용 필터, 수면 병합, 42일 필터, 요일, 연구 기간 나누기)
  baseline      : 기준선과 이탈 점수 (personal_z, group_z, cusum, bad_score ...)
  evaluation    : 방법 비교와 평가 (compare_alarms, 가짜 이탈 주입, 결과표, log_result)

오늘 밤 수면 예측 (이상 탐지 하루 표와 별개, docs/forecast.md)
  forecast_dataset   : 밤 단위 예측 표 (주수면 정답 3개 + 과거 7밤 입력, 겹침 밤 제외)
  forecast_split     : 최종 평가 15% + 개발 85% 참가자 5-fold
  forecast_eval      : 공통 평가 밤, MAE 채점, 기본 모델(CatBoost 포함), run_cv, compare
  forecast_run       : python -m src.forecast_run
  forecast_synthetic : 가짜 원본
"""
