"""
NetHealth 공통 분석 코드.

    from src import data_loader as dl, preprocessing as pp, baseline as bl, evaluation as ev

  data_loader   : 원본 CSV → 하루 한 줄 표 (load_nethealth)
  preprocessing : 전처리 규칙 (중복 제거, 착용 필터, 수면 병합, 42일 필터, 요일, 기기 교체 구간)
  baseline      : 기준선과 이탈 점수 (personal_z, group_z, cusum, bad_score ...)
  evaluation    : 방법 비교와 평가 (compare_alarms, 가짜 이탈 주입, 결과표, log_result)
"""
