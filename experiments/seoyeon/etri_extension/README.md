# ETRI 확장실험 (조서연)

ETRI 라이프로그 2024(10명, 라벨 450일)에서 제5회 ETRI 휴먼이해 AI 논문경진대회 수상작 두 편
(State, not Trait / Structure over Scale, ICTC 2026 C6 워크숍)이 남긴 빈칸 세 가지를 실험한 코드다.

| 방향 | 질문 | 실행 |
|---|---|---|
| 기준 | 전체 비율과 개인 비율 중 무엇이 나은가 | `run_baseline.py` |
| 1 | 개인 기록이 짧을 때 무엇을 믿나 (전체 비율·개인 비율·섞는 방식) | `run_coldstart.py` |
| 2 | 센서 결측(기록률)이 신호인가 | `run_coverage.py`, `run_sensor.py` |
| 3 | 화면 사용 리듬 지표가 예측을 더하나 | `run_rhythm.py` |

## 데이터 위치

데이터는 저장소에 올리지 않는다. 저장소 맨 위의 `data/etri2024/` 아래에 다음이 있어야 한다.

```
data/etri2024/
├── ch2025_metrics_train.csv
└── data_items/ch2025_*.parquet   # 센서 12종
```

## 실행

이 폴더를 작업 위치로 두고 실행한다(모듈을 폴더 안에서 바로 불러온다).

```sh
cd experiments/seoyeon/etri_extension
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python run_baseline.py
.venv/bin/python run_coldstart.py
.venv/bin/python run_coverage.py
.venv/bin/python run_sensor.py
.venv/bin/python run_rhythm.py
```

결과는 이 폴더의 `results/`에 CSV로 쌓인다. `results/`는 저장소에 올리지 않는다
(`split.csv`, `coverage.csv`, `rhythm_features.csv`에는 날짜별 라벨이 들어 있다).

## 공통 절차

- 라벨: Q1·Q2·Q3·S1·S2·S3. S1은 원값 0·1·2 중 1 이상을 양성으로 묶는다.
- 분할: 사람마다 라벨 있는 날의 앞 절반이 학습, 학습 마지막 날 뒤 달력 7일은 버리고 그 뒤가 평가.
- 채점: log loss가 주 지표, AUROC가 보조 지표. 차이는 사람 단위 부트스트랩 2,000회의 95% 백분위 구간.
- 판정: 차이의 구간이 0을 포함하지 않을 때만 이김 또는 짐.

## 파일

| 파일 | 내용 |
|---|---|
| `labels.py`, `split.py`, `scoring.py`, `compare.py` | 라벨 읽기, 분할, 채점, 두 방법의 라벨별 비교 |
| `baselines.py` | 전체 비율, 고정 개인 비율 |
| `shrinkage.py`, `coldstart.py` | 섞는 방식(베타-이항 수축), 기록 일수를 제한한 예측 |
| `record_coverage.py`, `association.py` | 센서 하루 기록률, 사람 내 상관 |
| `offset_logit.py`, `sensor_model.py` | 기준값이 있는 ℓ2 로지스틱, 섞은 비율 위에 피처를 더하는 예측 |
| `rhythm.py` | 화면 사용의 첫·마지막 시각과 최근 7일 흔들림 |
| `tests/` | 위 모듈의 테스트 77개. `run_*.py`는 테스트가 없고 실제 실행으로만 확인했다 |

## 근거를 찾지 못한 결정

논문이나 팀 코드에서 근거를 찾은 결정은 해당 모듈의 맨 위 설명에 출처를 적었다.
아래는 근거를 찾지 못한 채 쓴 것이다.

- 분할 경계: 앞 절반, 7일 간격, 달력 기준 (`split.py`). 절반과 7일의 출처는 승호님 설계 문서뿐이다.
- 전체 비율에서 평가 대상자를 빼고, 다른 사람도 학습 구간만 쓰는 것 (`coldstart.py`)
- 학습 구간의 마지막 k일을 공개하고 평가 날을 고정하는 것 (`coldstart.py`)
- 난수 시드 0
- ℓ2 후보값 0.1 ~ 1,000,000, 손실이 같으면 큰 벌점을 고르는 규칙 (`run_sensor.py`, `sensor_model.py`)
- 기록률을 1분 간격 센서 7종으로 한정하고 7개를 전부 피처로 쓰는 것 (`run_coverage.py`)
- 기록이 첫 1시간·마지막 1시간에 없으면 시각을 비우는 규칙, 화면 센서 하나만 쓰는 것 (`rhythm.py`)
