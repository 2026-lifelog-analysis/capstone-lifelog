# capstone-lifelog

웨어러블·스마트홈 라이프로그로 개인 기준선 이탈을 감지하는 일상생활 붕괴 조기 탐지 (동국대 CSC종합설계)

- 팀: 김민준, 박고은, 조서연, 진승호
- 지도교수: 장혜령 교수님

## 프로젝트 개요
- **모듈 1. 기준선 이탈 탐지**: 개인의 과거 기록으로 "평소"를 만들고, 오늘이 평소에서 얼마나 벗어났는지 계산
- **모듈 2. 상태 판단**: 그 이탈이 실제 상태 악화와 관련 있는지 확인
- **현재 작업: 오늘 밤 수면 예측**: 과거 기록으로 오늘 밤 주수면의 총수면시간·취침 시각·기상 시각을 18시에 미리 예측. 지금은 팀원들이 같은 조건으로 비교하기 위한 **공통 실험 기반**(데이터·참가자 분할·채점·실행)까지 만든 상태 → [아래 설명](#오늘-밤-수면-예측-공통-실험-기반), 상세는 [docs/forecast.md](docs/forecast.md)

## 코드 흐름 문서
- [docs/repo_structure.md](docs/repo_structure.md): 자료가 경보가 되기까지의 일곱 단계와 각 단계의 코드 위치, PR별로 추가된 지표·점수, 같은 조건에서 다시 측정한 결과 정리. 저장소를 처음 보는 사람은 이 문서부터 읽으면 됩니다.
- [docs/forecast.md](docs/forecast.md): 오늘 밤 수면 예측의 밤·정답 정의, 원본 정의와 채택한 가정, 겹침 조각 처리, 참가자 분할, 채점, 실행 방법.

## 기술 스택
| 구분 | 사용 |
| --- | --- |
| 언어 | Python 3.12 |
| 분석 | pandas, numpy, scipy, scikit-learn, catboost(수면 예측) |
| 시각화 | matplotlib |
| 테스트 | pytest |
| 개발 환경 | VS Code + Jupyter Notebook |

## 폴더 구조
```
capstone-lifelog/
├── src/
│   ├── data_loader.py      # load_nethealth (+ Calendar.csv 있으면 study_week, term_tag, event_tag)
│   ├── preprocessing.py    # 중복 제거, 착용 필터, 수면 병합, 42일 필터, 요일 보정, 연구 기간 나누기
│   ├── baseline.py         # personal_z, group_z, rolling_mean, rolling_slope, cusum, bad_score
│   ├── evaluation.py       # compare_alarms, 가짜 이탈 주입, 주입 평가, 결과표, log_result
│   ├── forecast_dataset.py # [수면 예측] 밤·주수면·정답 3개·겹침 밤 제외·과거 7밤 입력, 특징 추가
│   ├── forecast_split.py   # [수면 예측] 최종 평가 15% + 개발 85% 참가자 5-fold
│   ├── forecast_eval.py    # [수면 예측] 공통 평가 밤, MAE 채점, 기본 모델(CatBoost 포함), run_cv, compare
│   ├── forecast_run.py     # [수면 예측] python -m src.forecast_run
│   └── forecast_synthetic.py # [수면 예측] 가짜 원본 (실제 데이터 없이 흐름 확인)
├── notebooks/
│   ├── 01_preprocessing.ipynb        # 전처리 과정과 하루 표 확인
│   └── 02_baseline_nethealth.ipynb   # 개인 vs 집단 기준선 비교 + 주입 평가
├── experiments/            # PR별 실험 코드 (common.py, minjun/, integrated/)
├── docs/                   # 코드 흐름 문서 repo_structure.md, 그림, 수면 예측 forecast.md
├── tests/                  # pytest
├── data/                   # 데이터 (깃허브에 안 올림)
└── results/                # 실험 결과 results.csv, 수면 예측 forecast/<protocol>/ (깃허브에 안 올림)
```

## 설치
```bash
git clone https://github.com/2026-lifelog-analysis/capstone-lifelog.git
cd capstone-lifelog
pip install -r requirements.txt
```

## 데이터 준비 (NetHealth)
[NetHealth](https://sites.nd.edu/nethealth/)에서 받은 파일을 아래처럼 넣습니다. 파일 이름 앞부분만 맞으면 됩니다.
```
data/nethealth/
├── FitbitActivity(1-30-20).csv   # 하루 활동 (필수)
├── FitbitSleep(1-30-20).csv      # 수면 조각 (필수)
└── Calendar.csv                  # 학사 달력 (선택) — 있으면 study_week, term_tag, event_tag 열이 붙음
```
- `Calendar.csv` 형식 (NetHealth 원본 그대로): 주 단위 한 줄, 0~194주 (2015-08-16 ~ 2019-05-11)
  - 열: `Study Week, Begin Date, End date, Event, Beginning event date, End event date`
  - 날짜는 `20150823` 같은 숫자. 사건 날짜가 비어 있으면(Christmas, Orientation 등) 그 주 Begin~End 전체를 사건 기간으로 봄
  - Event 이름 표기가 제각각이라(Class begins/Classes begin, Final exams/Final exam, Mid-term/Mid-Term/fall/Spring break 등) 대소문자·단수/복수를 무시하고 같은 사건으로 처리함 (`dl.CAL_EVENTS`)
  - 파일이 없으면 세 열 없이 경고만 내고 진행함
- `data/`는 `.gitignore`에 있어서 커밋되지 않습니다. **데이터는 절대 커밋하지 않습니다**

## 사용법
```python
from src import data_loader as dl, baseline as bl, evaluation as ev

nh = dl.load_nethealth()                                            # 하루 한 줄 표
nh['p'] = bl.bad_score(bl.personal_z(nh, 'steps'), 'steps')         # 개인 기준선 ('클수록 나쁨')
nh['g'] = bl.bad_score(bl.group_z(nh, 'steps'), 'steps')            # 집단 기준선
ev.compare_alarms(nh, 'p', 'g')                                     # 라벨 없이 비교

res = ev.evaluate_injection(nh, 'steps', -0.3)                      # 걸음 30% 감소 주입 평가
ev.results_table(res)
```
노트북 02는 결과를 남길 때만 `WHO`를 내 이름으로, `SAVE = True`로 바꿔 실행 (기본은 `results/results.csv`에 기록 안 함)

테스트: 레포 맨 위에서 `pytest`

## 하루 표 열 (이상 탐지 `load_nethealth` 전용)
수면 예측은 이 표를 쓰지 않음. 예측용 정답·시각은 [docs/forecast.md](docs/forecast.md) 참고

| 열 | 뜻 |
| --- | --- |
| `pid`, `date` | 사람 ID(egoid), 날짜 |
| `steps` | 걸음 수 |
| `steps_per_hour` | 걸음 ÷ (24 × 착용률/100) |
| `active_min` | 움직인 시간(분) = lightly + fairly + very |
| `sleep_min` | 총 수면(분), 그날 조각 모두 합산 |
| `bed_min` | 잠든 시각, **정오부터 분** (23:30 → 690, 01:00 → 780) |
| `meanrate` | 하루 평균 심박 |
| `complypercent` | 착용률(%) |
| `dow` | 요일 (0=월 ~ 6=일) |
| `wake_min` | 일어난 시각, 18시부터 분 (07:00 → 780) |
| `n_episodes` | 그날 수면 조각 수 |
| `study_week` | 연구 주차 0~194 (`Calendar.csv`가 있을 때만. 아래 두 열도 같음, 달력 범위 밖 날은 빈칸) |
| `term_tag` | `학기` = 개강일 ~ 그 학기 기말고사 끝난 날, 나머지는 `방학` |
| `event_tag` | `개강주`(개강일부터 7일) / `시험`(기말고사) / `휴일`(break, Thanksgiving, Easter, Christmas) / `없음`. 겹치면 시험 > 휴일 > 개강주 |

## 전처리 규칙 (2026-10-13 회의 확정, 이상 탐지 하루 표에 적용)
아래 규칙 3~7(하루 수면 합산, 잠든 시각 단위, 28일 기준선, 42일 필터)은 수면 예측에 그대로 쓰지 않음. 예측 쪽 규칙은 [docs/forecast.md](docs/forecast.md)

| # | 규칙 | 구현 |
| --- | --- | --- |
| 1 | 중복 제거: `(egoid, datadate)` 기준, 빈칸이 가장 적은 줄 하나만 (같으면 파일에서 먼저 나온 줄) | `pp.drop_duplicate_days` |
| 2 | 착용률 `complypercent` 80% 미만인 날은 활동·심박 값을 **빈칸(NaN)**으로 (0 아님). 수면은 유지 | `pp.build_activity(wear=80)` |
| 3 | 총 수면 `sleep_min` = 그날 수면 조각 **모두 합산** | `pp.merge_sleep` |
| 4 | 잠든 시각 `bed_min` = 90분 안에 이어지는 조각을 한 덩어리로 묶은 뒤, 가장 길게 잔 덩어리의 시작 시각 | `pp.merge_sleep(gap=90)`, `load_nethealth(sleep_gap=90)` |
| 5 | 잠든 시각 단위 = **정오부터 몇 분** (23:30 → 690, 01:00 → 780) | `pp.hms_to_min` |
| 6 | 개인 기준선 = 직전 **달력 28일** (기록 28줄이 아님). 빈 날은 빈칸으로 두고 계산, 창 안 기록 14일 이상일 때만 | `bl.personal_z(window=28, min_periods=14)` |
| 7 | 기록 있는 날(걸음 또는 수면)이 **42일 미만인 사람 제외** (기준선 28 + 평가 14) | `pp.filter_min_days(min_days=42)` |
| 8 | **빈칸 채우기(보간, ffill, bfill) 금지** (미래 정보 누출) | 어디에서도 안 함 |
| 9 | 이상치는 지우지 않고 **z를 ±8로 자름** | `personal_z(clip=8)`, `group_z(clip=8)`, `clip=None`이면 끔 |
| 10 | 방향 통일: "클수록 나쁨" (걸음↓, 활동↓, 수면↓, 잠든 시각↑, 심박↑) | `bl.bad_score`, `bl.BAD_DIRECTION` |
| 11 | 연구 기간 나누기: 지우지 않고 1~2학년(`before`) / 3~4학년(`after`)으로 나눠 돌릴 수 있게. 경계 `2017-08-13` = Calendar 104주차(3학년 오리엔테이션 주) 시작. 2학년 이후 참여 인원이 크게 줄어(NetHealth Table 1 Attrition Counts) 데이터 성격이 바뀌는 시점임. 기기 교체 날짜는 공식 문서에서 확인되지 않음 | `load_nethealth(period='all'\|'before'\|'after', cutoff=...)`, `pp.split_study_phase` |
| 12 | 경보 기준 = 오경보 사람당 30일 1회 = **3.3%** (`ALARM_RATE = 1/30`). 모든 사람에게 **같은 기준값 하나** | `ev.ALARM_RATE` |
| + | 요일 보정(선택): 기준선 창 안의 **과거** 값으로만 요일 평균을 구해서 뺌 | `personal_z(dow_adjust=True)` |

## 함수 목록
| 파일 | 함수 | 하는 일 |
| --- | --- | --- |
| `data_loader` | `load_nethealth` | 원본 → 하루 표 (위 규칙 전부 적용) |
| | `read_raw`, `load_calendar`, `add_term_tag` | 원본 읽기, 학사 달력 → 하루 표(study_week, term_tag, event_tag), 하루 표에 붙이기 |
| `preprocessing` | `hms_to_min`, `drop_duplicate_days`, `build_activity`, `merge_sleep` | 시각 변환, 중복 제거, 착용 필터, 수면 병합 |
| | `split_study_phase`, `filter_min_days`, `add_dow`, `dow_offsets` | 연구 기간 나누기, 42일 필터, 요일, 요일 보정값 |
| `baseline` | `personal_z` | 개인 기준선 z (달력 28일, ±8, 요일 보정 옵션) |
| | `group_z` | 집단 기준선 z (같은 날 30명 이상) |
| | `bad_score` | '클수록 나쁨'으로 방향 통일 |
| | `cusum`, `cusum_threshold` | 나빠지는 쪽 누적합 (허용치 `k`, 경보선 `h`), 3.3%가 되는 `h` 찾기 |
| | `rolling_mean`, `rolling_slope` | 달력 k일 평균, 기울기 |
| `evaluation` | `compare_alarms` | 라벨 없이 두 방법 비교 (같은 날 비율, 0번인 사람, 상위 10% 몫, 사람별 오경보율 분포) |
| | `inject_drift` | 가짜 이탈 주입 (크기, 길이, 갑자기/서서히, 시드 고정) |
| | `method_scores` | 비교 방법 5개 점수 (무작위, 단일 z, 개인, 개인+CUSUM, 집단) |
| | `evaluate_injection` | 주입 평가: 대조군, 공통 기준값, 사람별 탐지율·오경보율·차이·탐지 지연, PR-AUC(보조) |
| | `results_table`, `wilcoxon_table`, `plot_person_box` | 결과표, 방법 간 Wilcoxon 검정, 사람별 박스플롯 |
| | `log_result`, `log_table` | `results/results.csv`에 결과 기록 |
| `forecast_dataset` | `build_forecast_table`, `load_forecast_data` | 수면 원본 → 밤 단위 예측 표 (`ForecastData`) |
| | `sleep_episodes`, `overlap_flags`, `assign_nights`, `sleep_blocks`, `night_summary` | 조각 시각 복원, 겹침 표시, 밤 배정, 90분 병합, 주수면 |
| | `base_features`, `check_features`, `add_features`, `date_assumption_check` | 목표별 고정 기본 입력, 금지 열 검사, 특징 추가, dataDate 해석 진단 |
| `forecast_split` | `make_split`, `load_or_make_split`, `fold_pids`, `check_split` | 참가자 분할, 저장·검사, fold별 학습/검증 참가자 |
| `forecast_eval` | `eval_nights`, `score` | 목표별 공통 평가 밤, MAE 채점 (누락·NaN·inf는 불완전) |
| | `run_cv`, `summarize`, `compare` | 5-fold 실행, fold 평균·표준편차·OOF, 기본−개선 차이 (완전한 5-fold일 때만) |
| | `EXPERIMENTS`, `Experiment`, `CatBoostModel`, `RidgeModel`, `PersonalMean`, `GroupMedian` | 기본 실험, 실험 정의, 모델 어댑터 |
| `forecast_run` | `prepare`, `run`, `main` | 공통 데이터+분할 준비, 실행·저장, 명령줄 |

## 평가 방법 (모듈 1, 가짜 이탈 주입)
- 사람마다 정상 구간(직전 28일 기록 14일 이상, 구간 안 기록 7일 이상)을 무작위로 골라 **14일** 동안 지표를 바꿈 (예: 걸음 −30%, 잠든 시각 +120분)
- 구간 14일 안에 하루라도 울리면 **탐지**. 같은 구간을 주입 없이 한 번 더 돌린 **대조군**에서 울리면 **오경보**
- 기준값: 대조군 전체에서 하루 오경보율이 3.3%가 되도록 방법마다 **하나**
- 지표: 탐지율(주), 오경보율, 차이(Youden's J), **탐지 지연일수 중앙값**, PR-AUC(보조)
- **사람마다** 계산한 뒤 분포로 보고 (중앙값 + 박스플롯 + 방법 간 Wilcoxon). 정확도·F1·AUROC만 단독으로 쓰지 않음
- 오경보율은 두 가지: **하루 오경보율**은 기준값 정의상 3.3%, 결과표의 **오경보율**은 14일 구간 안에서 하루라도 울린 비율이라 더 크게 나옴 (탐지율과 같은 단위로 비교하기 위함)

## 오늘 밤 수면 예측 (공통 실험 기반)
밤 D의 **18:00**(`fd.FORECAST_HOUR`, 팀 실험 설정값이며 의료 기준 아님)에 그때까지 확정된 기록만으로 밤 D의 **주수면**을 예측함. 상세 규칙·근거·한계는 [docs/forecast.md](docs/forecast.md)

| 목표 | 정답 열 | 정의 (MAE 단위: 분) | 기본 입력 (목표별 고정) |
| --- | --- | --- | --- |
| 총수면시간 | `target_sleep_min` | 주수면 조각의 `minsasleep` **합** (종료−시작 아님) | 과거 달력 7밤 총수면시간 `sleep_lag1..7` + 요일 `dow` |
| 취침 시각 | `target_bedtime` | 주수면 시작 `timetobed`("Time went to bed", 실제 잠든 시각 아님), **D 12:00부터 분** | 과거 달력 7밤 취침 시각 `bedtime_lag1..7` + `dow` |
| 기상 시각 | `target_waketime` | 주수면 종료 `timeoutofbed`("Time out of bed"), **D 12:00부터 분** | 과거 달력 7밤 기상 시각 `waketime_lag1..7` + `dow` |

- **밤 D** = 시작이 [D 12:00, D+1 12:00)인 수면 조각. 23시 시작과 다음 날 02시 시작이 모두 밤 D. 종료는 자르지 않음
- **주수면** = 같은 밤에서 90분 이하로 이어진 조각을 묶은 덩어리 중 `minsasleep` 합이 가장 큰 것 (같으면 먼저 시작)
- 시각은 같은 밤 기준의 연속된 분 (23:50 → 710, 다음 날 00:10 → 730 → 오차 20분). 24시간 나머지 연산 안 함. 이상 탐지의 `bed_min`(정오 기준)·`wake_min`(18시 기준)과 섞지 않음
- **이상 탐지 하루 표와 다름**: 하루 표 `sleep_min`은 그날 조각 **모두** 합(낮잠 포함), 예측 정답은 **주수면 하나**. 예측에는 42일 필터를 적용하지 않음 (기록 적은 사람·초기 밤도 입력 NaN으로 남김)
- **채택한 가정**: 원본 `dataDate`는 코드북에 뜻이 없음("unlabeled"). 같은 사람 조각끼리의 겹침 비율(종료일 가정 0.018% vs 시작일 가정 8.1%)을 근거로 **종료일**로 봄. 시간대·서머타임 변환은 안 함(확인 못 함). `python -m src.forecast_run --diagnose`로 다시 확인 가능
- **겹치는 조각**: 완전 중복(시작·종료·`minsasleep` 같음)은 하나만 남김. 시간이 겹치는 서로 다른 기록(포함·부분 겹침)이 있는 밤은 겹친 시간에 몇 분 잤는지 알 수 없어 **정답·과거 입력에서 통째로 제외** (실제 59밤/46명). 맞닿기만 한 조각은 겹침 아님
- **수면시간 결측·무효**: 조각의 `minsasleep`이 빈칸·숫자 아님·±inf·음수·그 조각 `bedtimedur` 초과이면, 그 조각이 있는 밤을 **정답·과거 입력에서 통째로 제외**함. 0으로 채우거나 일부 조각만 더하거나 다른 덩어리를 주수면으로 고르지 않음. 0분은 정상 값 (실제 데이터는 해당 0건)
- 주수면이 18시 전에 시작한 밤은 '시작 전 예측' 대상에서 빼고 개수를 출력함. 정답 없는 밤은 채우지 않음
- 빠진 밤은 NaN 유지(최근 기록 7개를 당겨 오지 않음), 당일 하루 집계(걸음·심박·착용률)는 안 넣음
- 세 목표는 **따로** 채점함 (평균 내서 하나로 합치지 않음)

### 참가자 분할: 최종 평가 15% 고정 + 개발 85% 5-fold
- 참가자 ID를 문자열로 바꾸고 정렬한 뒤 시드 0으로 섞음. `floor(n × 0.15 + 0.5)`명을 **최종 평가**로 한 번 떼어 놓고, 나머지를 섞은 순서대로 fold 1~5에 돌아가며 배정
- 매회 학습 4 fold / 검증 1 fold → 전체 기준 **약 68% / 17% / 15%** (70/15/15 아님, 독립 랜덤 분할 다섯 번 아님)
- 한 사람의 모든 밤은 같은 역할. 개발 참가자는 다섯 번 중 정확히 한 번 검증됨. 세 목표·모든 팀원 실험이 같은 분할표를 씀
- 분할표 `results/forecast/v3/split.csv`(원본 ID 포함, **커밋 안 함**). 데이터 규칙 버전(`fd.PROTOCOL`)별 폴더. 다시 돌리면 저장된 것을 읽고 검사함 (다르면 오류, 조용히 다시 만들지 않음). 새 버전 분할표를 만들 때 이전 버전과 최종 평가 참가자가 다르면 저장하지 않고 오류
- 팀원끼리 같은지는 `split_manifest.json`의 `split_sha256`, `data_fingerprint`로 비교 (`--check`가 출력)

### 채점
- 목표별 **평가 밤**은 검증 참가자 중 정답 있는 밤 전부. 모델·입력과 관계없이 고정. fold별 평가 밤 목록의 해시(`eval_keys_sha256`)를 남기고, 비교할 때 이 해시가 다르면 오류
- 예측이 빠지거나 NaN·inf면 그 fold는 **불완전**(MAE 칸 비움). 빠진 밤을 조용히 빼지 않음
- 출력: fold별 MAE, 다섯 fold MAE의 평균·표본 표준편차(ddof=1), 다섯 fold 예측을 합친 **OOF MAE**(fold 평균과 다를 수 있음), 사람별 MAE 평균(보조), 참가자 수·밤 수
- 비교 차이 = **기본 MAE − 개선 MAE** (양수 = 개선), fold별 차이와 개선된 fold 수. 두 실험 모두 fold 1~5가 빠짐없이 완전할 때만(`status=complete`) 평균·개선 fold 수를 냄. 부분 실행은 `partial`, 불완전은 `incomplete`로 표시. 5개 fold에서 개선됐다고 유의하다고 하지 않음

### 실행
```bash
pip install -r requirements.txt                    # catboost 포함
python -m src.forecast_run --synthetic             # 실제 데이터 없이 전체 흐름 확인 (숫자는 실제 성능 아님)
python -m src.forecast_run --diagnose              # 실제 데이터: dataDate 해석 진단
python -m src.forecast_run --check                 # 실제 데이터: 예측 표 요약 + 분할표 생성·검사만
python -m src.forecast_run --name base             # 기본 실험 3개 (group_median, personal_mean7, ridge_base)
python -m src.forecast_run --experiments catboost_base --baseline catboost_base --name cb   # CatBoost 기본 실험
pytest                                             # 전체 테스트 (실제 데이터 필요 없음)
```
기본 실행은 개발 참가자 5-fold만 돌리고 최종 평가는 하지 않음. 결과는 `results/forecast/v3/runs/<시각>-<이름>/`(가짜 데이터면 `results/forecast_synthetic/v3/`)에 저장: `scores.csv`, `summary.csv`, `compare.csv`, `predictions.csv`(pid·밤·목표·fold별 예측), `run_meta.json`(커밋, 시드, 설정, 데이터 지문, 분할 해시). `results/`는 `.gitignore`라 커밋 안 됨

CatBoost 기본(`catboost_base`): 결측은 채우지 않음(`nan_mode='Min'`), `dow`는 범주형(CatBoost 안에서 원-핫), `random_seed=0`, 목표·fold마다 새 모델, 검증 데이터는 조기 종료 등에 쓰지 않음. 설정은 `fe.CATBOOST_PARAMS`(튜닝 안 함)

내 실험은 실행 파일을 복사하지 말고 불러서 씀. 같은 목표의 기본 입력 `fd.base_features(t)`에서 출발 (전체 예시는 [docs/forecast.md](docs/forecast.md) 8절):
```python
from src import forecast_dataset as fd, forecast_eval as fe, forecast_run as fr
data, split, man, out = fr.prepare()                       # 공통 데이터 + 저장된 분할표
table = fd.add_features(data, my_feat)                     # 내 특징 (줄은 그대로)
exps = [fe.EXPERIMENTS['catboost_base'],
        fe.Experiment('catboost_mine', fe.CatBoostModel, lambda t: fd.base_features(t) + ['my_col'])]
fr.run(data, split, man, out, experiments=exps, table=table, baseline='catboost_base', name='mine')
```
공통 열 보호: `fd.add_features`의 특징 함수는 표 복사본을 받고, 받은 표를 직접 고치면 오류(원본은 그대로). `fr.run(..., table=...)`은 학습 전에 원래 열(키·예측 시각·정답·정답 보조 열·기본 입력)이 모두 같은 순서·같은 값인지 검사하고, 바뀌거나 빠지거나 행·열이 어긋나면 오류를 내며 결과를 저장하지 않음 (같은 위치 NaN은 같은 결측, 추가 열만 허용). `data.table`이 분할표의 데이터 지문과 달라도 오류

### 팀원 작업 순서
1. 진행 중인 작업 보존: `git status`로 확인하고, 변경이 있으면 내 브랜치에 커밋하거나 `git stash`
2. 팀에서 정한 공통 브랜치를 최신으로: `git switch <공통 브랜치>` → `git pull --ff-only`
3. 공통 데이터·분할 생성·확인: `pip install -r requirements.txt` → `python -m src.forecast_run --check` → 출력된 `split_sha256`·`data_fingerprint`를 팀원과 비교
4. 내 실험 브랜치에서 개선안 실행: `git switch -c feat/<이름>-<작업>` → 위 예시처럼 `fr.run(...)`
5. 기본 방식과 개선 방식을 **같은 다섯 fold**에서 비교: `compare.csv`의 `fold='mean'` 줄 중 `status=complete`인 `diff`와 `n_improved`

### 최종 평가
기본 실행은 최종 평가 참가자에 대해 예측·채점을 하지 않고(`run_meta.json`의 `final_test_scored: false`), 개발 중 모델 선택에도 쓰지 않음. 팀이 조합을 확정한 뒤 개발 참가자 전체(약 85%)로 새로 학습하고, 고정된 최종 평가(약 15%)로 한 번 채점함 (이 단계 코드는 아직 없음, [docs/forecast.md](docs/forecast.md) 9절)

## 브랜치 전략
| 브랜치 | 용도 |
| --- | --- |
| `main` | 항상 돌아가는 상태. PR로만 합치기 |
| `feat/<이름>-<작업>` | 새 기능·분석 |
| `fix/<이름>-<작업>` | 버그 수정 |
| `docs/<작업>` | 문서 수정 |

## 커밋 규칙
`<타입>: <내용>` (예: `feat: 수면 기준선 함수 추가`)

| 타입 | 언제 |
| --- | --- |
| `feat` | 새 기능·분석 |
| `fix` | 버그 수정 |
| `exp` | 실험 결과 추가 |
| `refactor` | 코드 정리 |
| `docs` | 문서 |
| `chore` | 설정 |

## PR 규칙
- 본문에 무엇을, 왜, 결과 적기
- 1명 이상 리뷰 후 merge
- 노트북은 충돌이 잦으니 각자 노트북 파일을 따로 만들어 작업
- 노트북은 **출력을 지우고** 커밋 (출력에 데이터 표가 들어가서)
- src/를 고치는 PR은 팀원 모두 확인 후 merge함

## 규칙
- `data/`, `results/`, `_archive/`는 커밋하지 않기 (`.gitignore`에 있음). 수면 예측 분할표·예측·실행 결과(`results/forecast*/`)도 여기에 포함되며, 원본 참가자 ID가 들어 있음
- 랜덤이 들어가는 곳은 시드 고정 (`ev.SEED = 0`, 수면 예측 분할 `fs.SEED = 0`, 모델 `fe.SEED = 0`)
- `src/`를 고치면 `pytest` 돌리기
