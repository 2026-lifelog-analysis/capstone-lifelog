# 오늘 밤 수면 예측: 공통 실험 기반 (데이터 프로토콜 v2)

팀원들이 **같은 데이터, 같은 참가자 분할, 같은 평가 밤, 같은 채점 함수**로 개선안을 비교할 수 있게 만든 공통 코드다.
팀원별 개선 실험(최근 수면 이탈, 개인+집단 기준선 결합, 마지막 기록 이후 일수 등)은 여기에 들어 있지 않다. 각자 이 기반 위에서 추가한다.

코드: `src/forecast_dataset.py`, `src/forecast_split.py`, `src/forecast_eval.py`, `src/forecast_run.py`, `src/forecast_synthetic.py`
테스트: `tests/test_forecast_*.py`

## 1. 무엇을 예측하나

밤 D의 예측 기준 시각 **D 18:00**(`fd.FORECAST_HOUR`)에, 그 시각까지 확인할 수 있는 기록만으로 밤 D의 **주수면** 세 가지를 예측한다.
18시는 팀 실험을 위한 설정값이고 의료적 기준이 아니다.

| 목표 이름 | 정답 열 | 정의 | 단위 |
| --- | --- | --- | --- |
| `sleep` | `target_sleep_min` | 주수면에 속한 조각의 `minsasleep` **합** (종료−시작이 아님. 중간에 깬 시간 제외) | 분 |
| `bed` | `target_bedtime` | 주수면 첫 조각의 `timetobed` = **취침 시각** | D 12:00부터 분 |
| `wake` | `target_waketime` | 주수면 마지막 조각의 `timeoutofbed` = **기상 시각**(침대에서 나온 시각) | D 12:00부터 분 |

- 두 시각은 **같은 밤 D의 정오를 0으로 둔 연속된 분**이다. 23:50 → 710, 다음 날 00:10 → 730, 다음 날 07:00 → 1140, 다음 날 13:30 → 1530. 24시간 나머지 연산을 하지 않으므로 그대로 빼면 오차가 된다(23:50 vs 00:10 = 20분).
- 정답이 없는 밤은 채우거나 보간하지 않는다.

### 이상 탐지 하루 표와 다른 점

| | 이상 탐지 하루 표 (`dl.load_nethealth`) | 예측 표 (`fd.build_forecast_table`) |
| --- | --- | --- |
| 한 줄 | (사람, `dataDate`) | (사람, 밤 D) |
| 날짜 묶기 | `dataDate` | 조각 **시작**이 [D 12:00, D+1 12:00)이면 밤 D |
| 총 수면 | `sleep_min` = 그날 조각 **모두** 합 (낮잠 포함) | `target_sleep_min` = **주수면 덩어리**의 합 |
| 시각 | `bed_min`(정오 기준), `wake_min`(18시 기준), 각각 0~1439로 감음 | `target_bedtime`, `target_waketime` 둘 다 D 12:00 기준, 감지 않음 |
| 병합 | 같은 `dataDate` 안에서 90분 | 같은 밤 안에서 datetime으로 90분 (자정을 사이에 둔 조각도 묶임) |
| 겹치는 조각 | 그대로 합산 | 겹침 조각이 있는 밤은 제외 (2절) |
| 42일 필터 | 적용 (기준선 28일 + 평가 14일) | 기본 **적용 안 함** (`min_nights=None`) |

이상 탐지 쪽 함수(`load_nethealth`, `merge_sleep`)와 그 테스트는 바꾸지 않았다. `bed_min`과 `target_bedtime`은 기준이 달라 섞어 쓰지 않는다.

## 2. 원본 정의와 채택한 가정

### 2.1 확인된 원본 정의 (로컬 `data/nethealth/Codebook_Fitbit_Sleep(1-30).pdf`)

| 열 | 코드북 설명 |
| --- | --- |
| `timetobed` | "Time went to bed", 문자열 `HH:MM:SS` |
| `timeoutofbed` | "Time out of bed", 문자열 `HH:MM:SS` (빈칸 2줄) |
| `bedtimedur` | "Minutes in bed", 1~1370 |
| `minsasleep` | "Minutes asleep", 조각 하나 전체의 값 |
| `minstofallasleep`, `minsafterwakeup`, `minsawake` | 잠들기까지, 깬 뒤 누워 있던, 수면 중 깨어 있던 분 |
| `dataDate` | **"(unlabeled)"** — 뜻이 적혀 있지 않음 |

그래서 시작 시각은 **취침 시각**이라고 부르고 "실제로 잠든 시각"이라고 하지 않는다. `minsasleep`에는 조각 전체의 합계만 있으므로, 조각 안 어느 시간대에 잤는지는 알 수 없다.
코드북에는 시간대나 서머타임에 대한 설명도 없다.

### 2.2 분석을 통해 채택한 가정

**A1. `dataDate` = 조각이 끝난 날 (종료일).**
`end = dataDate + timeoutofbed`, `start = end − ((timeoutofbed − timetobed) mod 24h)`로 복원한다.

근거는 `python -m src.forecast_run --diagnose`(`fd.date_assumption_check`)로 다시 볼 수 있다.
- 분모: 시각 검사를 통과한 조각 333,706개. 전체 333,708줄 중 종료 시각이 빈 2줄을 뺀 수다.
- 분자: 같은 사람의 조각을 시작 순으로 놓았을 때, 앞 조각들 중 가장 늦은 종료보다 **먼저 시작하는** 조각 수
- 결과:

| 가정 | 겹치는 조각 / 분모 | 비율 |
| --- | --- | --- |
| 종료일 | 59 / 333,706 | 0.0177% |
| 시작일 | 27,028 / 333,706 | 8.0993% |

- 같은 사람의 수면 조각은 시간상 서로 겹치지 않는 것이 자연스럽다. 그래서 겹침이 훨씬 적은 종료일 해석을 골랐다.
- 시계 차이(`timeoutofbed − timetobed` mod 24h)가 `bedtimedur − 1`분과 1분 이내로 맞는 조각은 333,706개 전부다. 이것은 시작·종료 시각과 길이가 서로 맞는다는 뜻일 뿐, 날짜 해석을 결정해 주지는 않는다.
- 이것은 해석을 고른 **근거**이지 원본 정의를 확인한 것이 아니다.

**A2. 시각은 원본 표기 그대로(현지 시각으로 추정) 쓰고 시간대·서머타임 변환은 하지 않는다.**
확인되지 않은 한계다. 서머타임이 바뀌는 밤에는 시계 차이와 실제 경과 시간이 1시간 어긋날 수 있다. 하지만 `bedtimedur` 검사가 전부 통과하므로, 원본이 시계 차이 기준으로 기록됐을 가능성이 있다. 이것도 확인하지 못했다.

**A3. 서로 겹치는 조각이 있는 밤은 정답과 과거 입력에서 뺀다.** (2.3)

### 2.3 겹치는 조각 처리

| 구분 | 정의 | 처리 |
| --- | --- | --- |
| 완전 중복 | 같은 사람·같은 시작·같은 종료·같은 `minsasleep` | 파일에서 먼저 나온 줄 하나만 남김 (`dropped_episodes['duplicate']`) |
| 같은 구간, 다른 값 | 시작·종료는 같고 `minsasleep`이 다름 | 겹침 (`same_span`) |
| 포함 | 한 조각이 다른 조각 구간 안에 들어감 (시작이나 종료가 같아도) | 겹침 (`contained`) |
| 부분 겹침 | 일부 시간만 겹침 | 겹침 (`partial`) |
| 맞닿음 | 앞 조각 종료 = 다음 조각 시작 | **겹침 아님** (열린 구간 기준) |

- 겹친 시간에 몇 분 잤는지는 `minsasleep`만으로 알 수 없다. 그래서 겹친 분량을 빼거나 비례로 나누지 않는다.
- 겹침 조각이 하나라도 있는 밤은 통째로 `overlap_conflict`로 처리한다. 정답에서 빠지고, 다음 밤들의 `*_lag` 입력에서도 NaN이 된다. 겹친 조각만 빼서 다른 덩어리를 주수면으로 고르지 않는다.
- 실제 데이터: 겹치는 쌍 59개, 조각 118개, 46명. 포함 48쌍(그중 같은 시작 46쌍), 부분 겹침 11쌍, 같은 구간 다른 값 0쌍, 완전 중복 0줄, 맞닿음 0쌍. 모든 쌍이 같은 밤 안에 있다. 겹친 길이는 26~476.5분(중앙값 101분)이다. 59쌍 중 51쌍은 두 조각의 `minsasleep` 합이 두 조각이 차지한 전체 시간보다 길어서, 그대로 더하면 이중 계산이 확실하다.
- 결과: 정답에서 59밤(46명)이 빠지고, 과거 입력 칸 358개(줄 × k)가 NaN이 된다.

## 3. 밤과 주수면 만들기

1. **조각 시각 복원** (`sleep_episodes`): 가정 A1. 제외 사유와 건수는 `info['dropped_episodes']`에 남는다. 사유는 종료·시작 시각 없음/형식 오류, `bedtimedur` ≥ 24시간, 시계 차이 불일치, 완전 중복이다. 실제 파일에서는 종료 시각이 빈 2줄만 빠진다.
2. **겹침 표시** (`overlap_flags`, `overlap_pairs`): 2.3
3. **밤 배정** (`assign_nights`): 조각 시작이 [D 12:00, D+1 12:00)이면 밤 D다. 밤 23시 시작과 다음 날 02시 시작은 둘 다 밤 D다. 한 조각은 한 밤에만 들어가고, 종료가 다음 정오를 넘어도 자르지 않는다.
4. **덩어리** (`sleep_blocks`): 같은 밤 안에서 시작 순으로 놓고, 앞 조각들의 가장 늦은 종료에서 **90분**(`pp.SLEEP_GAP`) 넘게 지나 시작하면 새 덩어리를 만든다. 이상 탐지와 같은 90분 규칙이다. 다만 `merge_sleep`는 `dataDate` 안에서만 묶기 때문에 (21:30~23:30, 00:30~07:00)처럼 자정을 사이에 둔 조각을 갈라 놓는다. 그래서 예측은 datetime으로 따로 묶는다. 정오 경계를 넘어서는 묶지 않는다.
5. **주수면** (`night_summary`): 덩어리 중 `minsasleep` 합이 가장 큰 것을 고른다. 같으면 **먼저 시작한** 덩어리를 고른다.
6. **시작 전 예측 대상만** (`build_forecast_table`): 밤마다 사유를 하나 붙이며, 앞의 사유가 우선한다.
   - `no_sleep_record`: 그 밤에 시작한 조각이 없음
   - `overlap_conflict`: 겹침 밤
   - `started_before_forecast`: 주수면이 D 18:00 전에 시작함
   - `ok`: 나머지

실제 데이터(프로토콜 v2, 전체 기간): 후보는 622명의 455,978밤이다.

| 사유 | 밤 수 | 비율 |
| --- | --- | --- |
| `ok` | 294,243 | 64.5% |
| `overlap_conflict` | 59 | 0.01% |
| `started_before_forecast` | 3,127 | 0.7% |
| `no_sleep_record` | 158,549 | 34.8% |

예측 표는 294,243밤, 621명이다. **예측 표의 밤 294,243개 중 232,400개(79.0%)**는 주수면 취침이 다음 날 00:00 이후다(`target_bedtime ≥ 720`). 숫자는 `--check` 출력으로 다시 볼 수 있다.

### 포함·제외 기준

- 참가자: 예측 표에 유효한 밤이 하나라도 있는 사람(621명). 기록 수로 사람을 빼지 않는다. 수면 기록 밤이 42개 미만인 사람(1차 확인 51명)도 남긴다.
- 밤: 사람마다 첫 수면 밤부터 마지막 수면 밤까지. 초기 밤(과거 7밤이 비어 있음)도 빼지 않고, 입력을 NaN으로 둔다.
- 옵션: `min_nights=k`, `period='before'|'after'`. 기본값은 둘 다 끔이다. 옵션을 바꾸면 데이터 지문이 달라져 저장된 분할표와 맞지 않는다는 오류가 난다.

## 4. 입력 (예측 시점 이후 정보 차단)

**목표별 기본 입력은 고정**이다(`fd.base_features(target)`). 팀원 실험은 같은 목표끼리 이 목록에서 출발해 열을 더한다.

| 목표 | 기본 입력 |
| --- | --- |
| 총수면시간 `sleep` | `sleep_lag1..7` (과거 달력 7밤 총수면시간) + `dow` |
| 취침 시각 `bed` | `bedtime_lag1..7` (과거 달력 7밤 취침 시각) + `dow` |
| 기상 시각 `wake` | `waketime_lag1..7` (과거 달력 7밤 기상 시각) + `dow` |

- `*_lag{k}` = 밤 D−k 주수면의 값. 기준은 정답과 같고, 시각은 밤 D−k 정오부터 분이다. **달력 기준**이라 밤 D−3이 비면 `lag3`이 NaN이고 `lag4`는 여전히 밤 D−4다(최근 기록 7개를 당겨 오지 않음). ffill/bfill/보간은 하지 않는다.
- 밤 D−k에 시작한 조각 중 하나라도 D 18:00 이후에 끝나면(`known_at > forecast_at`), 그 밤의 세 값은 NaN이다(실제 138건). 겹침 밤도 NaN이다(358건).
- `dow` = 밤 D의 요일(0=월~6=일). 미리 알 수 있는 값이다.
- 밤 D의 정답, 그리고 D 당일의 하루 집계(걸음, 평균 심박, 착용률)는 넣지 않는다.

입력으로 쓸 수 없는 열(`fd.NON_FEATURE_COLS`, `target_*`)은 `fd.check_features`가 막는다. 해당 열은 `pid`, `date`, `forecast_at`, 세 정답, `target_bed_at`, `target_wake_at`, `target_n_pieces`, `target_n_blocks`다. `run_cv`는 검증 쪽 모델에 키, `forecast_at`, 입력 열만 넘기고 정답 열은 넘기지 않는다.

### 추가 특징 붙이기

```python
def my_feat(data):                     # data = ForecastData
    t = data.table
    lags = t[[f'sleep_lag{k}' for k in range(1, 8)]]
    return pd.DataFrame({'sleep_lag_mean': lags.mean(axis=1)}, index=t.index)

table = fd.add_features(data, my_feat)        # 새 표 (data.table은 그대로)
```
- 함수는 `data.table`과 같은 index의 DataFrame을 돌려준다. 이름이 겹치거나 `target_`으로 시작하면 오류가 난다.
- 각 줄의 `forecast_at` 이전에 확정된 기록만 써야 한다. 더 긴 과거가 필요하면 `data.nights`에서 `known_at <= forecast_at`이고 `overlap == False`인 밤만 쓴다.
- 검증·최종 평가 참가자의 과거 수면을 그 사람의 입력으로 쓰는 것은 허용된다. 그 사람을 **학습 데이터**에 넣는 것과는 다르다.
- 집단 통계, 결측 대체값, 스케일러처럼 데이터로 맞춰야 하는 값은 `add_features`에서 만들지 말고 모델의 `fit(train, ...)` 안에서 학습 참가자로만 맞춘다.
- `fd.feat_steps_lags`는 예시다(D−1 ~ D−7 하루 걸음 수).

## 5. 참가자 분할: 최종 평가 15% 고정 + 개발 85% 5-fold

`src/forecast_split.py`

1. 참가자 = 예측 표의 `pid`. 문자열로 바꾸고 중복을 없앤 뒤 사전순으로 정렬한다.
2. 최종 평가 인원 `n_final = floor(n × 0.15 + 0.5)` (반올림, .5는 올림)
3. `np.random.default_rng(0).permutation(n)` 순서의 앞 `n_final`명 = `role='final'`, `fold=0`
4. 나머지(`role='dev'`)는 섞은 순서에서 i번째 사람이 `fold = (i mod 5) + 1`이 된다. fold 1~5이고 크기 차이는 최대 1명이다.

매회 학습 = 개발 중 4 fold(전체의 약 **68%**), 검증 = 1 fold(약 **17%**), 최종 평가 약 **15%**는 쓰지 않는다. 70/15/15 분할도, 다섯 번 독립 랜덤 분할도 아니다.
실제 데이터 v2: 621명 = 최종 평가 93 + 개발 528(fold 106/106/106/105/105).

- 한 참가자의 모든 밤은 같은 역할에 속한다. 각 개발 참가자는 다섯 번 중 정확히 한 번 검증된다.
- 세 목표와 모든 팀원 실험이 **같은 분할표**를 쓴다. 목표마다 정답 없는 밤을 빼도 분할표를 다시 만들지 않는다.
- 모델은 목표·fold마다 새로 만들어 처음부터 학습한다. 저장된 모델을 불러 이어서 학습하지 않는다.
- 분할표는 **데이터 프로토콜별 폴더**에 저장된다: `results/forecast/<protocol>/split.csv`, `split_manifest.json`. 다시 실행하면 저장된 것을 읽고 참가자 목록, 설정, 프로토콜, 데이터 지문, 같은 시드로 다시 만든 결과와 같은지 검사한다. 다르면 `SplitMismatch` 오류가 난다. 조용히 다시 만들지 않으며, 의도한 변경일 때만 `--rebuild-split`을 쓴다.
- `split.csv`에는 원본 참가자 ID가 들어 있으므로 **커밋하지 않는다**(`results/`는 `.gitignore`). 팀원끼리는 `split_sha256`과 `data_fingerprint` 값을 비교한다.

### 프로토콜 버전

| 버전 | 내용 | 결과 위치 |
| --- | --- | --- |
| v1 | 1차 구현. 겹침 조각을 그대로 합산 | `results/forecast/split.csv`, `results/forecast/runs/` (점검용. 최종 수치로 쓰지 않음) |
| **v2** | 겹침 밤 제외, 완전 중복 = (시작, 종료, `minsasleep`) | `results/forecast/v2/` |

v1에서 v2로 바뀌어도 참가자 목록(621명)은 같았다. 그래서 분할표 내용과 `split_sha256`(`de80b681…`)은 같고, 데이터 지문만 바뀌었다(`d5a8fa88…` → `736ccdaa…`). v1 결과 파일은 지우거나 덮어쓰지 않았다.

## 6. 채점

`src/forecast_eval.py`

- 주 지표는 목표별 **MAE(분)**다. 세 목표를 평균 내 하나의 점수로 만들지 않는다.
- **평가 밤**(`eval_nights`): 검증 참가자 중 그 목표 정답이 있는 밤 전부. 모델과 입력에 관계없이 고정된다. 입력이 NaN이어도 빼지 않는다.
- 예측이 빠지거나 **NaN 또는 ±inf**이면 그 fold는 불완전(`complete=False`, `mae=NaN`)이 되고, 개수가 `note`에 남는다. 남은 밤만으로 점수를 내지 않는다. (pid, date) 중복이나 평가 밤 밖의 예측은 오류다.
- `eval_keys_sha256` = 목표·fold별 평가 (pid, date) 목록의 해시. `scores.csv`에 남는다.
- `fold MAE` = 그 fold 평가 밤 전체의 |정답−예측| 평균(밤 단위)
- `fold_mae_mean`, `fold_mae_std` = 다섯 fold MAE의 평균과 **표본 표준편차(ddof=1)**
- `oof_mae` = 다섯 fold 검증 예측을 합친 밤 단위 MAE. fold마다 밤 수가 달라 `fold_mae_mean`과 다를 수 있다.
- `oof_person_mae_mean` = 사람마다 MAE를 구한 뒤 평균 낸 값(보조)
- **status**: 목표·실험마다 아래 넷 중 하나다. 요약 숫자는 `complete`일 때만 채운다.
  - `complete`: fold 1~5가 정확히 한 번씩 있고 모두 완전함
  - `partial`: fold가 빠짐(부분 실행)
  - `incomplete`: 불완전한 fold가 있음
  - `duplicate`: fold가 중복됨
- 비교(`fe.compare(scores, base, new)`): `diff = 기본 MAE − 개선 MAE`로, **양수면 개선**이다.
  - fold별 차이와 개선된 fold 수 `n_improved`를 함께 본다.
  - 요약 줄(`fold='mean'`)은 두 실험이 모두 `complete`일 때만 숫자를 낸다. 아니면 `partial`/`incomplete`로 표시하고 평균과 `n_improved`를 비운다.
  - 같은 목표·fold의 `eval_keys_sha256`가 다르면 밤 수가 같아도 오류다.
  - 다섯 fold에서 모두 개선됐다고 해서 통계적으로 유의하다고 말하지 않는다.

## 7. 모델

기본 비교 모델은 아래 넷이다. 모델 구조 비교를 넓히려는 것이 아니며, 하이퍼파라미터 탐색은 하지 않았다.

| 실험 이름 (`fe.EXPERIMENTS`) | 모델 | 결측 | 요일 |
| --- | --- | --- | --- |
| `group_median` | 학습 참가자 정답 중앙값 | 입력 안 씀 | 안 씀 |
| `personal_mean7` | 그 사람 과거 7밤 평균. 7밤이 모두 NaN이면 학습 참가자 중앙값 | 있는 밤만 평균 | 안 씀 |
| `ridge_base` | Ridge(α=1) | 학습 참가자 중앙값으로 대체 + 결측 표시 열, 표준화 | 원-핫 |
| `catboost_base` | CatBoost 회귀 `fe.CatBoostModel` | **채우지 않음** (`nan_mode='Min'`) | 범주형 `cat_features=['dow']`, `one_hot_max_size=7`이라 CatBoost 안에서 원-핫 (정답 기반 통계 CTR 안 씀) |

CatBoost 고정 설정(`fe.CATBOOST_PARAMS`)은 `iterations=500`, `learning_rate=0.05`, `depth=6`, `loss_function='RMSE'`, `random_seed=0`, `thread_count=4`, `allow_writing_files=False`다.
- 검증 데이터는 `eval_set`이나 조기 종료에 쓰지 않는다. 경계값(quantization)과 트리는 학습 참가자로만 만든다.
- 테스트로 확인한 것은 세 가지다. 같은 시드로 두 번 돌리면 예측이 같다. 한 검증 줄의 예측은 다른 검증 줄에 영향을 받지 않는다. 입력 열은 `fd.base_features(target)`와 정확히 같다.
- 설정을 바꿔 돌릴 때는 `fe.Experiment('catboost_d4', lambda: fe.CatBoostModel(depth=4))`처럼 **다른 실험 이름**을 붙인다.
- 필요 패키지: `pip install -r requirements.txt` (`catboost>=1.2`)

## 8. 실행

```bash
pip install -r requirements.txt
python -m src.forecast_run --synthetic --check     # 가짜 데이터: 데이터·분할 생성만 확인
python -m src.forecast_run --synthetic             # 가짜 데이터: 전체 흐름 (숫자는 실제 성능 아님)
python -m src.forecast_run --diagnose              # 실제 데이터: dataDate 해석 진단 (2.2)
python -m src.forecast_run --check                 # 실제 데이터: 예측 표 요약 + v2 분할표 생성/검사
python -m src.forecast_run --name base             # 실제 데이터: group_median, personal_mean7, ridge_base
python -m src.forecast_run --experiments catboost_base --baseline catboost_base --name cb   # CatBoost 기본
python -m src.forecast_run --experiments catboost_base ridge_base --baseline catboost_base --targets sleep
pytest                                             # 전체 테스트 (실제 데이터 필요 없음)
```

결과는 `results/forecast/v2/`에, 가짜 데이터는 `results/forecast_synthetic/v2/`에 저장된다. 모두 커밋하지 않는다.

| 파일 | 내용 |
| --- | --- |
| `split.csv` | `pid, role, fold` (원본 ID 포함) |
| `split_manifest.json` | 시드, 반올림·fold 규칙, 인원, `split_sha256`, `pid_list_sha256`, `data_fingerprint`, `protocol` |
| `runs/<시각>-<이름>/scores.csv` | `target, experiment, fold, mae, person_mae_mean, n_people, n_nights, n_missing, n_nonfinite, complete, eval_keys_sha256, note` |
| `runs/.../summary.csv` | `target, experiment, status, n_folds, fold_mae_mean, fold_mae_std, oof_mae, oof_person_mae_mean, n_people, n_nights, complete, note` |
| `runs/.../compare.csv` | `target, fold, base_mae, new_mae, diff, n_improved, status, note, base, new` (`fold='mean'` 줄 = 요약) |
| `runs/.../predictions.csv` | `pid, date, target, experiment, fold, y_true, y_pred` |
| `runs/.../run_meta.json` | git 커밋·`src` 변경 여부, 패키지 버전, 시드, 프로토콜, 데이터 설정·지문, 분할 해시, 실험별 모델·설정·입력 열, `final_test_scored: false` |

### 팀원 실험 예시: CatBoost 기본에서 출발 (파일을 복사하지 않고 불러 쓰기)

```python
from src import forecast_dataset as fd, forecast_eval as fe, forecast_run as fr
import pandas as pd

data, split, man, out = fr.prepare()                 # 공통 데이터 + 저장된 v2 분할표 (검사 포함)

def my_feat(data):                                   # 예측 시점 이전 정보만
    lags = data.table[[f'sleep_lag{k}' for k in range(1, 8)]]
    return pd.DataFrame({'sleep_lag_mean': lags.mean(axis=1)}, index=data.table.index)

table = fd.add_features(data, my_feat)               # 줄은 그대로, 열만 추가
exps = [fe.EXPERIMENTS['catboost_base'],             # 목표별 고정 기본 입력
        fe.Experiment('catboost_mine', fe.CatBoostModel,
                      lambda t: fd.base_features(t) + ['sleep_lag_mean'])]
scores, summary, comp, run_dir = fr.run(data, split, man, out, experiments=exps, table=table,
                                        baseline='catboost_base', name='mine')
print(comp[comp['fold'] == 'mean'])                  # status == 'complete'인 줄만 비교 결론에 씀
```
새 모델은 `fit(train_df, y_col, features)`와 `predict(df)`가 있는 클래스이면 된다.

## 9. 최종 평가 (조합을 정한 뒤에만)

기본 실행(`forecast_run`)은 최종 평가 참가자에 대해 예측도 채점도 하지 않는다(`run_meta.json`의 `final_test_scored: false`). 개발 중에 모델을 고를 때도 최종 평가 점수를 쓰지 않는다. 분할 무결성 검사(`fs.check_split`)는 최종 평가 참가자에게도 돌아간다.

팀이 입력·모델 조합을 확정한 뒤의 흐름은 다음과 같다. 아직 코드로 만들지 않았다.
1. 개발 참가자 전체(약 85%)로 확정한 조합을 처음부터 새로 학습한다. fold 모델은 재사용하지 않는다.
2. 고정된 최종 평가 참가자(약 15%, `role='final'`)의 정답 있는 밤 전부를 같은 `fe.score`로 목표별로 한 번 채점한다.
3. 그 결과로 다시 조합을 바꾸지 않는다.

## 10. 남은 한계

- `dataDate` = 종료일은 코드북 정의가 아니라 겹침 비율로 채택한 가정이다(2.2).
- 시간대·서머타임은 확인하지 못했고 변환하지 않는다.
- 겹침 밤 제외는 근거가 없을 때의 보수적 규칙이다. 원본 기록 방식(기기 동기화 중복 등)을 확인하면 다시 정할 수 있다. 그때는 프로토콜 버전을 올린다.
- 최종 평가 재학습 코드는 아직 없다.
