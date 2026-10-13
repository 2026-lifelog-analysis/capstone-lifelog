"""
preprocessing.py — NetHealth 전처리 규칙 (2026-10-13 회의 확정)

원본 표를 '하루 한 줄' 표로 만들 때 쓰는 작은 함수들. 순서대로 쓰면 load_nethealth와 같은 결과가 나옴.
  1. drop_duplicate_days : 같은 사람·같은 날 중복 줄 → 빈칸 가장 적은 줄 하나만
  2. build_activity      : 활동 표 정리 + 착용 80% 미만인 날의 활동·심박은 빈칸
  3. merge_sleep         : 수면 조각 → 하루 총 수면, 잠든 시각
  4. split_study_phase   : 연구 기간(1~2학년 / 3~4학년)으로 나누기
  5. filter_min_days     : 기록 42일 미만인 사람 제외
  6. add_dow             : 요일 열 추가
  + dow_offsets          : 요일 보정용 (baseline.personal_z(dow_adjust=True)가 사용)

빈칸 채우기(보간, ffill, bfill)는 어디에서도 하지 않음. 미래 정보가 섞이기 때문임.
"""
import numpy as np
import pandas as pd

WEAR_MIN = 80                 # 착용률(complypercent) 기준. 이보다 낮은 날의 활동·심박은 빈칸
SLEEP_GAP = 90                # 수면 조각 사이가 이 분 이하면 같은 잠으로 묶음
MIN_DAYS = 42                 # 기준선 28일 + 평가 14일. 기록이 이보다 적은 사람은 제외
PHASE_CUTOFF = '2017-08-13'   # 연구 기간 경계 = Calendar 104주차(3학년 오리엔테이션 주) 시작

# 착용률이 낮으면 믿을 수 없는 열 (수면은 착용률과 따로 기록되므로 유지)
WEAR_COLS = ['steps', 'steps_per_hour', 'active_min', 'meanrate']


def hms_to_min(t, start_hour=12):
    """
    'HH:MM:SS' 시각을 'start_hour시부터 몇 분 지났는지'로 바꿈 (0 ~ 1439).
    잠든 시각은 정오(12시) 기준이라 밤을 넘겨도 숫자가 이어짐.
      23:30 → 690,  00:00 → 720,  01:00 → 780
    빈칸이면 빈칸(NaN)을 돌려줌.
    """
    if pd.isna(t):
        return np.nan
    h, m, s = str(t).split(':')
    return (int(h) * 60 + int(m) + float(s) / 60 - start_hour * 60) % 1440


def drop_duplicate_days(A, keys=('egoid', 'datadate')):
    """
    같은 사람·같은 날이 여러 줄이면 빈칸이 가장 적은 줄 하나만 남김.
    빈칸 수가 같으면 원본 파일에서 먼저 나온 줄을 남김 (실행할 때마다 같은 결과).
    """
    n_nan = A.isna().sum(axis=1)
    order = n_nan.sort_values(kind='stable').index
    return A.loc[order].drop_duplicates(list(keys), keep='first').sort_index()


def build_activity(A, wear=WEAR_MIN):
    """
    Fitbit 활동 원본 → pid | date | steps | steps_per_hour | active_min | meanrate | complypercent
      steps_per_hour : 걸음 ÷ 착용 시간 (착용 시간 = 24시간 × 착용률/100)
      active_min     : 가볍게 + 적당히 + 활발히 움직인 분의 합
    - 중복 날은 drop_duplicate_days로 먼저 정리
    - 착용률이 wear% 미만인 날은 활동·심박 값을 빈칸(NaN)으로. 0이 아니라 '모름'이라는 뜻
      (줄 자체는 남겨서 그날 수면 기록은 그대로 쓸 수 있게)
    """
    A = drop_duplicate_days(A)
    ok = A['complypercent'] >= wear
    act = pd.DataFrame({
        'pid': A['egoid'].astype(str),
        'date': pd.to_datetime(A['datadate']),
        'steps': A['steps'],
        'steps_per_hour': A['steps'] / (24 * A['complypercent'] / 100),
        'active_min': A['lightlyactiveminutes'] + A['fairlyactiveminutes'] + A['veryactiveminutes'],
        'meanrate': A['meanrate'],
        'complypercent': A['complypercent'],
    })
    act[WEAR_COLS] = act[WEAR_COLS].where(ok, axis=0)
    return act.reset_index(drop=True)


def merge_sleep(S, gap=SLEEP_GAP):
    """
    Fitbit 수면 조각 → 하루 한 줄: pid | date | sleep_min | bed_min | wake_min | n_episodes
      sleep_min  : 그날 수면 조각을 **모두** 더한 총 수면(분). 낮잠 포함
      bed_min    : 잠든 시각(정오부터 분). gap분 안에 이어지는 조각을 한 덩어리로 묶은 뒤,
                   가장 오래 잔 덩어리의 시작 시각
                   예) 23:30~02:00, 02:40~07:00 (40분 깸) → 한 덩어리 → bed_min = 23:30 = 690
      wake_min   : 그 덩어리가 끝난 시각 (18시부터 분. 예: 07:00 → 780)
      n_episodes : 그날 수면 조각 수
    S 열: egoid, dataDate, timetobed('HH:MM:SS'), bedtimedur(분), minsasleep(분)
    """
    S = S.copy()
    S['pid'] = S['egoid'].astype(str)
    S['_start'] = S['timetobed'].apply(hms_to_min)
    S['_end'] = S['_start'] + S['bedtimedur']
    S = S.sort_values(['pid', 'dataDate', '_start'], kind='stable')

    # 앞 조각이 끝난 뒤 gap분 넘게 지나서 시작하면 새 덩어리
    prev_end = S.groupby(['pid', 'dataDate'])['_end'].shift()
    S['_block'] = (prev_end.isna() | (S['_start'] - prev_end > gap)).cumsum()
    blocks = S.groupby('_block').agg(pid=('pid', 'first'), date=('dataDate', 'first'),
                                     bed_min=('_start', 'min'), _end=('_end', 'max'),
                                     block_asleep=('minsasleep', 'sum'))
    main = blocks.sort_values('block_asleep', kind='stable').groupby(['pid', 'date']).tail(1)
    main = main.assign(wake_min=(main['_end'] - 360) % 1440)[['pid', 'date', 'bed_min', 'wake_min']]

    tot = S.groupby(['pid', 'dataDate']).agg(sleep_min=('minsasleep', 'sum'),
                                             n_episodes=('minsasleep', 'size')).reset_index()
    tot = tot.rename(columns={'dataDate': 'date'})
    d = tot.merge(main, on=['pid', 'date'])
    d['date'] = pd.to_datetime(d['date'])
    return d[['pid', 'date', 'sleep_min', 'bed_min', 'wake_min', 'n_episodes']]


def split_study_phase(d, period='all', cutoff=PHASE_CUTOFF):
    """
    연구 기간(1~2학년 / 3~4학년)을 나눠서 분석할 때 씀. 데이터를 지우지는 않고 고르기만 함.
      period='all'    : 전부 (기본)
      period='before' : cutoff 날짜 이전만 = 1~2학년
      period='after'  : cutoff 날짜부터 = 3~4학년
    2017-08-13 = Calendar 104주차(3학년 오리엔테이션 주) 시작. 2학년 이후 참여 인원이 크게 줄어
    (NetHealth Table 1 Attrition Counts) 데이터 성격이 바뀌는 시점임.
    """
    if period in (None, 'all'):
        return d
    cut = pd.Timestamp(cutoff)
    if period == 'before':
        return d[d['date'] < cut]
    if period == 'after':
        return d[d['date'] >= cut]
    raise ValueError("period는 'all', 'before', 'after' 중 하나여야 합니다")


def filter_min_days(d, min_days=MIN_DAYS, cols=('steps', 'sleep_min')):
    """
    '기록 있는 날'(cols 중 하나라도 값이 있는 날)이 min_days일 미만인 사람을 뺌.
    기본 42일 = 기준선 28일 + 평가 14일. min_days가 None/0이면 아무도 안 뺌.
    """
    if not min_days:
        return d
    has = d[list(cols)].notna().any(axis=1)
    n = has.groupby(d['pid']).sum()
    return d[d['pid'].isin(n[n >= min_days].index)]


def add_dow(d):
    """요일 열 dow를 붙임 (0=월, 1=화, ..., 6=일)."""
    d = d.copy()
    d['dow'] = d['date'].dt.dayofweek
    return d


def dow_offsets(win, win_dow):
    """
    요일 보정값 계산 (기준선 창 안의 과거 데이터만 사용).
      win     : (날 수, 창 길이) 배열. 각 줄 = 그날의 기준선 창에 들어가는 과거 값들
      win_dow : win과 같은 모양의 요일(0~6) 배열
    반환: (날 수, 7) 배열. [i, k] = 창 i 안에서 '요일 k 평균 − 창 전체 평균'
          창 안에 그 요일 기록이 없으면 0 (보정 안 함)
    오늘 값과 창 값에서 이 보정값을 빼면 '요일 효과를 뺀 값'이 됨.
    (요일 평균을 그대로 빼는 것과 z점수는 같고, 숫자 크기만 원래 단위로 남음.)
    """
    ok = ~np.isnan(win)
    total = np.where(ok, win, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        overall = total.sum(axis=1) / ok.sum(axis=1)
        out = np.zeros((win.shape[0], 7))
        for k in range(7):
            m = ok & (win_dow == k)
            cnt = m.sum(axis=1)
            mean_k = np.where(m, win, 0.0).sum(axis=1) / cnt
            out[:, k] = np.where(cnt > 0, mean_k - overall, 0.0)
    return np.nan_to_num(out)
