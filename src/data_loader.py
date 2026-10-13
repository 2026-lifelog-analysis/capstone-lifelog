"""
data_loader.py — NetHealth 원본 CSV를 읽어 '하루 한 줄' 표로 만들어요.

    from src.data_loader import load_nethealth
    nh = load_nethealth()        # pid | date | steps | ... | dow (| term_tag)

데이터는 data/nethealth/ 아래에 두세요 (깃허브에는 올리지 않아요).
"""
import os
import warnings
from pathlib import Path

import pandas as pd

from . import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]     # 레포 맨 위 폴더 (노트북 어디서 열어도 같은 경로)

PATHS = {
    'nethealth': ROOT / 'data' / 'nethealth',
    'calendar': ROOT / 'data' / 'nethealth' / 'Calendar.csv',
    'results': ROOT / 'results' / 'results.csv',
}

DAILY_COLS = ['pid', 'date', 'steps', 'steps_per_hour', 'active_min', 'sleep_min', 'bed_min',
              'meanrate', 'complypercent', 'dow', 'wake_min', 'n_episodes']


def _find_csv(folder, prefix):
    """folder 안에서 prefix로 시작하는 CSV 파일 경로를 찾아요 (대소문자 무시)."""
    for f in sorted(os.listdir(folder)):
        if f.lower().startswith(prefix) and f.lower().endswith('.csv'):
            return os.path.join(folder, f)
    raise FileNotFoundError(f'{folder} 안에 {prefix}*.csv 파일이 없어요')


def read_raw(path=None):
    """
    NetHealth 원본 두 개를 그대로 읽어요.
    반환: (활동 표 A, 수면 조각 표 S)
      A : FitbitActivity*.csv  (하루 한 줄, 중복 있음)
      S : FitbitSleep*.csv     (수면 조각 한 줄. 하루에 여러 줄일 수 있음)
    """
    p = path or PATHS['nethealth']
    A = pd.read_csv(_find_csv(p, 'fitbitactivity'))
    S = pd.read_csv(_find_csv(p, 'fitbitsleep'))
    return A, S


def load_calendar(path=None):
    """
    학사 달력(Calendar.csv)을 '날짜 | term_tag' 표로 읽어요. 파일이 없으면 None.
    받아들이는 형식 (⚠️ 실제 파일을 받으면 형식 확인 필요)
      - 하루 한 줄 : date, term_tag
      - 기간 한 줄 : start, end, term_tag   (start~end 모든 날에 같은 태그)
    term_tag 예: 학기 / 방학 / 시험 / 휴일
    """
    p = Path(path or PATHS['calendar'])
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c.columns = [x.strip().lower() for x in c.columns]
    tag = 'term_tag' if 'term_tag' in c else ('tag' if 'tag' in c else None)
    if tag is None:
        raise ValueError(f'{p}에 term_tag 열이 없어요. 열: {list(c.columns)}')
    if 'date' in c:
        return pd.DataFrame({'date': pd.to_datetime(c['date']), 'term_tag': c[tag]})
    if {'start', 'end'} <= set(c.columns):
        rows = [pd.DataFrame({'date': pd.date_range(r['start'], r['end']), 'term_tag': r[tag]})
                for _, r in c.iterrows()]
        return pd.concat(rows, ignore_index=True).drop_duplicates('date', keep='last')
    raise ValueError(f'{p}에 date 또는 start/end 열이 없어요. 열: {list(c.columns)}')


def add_term_tag(d, calendar_path=None):
    """
    Calendar.csv가 있으면 term_tag(학기/방학/시험/휴일) 열을 붙여요.
    없으면 열을 붙이지 않고 경고만 출력해요.
    """
    cal = load_calendar(calendar_path)
    if cal is None:
        warnings.warn('Calendar.csv가 없어 term_tag 열 없이 진행해요 '
                      f'(넣을 위치: {calendar_path or PATHS["calendar"]})')
        return d
    return d.merge(cal, on='date', how='left')


def load_nethealth(path=None, wear=pp.WEAR_MIN, sleep_gap=pp.SLEEP_GAP, min_days=pp.MIN_DAYS,
                   period='all', cutoff=pp.DEVICE_CUTOFF, calendar_path=None):
    """
    NetHealth 하루 표를 만들어요. 전처리 규칙은 README '전처리 규칙' 표와 같아요.

    열
      pid, date        : 사람 ID(문자열), 날짜
      steps            : 걸음 수
      steps_per_hour   : 걸음 ÷ 착용 시간 (24 × 착용률/100)
      active_min       : 움직인 시간(분) = 가볍게 + 적당히 + 활발히
      sleep_min        : 총 수면(분), 그날 조각 모두 합산
      bed_min          : 잠든 시각, 정오부터 분 (23:30 → 690, 01:00 → 780)
      meanrate         : 하루 평균 심박
      complypercent    : 착용률(%)
      dow              : 요일 (0=월 ~ 6=일)
      wake_min         : 일어난 시각, 18시부터 분 (07:00 → 780)
      n_episodes       : 그날 수면 조각 수
      term_tag         : 학기/방학/시험/휴일 (Calendar.csv가 있을 때만)

    인자
      wear       : 착용률 기준(%). 미만인 날의 활동·심박은 빈칸 (기본 80)
      sleep_gap  : 수면 조각을 한 덩어리로 묶는 간격(분) (기본 90)
      min_days   : 기록 있는 날이 이보다 적은 사람 제외 (기본 42, None이면 제외 안 함)
      period     : 'all' | 'before' | 'after' — 기기 교체 전/후만 보기
      cutoff     : 기기 교체 날짜 (⚠️ 기본 '2018-01-01'은 임시값, 확인 필요)
    """
    A, S = read_raw(path)
    act = pp.build_activity(A, wear=wear)
    sl = pp.merge_sleep(S, gap=sleep_gap)
    d = act.merge(sl, on=['pid', 'date'], how='outer')
    d = pp.split_device_period(d, period=period, cutoff=cutoff)
    d = pp.filter_min_days(d, min_days=min_days)
    d = pp.add_dow(d)
    d = d[DAILY_COLS].sort_values(['pid', 'date']).reset_index(drop=True)
    return add_term_tag(d, calendar_path)
