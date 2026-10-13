"""
data_loader.py — NetHealth 원본 CSV를 읽어 '하루 한 줄' 표로 만듦.

    from src.data_loader import load_nethealth
    nh = load_nethealth()        # pid | date | steps | ... | dow (| study_week | term_tag | event_tag)

데이터는 data/nethealth/ 아래에 둠 (깃허브에는 올리지 않음).
"""
import os
import re
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
    """folder 안에서 prefix로 시작하는 CSV 파일 경로를 찾음 (대소문자 무시)."""
    for f in sorted(os.listdir(folder)):
        if f.lower().startswith(prefix) and f.lower().endswith('.csv'):
            return os.path.join(folder, f)
    raise FileNotFoundError(f'{folder} 안에 {prefix}*.csv 파일이 없습니다')


def read_raw(path=None):
    """
    NetHealth 원본 두 개를 그대로 읽음.
    반환: (활동 표 A, 수면 조각 표 S)
      A : FitbitActivity*.csv  (하루 한 줄, 중복 있음)
      S : FitbitSleep*.csv     (수면 조각 한 줄. 하루에 여러 줄일 수 있음)
    """
    p = path or PATHS['nethealth']
    A = pd.read_csv(_find_csv(p, 'fitbitactivity'))
    S = pd.read_csv(_find_csv(p, 'fitbitsleep'))
    return A, S


CAL_EVENTS = [
    # (정규식, 사건 종류). 이름을 소문자로 바꾸고 '-'를 공백으로 바꾼 뒤 비교 (대소문자·단수/복수 무시)
    (r'^class(es)? begins?$', 'class_begin'),
    (r'^final exams?$', 'final'),
    (r'^(mid ?term|fall|spring) breaks?$', 'holiday'),
    (r'^(thanksgiving|easter|christmas)$', 'holiday'),
    (r'^orientation weeks?$', 'orientation'),
]
EVENT_PRIORITY = ['시험', '휴일', '개강주']   # 겹치면 앞쪽이 우선


def _event_kind(name):
    """Calendar.csv의 Event 이름 → 사건 종류. 모르는 이름이면 None."""
    s = re.sub(r'\s+', ' ', str(name).lower().replace('-', ' ')).strip()
    for pat, kind in CAL_EVENTS:
        if re.match(pat, s):
            return kind
    return None


def _ymd(x):
    """20150823 같은 숫자 → 날짜. 빈칸이면 NaT."""
    return pd.NaT if pd.isna(x) else pd.to_datetime(str(int(x)), format='%Y%m%d')


def load_calendar(path=None):
    """
    학사 달력(Calendar.csv)을 하루 한 줄 표로 읽음. 파일이 없으면 None.
    반환: date | study_week | term_tag | event_tag

    입력 형식 (NetHealth 원본, 주 단위 한 줄, 0~194주)
      Study Week, Begin Date, End date, Event, Beginning event date, End event date
      날짜는 20150823 같은 숫자
    규칙
      - 사건 기간 = Beginning~End event date. 비어 있으면 그 주 Begin~End date 전체
        (사건 날짜가 그 주 범위를 벗어나도 사건 날짜를 우선)
      - Class begins는 시작일만 있음 → 개강주 = 개강일부터 7일
      - term_tag  : '학기' = 개강일 ~ 그 뒤 첫 기말고사 끝난 날, 나머지는 '방학'
      - event_tag : '시험'(Final exams), '휴일'(break, Thanksgiving, Easter, Christmas),
                    '개강주', 그 외 '없음'. 겹치면 시험 > 휴일 > 개강주
      - Event 이름의 대소문자·단수/복수 차이는 같은 사건으로 처리 (CAL_EVENTS)
      - Orientation week는 사건으로 표시하지 않음 ('없음')
    달력 범위(2015-08-16 ~ 2019-05-11) 밖의 날은 load_nethealth에서 세 열 모두 빈칸이 됨.
    """
    p = Path(path or PATHS['calendar'])
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c.columns = [x.strip().lower() for x in c.columns]
    need = {'study week', 'begin date', 'end date', 'event', 'beginning event date', 'end event date'}
    if not need <= set(c.columns):
        raise ValueError(f'{p}에 필요한 열이 없습니다. 필요: {sorted(need)}, 있음: {list(c.columns)}')

    # 하루 한 줄 뼈대 + 주차
    days = pd.concat([pd.DataFrame({'date': pd.date_range(_ymd(r['begin date']), _ymd(r['end date'])),
                                    'study_week': int(r['study week'])})
                      for _, r in c.iterrows()], ignore_index=True)
    days = days.drop_duplicates('date').sort_values('date').reset_index(drop=True)

    # 사건 기간 표
    ev = []
    for _, r in c[c['event'].notna()].iterrows():
        kind = _event_kind(r['event'])
        if kind is None:
            warnings.warn(f"Calendar.csv의 알 수 없는 사건 이름은 건너뜁니다: {r['event']!r}")
            continue
        start, end = _ymd(r['beginning event date']), _ymd(r['end event date'])
        if pd.isna(start):                       # 사건 날짜가 없으면 그 주 전체
            start, end = _ymd(r['begin date']), _ymd(r['end date'])
        elif kind == 'class_begin' or pd.isna(end):
            end = start
        ev.append({'kind': kind, 'start': start, 'end': end})
    ev = pd.DataFrame(ev, columns=['kind', 'start', 'end']).sort_values('start', kind='stable')
    begins = ev.loc[ev['kind'] == 'class_begin', 'start']
    finals = ev[ev['kind'] == 'final']

    # 학기: 개강일 ~ 그 뒤 첫 기말고사 끝
    days['term_tag'] = '방학'
    for s in begins:
        f = finals[finals['start'] >= s]
        if len(f):
            days.loc[days['date'].between(s, f['end'].iloc[0]), 'term_tag'] = '학기'

    # 사건: 우선순위 낮은 것부터 칠해서 높은 것이 덮어쓰게 함
    days['event_tag'] = '없음'
    spans = {'개강주': [(s, s + pd.Timedelta(days=6)) for s in begins],
             '휴일': list(ev.loc[ev['kind'] == 'holiday', ['start', 'end']].itertuples(index=False)),
             '시험': list(finals[['start', 'end']].itertuples(index=False))}
    for tag in reversed(EVENT_PRIORITY):
        for s, e in spans[tag]:
            days.loc[days['date'].between(s, e), 'event_tag'] = tag
    return days[['date', 'study_week', 'term_tag', 'event_tag']]


def add_term_tag(d, calendar_path=None):
    """
    Calendar.csv가 있으면 study_week, term_tag(학기/방학), event_tag(개강주/시험/휴일/없음) 열을 붙임.
    없으면 열을 붙이지 않고 경고만 출력함.
    """
    cal = load_calendar(calendar_path)
    if cal is None:
        warnings.warn('Calendar.csv가 없어 study_week, term_tag, event_tag 열 없이 진행합니다 '
                      f'(넣을 위치: {calendar_path or PATHS["calendar"]})')
        return d
    return d.merge(cal, on='date', how='left')


def load_nethealth(path=None, wear=pp.WEAR_MIN, sleep_gap=pp.SLEEP_GAP, min_days=pp.MIN_DAYS,
                   period='all', cutoff=pp.PHASE_CUTOFF, calendar_path=None):
    """
    NetHealth 하루 표를 만듦. 전처리 규칙은 README '전처리 규칙' 표와 같음.

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
      study_week       : 연구 주차 0~194 (Calendar.csv가 있을 때만. 아래 두 열도 같음)
      term_tag         : 학기 / 방학
      event_tag        : 개강주 / 시험 / 휴일 / 없음

    인자
      wear       : 착용률 기준(%). 미만인 날의 활동·심박은 빈칸 (기본 80)
      sleep_gap  : 수면 조각을 한 덩어리로 묶는 간격(분) (기본 90)
      min_days   : 기록 있는 날이 이보다 적은 사람 제외 (기본 42, None이면 제외 안 함)
      period     : 'all' | 'before'(1~2학년) | 'after'(3~4학년) — 연구 기간 나눠 보기
      cutoff     : 연구 기간 경계 (기본 '2017-08-13' = Calendar 104주차, 3학년 오리엔테이션 주 시작)
    """
    A, S = read_raw(path)
    act = pp.build_activity(A, wear=wear)
    sl = pp.merge_sleep(S, gap=sleep_gap)
    d = act.merge(sl, on=['pid', 'date'], how='outer')
    d = pp.split_study_phase(d, period=period, cutoff=cutoff)
    d = pp.filter_min_days(d, min_days=min_days)
    d = pp.add_dow(d)
    d = d[DAILY_COLS].sort_values(['pid', 'date']).reset_index(drop=True)
    return add_term_tag(d, calendar_path)
