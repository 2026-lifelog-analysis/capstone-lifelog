"""
forecast_dataset.py — '오늘 밤 수면 예측' 공통 데이터 (참가자 × 밤 한 줄), 프로토콜 PROTOCOL

밤 D의 예측 기준 시각(D 18:00, FORECAST_HOUR)에, 그때까지 확인 가능한 기록만으로
밤 D의 주수면 세 가지를 예측한다.
  target_sleep_min : 총수면시간(분) = 주수면에 들어간 조각의 minsasleep 합
  target_bedtime   : 취침 시각 = 주수면 첫 조각의 timetobed ('Time went to bed').
                     실제로 잠든 시각이 아님. 단위는 'D 12:00부터 분' (23:50 → 710, 다음 날 00:10 → 730)
  target_waketime  : 기상 시각 = 주수면 마지막 조각의 timeoutofbed ('Time out of bed').
                     단위는 target_bedtime과 같은 'D 12:00부터 분' (다음 날 07:00 → 1140)
  두 시각 모두 24시간 나머지 연산을 하지 않는다. 같은 밤 기준의 연속된 분이라 그대로 빼면 오차가 된다.

    from src import forecast_dataset as fd
    data = fd.load_forecast_data()          # data/nethealth 원본에서
    data.table                              # 한 줄 = (pid, date=밤 D), 정답 + 입력
    fd.print_summary(data.info)

이 모듈은 이상 탐지용 하루 표(data_loader.load_nethealth, preprocessing.merge_sleep)를 쓰지도,
바꾸지도 않는다. 차이는 docs/forecast.md 참고.

처리 순서 (원본 열: egoid, dataDate, timetobed, timeoutofbed, bedtimedur, minsasleep)
  1. sleep_episodes : 조각마다 시작·종료 datetime 복원. dataDate = '조각이 끝난 날'로 **가정**
                      (코드북에 정의 없음. date_assumption_check로 재현 가능한 진단 근거, docs/forecast.md)
                      시각은 원본 표기 그대로 쓰고 시간대 변환은 하지 않는다.
                      완전 중복(같은 사람·시작·종료·minsasleep)은 하나만 남김
                      minsasleep 검사(asleep_check): 결측·숫자 아님·inf·음수·bedtimedur 초과 → NaN 표시
                      (그 조각이 있는 밤은 5단계에서 통째로 제외. 0으로 채우지 않음)
  2. overlap_flags  : 같은 사람의 다른 조각과 시간이 겹치는 조각 표시 (맞닿기만 한 것은 겹침 아님)
  3. assign_nights  : 조각의 시작이 [D 12:00, D+1 12:00)에 있으면 밤 D. 한 조각은 한 밤에만 들어간다.
                      종료가 다음 정오를 넘어도 자르지 않는다.
  4. sleep_blocks   : 같은 밤 안에서, 앞 조각들이 끝난 뒤 gap분(기본 90, pp.SLEEP_GAP) 이하로 이어지면 한 덩어리
  5. night_summary  : 덩어리 중 minsasleep 합이 가장 큰 것이 주수면. 같으면 먼저 시작한 덩어리
                      겹침 조각이 하나라도 있는 밤은 overlap=True → 정답·과거 입력 모두에서 제외
                      (minsasleep은 조각 전체 집계라 겹친 시간에 몇 분 잤는지 알 수 없음. 빼거나 나누지 않음)
  6. build_forecast_table : 겹침 밤·예측 기준 시각 전에 시작한 주수면은 제외(사유 기록), 과거 7밤 입력을 붙임

입력 (base_features)
  {sleep,bedtime,waketime}_lag{k} : 밤 D-k의 주수면 값 (k = 1..7, 달력 기준. 기록 7개가 아님)
                                    빠진 밤·겹침 밤은 NaN 그대로. 채우지 않음
                                    밤 D-k의 조각이 예측 기준 시각까지 다 끝나지 않았으면 NaN (known_at 검사)
  dow : 밤 D의 요일 (0=월 ~ 6=일). 미리 알 수 있는 값
  예측일 D 당일의 하루 집계(걸음, 평균 심박, 착용률)는 18시에 확정되지 않으므로 넣지 않는다.
"""
import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import preprocessing as pp

# 공통 데이터 규칙 버전. 정답·입력 규칙이 바뀌면 올림 (분할표·결과 폴더가 버전별로 나뉨)
#   v1: 1차 구현 (겹침 조각 그대로 합산) — 점검용, 결과를 최종 수치로 쓰지 않음
#   v2: 겹침 조각이 있는 밤 제외, 완전 중복 정의를 (시작, 종료, minsasleep)로
#   v3: minsasleep이 결측·무효인 조각이 있는 밤 제외 (0으로 채우거나 일부 조각만 더하지 않음)
PROTOCOL = 'v3'
PREVIOUS_PROTOCOL = 'v2'   # 새 분할표를 만들 때 최종 평가 참가자가 이 버전과 같은지 확인

# ---------- 설정 (팀 실험용 약속. 의료적 기준이 아님) ----------
FORECAST_HOUR = 18          # 예측 기준 시각: 밤 D의 날짜 D 18:00
NIGHT_START_HOUR = 12       # 밤 D = [D 12:00, D+1 12:00) 에 시작한 조각
N_LAGS = 7                  # 과거 입력: 달력 기준 밤 D-1 ~ D-7
SLEEP_GAP = pp.SLEEP_GAP    # 90분. 이상 탐지 하루 표와 같은 병합 간격
CLOCK_TOL = 1               # (시계 차이) vs (bedtimedur - 1) 허용 오차(분)

# 목표 이름 → 정답 열. 세 목표는 각각 따로 채점한다 (평균 내서 하나로 합치지 않음)
TARGETS = {'sleep': 'target_sleep_min', 'bed': 'target_bedtime', 'wake': 'target_waketime'}
TARGET_LABELS = {'sleep': '총수면시간(분)', 'bed': '취침 시각(D 12:00부터 분)',
                 'wake': '기상 시각(D 12:00부터 분)'}
# 목표별 과거 값 열 이름 앞부분 (밤 D-k의 같은 양)
LAG_PREFIX = {'sleep': 'sleep', 'bed': 'bedtime', 'wake': 'waketime'}
NIGHT_VALUE = {'sleep': 'sleep_min', 'bed': 'bedtime', 'wake': 'waketime'}

KEY_COLS = ['pid', 'date']
META_COLS = ['forecast_at']
# 정답과 정답에서 나온 값. 입력으로 쓰면 안 됨
TARGET_AUX_COLS = ['target_bed_at', 'target_wake_at', 'target_n_pieces', 'target_n_blocks']
NON_FEATURE_COLS = KEY_COLS + META_COLS + list(TARGETS.values()) + TARGET_AUX_COLS

DROP_REASONS = {
    'no_end_time': '종료 시각(timeoutofbed) 없음·형식 오류',
    'no_start_time': '시작 시각(timetobed) 없음·형식 오류',
    'too_long': 'bedtimedur가 24시간 이상이라 날짜를 정할 수 없음',
    'clock_mismatch': '시계 차이와 bedtimedur가 맞지 않음',
    'duplicate': '완전 중복 (같은 사람·시작·종료·minsasleep 조각이 이미 있음)',
}
ASLEEP_ISSUES = {
    'missing': 'minsasleep 빈칸',
    'non_numeric': 'minsasleep이 숫자로 읽히지 않음',
    'nonfinite': 'minsasleep이 ±inf',
    'negative': 'minsasleep이 음수',
    'exceeds_bedtimedur': 'minsasleep이 그 조각의 bedtimedur(누워 있던 분)보다 큼',
}
NIGHT_REASONS = {
    'ok': '유효 (주수면이 예측 기준 시각 이후 시작)',
    'invalid_sleep_value': 'minsasleep이 결측·무효인 조각이 있는 밤 → 주수면 선택·총수면을 알 수 없어 정답·입력에서 제외',
    'overlap_conflict': '겹치는 조각이 있는 밤 → 총수면 중복 계산 위험이라 정답·입력에서 제외',
    'started_before_forecast': '주수면이 예측 기준 시각 전에 시작 → 시작 전 예측 대상 아님',
    'no_sleep_record': '그 밤에 시작한 수면 조각 없음 (정답 없음, 채우지 않음)',
}
OVERLAP_KINDS = {
    'same_span': '시작·종료가 같은데 minsasleep이 다름',
    'contained': '한 조각이 다른 조각 안에 들어감 (시작 또는 종료가 같은 경우 포함)',
    'partial': '일부만 겹침',
}


def _clock_min(t):
    """'HH:MM:SS' Series → 0시부터 분 (실수). 형식이 틀리면 NaN."""
    p = t.astype('string').str.extract(r'^\s*(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*$')
    h, m, s = (pd.to_numeric(p[i], errors='coerce') for i in range(3))
    out = h * 60 + m + s / 60
    return out.where((h < 24) & (m < 60) & (s < 60))


def _times(S, tol=CLOCK_TOL):
    """원본 → (dataDate, 시작 시계 분, 종료 시계 분, 시계 차이, 제외 사유 Series)."""
    date = pd.to_datetime(S['dataDate'], errors='coerce')
    t_in, t_out = _clock_min(S['timetobed']), _clock_min(S['timeoutofbed'])
    clock = (t_out - t_in) % 1440
    reason = pd.Series(pd.NA, index=S.index, dtype='object')
    rules = [('no_end_time', t_out.isna() | date.isna()),
             ('no_start_time', t_in.isna()),
             ('too_long', S['bedtimedur'] >= 1440),
             ('clock_mismatch', (clock - (S['bedtimedur'] - 1)).abs() > tol)]
    for name, bad in rules:                      # 앞 사유가 우선
        reason = reason.mask(reason.isna() & bad.fillna(True), name)
    return date, t_in, t_out, clock, reason


def asleep_check(raw, bedtimedur):
    """
    minsasleep 검사. 반환: (숫자 값 Series(무효면 NaN), 문제 종류 Series('' = 정상))
      정상: 숫자로 읽히고, 유한하고, 0 이상이고, 그 조각의 bedtimedur 이하 (0분도 정상 값)
      상한을 bedtimedur로 둔 이유: 코드북상 '누워 있던 분'이고, bedtimedur는 이미 시계 차이 + 1분과
      CLOCK_TOL 안에서 맞는지 검사됨. 의학적 기준은 쓰지 않음
    """
    num = pd.to_numeric(raw, errors='coerce').astype(float)
    bed = pd.to_numeric(bedtimedur, errors='coerce').astype(float)
    with np.errstate(invalid='ignore'):
        issue = np.select([raw.isna().to_numpy(), num.isna().to_numpy(), ~np.isfinite(num.to_numpy()),
                           (num < 0).to_numpy(), (num > bed).to_numpy()],
                          list(ASLEEP_ISSUES), '')
    issue = pd.Series(issue, index=raw.index, dtype=object)
    return num.where(issue == ''), issue


def sleep_episodes(S, tol=CLOCK_TOL):
    """
    수면 원본 → 조각별 시작·종료 datetime (dataDate = 종료일 가정).
      end   = dataDate + timeoutofbed
      start = end - ((timeoutofbed - timetobed) mod 24시간)
    (시계 차이 = bedtimedur - 1분인지로 기록을 검사함)
    반환: (ep, dropped)
      ep      : pid | end_date | start | end | minsasleep | asleep_issue | bedtimedur
                (시각을 복원할 수 있으면 minsasleep이 무효여도 조각은 남김. 그 밤을 통째로 제외하기 위해)
      dropped : {사유: 건수} (DROP_REASONS 순서, 0건 포함)
    """
    end_date, t_in, t_out, clock, reason = _times(S, tol)
    ok = reason.isna()
    end = end_date[ok] + pd.to_timedelta(t_out[ok], unit='min')
    asleep, issue = asleep_check(S.loc[ok, 'minsasleep'], S.loc[ok, 'bedtimedur'])
    ep = pd.DataFrame({'pid': S.loc[ok, 'egoid'].astype(str),
                       'end_date': end_date[ok],
                       'start': end - pd.to_timedelta(clock[ok], unit='min'),
                       'end': end,
                       'minsasleep': asleep,                 # 무효면 NaN (0으로 채우지 않음)
                       'asleep_issue': issue,                # '' = 정상, 아니면 ASLEEP_ISSUES 키
                       'bedtimedur': S.loc[ok, 'bedtimedur']})
    dup = ep.duplicated(['pid', 'start', 'end', 'minsasleep'], keep='first')   # 파일에서 먼저 나온 줄을 남김
    reason.loc[dup[dup].index] = 'duplicate'
    ep = ep[~dup]
    dropped = {k: int((reason == k).sum()) for k in DROP_REASONS}
    return ep.reset_index(drop=True), dropped


def _overlap_rate(pid, start, end):
    """같은 사람 안에서 시작 순으로 놓았을 때, 앞 조각들 중 가장 늦은 종료보다 먼저 시작하는 조각의 비율."""
    e = pd.DataFrame({'pid': pid, 'start': start, 'end': end}).dropna()
    e = e.sort_values(['pid', 'start', 'end'], kind='stable')
    prev_end = e.groupby('pid')['end'].cummax().groupby(e['pid']).shift()
    n = int((e['start'] < prev_end).sum())
    return {'pieces': int(len(e)), 'overlapping': n, 'rate': n / max(len(e), 1)}


def date_assumption_check(S, tol=CLOCK_TOL):
    """
    dataDate를 '종료일'로 볼지 '시작일'로 볼지에 대한 진단 (재현용).
      분모 = 시각 검사를 통과한 조각 수 (완전 중복 제거 전)
      분자 = 같은 사람 안에서 시작 순으로 놓았을 때, 앞 조각들 중 가장 늦은 종료보다 먼저 시작하는 조각 수
      clock_matches_bedtimedur = |시계 차이 − (bedtimedur − 1)| ≤ tol 인 조각 비율 (분모 = 시각이 둘 다 있는 조각)
    겹침이 적은 쪽이 같은 사람의 수면 조각이 시간상 서로 배타적이라는 상식과 맞는 해석이다.
    이것은 해석을 고르는 근거이지 원본 정의를 확인한 것이 아니다.
    """
    date, t_in, t_out, clock, reason = _times(S, tol)
    ok = reason.isna()
    pid = S.loc[ok, 'egoid'].astype(str)
    end_e = date[ok] + pd.to_timedelta(t_out[ok], unit='min')
    start_s = date[ok] + pd.to_timedelta(t_in[ok], unit='min')
    span = pd.to_timedelta(clock[ok], unit='min')
    both = t_in.notna() & t_out.notna()
    return {'end_date': _overlap_rate(pid, end_e - span, end_e),
            'start_date': _overlap_rate(pid, start_s, start_s + span),
            'clock_matches_bedtimedur': float(((clock - (S['bedtimedur'] - 1)).abs() <= tol)[both].mean()),
            'pieces_with_both_times': int(both.sum())}


def overlap_flags(ep):
    """
    조각마다 같은 사람의 다른 조각과 시간이 겹치는지 (열린 구간 기준: a.start < b.end 이고 b.start < a.end).
    한 조각의 종료 = 다음 조각의 시작(맞닿음)은 겹침이 아님.
    반환: ep와 같은 index의 bool Series
    """
    e = ep.sort_values(['pid', 'start', 'end'], kind='stable')
    run_end = e.groupby('pid')['end'].cummax()
    prev_end = run_end.groupby(e['pid']).shift()
    next_start = e.groupby('pid')['start'].shift(-1)
    flag = (e['start'] < prev_end) | (next_start < e['end'])     # 바로 다음 시작이 내 종료 전이면 겹침
    return flag.reindex(ep.index).fillna(False).astype(bool)


def overlap_pairs(ep):
    """
    겹치는 조각 쌍 목록 (진단·문서용). 반환: pid | i | j | kind | overlap_min | same_night
      kind: OVERLAP_KINDS. same_night는 assign_nights의 date가 같은지 (ep에 date가 있을 때)
    """
    flag = overlap_flags(ep)
    e = ep[flag].sort_values(['pid', 'start', 'end'], kind='stable')
    rows = []
    for pid, g in e.groupby('pid', sort=False):
        idx, s, en = g.index.to_numpy(), g['start'].to_numpy(), g['end'].to_numpy()
        for a in range(len(g)):
            b = a + 1
            while b < len(g) and s[b] < en[a]:
                rows.append((pid, idx[a], idx[b]))
                b += 1
    out = pd.DataFrame(rows, columns=['pid', 'i', 'j'])
    if out.empty:
        return out.assign(kind=pd.Series(dtype=str), overlap_min=pd.Series(dtype=float),
                          same_night=pd.Series(dtype=bool))
    A, B = ep.loc[out['i']].reset_index(drop=True), ep.loc[out['j']].reset_index(drop=True)
    same = (A['start'] == B['start']) & (A['end'] == B['end'])
    inside = ((B['start'] >= A['start']) & (B['end'] <= A['end'])) | \
             ((A['start'] >= B['start']) & (A['end'] <= B['end']))
    out['kind'] = np.select([same, inside], ['same_span', 'contained'], 'partial')
    out['overlap_min'] = ((np.minimum(A['end'], B['end']) - B['start']) / pd.Timedelta(minutes=1)).to_numpy()
    out['same_night'] = (A['date'] == B['date']).to_numpy() if 'date' in ep else np.nan
    return out


def assign_nights(ep, night_start_hour=NIGHT_START_HOUR):
    """조각마다 밤 날짜 date를 붙임. 시작이 [D 12:00, D+1 12:00)이면 밤 D (23:00 시작, 다음 날 02:00 시작 모두 D)."""
    ep = ep.copy()
    ep['date'] = (ep['start'] - pd.Timedelta(hours=night_start_hour)).dt.normalize()
    return ep


def sleep_blocks(ep, gap=SLEEP_GAP):
    """
    같은 사람·같은 밤의 조각을 덩어리로 묶음 (assign_nights 결과를 받음).
    시작 순으로 놓고, 그때까지 나온 조각들의 가장 늦은 종료에서 gap분 넘게 지나 시작하면 새 덩어리.
    (이상 탐지 pp.merge_sleep와 같은 90분 규칙. 다만 merge_sleep는 dataDate(종료일) 안에서
     시계 분으로 묶으므로, 자정을 사이에 둔 조각이 다른 날로 갈라질 수 있어 예측에서는 datetime으로 묶음)
    minsasleep이 NaN(결측·무효)인 조각이 하나라도 있는 덩어리는 sleep_min = NaN (0으로 더하지 않음).
    반환: pid | date | block | start | end | sleep_min | n_pieces | n_invalid
    """
    e = ep.sort_values(['pid', 'date', 'start', 'end'], kind='stable').reset_index(drop=True)
    keys = [e['pid'], e['date']]
    run_end = e.groupby(keys)['end'].cummax()
    prev_end = run_end.groupby(keys).shift()
    new = prev_end.isna() | ((e['start'] - prev_end) > pd.Timedelta(minutes=gap))
    e['block'] = new.cumsum()
    e['_bad'] = pd.to_numeric(e['minsasleep'], errors='coerce').isna()
    b = (e.groupby('block')
          .agg(pid=('pid', 'first'), date=('date', 'first'), start=('start', 'min'),
               end=('end', 'max'), sleep_min=('minsasleep', 'sum'), n_pieces=('minsasleep', 'size'),
               n_invalid=('_bad', 'sum'))
          .reset_index())
    b['sleep_min'] = b['sleep_min'].where(b['n_invalid'] == 0)    # 일부 조각만 더한 합을 만들지 않음
    return b


def night_summary(ep, gap=SLEEP_GAP, night_start_hour=NIGHT_START_HOUR):
    """
    조각 → (사람, 밤) 한 줄. 주수면 = minsasleep 합이 가장 큰 덩어리, 같으면 먼저 시작한 덩어리.
    반환: pid | date | sleep_min | bedtime | waketime | bed_at | wake_at | n_pieces | n_blocks | known_at | overlap
      bedtime, waketime : 그 밤 D 12:00부터 분 (나머지 연산 없음)
      known_at          : 그 밤에 시작한 모든 조각 중 가장 늦은 종료. 이 시각이 지나야 그 밤 기록이 확정
      asleep_invalid    : 그 밤의 조각 중 하나라도 minsasleep이 결측·무효 → 정답·입력으로 쓰지 않음
                          (무효 조각만 빼고 다른 덩어리를 주수면으로 고르지 않음. 그 조각이 선택을 바꿀 수 있음)
      overlap           : 그 밤의 조각 중 하나라도 다른 조각과 겹침 → 정답·입력으로 쓰지 않음
                          (주수면 값은 확인용으로 계산만 해 둠. 겹친 조각만 빼서 다른 주수면을 고르지 않음)
    """
    ep = assign_nights(ep, night_start_hour)
    ep['overlap'] = overlap_flags(ep)
    b = sleep_blocks(ep, gap)
    main = (b.sort_values(['sleep_min', 'start'], ascending=[False, True], kind='stable')
             .drop_duplicates(['pid', 'date'], keep='first'))
    g = b.groupby(['pid', 'date'])
    extra = pd.DataFrame({'n_blocks': g.size(), 'known_at': g['end'].max()}).reset_index()
    ov = ep.groupby(['pid', 'date'])['overlap'].any().reset_index()
    inv = (b.groupby(['pid', 'date'])['n_invalid'].sum() > 0).rename('asleep_invalid').reset_index()
    n = main.merge(extra, on=['pid', 'date']).merge(ov, on=['pid', 'date']).merge(inv, on=['pid', 'date'])
    noon = n['date'] + pd.Timedelta(hours=night_start_hour)
    n['bedtime'] = (n['start'] - noon) / pd.Timedelta(minutes=1)
    n['waketime'] = (n['end'] - noon) / pd.Timedelta(minutes=1)
    n = n.rename(columns={'start': 'bed_at', 'end': 'wake_at'})
    return (n[['pid', 'date', 'sleep_min', 'bedtime', 'waketime', 'bed_at', 'wake_at',
               'n_pieces', 'n_blocks', 'known_at', 'overlap', 'asleep_invalid']]
            .sort_values(['pid', 'date']).reset_index(drop=True))


def lag_features(base, nights, n_lags=N_LAGS):
    """
    base(pid, date, forecast_at)에 밤 D-k (k=1..n_lags) 값을 붙임. 달력 날짜로 맞추므로 빠진 밤은 NaN.
      - 밤 D-k의 known_at이 forecast_at보다 늦으면(아직 안 끝난 조각이 있으면) 세 값 모두 NaN
      - 밤 D-k가 minsasleep 결측·무효 밤(asleep_invalid)이나 겹침 밤(overlap)이면 세 값 모두 NaN (0분으로 쓰지 않음)
    반환: (새 열 표, {'not_known_yet', 'invalid_sleep', 'overlap': (줄, k) 수})
    """
    out = pd.DataFrame(index=base.index)
    masked = {'not_known_yet': 0, 'invalid_sleep': 0, 'overlap': 0}
    vals = ['sleep_min', 'bedtime', 'waketime']
    for k in range(1, n_lags + 1):
        lag = nights[['pid', 'date', 'known_at', 'asleep_invalid', 'overlap'] + vals].copy()
        lag['date'] = lag['date'] + pd.Timedelta(days=k)
        m = base[['pid', 'date', 'forecast_at']].merge(lag, on=['pid', 'date'], how='left')
        m.index = base.index
        late = m['known_at'].notna() & (m['known_at'] > m['forecast_at'])
        inv = m['asleep_invalid'].fillna(False).astype(bool) & ~late
        ov = m['overlap'].fillna(False).astype(bool) & ~late & ~inv
        masked['not_known_yet'] += int(late.sum())
        masked['invalid_sleep'] += int(inv.sum())
        masked['overlap'] += int(ov.sum())
        for t, v in NIGHT_VALUE.items():
            out[f'{LAG_PREFIX[t]}_lag{k}'] = m[v].where(~late & ~inv & ~ov)
    return out, masked


@dataclass
class ForecastData:
    """공통 예측 데이터. table만 모델에 들어가고, 나머지는 확인·추가 특징용."""
    table: pd.DataFrame                 # (pid, date) 한 줄, 정답 있는 '시작 전 예측' 밤만
    nights: pd.DataFrame                # 모든 (사람, 밤)의 주수면 요약 (night_summary, overlap 표시 포함)
    candidates: pd.DataFrame            # 사람별 첫 밤 ~ 마지막 밤 전체 + reason
    episodes: pd.DataFrame              # 조각 표 (밤 날짜·overlap 포함)
    info: dict
    settings: dict
    activity: pd.DataFrame | None = None   # pp.build_activity 결과 (추가 특징용, 기본 입력엔 안 씀)
    extra_features: list = field(default_factory=list)


def build_forecast_table(S, A=None, forecast_hour=FORECAST_HOUR, n_lags=N_LAGS, gap=SLEEP_GAP,
                         night_start_hour=NIGHT_START_HOUR, min_nights=None, period='all',
                         cutoff=pp.PHASE_CUTOFF, wear=pp.WEAR_MIN):
    """
    원본 S(FitbitSleep), A(FitbitActivity, 선택) → ForecastData.
      min_nights : 수면 기록 있는 밤이 이보다 적은 사람 제외. 기본 None = 제외 안 함
                   (이상 탐지의 42일 필터는 기준선 28일 + 평가 14일 때문이라 예측에는 그대로 쓰지 않음.
                    기록이 적은 사람·초기 밤도 남겨 두고, 입력이 NaN인 채로 둔다)
      period     : 'all' | 'before' | 'after' (밤 날짜 기준, pp.split_study_phase)
    """
    if not 0 <= forecast_hour < 24:
        raise ValueError('forecast_hour는 0 이상 24 미만')
    if n_lags < 1:
        raise ValueError('n_lags는 1 이상 (밤 D 자체는 정답이라 입력에 못 씀)')
    if forecast_hour < night_start_hour:
        raise ValueError('forecast_hour가 밤 시작(정오)보다 앞서면 밤 D의 조각이 섞일 수 있음')

    ep, dropped = sleep_episodes(S)
    ep = assign_nights(ep, night_start_hour)
    ep['overlap'] = overlap_flags(ep)
    pairs = overlap_pairs(ep)
    nights = night_summary(ep.drop(columns=['date', 'overlap']), gap, night_start_hour)

    # 사람별 첫 밤 ~ 마지막 밤 달력
    span = nights.groupby('pid')['date'].agg(['min', 'max'])
    cand = pd.concat([pd.DataFrame({'pid': p, 'date': pd.date_range(r['min'], r['max'])})
                      for p, r in span.iterrows()], ignore_index=True) if len(span) else \
        pd.DataFrame({'pid': pd.Series(dtype=str), 'date': pd.Series(dtype='datetime64[ns]')})
    cand = pp.split_study_phase(cand, period=period, cutoff=cutoff)
    n_people_before_filter = int(cand['pid'].nunique())
    if min_nights:
        n_obs = nights.groupby('pid').size()
        cand = cand[cand['pid'].isin(n_obs[n_obs >= min_nights].index)]
    cand = cand.merge(nights, on=['pid', 'date'], how='left')
    cand['forecast_at'] = cand['date'] + pd.Timedelta(hours=forecast_hour)
    cand['reason'] = np.select([cand['bed_at'].isna(),
                                cand['asleep_invalid'].fillna(False).astype(bool),
                                cand['overlap'].fillna(False).astype(bool),
                                cand['bed_at'] < cand['forecast_at']],
                               ['no_sleep_record', 'invalid_sleep_value', 'overlap_conflict',
                                'started_before_forecast'], 'ok')
    cand = cand.reset_index(drop=True)

    t = cand[cand['reason'] == 'ok'].reset_index(drop=True)
    table = pd.DataFrame({'pid': t['pid'].astype(str), 'date': t['date'], 'forecast_at': t['forecast_at'],
                          'target_sleep_min': t['sleep_min'], 'target_bedtime': t['bedtime'],
                          'target_waketime': t['waketime'], 'target_bed_at': t['bed_at'],
                          'target_wake_at': t['wake_at'], 'target_n_pieces': t['n_pieces'],
                          'target_n_blocks': t['n_blocks']})
    lags, masked = lag_features(table, nights, n_lags)
    table = pd.concat([table, lags], axis=1)
    table['dow'] = table['date'].dt.dayofweek
    table = table.sort_values(KEY_COLS).reset_index(drop=True)

    ov_n = cand[cand['reason'] == 'overlap_conflict']
    inv_n = cand[cand['reason'] == 'invalid_sleep_value']
    sl = [f'sleep_lag{k}' for k in range(1, n_lags + 1)]
    settings = {'protocol': PROTOCOL, 'forecast_hour': forecast_hour, 'night_start_hour': night_start_hour,
                'n_lags': n_lags, 'sleep_gap': gap, 'min_nights': min_nights, 'period': period,
                'cutoff': str(cutoff), 'wear': wear}
    info = {
        'settings': settings,
        'raw_sleep_rows': int(len(S)), 'episodes': int(len(ep)), 'dropped_episodes': dropped,
        'overlap': {'pieces': int(ep['overlap'].sum()), 'pairs': int(len(pairs)),
                    'pairs_by_kind': pairs['kind'].value_counts().reindex(list(OVERLAP_KINDS), fill_value=0)
                                                  .astype(int).to_dict(),
                    'pairs_cross_night': int((~pairs['same_night'].astype(bool)).sum()) if len(pairs) else 0,
                    'people': int(ep.loc[ep['overlap'], 'pid'].nunique()),
                    'nights_excluded': int(len(ov_n)), 'people_with_excluded_nights': int(ov_n['pid'].nunique()),
                    'overlap_min': pairs['overlap_min'].describe()[['min', '50%', 'max']].round(1).to_dict()
                    if len(pairs) else {}},
        'invalid_sleep': {'pieces_by_issue': ep['asleep_issue'].value_counts()
                                                .reindex(list(ASLEEP_ISSUES), fill_value=0).astype(int).to_dict(),
                          'nights_excluded': int(len(inv_n)),
                          'people_with_excluded_nights': int(inv_n['pid'].nunique())},
        'people_candidates': n_people_before_filter,
        'people_after_min_nights': int(cand['pid'].nunique()),
        'nights_candidates': int(len(cand)),
        'night_reasons': cand['reason'].value_counts().reindex(list(NIGHT_REASONS), fill_value=0)
                                       .astype(int).to_dict(),
        'people_table': int(table['pid'].nunique()), 'nights_table': int(len(table)),
        'lag_nights_masked_not_known_yet': masked['not_known_yet'],
        'lag_nights_masked_overlap': masked['overlap'],
        'lag_nights_masked_invalid_sleep': masked['invalid_sleep'],
        'nights_no_sleep_input': int(table[sl].isna().all(axis=1).sum()),
        'sleep_input_missing_rate': float(table[sl].isna().to_numpy().mean()) if len(table) else np.nan,
        # 분모 = 예측 표의 밤 (정답 있는 '시작 전 예측' 밤). 분자 = 주수면 취침이 다음 날 00:00 이후
        'table_bed_after_midnight': {'nights': int((table['target_bedtime'] >= 720).sum()),
                                     'of': int(len(table))},
    }
    act = pp.build_activity(A, wear=wear) if A is not None else None
    return ForecastData(table=table, nights=nights, candidates=cand, episodes=ep, info=info,
                        settings=settings, activity=act)


def load_forecast_data(path=None, with_activity=True, **kw):
    """data/nethealth 원본을 읽어 build_forecast_table. 인자는 build_forecast_table과 같음."""
    from . import data_loader as dl
    A, S = dl.read_raw(path)
    return build_forecast_table(S, A if with_activity else None, **kw)


# ---------- 입력 열 관리 ----------

def base_features(target, n_lags=N_LAGS):
    """
    공통 기본 입력 (목표별로 고정): 그 목표와 같은 양의 최근 달력 7밤 + 밤 D의 요일.
      sleep → sleep_lag1..7 + dow     (과거 7밤 총수면시간 + 요일)
      bed   → bedtime_lag1..7 + dow   (과거 7밤 취침 시각 + 요일)
      wake  → waketime_lag1..7 + dow  (과거 7밤 기상 시각 + 요일)
    팀원 실험은 이 목록에서 출발해 열을 더한다 (fd.base_features(t) + [...]).
    """
    if target not in TARGETS:
        raise KeyError(f'목표는 {list(TARGETS)} 중 하나')
    return [f'{LAG_PREFIX[target]}_lag{k}' for k in range(1, n_lags + 1)] + ['dow']


CATEGORICAL_FEATURES = ['dow']      # 모델 어댑터가 범주형으로 다루는 입력


def check_features(cols, table=None):
    """입력 열 목록 검사. 사람 ID·날짜·정답·정답에서 나온 열이 섞이면 오류."""
    cols = list(cols)
    bad = [c for c in cols if c in NON_FEATURE_COLS or c.startswith('target_') or c == 'reason']
    if bad:
        raise ValueError(f'입력으로 쓸 수 없는 열: {bad} (ID·날짜·정답·정답 파생 값)')
    if len(set(cols)) != len(cols):
        raise ValueError('입력 열 이름이 중복됨')
    if table is not None:
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise KeyError(f'표에 없는 입력 열: {missing}')
    return cols


def _same_values(a, b):
    """두 열이 같은 위치에서 같은 값인지. 같은 위치의 NaN/NaT는 같은 결측. 정수·실수 표현 차이는 무시."""
    a_dt = pd.api.types.is_datetime64_any_dtype(a)
    b_dt = pd.api.types.is_datetime64_any_dtype(b)
    if a_dt or b_dt:
        if not (a_dt and b_dt):
            return False
        x, y = pd.to_datetime(a).to_numpy(), pd.to_datetime(b).to_numpy()
        na_x, na_y = pd.isna(x), pd.isna(y)
        return bool((na_x == na_y).all() and (x[~na_x] == y[~na_y]).all())
    a_num = pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a)
    b_num = pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b)
    if a_num or b_num:
        if not (a_num and b_num):
            return False
        return bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))
    na_a, na_b = a.isna().to_numpy(), b.isna().to_numpy()
    if not (na_a == na_b).all():
        return False
    return bool((a.astype(str).to_numpy()[~na_a] == b.astype(str).to_numpy()[~na_b]).all())


def check_common_columns(base, table, who='table'):
    """
    table이 공통 예측 표 base의 모든 열을 같은 행 순서·같은 값으로 갖고 있는지 검사 (추가 열만 허용).
    검사: base·table의 중복 열 이름, base의 (pid, date) 중복, 열 삭제, 행 수·순서, 값 변경
    (키·예측 시각·정답·정답 보조 열·기본 입력 등 base의 모든 열). 같은 위치의 NaN은 같은 결측.
    문제가 있으면 ValueError, 없으면 추가된 열 이름 목록을 돌려줌.
    """
    if base.columns.duplicated().any():
        raise ValueError(f'공통 예측 표에 중복 열 이름: {list(base.columns[base.columns.duplicated()])}')
    if base.duplicated(KEY_COLS).any():
        raise ValueError('공통 예측 표에 (pid, date) 중복이 있음')
    if table.columns.duplicated().any():
        raise ValueError(f'{who}에 중복 열 이름: {list(table.columns[table.columns.duplicated()])}')
    missing = [c for c in base.columns if c not in table.columns]
    if missing:
        raise ValueError(f'{who}에서 공통 열이 빠짐: {missing}')
    if len(table) != len(base):
        raise ValueError(f'{who}의 행 수({len(table)})가 공통 예측 표({len(base)})와 다름. 행을 빼거나 더하지 마세요')
    changed = [c for c in base.columns if not _same_values(base[c], table[c])]
    if changed:
        raise ValueError(f'{who}에서 공통 열 값(또는 행 순서)이 바뀜: {changed}. 추가 특징 열만 더할 수 있음')
    return [c for c in table.columns if c not in base.columns]


def add_features(data, *fns):
    """
    추가 특징을 붙인 새 표를 돌려줌 (data.table은 바꾸지 않음).
    fn(data) → data.table과 같은 index의 DataFrame. 열 이름은 기존 열·정답 이름과 겹치면 안 됨.
    fn 안에서는 각 줄의 forecast_at 이전에 확정된 기록만 써야 함
    (예: data.nights에서 known_at <= forecast_at 이고 overlap == False, asleep_invalid == False 인 밤만).
    fn에는 표들의 복사본이 넘어가므로 원본 data는 바뀌지 않는다. fn이 넘겨받은 table을 고치면 오류.
    """
    base = data.table
    check_common_columns(base, base, 'data.table')
    table = base.copy()
    for fn in fns:
        name = getattr(fn, '__name__', repr(fn))
        view = replace(data, table=base.copy(deep=True), nights=data.nights.copy(deep=True),
                       candidates=data.candidates.copy(deep=True), episodes=data.episodes.copy(deep=True),
                       activity=None if data.activity is None else data.activity.copy(deep=True))
        new = fn(view)
        try:
            check_common_columns(base, view.table, f'{name}가 받은 data.table')
        except ValueError as e:
            raise ValueError(f'{name}: 특징 함수가 data.table을 직접 고쳤음 (원본은 보호됨). 새 열은 반환값으로만 주세요 — {e}')
        if not isinstance(new, pd.DataFrame) or not new.index.equals(base.index):
            raise ValueError(f'{name}: data.table과 같은 index의 DataFrame을 돌려줘야 함')
        check_features(new.columns)
        clash = [c for c in new.columns if c in table.columns]
        if clash:
            raise ValueError(f'이미 있는 열 이름: {clash}')
        table = pd.concat([table, new], axis=1)
    return table


def feat_steps_lags(data, n_lags=N_LAGS):
    """
    추가 특징 예시: 날짜 D-k (k=1..n_lags)의 하루 걸음 수 steps_lag{k}.
    D-1 하루 집계는 D 00:00에 끝나므로 D 18:00에 확정됨. D 당일 걸음은 넣지 않음.
    착용률 80% 미만이거나 빠진 날은 NaN.
    """
    if data.activity is None:
        raise ValueError('활동 원본(A) 없이 만든 데이터라 걸음 특징을 만들 수 없음')
    st = data.activity[['pid', 'date', 'steps']]
    base = data.table[['pid', 'date']]
    out = pd.DataFrame(index=base.index)
    for k in range(1, n_lags + 1):
        s = st.assign(date=st['date'] + pd.Timedelta(days=k))
        m = base.merge(s, on=['pid', 'date'], how='left')
        out[f'steps_lag{k}'] = m['steps'].to_numpy()
    return out


# ---------- 확인용 ----------

def table_fingerprint(table, cols=None):
    """표 내용의 sha256 (키·정답·기본 입력). 팀원끼리 같은 데이터인지 확인용. 원본 ID는 해시 안에만 있음."""
    if cols is None:
        cols = KEY_COLS + list(TARGETS.values()) + sorted(
            {c for t in TARGETS for c in base_features(t) if c in table.columns})
    t = table[cols].sort_values(KEY_COLS).copy()
    t['date'] = t['date'].dt.strftime('%Y-%m-%d')
    return hashlib.sha256(t.to_csv(index=False, float_format='%.6f').encode()).hexdigest()


def print_summary(info):
    """build_forecast_table의 info를 읽기 좋게 출력."""
    s = info['settings']
    print(f"[프로토콜 {s['protocol']}] 예측 기준 시각 D {s['forecast_hour']:02d}:00 | 밤 = [D {s['night_start_hour']:02d}:00, "
          f"D+1 {s['night_start_hour']:02d}:00) 시작 | 입력 밤 D-{s['n_lags']} ~ D-1 | 병합 간격 {s['sleep_gap']}분 | "
          f"min_nights {s['min_nights']} | period {s['period']}")
    print(f"수면 원본 {info['raw_sleep_rows']:,}줄 → 조각 {info['episodes']:,}개")
    for k, n in info['dropped_episodes'].items():
        print(f"  제외 {n:>7,}  {DROP_REASONS[k]}")
    o = info['overlap']
    kinds = ', '.join(f"{OVERLAP_KINDS[k]} {v}" for k, v in o['pairs_by_kind'].items())
    print(f"겹치는 조각 {o['pieces']:,}개 / {o['people']}명, 쌍 {o['pairs']} ({kinds}; 서로 다른 밤 사이 "
          f"{o['pairs_cross_night']}), 겹친 길이(분) {o['overlap_min']}")
    print(f"  → 겹침 밤 {o['nights_excluded']:,}개 / {o['people_with_excluded_nights']}명을 정답·과거 입력에서 제외")
    iv = info['invalid_sleep']
    bad = ', '.join(f"{ASLEEP_ISSUES[k]} {v}" for k, v in iv['pieces_by_issue'].items())
    print(f"minsasleep 결측·무효 조각: {bad}")
    print(f"  → 그 조각이 있는 밤 {iv['nights_excluded']:,}개 / {iv['people_with_excluded_nights']}명을 정답·과거 입력에서 제외")
    print(f"후보 (사람, 밤) {info['nights_candidates']:,}개 / {info['people_after_min_nights']}명 "
          f"(min_nights 적용 전 {info['people_candidates']}명)")
    for k, n in info['night_reasons'].items():
        print(f"  {n:>8,} ({n / max(info['nights_candidates'], 1):6.1%})  {NIGHT_REASONS[k]}")
    am = info['table_bed_after_midnight']
    print(f"예측 표 {info['nights_table']:,}밤 / {info['people_table']}명 "
          f"(그중 주수면 취침이 자정 이후 {am['nights']:,}밤 = {am['nights'] / max(am['of'], 1):.1%})")
    print(f"  과거 7밤 수면 입력이 전부 NaN인 밤 {info['nights_no_sleep_input']:,}, "
          f"수면 입력 칸 결측률 {info['sleep_input_missing_rate']:.1%}, 과거 밤을 NaN으로 지운 수: "
          f"예측 시점 미확정 {info['lag_nights_masked_not_known_yet']:,}, "
          f"minsasleep 무효 밤 {info['lag_nights_masked_invalid_sleep']:,}, 겹침 밤 {info['lag_nights_masked_overlap']:,}")
