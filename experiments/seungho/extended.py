# 작성 2026-11-06 08:00 (KST)
"""
extended.py — ① 확장: 변화량 외의 생활패턴·건강 정보까지 넣어 보는 특징 계열 (공통 프로토콜 v3 위에서)

sleep_delta.py(평소 대비 변화량)에 더해, 아래 계열을 "기본 입력 + 계열" 형태로 하나씩 비교하고,
마지막에 계열 단위 전진 선택(run_extended.py greedy)으로 조합을 고른다.

  W  창 길이      : 과거 14·28밤 lag, 최근 14·28밤 평균의 변화량
  Q  양 조합      : 총수면·취침·기상 중 어떤 양의 과거를 넣나 (7가지 부분집합, 기본 입력을 대체)
  T  평일/쉬는 날 : 다음 날이 쉬는 날인지, 같은 종류(쉬는 날 전/일하는 날 전) 밤끼리의 개인 평소
  H  공휴일·학사  : 미국 연방 공휴일, 학사 달력(수업·시험·방학·겨울·여름), 학기 주차
  A  활동·건강    : 전날까지의 걸음, 심박 평균·표준편차, 좌식·활동 시간, 심박 구간, 계단, 착용률
  N  수면 부가    : 수면 효율, 주수면 외 수면(낮잠), 덩어리·조각 수, 수면 중앙 시각과 그 흔들림, 침대에 있던 시간
  P  개인 정적    : 입학 시(1차) 설문 — 성별·BMI, PSQI(평소 취침·기상·수면시간·잠들기까지·총점),
                    CES-D·STAI 특성 불안, 건강 자기평가·카페인·운동·수면장애

예측 시점(D 18:00) 이후 정보 차단
  - 과거 밤 값은 fd.lag_features와 같은 가림 규칙(달력 기준, D 18:00까지 안 끝난 밤·겹침·무효 밤 → NaN)
  - 활동은 D−1까지의 하루 집계만 (D−1 하루는 D 00:00에 끝남. fd.feat_steps_lags와 같은 규칙). D 당일은 안 씀
  - 착용률 80% 미만인 날의 활동·심박은 NaN (pp.build_activity와 같은 규칙). 착용률 자체는 남김
  - 달력·공휴일은 미리 정해진 일정이라 D+1 값도 쓸 수 있다 (내일이 쉬는 날인지)
  - 설문은 입학 시 1회(1차) 값만. 시간에 따라 바뀌는 이후 차수는 쓰지 않는다
    한계: 1차 설문 응답일이 일부 사람의 첫 수면 기록보다 늦을 수 있다(2015-08~10). 그 기간 수면을 직접 담진 않지만
    "응답 시점에 알려진 정보"라는 가정은 완전하지 않다 → 결과에 따로 적는다
  - ffill / bfill / 보간 없음. 집단 통계는 만들지 않는다(줄마다 그 사람의 과거만)
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import preprocessing as pp
from experiments.seungho import sleep_delta as sd

QS = list(fd.TARGETS)                         # sleep, bed, wake
QL = {'sleep': 'S', 'bed': 'B', 'wake': 'W'}
WEAR_MIN = pp.WEAR_MIN
F32 = np.float32


def _med(M, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(M, axis=axis)


def _mean(M, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(M, axis=axis)


# ---------- 원본 (설문·달력·활동) ----------

def nethealth_dir():
    from src import data_loader as dl
    return Path(dl.PATHS['nethealth'])


def _find(p, key):
    hits = [f for f in Path(p).iterdir() if f.name.lower().startswith(key)]
    if not hits:
        raise FileNotFoundError(f'{p}에 {key}* 파일 없음')
    return hits[0]


def _clock(s, wake=False):
    """'11:30 PM' → 정오부터 분. 취침은 자정 넘어가면 +, 기상은 다음 날 오전으로 봄 (fd 정답과 같은 축)."""
    if not isinstance(s, str):
        return np.nan
    m = re.match(r'\s*(\d{1,2})(?::(\d{2}))?\s*([AP])?\.?\s*M?\.?\s*$', s.upper())
    if not m or int(m.group(1)) > 12:
        return np.nan
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap is None:            # 오전/오후 표기 없음 → 취침은 6~11시를 밤으로, 기상은 4~11시를 아침으로 본다
        if wake:
            ap = 'A' if 4 <= h <= 11 else 'P'
        else:
            ap = 'P' if 6 <= h <= 11 else 'A'
    ap = ap + 'M'
    h %= 12
    h24 = h + (12 if ap == 'PM' else 0)
    t = (h24 - 12) * 60 + mi                 # 정오 기준 (−720 ~ 719)
    if t < 0:
        t += 1440                            # 오전 → 다음 날
    if wake and t < 720:                     # 기상인데 정오~자정 사이 → 다음 날 오후로
        t += 1440
    return float(t)


ORD = {
    'Health_1': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
    'usecaffine_1': {'Not at all': 0, 'Less than 1-2 times a month': 1, '1-2 times a month': 2,
                     '1-2 times a week': 3, 'Three times a week or more': 4},
    'exercise_1': {'0': 0, 'Less than 1-2 times a month': 1, '1-2 times a month': 2, '1-2 times a week': 3,
                   'Three times a week or more': 4},
}


def survey_static(path=None):
    """1차 설문 → pid 한 줄. 반환 열은 P_COLS."""
    b = pd.read_csv(_find(path or nethealth_dir(), 'basicsurvey'), low_memory=False)
    num = lambda c: pd.to_numeric(b[c], errors='coerce')
    h_in = num('Height_1') * 12
    out = pd.DataFrame({
        'pid': b['egoid'].astype(str),
        'p_male': (b['gender_1'] == 'Male').astype(float).where(b['gender_1'].notna()),
        'p_bmi': 703 * num('Weight_1') / h_in ** 2,
        'p_psqi_bed': b['PSQI1_1'].map(_clock),
        'p_psqi_wake': b['PSQI3_1'].map(lambda s: _clock(s, wake=True)),
        'p_psqi_latency': num('PSQI2_1'),
        'p_psqi_hours': num('PSQI4_1') * 60,
        'p_psqi_global': num('PSQIGlobal_1'),
        'p_cesd': num('CESDOverall_1'),
        'p_stai': num('STAITraitTotal_1'),
        'p_health': b['Health_1'].map(ORD['Health_1']),
        'p_caffeine': b['usecaffine_1'].map(ORD['usecaffine_1']),
        'p_exercise': b['exercise_1'].astype(str).map(ORD['exercise_1']),
        'p_sleepdisorder': b['SleepDisorder_1'].map({'yes': 1.0, 'no': 0.0}),
    })
    out.loc[(out['p_bmi'] < 10) | (out['p_bmi'] > 70), 'p_bmi'] = np.nan
    return out.drop_duplicates('pid')


P_GROUPS = {
    'demo': ['p_male', 'p_bmi'],
    'psqi': ['p_psqi_bed', 'p_psqi_wake', 'p_psqi_latency', 'p_psqi_hours', 'p_psqi_global'],
    'mental': ['p_cesd', 'p_stai'],
    'life': ['p_health', 'p_caffeine', 'p_exercise', 'p_sleepdisorder'],
}


def calendar_days(path=None, start='2015-06-01', end='2019-12-31'):
    """
    날짜 한 줄: 학사 상태(수업·시험·방학·겨울·여름) 플래그, 학기 주차, 시험까지 남은 날, 연방 공휴일.
    Calendar.csv의 사건만 쓰고, 사건 사이 구간은 규칙으로 채운다:
      수업 = '수업 시작' ~ 시험 전날, 시험 = 시험 주 시작~끝, 방학 = 중간 방학·추수감사·부활절 사건 기간,
      겨울 = 가을 시험 끝 다음 날 ~ 봄 수업 시작 전날, 여름 = 봄 시험 끝 다음 날 ~ 가을 수업 시작 전날
    """
    c = pd.read_csv(_find(path or nethealth_dir(), 'calendar'))
    c['ev'] = c['Event'].fillna('').str.lower()
    d8 = lambda s: pd.to_datetime(s.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
    c['b'] = d8(c['Beginning event date']).fillna(d8(c['Begin Date']))
    c['e'] = d8(c['End event date']).fillna(d8(c['End date']))
    days = pd.DataFrame({'date': pd.date_range(start, end)})
    status = pd.Series('summer', index=days.index)
    starts = c[c['ev'].str.contains('class')][['b']].sort_values('b')['b'].tolist()
    finals = c[c['ev'].str.contains('final')][['b', 'e']].sort_values('b').values.tolist()
    for s in starts:                                    # 수업: 시작 ~ 그 뒤 첫 시험 전날
        f = [x for x in finals if x[0] > s]
        if not f:
            continue
        fb, fe_ = f[0]
        status[(days['date'] >= s) & (days['date'] < fb)] = 'class'
        status[(days['date'] >= fb) & (days['date'] <= fe_)] = 'finals'
        nxt = [x for x in starts if x > fe_]
        if fb.month == 12 and nxt:                      # 겨울: 가을 시험 끝 ~ 봄 수업 전
            status[(days['date'] > fe_) & (days['date'] < nxt[0])] = 'winter'
    for _, r in c[c['ev'].str.contains('break|thanksgiving|easter')].iterrows():
        status[(days['date'] >= r['b']) & (days['date'] <= r['e'])] = 'break'
    days['status'] = status.values
    for s in ('class', 'finals', 'break', 'winter', 'summer'):
        days[f'h_{s}'] = (days['status'] == s).astype(F32)
    st = pd.Series(pd.NaT, index=days.index)
    for s in starts:
        st[days['date'] >= s] = s
    wk = ((days['date'] - st).dt.days // 7).astype(float)
    days['h_term_week'] = wk.where(days['status'].isin(['class', 'finals', 'break']))
    fb_list = sorted(x[0] for x in finals)
    nxt_f = days['date'].map(lambda d: next((f for f in fb_list if f >= d), pd.NaT))
    dd = (nxt_f - days['date']).dt.days.astype(float)
    days['h_days_to_finals'] = dd.where(days['status'].isin(['class']))
    hol = set(USFederalHolidayCalendar().holidays(start, end))
    days['h_fed'] = days['date'].isin(hol).astype(F32)
    off = (days['date'].dt.dayofweek >= 5) | days['date'].isin(hol) | days['status'].isin(['break', 'winter', 'summer'])
    days['off'] = off.astype(F32)
    return days


def raw_activity(path=None):
    """활동 원본 → pid | date | 하루 집계 열 (착용률 80% 미만 날은 착용률 외 NaN)."""
    A = pd.read_csv(_find(path or nethealth_dir(), 'fitbitactivity'))
    A = pp.drop_duplicate_days(A)
    ok = A['complypercent'] >= WEAR_MIN
    a = pd.DataFrame({
        'pid': A['egoid'].astype(str), 'date': pd.to_datetime(A['datadate']),
        'steps': A['steps'], 'hr': A['meanrate'], 'hrsd': A['sdrate'], 'floors': A['floors'],
        'sed': A['sedentaryminutes'], 'light': A['lightlyactiveminutes'],
        'modvig': A['fairlyactiveminutes'] + A['veryactiveminutes'],
        'fatburn': A['fatburnmins'], 'cardiopeak': A['cardiomins'] + A['peakmins'],
    })
    vals = [c for c in a.columns if c not in ('pid', 'date')]
    a[vals] = a[vals].where(ok, axis=0)
    a['wear'] = A['complypercent'].to_numpy()
    return a


ACT = ['steps', 'hr', 'hrsd', 'floors', 'sed', 'light', 'modvig', 'fatburn', 'cardiopeak', 'wear']


# ---------- 과거 값 붙이기 ----------

def day_lags(base, daily, cols, ks):
    """날짜 D−k의 하루 값 (달력 기준, 빠진 날 NaN)."""
    out = {}
    for k in ks:
        s = daily[['pid', 'date'] + cols].copy()
        s['date'] = s['date'] + pd.Timedelta(days=k)
        m = base[['pid', 'date']].merge(s, on=['pid', 'date'], how='left')
        for c in cols:
            out[f'{c}_l{k}'] = m[c].to_numpy(dtype=F32)
    return out


def night_metrics(data):
    """밤마다 부가 수면 지표 + 가림용 열."""
    n = data.nights.copy()
    ep = data.episodes
    tot = ep.groupby(['pid', 'date'])['minsasleep'].sum().rename('tot').reset_index()
    n = n.merge(tot, on=['pid', 'date'], how='left')
    inbed = (n['wake_at'] - n['bed_at']) / pd.Timedelta(minutes=1)
    n['n_inbed'] = inbed
    n['n_eff'] = n['sleep_min'] / inbed.where(inbed > 0)
    n['n_extra'] = n['tot'] - n['sleep_min']
    n['n_blocks'] = n['n_blocks'].astype(float)
    n['n_pieces'] = n['n_pieces'].astype(float)
    n['n_mid'] = (n['bedtime'] + n['waketime']) / 2
    return n


N_COLS = ['n_inbed', 'n_eff', 'n_extra', 'n_blocks', 'n_pieces', 'n_mid']


def night_lags(table, nm, cols, ks):
    """fd.lag_features와 같은 가림 규칙으로 부가 지표의 밤 D−k 값."""
    out = {}
    base = table[['pid', 'date', 'forecast_at']]
    for k in ks:
        lag = nm[['pid', 'date', 'known_at', 'asleep_invalid', 'overlap'] + cols].copy()
        lag['date'] = lag['date'] + pd.Timedelta(days=k)
        m = base.merge(lag, on=['pid', 'date'], how='left')
        bad = ((m['known_at'].notna() & (m['known_at'] > m['forecast_at']))
               | m['asleep_invalid'].astype('boolean').fillna(False).astype(bool)
               | m['overlap'].astype('boolean').fillna(False).astype(bool))
        for c in cols:
            out[f'{c}_l{k}'] = m[c].where(~bad).to_numpy(dtype=F32)
    return out


# ---------- 특징 ----------

def extended_features(data, survey=None, cal=None, act=None):
    t = data.table
    cols = {}
    mats = sd.lag_matrices(t, data.nights, max_lag=56)

    # W: 창 길이
    for q in QS:
        M = mats[q]
        for k in range(8, 57):
            cols[f'x_{q}_lag{k}'] = M[:, k - 1].astype(F32)
        L56 = sd.level(M, 56, 'med', 'in', 7)
        L28 = sd.level(M, 28, 'med', 'in', 7)
        cols[f'x_{q}_dR14_L56'] = (_mean(M[:, :14]) - L56).astype(F32)
        cols[f'x_{q}_dR28_L56'] = (_mean(M[:, :28]) - L56).astype(F32)
        cols[f'x_{q}_dR14_L28'] = (_mean(M[:, :14]) - L28).astype(F32)

    # T, H: 달력
    cal = calendar_days() if cal is None else cal
    cidx = cal.set_index('date')
    d0 = t['date']
    d1 = t['date'] + pd.Timedelta(days=1)
    for c in ['h_class', 'h_finals', 'h_break', 'h_winter', 'h_summer', 'h_term_week', 'h_days_to_finals']:
        cols[f'x_{c}'] = cidx[c].reindex(d1).to_numpy(dtype=F32)      # 깨어나는 날(D+1) 기준
    cols['x_h_fed_today'] = cidx['h_fed'].reindex(d0).to_numpy(dtype=F32)
    cols['x_h_fed_tomorrow'] = cidx['h_fed'].reindex(d1).to_numpy(dtype=F32)
    cols['x_t_off_tomorrow'] = cidx['off'].reindex(d1).to_numpy(dtype=F32)
    cols['x_t_off_today'] = cidx['off'].reindex(d0).to_numpy(dtype=F32)
    # 같은 종류 밤(내일 쉬는지)끼리의 개인 평소 (과거 W밤, W=28은 접미사 없음)
    offall = np.stack([cidx['off'].reindex(t['date'] - pd.Timedelta(days=k - 1)).to_numpy(dtype=float)
                       for k in range(1, 57)], axis=1)                 # 밤 D−k의 다음 날 = D−k+1
    today_off = cols['x_t_off_tomorrow'].astype(float)
    vac = (cidx['h_break'] + cidx['h_winter'] + cidx['h_summer']).clip(upper=1)
    for k in range(1, 57):                                   # 밤 D−k의 깨어나는 날 = D−k+1
        cols[f'x_offlag{k}'] = offall[:, k - 1].astype(F32)
        cols[f'x_vaclag{k}'] = vac.reindex(t['date'] - pd.Timedelta(days=k - 1)).to_numpy(dtype=F32)
    for (q, W) in [(q, W) for W in (28, 42, 56) for q in QS]:
        sfx = '' if W == 28 else f'_w{W}'
        offmat = offall[:, :W]
        M = mats[q][:, :W]
        Mo = np.where(offmat == 1, M, np.nan)
        Mw = np.where(offmat == 0, M, np.nan)
        Lo, Lw = _med(Mo), _med(Mw)
        Lo[np.sum(~np.isnan(Mo), 1) < 3] = np.nan
        Lw[np.sum(~np.isnan(Mw), 1) < 3] = np.nan
        same = np.where(today_off == 1, Lo, Lw)
        L28 = sd.level(mats[q], 28, 'med', 'in', 7)
        cols[f'x_{q}_Lsame{sfx}'] = same.astype(F32)
        cols[f'x_{q}_gap_same{sfx}'] = (same - L28).astype(F32)
        cols[f'x_{q}_Loffwork{sfx}'] = (Lo - Lw).astype(F32)
        cols[f'x_{q}_dR7_Lsame{sfx}'] = (_mean(mats[q][:, :7]) - same).astype(F32)

    # A: 활동 (D−1..D−7, 평소 28일 대비)
    act = raw_activity() if act is None else act
    cols.update({f'x_a_{k}': v for k, v in day_lags(t, act, ACT, range(1, 8)).items()})
    for c in ['steps', 'hr', 'sed', 'modvig']:
        L = day_lags(t, act, [c], range(1, 29))
        M = np.stack([L[f'{c}_l{k}'] for k in range(1, 29)], axis=1).astype(float)
        Lm = _med(M)
        Lm[np.sum(~np.isnan(M), 1) < 7] = np.nan
        cols[f'x_a_{c}_L28'] = Lm.astype(F32)
        cols[f'x_a_{c}_dR3'] = (_mean(M[:, :3]) - Lm).astype(F32)

    # N: 수면 부가
    nm = night_metrics(data)
    cols.update({f'x_{k}': v for k, v in night_lags(t, nm, N_COLS, range(1, 8)).items()})
    mid = np.stack([cols[f'x_n_mid_l{k}'] for k in range(1, 8)], axis=1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sdm = np.nanstd(mid, axis=1, ddof=1)
    sdm[np.sum(~np.isnan(mid), 1) < 3] = np.nan
    cols['x_n_mid_sd7'] = sdm.astype(F32)

    # P: 개인 정적
    sv = survey_static() if survey is None else survey
    m = t[['pid']].merge(sv, on='pid', how='left')
    for c in [c for g in P_GROUPS.values() for c in g]:
        cols[f'x_{c}'] = m[c].to_numpy(dtype=F32)
    return pd.DataFrame(cols, index=t.index)


def feat_extended(data):
    return extended_features(data)


# ---------- 블록 (계열의 최소 단위). 목표 t → 열 목록 ----------

def _q(t, names):
    return [sd.col(t, n) for n in names]


def own_lags(q, ks):
    return [f'{fd.LAG_PREFIX[q]}_lag{k}' if k <= 7 else f'x_{q}_lag{k}' for k in ks]


ACT_L3 = lambda cs: [f'x_a_{c}_l{k}' for c in cs for k in (1, 2, 3)]
NIGHT_L3 = lambda cs: [f'x_{c}_l{k}' for c in cs for k in (1, 2, 3)]

BLOCKS = {
    # sleep_delta 쪽
    'delta': lambda t: _q(t, [sd.D0]),
    'level': lambda t: _q(t, [sd.L0]),
    'shapes': lambda t: _q(t, sd.SHAPES),
    'delta_z': lambda t: _q(t, [sd.dname(form='z')]),
    'delta_multiR': lambda t: _q(t, [sd.dname(r=1), sd.dname(r=3)]),
    'cross_delta': lambda t: [sd.col(q, sd.D0) for q in QS if q != t],
    # W
    'lag8_14': lambda t: own_lags(t, range(8, 15)),
    'lag15_28': lambda t: own_lags(t, range(15, 29)),
    'other_lags7': lambda t: [c for q in QS if q != t for c in own_lags(q, range(1, 8))],
    'other_lags14': lambda t: [c for q in QS if q != t for c in own_lags(q, range(1, 15))],
    'dR14_28': lambda t: [f'x_{t}_dR14_L56', f'x_{t}_dR28_L56', f'x_{t}_dR14_L28'],
    # T
    'off': lambda t: ['x_t_off_tomorrow', 'x_t_off_today'],
    'offbase': lambda t: [f'x_{t}_Lsame', f'x_{t}_gap_same', f'x_{t}_Loffwork', f'x_{t}_dR7_Lsame'],
    # H
    'fed': lambda t: ['x_h_fed_today', 'x_h_fed_tomorrow'],
    'term': lambda t: ['x_h_class', 'x_h_finals', 'x_h_break', 'x_h_winter', 'x_h_summer',
                       'x_h_term_week', 'x_h_days_to_finals'],
    # A
    'steps': lambda t: ACT_L3(['steps']),
    'steps7': lambda t: [f'x_a_steps_l{k}' for k in range(1, 8)],
    'steps_delta': lambda t: ['x_a_steps_L28', 'x_a_steps_dR3'],
    'hr': lambda t: ACT_L3(['hr']) + ['x_a_hr_L28', 'x_a_hr_dR3'],
    'hrsd': lambda t: ACT_L3(['hrsd']),
    'sed': lambda t: ACT_L3(['sed']) + ['x_a_sed_L28', 'x_a_sed_dR3'],
    'active': lambda t: ACT_L3(['light', 'modvig']) + ['x_a_modvig_L28', 'x_a_modvig_dR3'],
    'zones': lambda t: ACT_L3(['fatburn', 'cardiopeak']),
    'floors': lambda t: ACT_L3(['floors']),
    'wear': lambda t: ACT_L3(['wear']),
    # N
    'eff': lambda t: NIGHT_L3(['n_eff']),
    'nap': lambda t: NIGHT_L3(['n_extra']),
    'frag': lambda t: NIGHT_L3(['n_blocks', 'n_pieces']),
    'mid': lambda t: NIGHT_L3(['n_mid']) + ['x_n_mid_sd7'],
    'inbed': lambda t: NIGHT_L3(['n_inbed']),
    # P
    **{f'p_{g}': (lambda cs: (lambda t: [f'x_{c}' for c in cs]))(cs) for g, cs in P_GROUPS.items()},
}

FAMILY = {'W': ['lag8_14', 'lag15_28', 'other_lags7', 'other_lags14', 'dR14_28'],
          'T': ['off', 'offbase'], 'H': ['fed', 'term'],
          'A': ['steps', 'steps7', 'steps_delta', 'hr', 'hrsd', 'sed', 'active', 'zones', 'floors', 'wear'],
          'N': ['eff', 'nap', 'frag', 'mid', 'inbed'],
          'P': [f'p_{g}' for g in P_GROUPS]}

# 전진 선택 후보 (겹치는 블록은 하나만: steps7은 steps를 포함, other_lags14는 other_lags7을 포함 → 큰 쪽만)
GREEDY_BLOCKS = ['delta', 'level', 'shapes', 'delta_z', 'delta_multiR', 'cross_delta',
                 'lag8_14', 'lag15_28', 'other_lags7', 'dR14_28', 'off', 'offbase', 'fed', 'term',
                 'steps7', 'steps_delta', 'hr', 'hrsd', 'sed', 'active', 'zones', 'floors', 'wear',
                 'eff', 'nap', 'frag', 'mid', 'inbed', 'p_demo', 'p_psqi', 'p_mental', 'p_life']


def feats_of(blocks, replace_base=None):
    """블록 목록 → 목표별 입력 함수. replace_base = 양 부분집합이면 기본 입력 대신 그 양들의 lag1..7 + dow."""
    def f(t):
        if replace_base is None:
            cols = fd.base_features(t)
        else:
            cols = [c for q in replace_base for c in own_lags(q, range(1, 8))] + ['dow']
        for b in blocks:
            for c in BLOCKS[b](t):
                if c not in cols:
                    cols.append(c)
        return cols
    return f


def exp(name, blocks, seed=0, replace_base=None):
    return fe.Experiment(name + (f'_s{seed}' if seed else ''), sd._cb(seed), feats_of(blocks, replace_base))


def family_exps():
    """계열 하나씩 (기본 입력 + 블록) + 계열 전체."""
    e = []
    for fam, bl in FAMILY.items():
        for b in bl:
            e.append(exp(f'{fam}_{b}', [b]))
        e.append(exp(f'{fam}_ALL', bl))
    return e


def subset_exps():
    """Q: 어떤 양의 과거를 넣나. 기본 입력을 대체한다 (목표 자신의 과거가 빠질 수도 있음)."""
    from itertools import combinations
    e = []
    for r in (1, 2, 3):
        for sub in combinations(QS, r):
            tag = ''.join(QL[q] for q in sub)
            e.append(exp(f'Q_{tag}', [], replace_base=list(sub)))
            e.append(fe.Experiment(f'Q_{tag}_delta', sd._cb(0),
                                   (lambda sub: lambda t: feats_of([], list(sub))(t)
                                    + [sd.col(q, sd.D0) for q in sub])(sub)))
    return e


ALL_EXT = [b for b in GREEDY_BLOCKS]


def combo2_exps():
    """미리 정한 큰 조합."""
    core = ['delta', 'off', 'fed', 'term']
    return [exp('K2_core', core),
            exp('K2_core_act', core + FAMILY['A']),
            exp('K2_core_night', core + FAMILY['N']),
            exp('K2_core_person', core + FAMILY['P']),
            exp('K2_core_wide', core + ['lag8_14', 'lag15_28', 'other_lags14', 'dR14_28']),
            exp('K2_everything', ALL_EXT + ['other_lags14', 'steps'])]


def offbase_w(W):
    sfx = '' if W == 28 else f'_w{W}'
    return lambda t: [f'x_{t}_Lsame{sfx}', f'x_{t}_gap_same{sfx}', f'x_{t}_Loffwork{sfx}', f'x_{t}_dR7_Lsame{sfx}']


def window_exps(weeks=(4, 6, 8)):
    """지난 W주 원값 입력 (자기 양 / 세 양) × (그대로 / + 같은 창의 쉬는 날 기준선 / + 학사·공휴일)."""
    out = []
    for wk in weeks:
        n = wk * 7
        def f(all3, ob, cal, n=n):
            def g(t):
                cols = [c for q in (QS if all3 else [t]) for c in own_lags(q, range(1, n + 1))] + ['dow']
                if ob:
                    cols += offbase_w(n)(t)
                if cal:
                    cols += BLOCKS['term'](t) + BLOCKS['fed'](t)
                return cols
            return g
        p = f'L{wk}w'
        if wk == 4:
            continue                      # 4주는 window28_exps (이미 실행한 이름 유지)
        out += [fe.Experiment(f'{p}_lag{n}', fe.CatBoostModel, f(False, False, False)),
                fe.Experiment(f'{p}_lag{n}_all3', fe.CatBoostModel, f(True, False, False)),
                fe.Experiment(f'{p}_lag{n}_offbase', fe.CatBoostModel, f(False, True, False)),
                fe.Experiment(f'{p}_lag{n}_offbase_term', fe.CatBoostModel, f(False, True, True)),
                fe.Experiment(f'{p}_lag{n}_all3_offbase_term', fe.CatBoostModel, f(True, True, True)),
                fe.Experiment(f'{p}_lag7_offbase{n}_term', fe.CatBoostModel,
                              lambda t, n=n: fd.base_features(t) + offbase_w(n)(t) + BLOCKS['term'](t) + BLOCKS['fed'](t))]
    return out


def tag_exps():
    """과거 밤마다 '쉬는 날 전이었나'·'방학 중이었나' 표시를 원값 옆에 직접 붙임 (요약하지 않음)."""
    def f(n, offbase, cal):
        def g(t):
            cols = [c for q in QS for c in own_lags(q, range(1, n + 1))] + ['dow']
            cols += [f'x_offlag{k}' for k in range(1, n + 1)] + [f'x_vaclag{k}' for k in range(1, n + 1)]
            if offbase:
                cols += offbase_w(n)(t)
            if cal:
                cols += BLOCKS['term'](t) + BLOCKS['fed'](t)
            return cols
        return g
    return [fe.Experiment('TAG8w_all3', fe.CatBoostModel, f(56, False, False)),
            fe.Experiment('TAG8w_all3_term', fe.CatBoostModel, f(56, False, True)),
            fe.Experiment('TAG8w_all3_offbase_term', fe.CatBoostModel, f(56, True, True)),
            fe.Experiment('TAG4w_all3_term', fe.CatBoostModel, f(28, False, True)),
            fe.Experiment('TAG4w_all3_offbase_term', fe.CatBoostModel, f(28, True, True))]


def window28_exps():
    """지난 4주(28밤) 원값을 입력으로: 기본 7밤 → 28밤, 세 양 28밤, + 쉬는 날 기준선, + 학사·공휴일."""
    lag28 = lambda q: own_lags(q, range(1, 29))
    def f(extra_blocks, all3=False):
        def g(t):
            qs = QS if all3 else [t]
            cols = [c for q in qs for c in lag28(q)] + ['dow']
            for b in extra_blocks:
                cols += [c for c in BLOCKS[b](t) if c not in cols]
            return cols
        return g
    return [fe.Experiment('L4w_lag28', fe.CatBoostModel, f([])),
            fe.Experiment('L4w_lag28_all3', fe.CatBoostModel, f([], True)),
            fe.Experiment('L4w_lag28_offbase', fe.CatBoostModel, f(['offbase'])),
            fe.Experiment('L4w_lag28_offbase_term', fe.CatBoostModel, f(['offbase', 'term', 'fed'])),
            fe.Experiment('L4w_lag28_all3_offbase_term', fe.CatBoostModel, f(['offbase', 'term', 'fed'], True)),
            fe.Experiment('L4w_lag7_offbase28_term', fe.CatBoostModel,
                          lambda t: fd.base_features(t) + offbase_w(28)(t) + BLOCKS['term'](t) + BLOCKS['fed'](t))]


def phase3_exps():
    out = {}
    for e in family_exps() + subset_exps() + combo2_exps() + window28_exps() + window_exps() + tag_exps():
        if e.name in out:
            raise ValueError(e.name)
        out[e.name] = e
    return out
