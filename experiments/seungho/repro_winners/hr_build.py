"""Human Rhythms (Cleaned_Raw_Data) -> person-day table (contract in common.py).

Usage: python3 -I hr_build.py <HumanRhythms_root> <out.pkl>
       HR_LIMIT=5 python3 -I hr_build.py ...   # first 5 participants only (quick check)
       HR_SLICE=0:80 python3 -I hr_build.py <root> part1.pkl   # build a slice only (no filtering / labels)
       HR_MERGE=part1.pkl,part2.pkl python3 -I hr_build.py <root> out.pkl   # merge slices and finalize

Alignment (fixed before running):
- the daily survey is answered in the evening (mostly 17-21h) about that day; one response per day (earliest).
- day t row uses: main sleep that ended on day t, activity / phone use of day t-1 and earlier.
- labels: sleep_quality (Q2), cognitive_energy (Q3), physical_energy (Q4), emotional_energy (Q5), 1-5,
  binarized as value > that person's whole-period mean; tst_label = main sleep 7-9 h.
- events (sensor-independent): ev_exam = 'Final Exams' days, ev_break = listed breaks + winter break
  (day after the last exam day .. day before the January term).
- scales (person level): b_psqi_q_{pre,post} = PSQI overall sleep-quality item (0 very good .. 3 very bad),
  b_sf12_gh_{pre,post} = SF-12 general health item (1 excellent .. 5 poor). Columns are found by their answer
  vocabulary because the post-study PSQI file has its columns in a different order from its header.
"""
import csv, glob, os, sys
import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
RAW = os.path.join(root, 'Cleaned_Raw_Data')
LIMIT = int(os.environ.get('HR_LIMIT', '0'))
SLICE = os.environ.get('HR_SLICE', '')
MERGE = os.environ.get('HR_MERGE', '')
ITEMS = {'Q2:': 'sleep_quality', 'Q3:': 'cognitive_energy', 'Q4:': 'physical_energy', 'Q5:': 'emotional_energy'}


def read_minute_daily(folder, pid, how='sum'):
    p = os.path.join(RAW, 'Fitbit', folder, f'{pid}.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=['dateTime', 'value'], dtype={'dateTime': str})
    d['date'] = pd.to_datetime(d['dateTime'].str[:10])
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    g = d.groupby('date')['value']
    return pd.DataFrame({'sum': g.sum(), 'n': g.size()})


def screen_daily(pid):
    """AWARE screen events (UTC ms) -> unlock counts per local day (America/New_York)."""
    p = os.path.join(RAW, 'Aware', 'Screen', f'{pid}.csv')
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=['timestamp', 'screen_status'])
    d = d[d['screen_status'] == 3]
    if d.empty:
        return None
    t = pd.to_datetime(d['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    s = pd.DataFrame({'date': t.dt.normalize(), 'night': t.dt.hour < 5})
    return pd.DataFrame({'unlocks': s.groupby('date').size(), 'night_unlocks': s.groupby('date')['night'].sum()})


def norm(x):
    return str(x).strip().strip('.').lower()


def scale_items():
    psqi_map = {'very good': 0, 'fairly good': 1, 'fairly bad': 2, 'very bad': 3}
    gh_map = {'excellent': 1, 'very good': 2, 'good': 3, 'fair': 4, 'poor': 5}
    res = {}
    for phase in ('pre', 'post'):
        for name, fname, vocab in [('psqi_q', 'PSQI.csv', psqi_map), ('sf12_gh', 'SF-12.csv', gh_map)]:
            p = os.path.join(RAW, 'Surveys', f'{phase.capitalize()}-Study_Baselines', fname)
            if not os.path.exists(p):
                continue
            rows = list(csv.reader(open(p, encoding='utf-8-sig')))
            body = rows[1:]
            # pick the column whose non-empty answers are all in the vocabulary (excluding PID)
            best, best_hits = None, 0
            for i in range(1, len(rows[0])):
                vals = [norm(r[i]) for r in body if len(r) > i and r[i].strip()]
                hits = sum(v in vocab for v in vals)
                if vals and hits / len(vals) > 0.9 and hits > best_hits:
                    best, best_hits = i, hits
            if best is None:
                continue
            for r in body:
                v = norm(r[best]) if len(r) > best else ''
                res.setdefault(r[0], {})[f'b_{name}_{phase}'] = vocab.get(v, np.nan)
    return pd.DataFrame.from_dict(res, orient='index')


def events():
    p = os.path.join(root, 'Additional_Data', 'University_Events.csv')
    e = pd.read_csv(p, encoding='utf-8-sig')
    e.columns = ['date', 'desc']
    e['date'] = pd.to_datetime(e['date'], format='%m/%d/%y')
    exam = set(e.loc[e.desc.str.contains('Final Exams'), 'date'])
    brk = set(e.loc[e.desc.str.contains('Break'), 'date'])
    for y in (2021, 2022):
        last_exam = max(d for d in exam if d.year == y and d.month == 12)
        jan = min(d for d in e.loc[e.desc.str.contains('January Term') & e.desc.str.startswith('Start'), 'date'] if d.year == y + 1)
        brk |= set(pd.date_range(last_exam + pd.Timedelta(days=1), jan - pd.Timedelta(days=1)))
    return exam, brk


exam_days, break_days = events()
scales = scale_items()
pids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RAW, 'Surveys', 'Daily_Surveys', '*.csv')))
if LIMIT:
    pids = pids[:LIMIT]
if SLICE:
    a, b = (int(x) for x in SLICE.split(':')); pids = pids[a:b]
if MERGE:
    pids = []

people = []
for k, pid in enumerate(pids):
    sv = pd.read_csv(os.path.join(RAW, 'Surveys', 'Daily_Surveys', f'{pid}.csv'))
    if sv.empty:
        continue
    cols = {c: ITEMS[p] for c in sv.columns for p in ITEMS if c.startswith(p)}
    sv = sv.rename(columns=cols)
    sv['t'] = pd.to_datetime(sv['StartDate']); sv['date'] = sv['t'].dt.normalize(); sv['resp_hour'] = sv['t'].dt.hour
    lab = sv.sort_values('t').groupby('date', as_index=False).first()[['date', 'resp_hour'] + list(ITEMS.values())]

    # sleep
    sp = os.path.join(RAW, 'Fitbit', 'Sleep_Summary', f'{pid}.csv')
    sl = pd.DataFrame()
    if os.path.exists(sp) and os.path.getsize(sp) > 0:
        s = pd.read_csv(sp)
        s = s[s['is_main_sleep'].astype(str).str.lower().isin(['true', '1'])].copy()
        if len(s):
            st = pd.to_datetime(s['local_start_date_time']); en = pd.to_datetime(s['local_end_date_time'])
            s['date'] = en.dt.normalize()
            s['bed_h'] = st.dt.hour + st.dt.minute / 60 + np.where(st.dt.hour < 12, 24, 0)
            s['wake_h'] = en.dt.hour + en.dt.minute / 60
            s = s.rename(columns={'minutes_asleep': 'tst_min', 'efficiency': 'eff', 'minutes_awake': 'awake_min', 'minutes_in_bed': 'tib_min'})
            sl = s.sort_values('tst_min').groupby('date')[['tst_min', 'eff', 'awake_min', 'tib_min', 'bed_h', 'wake_h']].last()
            sl.loc[(sl.tst_min < 120) | (sl.tst_min > 960)] = np.nan

    # activity
    act = pd.DataFrame()
    stp = read_minute_daily('Steps', pid)
    if stp is not None:
        act['steps'] = stp['sum'].where((stp['sum'] >= 100) & (stp['sum'] <= 45000))
    for folder, name in [('Minutes_Sedentary', 'sedentary_minutes'), ('Minutes_Very_Active', 'very_active_minutes')]:
        m = read_minute_daily(folder, pid)
        if m is not None:
            act[name] = m['sum']
    hr = os.path.join(RAW, 'Fitbit', 'Heart_Rate_Summary', f'{pid}.csv')
    if os.path.exists(hr):
        h = pd.read_csv(hr, usecols=['local_date_time', 'heartrate_daily_restinghr'])
        h['date'] = pd.to_datetime(h['local_date_time']).dt.normalize()
        act['resting_heart_rate'] = h.groupby('date')['heartrate_daily_restinghr'].last()
    scr = screen_daily(pid)
    if scr is not None:
        act = act.join(scr, how='outer')
    if act.empty or sl.empty:
        print(pid, 'skipped (no activity or sleep)'); continue
    if 'steps' in act:
        act.loc[act['steps'].isna(), [c for c in ('sedentary_minutes', 'very_active_minutes') if c in act]] = np.nan

    lo, hi = lab.date.min(), lab.date.max()
    cal = pd.DataFrame(index=pd.date_range(lo - pd.Timedelta(days=35), hi, freq='D')); cal.index.name = 'date'
    cal = cal.join(sl, how='left').join(act, how='left')
    act_cols, sleep_cols = list(act.columns), list(sl.columns)
    A = cal[act_cols].shift(1); A.columns = [f'a_{c}' for c in act_cols]
    S = cal[sleep_cols].copy(); S.columns = [f's_{c}' for c in sleep_cols]
    SP = cal[sleep_cols].shift(1); SP.columns = [f'sp_{c}' for c in sleep_cols]
    roll = pd.concat([A, SP], axis=1)
    feats = [A, S, SP]
    for kk in (3, 7):
        r = roll.rolling(kk, min_periods=max(2, kk // 2)).mean(); r.columns = [f'{c}_m{kk}' for c in roll.columns]; feats.append(r)
    z7 = (roll.rolling(7, min_periods=3).mean() - roll.rolling(28, min_periods=10).mean()) / roll.rolling(28, min_periods=10).std().replace(0, np.nan)
    z7.columns = [f'{c}_z7v28' for c in roll.columns]; feats.append(z7)
    F = pd.concat(feats, axis=1)
    for name, col in {'steps': 'a_steps', 'sedentary': 'a_sedentary_minutes', 'rhr': 'a_resting_heart_rate',
                      'unlocks': 'a_unlocks', 'tst': 'sp_tst_min', 'bed': 'sp_bed_h'}.items():
        if f'{col}_z7v28' in F:
            F[f'z_{name}'] = F[f'{col}_z7v28']
    F['dow'] = F.index.dayofweek
    F = F.loc[lo:hi].reset_index()
    d = F.merge(lab, on='date', how='left')
    d['tst_label'] = np.where(d['s_tst_min'].notna() & d['sleep_quality'].notna(),
                              ((d['s_tst_min'] >= 420) & (d['s_tst_min'] <= 540)).astype(float), np.nan)
    d['ev_exam'] = d['date'].isin(exam_days).astype(int)
    d['ev_break'] = d['date'].isin(break_days).astype(int)
    if pid in scales.index:
        for c in scales.columns:
            d[c] = scales.loc[pid, c]
    d.insert(0, 'pid', pid)
    people.append(d)
    if (k + 1) % 20 == 0:
        print(f'{k + 1}/{len(pids)} done', flush=True)

if SLICE:
    pd.concat(people, ignore_index=True).to_pickle(out); print('slice saved', SLICE, len(people), 'people'); sys.exit(0)
df = pd.concat([pd.read_pickle(p) for p in MERGE.split(',')], ignore_index=True) if MERGE else pd.concat(people, ignore_index=True)
# people need enough labelled days for a 50/50 split (same rule as the earlier HR work: >= 60 labelled days)
nlab = df.groupby('pid')['sleep_quality'].transform(lambda x: x.notna().sum())
df = df[nlab >= 60].copy()
for c in ITEMS.values():
    df[c + '_label'] = np.where(df[c].notna(), (df[c] > df.groupby('pid')[c].transform('mean')).astype(float), np.nan)
df.to_pickle(out)
lab_cols = [c for c in df.columns if c.endswith('_label')]
print(df.shape, df.pid.nunique(), 'people (>=60 labelled days);', 'labelled rows', int(df['sleep_quality'].notna().sum()))
print('positive rate', df[lab_cols].mean().round(3).to_dict())
print('exam days', int(df.ev_exam.sum()), 'break days', int(df.ev_break.sum()), 'z cols', [c for c in df.columns if c.startswith('z_')])
print('scales', [c for c in df.columns if c.startswith('b_')], 'people with both PSQI', int(df.groupby('pid')[['b_psqi_q_pre', 'b_psqi_q_post']].first().dropna().shape[0]) if 'b_psqi_q_post' in df else 0)
