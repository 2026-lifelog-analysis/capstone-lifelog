"""PMData -> person-day table for the 춘자/LOV3 reproduction.

Usage: python3 -I pm_build.py <pmdata_root> <out.pkl>

Alignment (fixed before running):
- wellness is answered in the morning; its local date = UTC timestamp + 1h (Norway, CET), normalized.
- day t row uses: the main sleep that ended on day t (Fitbit dateOfSleep == t) and activity of day t-1 and earlier.
- self-reported sleep_duration_h and other wellness items are NOT features.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
SUBJ = ['fatigue', 'mood', 'readiness', 'sleep_quality', 'soreness', 'stress']


def load_json(p):
    with open(p) as f:
        return json.load(f)


def daily_value_series(p):
    """resting_heart_rate / *_minutes.json -> Series indexed by date."""
    rows = []
    for r in load_json(p):
        v = r['value']
        if isinstance(v, dict):
            v = v.get('value')
        rows.append((pd.to_datetime(r['dateTime']).normalize(), float(v) if v not in (None, '') else np.nan))
    s = pd.DataFrame(rows, columns=['date', 'v']).groupby('date')['v'].last()
    return s


people = []
for pdir in sorted(glob.glob(os.path.join(root, 'p[0-9][0-9]'))):
    pid = os.path.basename(pdir)
    # --- labels
    w = pd.read_csv(os.path.join(pdir, 'pmsys', 'wellness.csv'))
    ts = pd.to_datetime(w['effective_time_frame'], utc=True).dt.tz_convert(None) + pd.Timedelta(hours=1)
    w['date'] = ts.dt.normalize()
    w['resp_hour'] = ts.dt.hour + ts.dt.minute / 60
    w = w.sort_values('resp_hour').groupby('date', as_index=False).first()  # one response per day (earliest)
    lab = w[['date', 'resp_hour'] + SUBJ].copy()

    # --- sleep (main sleep, by wake date)
    sl = []
    for r in load_json(os.path.join(pdir, 'fitbit', 'sleep.json')):
        if not r.get('mainSleep'):
            continue
        st = pd.to_datetime(r['startTime']); en = pd.to_datetime(r['endTime'])
        sl.append(dict(date=pd.to_datetime(r['dateOfSleep']).normalize(), tst_min=r['minutesAsleep'],
                       eff=r['efficiency'], awake_min=r['minutesAwake'], tib_min=r['timeInBed'],
                       bed_h=(st.hour + st.minute / 60 + (24 if st.hour < 12 else 0)),  # hours after previous noon
                       wake_h=en.hour + en.minute / 60))
    sl = pd.DataFrame(sl).groupby('date').last() if sl else pd.DataFrame()
    ss_p = os.path.join(pdir, 'fitbit', 'sleep_score.csv')
    if os.path.exists(ss_p):
        ss = pd.read_csv(ss_p)
        ss['date'] = (pd.to_datetime(ss['timestamp'], utc=True).dt.tz_convert(None) + pd.Timedelta(hours=1)).dt.normalize()
        ss = ss.groupby('date')[['overall_score', 'deep_sleep_in_minutes', 'restlessness']].last()
        sl = sl.join(ss, how='outer')

    # --- activity (calendar day)
    steps = pd.DataFrame(load_json(os.path.join(pdir, 'fitbit', 'steps.json')))
    steps['date'] = pd.to_datetime(steps['dateTime']).dt.normalize()
    steps['value'] = pd.to_numeric(steps['value'], errors='coerce')
    act = steps.groupby('date')['value'].sum().rename('steps').to_frame()
    act['steps_min_cov'] = steps.groupby('date')['value'].size()  # minutes present
    for name in ['sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 'very_active_minutes', 'resting_heart_rate']:
        p = os.path.join(pdir, 'fitbit', f'{name}.json')
        if os.path.exists(p):
            act[name] = daily_value_series(p)
    act.loc[act['steps_min_cov'] < 600, 'steps'] = np.nan  # too little coverage -> missing
    act = act.drop(columns='steps_min_cov')

    # --- daily calendar frame
    lo = min(lab.date.min(), act.index.min()); hi = max(lab.date.max(), act.index.max())
    cal = pd.DataFrame(index=pd.date_range(lo, hi, freq='D'))
    cal.index.name = 'date'
    cal = cal.join(sl, how='left').join(act, how='left')

    sleep_cols = [c for c in sl.columns]
    act_cols = list(act.columns)
    # activity of day t-1 for row t
    A = cal[act_cols].shift(1)
    A.columns = [f'a_{c}' for c in act_cols]
    S = cal[sleep_cols].copy()  # sleep that ended on day t
    S.columns = [f's_{c}' for c in sleep_cols]
    S_prev = cal[sleep_cols].shift(1)  # previous night's sleep (for the TST label)
    S_prev.columns = [f'sp_{c}' for c in sleep_cols]
    base = pd.concat([A, S, S_prev], axis=1)
    feats = [base]
    # rolling past-only summaries on the activity series and previous-night sleep series
    roll_src = pd.concat([A, S_prev], axis=1)
    for k in (3, 7):
        r = roll_src.rolling(k, min_periods=max(2, k // 2)).mean(); r.columns = [f'{c}_m{k}' for c in roll_src.columns]
        feats.append(r)
    mu28 = roll_src.rolling(28, min_periods=10).mean(); sd28 = roll_src.rolling(28, min_periods=10).std()
    z7 = (roll_src.rolling(7, min_periods=3).mean() - mu28) / sd28.replace(0, np.nan)
    z7.columns = [f'{c}_z7v28' for c in roll_src.columns]
    feats.append(z7)
    F = pd.concat(feats, axis=1)
    F['dow'] = F.index.dayofweek
    F = F.reset_index()

    d = lab.merge(F, on='date', how='left')
    d['tst_label'] = np.where(d['s_tst_min'].notna(), ((d['s_tst_min'] >= 420) & (d['s_tst_min'] <= 540)).astype(float), np.nan)
    d.insert(0, 'pid', pid)
    people.append(d)

df = pd.concat(people, ignore_index=True)
for c in SUBJ:
    df[c + '_label'] = (df[c] > df.groupby('pid')[c].transform('mean')).astype(float)
    df.loc[df[c].isna(), c + '_label'] = np.nan
df.to_pickle(out)
print(df.shape, df.pid.nunique(), 'people')
print(df[[c + '_label' for c in SUBJ] + ['tst_label']].mean().round(3).to_dict())
print('feature cols', len([c for c in df.columns if c.startswith(('a_', 's_', 'sp_')) or c == 'dow']))
