"""PMData -> person-day table (contract in common.py).

Usage: python3 -I pm_build.py <pmdata_root> <out.pkl>

Alignment (fixed before running):
- wellness is answered in the morning; local date = UTC timestamp + 1h (Norway, CET), normalized.
- day t row uses: the main sleep that ended on day t (Fitbit dateOfSleep == t) and activity of day t-1 and earlier.
- self-reported sleep_duration_h and other wellness items are NOT features.
- one row per calendar day from a person's first to last wellness response (days without a response keep NaN labels).
- ev_injury: day of a pmsys injury report that lists at least one injury (sensor-independent event).
"""
import ast, glob, json, os, sys
import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
SUBJ = ['fatigue', 'mood', 'readiness', 'sleep_quality', 'soreness', 'stress']


def load_json(p):
    with open(p) as f:
        return json.load(f)


def daily_value_series(p):
    rows = []
    for r in load_json(p):
        v = r['value']
        if isinstance(v, dict):
            v = v.get('value')
        rows.append((pd.to_datetime(r['dateTime']).normalize(), float(v) if v not in (None, '') else np.nan))
    return pd.DataFrame(rows, columns=['date', 'v']).groupby('date')['v'].last()


def local_date(s):
    return (pd.to_datetime(s, utc=True).dt.tz_convert(None) + pd.Timedelta(hours=1)).dt.normalize()


def add_rolling(cal, src_cols, prefix_map):
    """cal: calendar frame. Returns features for row t (activity of t-1, sleep of t, previous-night sleep)."""
    act_cols, sleep_cols = src_cols
    A = cal[act_cols].shift(1); A.columns = [f'a_{c}' for c in act_cols]
    S = cal[sleep_cols].copy(); S.columns = [f's_{c}' for c in sleep_cols]
    SP = cal[sleep_cols].shift(1); SP.columns = [f'sp_{c}' for c in sleep_cols]
    roll_src = pd.concat([A, SP], axis=1)
    feats = [A, S, SP]
    for k in (3, 7):
        r = roll_src.rolling(k, min_periods=max(2, k // 2)).mean(); r.columns = [f'{c}_m{k}' for c in roll_src.columns]
        feats.append(r)
    mu28 = roll_src.rolling(28, min_periods=10).mean(); sd28 = roll_src.rolling(28, min_periods=10).std()
    z7 = (roll_src.rolling(7, min_periods=3).mean() - mu28) / sd28.replace(0, np.nan)
    z7.columns = [f'{c}_z7v28' for c in roll_src.columns]
    feats.append(z7)
    F = pd.concat(feats, axis=1)
    # tracker inputs: a few interpretable 7-vs-28-day deviations (past only)
    for name, col in prefix_map.items():
        if f'{col}_z7v28' in F:
            F[f'z_{name}'] = F[f'{col}_z7v28']
    return F


people = []
for pdir in sorted(glob.glob(os.path.join(root, 'p[0-9][0-9]'))):
    pid = os.path.basename(pdir)
    w = pd.read_csv(os.path.join(pdir, 'pmsys', 'wellness.csv'))
    ts = pd.to_datetime(w['effective_time_frame'], utc=True).dt.tz_convert(None) + pd.Timedelta(hours=1)
    w['date'] = ts.dt.normalize(); w['resp_hour'] = ts.dt.hour + ts.dt.minute / 60
    w = w.sort_values('resp_hour').groupby('date', as_index=False).first()
    lab = w[['date', 'resp_hour'] + SUBJ].copy()

    sl = []
    for r in load_json(os.path.join(pdir, 'fitbit', 'sleep.json')):
        if not r.get('mainSleep'):
            continue
        st = pd.to_datetime(r['startTime']); en = pd.to_datetime(r['endTime'])
        sl.append(dict(date=pd.to_datetime(r['dateOfSleep']).normalize(), tst_min=r['minutesAsleep'], eff=r['efficiency'],
                       awake_min=r['minutesAwake'], tib_min=r['timeInBed'],
                       bed_h=st.hour + st.minute / 60 + (24 if st.hour < 12 else 0), wake_h=en.hour + en.minute / 60))
    sl = pd.DataFrame(sl).groupby('date').last()
    ss_p = os.path.join(pdir, 'fitbit', 'sleep_score.csv')
    if os.path.exists(ss_p):
        ss = pd.read_csv(ss_p); ss['date'] = local_date(ss['timestamp'])
        sl = sl.join(ss.groupby('date')[['overall_score', 'deep_sleep_in_minutes', 'restlessness']].last(), how='outer')

    steps = pd.DataFrame(load_json(os.path.join(pdir, 'fitbit', 'steps.json')))
    steps['date'] = pd.to_datetime(steps['dateTime']).dt.normalize(); steps['value'] = pd.to_numeric(steps['value'], errors='coerce')
    act = steps.groupby('date')['value'].sum().rename('steps').to_frame()
    cov = steps.groupby('date')['value'].size()
    for name in ['sedentary_minutes', 'lightly_active_minutes', 'moderately_active_minutes', 'very_active_minutes', 'resting_heart_rate']:
        p = os.path.join(pdir, 'fitbit', f'{name}.json')
        if os.path.exists(p):
            act[name] = daily_value_series(p)
    act.loc[cov.reindex(act.index).fillna(0) < 600, 'steps'] = np.nan

    lo, hi = lab.date.min(), lab.date.max()
    cal = pd.DataFrame(index=pd.date_range(lo - pd.Timedelta(days=35), hi, freq='D')); cal.index.name = 'date'
    cal = cal.join(sl, how='left').join(act, how='left')
    F = add_rolling(cal, (list(act.columns), list(sl.columns)),
                    {'steps': 'a_steps', 'sedentary': 'a_sedentary_minutes', 'rhr': 'a_resting_heart_rate',
                     'tst': 'sp_tst_min', 'bed': 'sp_bed_h'})
    F['dow'] = F.index.dayofweek
    F = F.loc[lo:hi].reset_index()

    d = F.merge(lab, on='date', how='left')
    d['tst_label'] = np.where(d['s_tst_min'].notna() & d[SUBJ[0]].notna(),
                              ((d['s_tst_min'] >= 420) & (d['s_tst_min'] <= 540)).astype(float), np.nan)
    # external event: injury reports with at least one listed injury
    ip = os.path.join(pdir, 'pmsys', 'injury.csv')
    inj = pd.read_csv(ip) if os.path.exists(ip) else pd.DataFrame(columns=['effective_time_frame', 'injuries'])
    inj['date'] = local_date(inj['effective_time_frame']) if len(inj) else pd.Series(dtype='datetime64[ns]')
    inj['n'] = inj['injuries'].map(lambda s: len(ast.literal_eval(s)) if isinstance(s, str) and s.strip() else 0)
    d['ev_injury'] = d['date'].isin(set(inj.loc[inj['n'] > 0, 'date'])).astype(int)
    d['ev_injury_reported'] = d['date'].isin(set(inj['date'])).astype(int)
    d.insert(0, 'pid', pid)
    people.append(d)

df = pd.concat(people, ignore_index=True)
for c in SUBJ:
    df[c + '_label'] = np.where(df[c].notna(), (df[c] > df.groupby('pid')[c].transform('mean')).astype(float), np.nan)
df.to_pickle(out)
lab_cols = [c for c in df.columns if c.endswith('_label')]
print(df.shape, df.pid.nunique(), 'people;', 'labelled rows', int(df[SUBJ[0]].notna().sum()))
print('positive rate', df[lab_cols].mean().round(3).to_dict())
print('injury days', int(df.ev_injury.sum()), 'z cols', [c for c in df.columns if c.startswith('z_')])
