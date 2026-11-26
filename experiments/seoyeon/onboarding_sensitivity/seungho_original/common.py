"""Shared helpers for the reproduction / follow-up experiments.

Person-day table contract (output of pm_build.py / hr_build.py):
- pid, date (datetime64, one row per calendar day inside each person's survey period)
- <item>        raw self-report value (NaN on days without a response)
- <item>_label  0/1, value > that person's whole-period mean (ETRI style; used by the reproduction)
- tst_label     0/1, Fitbit main-sleep minutes in [420, 540]
- a_*  activity up to day t-1, s_* sleep that ended on day t, sp_* previous night's sleep (+ rolling/z versions)
- z_*  7-day mean vs past-28-day personal baseline (inputs for the sensor-driven tracker)
- dow, ev_* external event flags (sensor-independent, used only by external_check.py)
"""
import numpy as np
import pandas as pd

EPS = 1e-6
GAP = 7


def targets(df):
    return [c for c in df.columns if c.endswith('_label')]


def raw_of(t):
    """raw column behind a personal-mean label ('' for absolute labels such as tst_label)."""
    return '' if t == 'tst_label' else t[:-len('_label')]


def feature_cols(df, t):
    f = [c for c in df.columns if c.startswith(('a_', 's_', 'sp_')) or c == 'dow']
    if t == 'tst_label':  # the label is computed from the night that ended on day t
        f = [c for c in f if not c.startswith('s_')]
    return f


def z_cols(df, t):
    z = [c for c in df.columns if c.startswith('z_')]
    if t == 'tst_label':
        z = [c for c in z if 'tst' not in c and 'bed' not in c]
    return z


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def split_person(d, t):
    """per person, labelled days in time order: first 50% train, last 50% eval, GAP-day gap.
    Returns (train_rows, eval_rows) restricted to days with a label for t, plus the cut date per person."""
    tr, ev, cuts = [], [], {}
    for pid, g in d[d[t].notna()].groupby('pid'):
        g = g.sort_values('date')
        cut = g['date'].iloc[len(g) // 2]
        cuts[pid] = cut
        tr.append(g[g['date'] < cut]); ev.append(g[g['date'] >= cut + pd.Timedelta(days=GAP)])
    return pd.concat(tr), pd.concat(ev), cuts


def icc1(y, groups):
    d = pd.DataFrame({'y': y, 'g': groups}).dropna()
    k = d.groupby('g').size(); n = len(k); N = len(d)
    gm = d['y'].mean(); means = d.groupby('g')['y'].mean()
    msb = (k * (means - gm) ** 2).sum() / (n - 1)
    msw = ((d['y'] - d['g'].map(means)) ** 2).sum() / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def person_bootstrap(E, value_cols, n=2000, seed=0):
    """E: rows with pid + per-row losses. Returns {col: (lo, hi)} of the mean over resampled people."""
    rng = np.random.default_rng(seed)
    pids = E['pid'].unique()
    sums = E.groupby('pid')[value_cols].sum(); cnt = E.groupby('pid').size()
    out = {c: [] for c in value_cols}
    for _ in range(n):
        s = rng.choice(pids, len(pids), replace=True)
        tot = sums.loc[s].sum(); m = cnt.loc[s].sum()
        for c in value_cols:
            out[c].append(tot[c] / m)
    return {c: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for c, v in out.items()}


def holm(pvals):
    """Holm-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj
