"""RQ4: does the tracked personal state move with sensor-independent events / standard scales?

Usage: python3 -I external_check.py <daily.pkl> <rq3_preds.pkl> <out_dir>

Layer 2 (events): every ev_* column of the daily table except *_reported.
  episode = run of consecutive event days; statistic = mean over episodes of
  [mean during the episode - mean over the 14 days before onset], for the actual labels and for
  tracked probabilities (budget k7 and real; methods B3, P1, P2). Only days in the scored (post-onboarding) range.
  null: the same episodes moved to random onsets inside the same person's scored range, 200 draws.
  criterion: |observed| > 95th percentile of |null|.
Layer 3 (scales, only if b_<scale>_pre / b_<scale>_post exist): per person, tracked probability over the last
  28 days minus the first 28 days of the scored range vs (post - pre) of the scale; Spearman + 2,000 permutations.
Outputs: rq4_events.csv, rq4_scales.csv
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

src, pred_p, outd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outd, exist_ok=True)
df = pd.read_pickle(src)
P = pd.read_pickle(pred_p)
rng = np.random.default_rng(0)
NNULL, PRE = 200, 14
SERIES = [('label', None, None)] + [(f'{m}_{b}', m, b) for b in ('k7', 'real') for m in ('B3', 'P1', 'P2')]
EV = [c for c in df.columns if c.startswith('ev_') and not c.endswith('_reported')]

# scored range per person/target = first..last scored day
rngs = P[P.scored].groupby(['target', 'pid'])['date'].agg(['min', 'max'])


def series_frame(t, name, m, b):
    if name == 'label':
        s = P[(P.target == t) & (P.method == 'B1') & (P.budget == 'k1')][['pid', 'date', 'y']].rename(columns={'y': 'v'})
    else:
        s = P[(P.target == t) & (P.method == m) & (P.budget == b)][['pid', 'date', 'p']].rename(columns={'p': 'v'})
    return s


def episodes(flags):
    on = np.where(flags & ~np.r_[False, flags[:-1]])[0]
    out = []
    for s in on:
        e = s
        while e + 1 < len(flags) and flags[e + 1]:
            e += 1
        out.append((s, e))
    return out


def stat_for(v, eps):
    vals = []
    for s, e in eps:
        if s - PRE < 0:
            continue
        a = np.nanmean(v[s:e + 1]) if np.isfinite(v[s:e + 1]).any() else np.nan
        bpre = np.nanmean(v[s - PRE:s]) if np.isfinite(v[s - PRE:s]).any() else np.nan
        if np.isfinite(a) and np.isfinite(bpre):
            vals.append(a - bpre)
    return vals


rows = []
for ev in EV:
    for t in P.target.unique():
        for name, m, b in SERIES:
            S = series_frame(t, name, m, b)
            per = []  # (values array, episodes) per person within scored range
            for pid, g in df[['pid', 'date', ev]].groupby('pid'):
                if (t, pid) not in rngs.index:
                    continue
                lo, hi = rngs.loc[(t, pid)]
                g = g[(g.date >= lo - pd.Timedelta(days=PRE)) & (g.date <= hi)].sort_values('date')
                v = g[['date']].merge(S[S.pid == pid], on='date', how='left')['v'].values.astype(float)
                fl = g[ev].values.astype(bool)
                eps = [(s, e) for s, e in episodes(fl) if s >= PRE]
                if eps:
                    per.append((v, eps))
            obs_vals = [x for v, eps in per for x in stat_for(v, eps)]
            if not obs_vals:
                continue
            obs = float(np.mean(obs_vals))
            null = []
            for _ in range(NNULL):
                vals = []
                for v, eps in per:
                    n = len(v)
                    for s, e in eps:
                        L = e - s + 1
                        if n - L - PRE <= 1:
                            continue
                        ns = rng.integers(PRE, n - L)
                        vals += stat_for(v, [(ns, ns + L - 1)])
                if vals:
                    null.append(np.mean(vals))
            null = np.abs(np.array(null))
            rows.append(dict(event=ev, target=t, series=name, n_episodes=len(obs_vals), change=round(obs, 4),
                             null_abs_p95=round(float(np.quantile(null, 0.95)), 4), pct=round(float((null < abs(obs)).mean() * 100), 1),
                             exceeds=bool(abs(obs) > np.quantile(null, 0.95))))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(outd, 'rq4_events.csv'), index=False)
pd.set_option('display.width', 200)
if len(E):
    print(E.to_string(index=False))

# layer 3
scales = sorted({c[2:-4] for c in df.columns if c.startswith('b_') and c.endswith('_pre')})
srows = []
for sc in scales:
    sdelta = (df.groupby('pid')[f'b_{sc}_post'].first() - df.groupby('pid')[f'b_{sc}_pre'].first()).dropna()
    for t in P.target.unique():
        for name, m, b in SERIES:
            S = series_frame(t, name, m, b)
            dv = {}
            for pid in sdelta.index:
                if (t, pid) not in rngs.index:
                    continue
                lo, hi = rngs.loc[(t, pid)]
                g = S[(S.pid == pid) & (S.date >= lo) & (S.date <= hi)].sort_values('date')['v'].dropna().values
                if len(g) >= 56:
                    dv[pid] = g[-28:].mean() - g[:28].mean()
            ids = [p for p in dv if p in sdelta.index]
            if len(ids) < 10:
                continue
            x = np.array([dv[p] for p in ids]); y = sdelta.loc[ids].values
            r = spearmanr(x, y).correlation
            perm = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(2000)])
            srows.append(dict(scale=sc, target=t, series=name, n_people=len(ids), spearman=round(float(r), 3),
                              p_perm=round(float((np.abs(perm) >= abs(r)).mean()), 4)))
Sc = pd.DataFrame(srows)
Sc.to_csv(os.path.join(outd, 'rq4_scales.csv'), index=False)
if len(Sc):
    print(Sc.to_string(index=False))
print('events', EV, 'scales', scales)
