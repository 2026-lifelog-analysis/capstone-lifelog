"""LOV3 (Structure over Scale) label-structure reproduction: lag-1 autocorrelation and within-person prevalence drift.

Usage: python3 -I repro_lov3.py <daily.pkl> <out_dir>
- lag1: Pearson r between y_t and y_{t-1} within person, consecutive calendar days only (pooled).
- half drift: mean over people of |rate(first half) - rate(second half)| (same halves as the 춘자 split, no gap).
- block drift: per person, 28-day calendar blocks with >=10 labelled days; mean |rate_b - rate_{b-1}| over consecutive blocks.
- null: shuffle labels within person (keeps each person's rate), 200 times; report the observed value's percentile.
"""
import json, os, sys
import numpy as np
import pandas as pd

src, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)
df = pd.read_pickle(src).sort_values(['pid', 'date']).reset_index(drop=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import targets  # noqa: E402
TARGETS = targets(df)
NPERM = int(os.environ.get('NPERM', '200'))
rng = np.random.default_rng(0)


def lag1(d, t):
    a, b = [], []
    for _, g in d.groupby('pid'):
        g = g.sort_values('date')
        nxt = g['date'].shift(-1) - g['date'] == pd.Timedelta(days=1)
        a.append(g[t].values[:-1][nxt.values[:-1]]); b.append(g[t].values[1:][nxt.values[:-1]])
    a, b = np.concatenate(a), np.concatenate(b)
    return float(np.corrcoef(a, b)[0, 1])


def half_drift(d, t):
    v = []
    for _, g in d.groupby('pid'):
        h = len(g) // 2
        v.append(abs(g[t].values[:h].mean() - g[t].values[h:].mean()))
    return float(np.mean(v))


def block_drift(d, t):
    v = []
    for _, g in d.groupby('pid'):
        blk = ((g['date'] - g['date'].min()).dt.days // 28).values
        rates = [g[t].values[blk == b].mean() for b in np.unique(blk) if (blk == b).sum() >= 10]
        v += [abs(rates[i] - rates[i - 1]) for i in range(1, len(rates))]
    return float(np.mean(v)) if v else np.nan


rows = []
for t in TARGETS:
    d = df[df[t].notna()][['pid', 'date', t]].copy()
    obs = dict(lag1=lag1(d, t), half=half_drift(d, t), block=block_drift(d, t))
    null = {k: [] for k in obs}
    for _ in range(NPERM):
        s = d.copy()
        s[t] = s.groupby('pid')[t].transform(lambda x: rng.permutation(x.values))
        null['lag1'].append(lag1(s, t)); null['half'].append(half_drift(s, t)); null['block'].append(block_drift(s, t))
    r = dict(target=t)
    for k in obs:
        n = np.array(null[k])
        r[k] = round(obs[k], 3); r[k + '_null_mean'] = round(float(n.mean()), 3); r[k + '_null_p95'] = round(float(np.quantile(n, 0.95)), 3)
        r[k + '_pct'] = round(float((n < obs[k]).mean() * 100), 1)
    rows.append(r)
tab = pd.DataFrame(rows)
tab['block_exceeds_p95'] = tab['block'] > tab['block_null_p95']
tab.to_csv(os.path.join(outd, 'lov3_table.csv'), index=False)
summary = dict(n_targets_block_drift=int(tab['block_exceeds_p95'].sum()), criterion=bool(tab['block_exceeds_p95'].sum() >= 2), nperm=NPERM)
json.dump(summary, open(os.path.join(outd, 'lov3_summary.json'), 'w'), indent=1)
pd.set_option('display.width', 250)
print(tab.to_string(index=False)); print(summary)
