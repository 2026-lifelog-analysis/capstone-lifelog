"""Implementation check on the ETRI 2024 labels: our ICC / lag-1 vs the values the two papers report.

Usage (run it yourself in a terminal; do NOT open the ETRI data with any AI tool, license):
    python3 -I etri_icc_check.py <ch2025_metrics_train.csv> <out_dir>

Reads ONLY the label file. Prints and writes aggregate numbers only (no rows, no ids).
Auto-detects: subject column (subject_id / id / pid / user), date column (sleep_date / lifelog_date / date),
label columns Q1..Q3, S1..S4 that exist.
Reference values: 춘자 (State, not Trait) Table I ICC on its test-like subset; LOV3 (Structure over Scale) Table I
ICC and lag-1 on the full training set. Both used the 5th-challenge split (10 people, 450 train nights).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import icc1  # noqa: E402

src, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)
df = pd.read_csv(src)
low = {c.lower(): c for c in df.columns}
sid = next(low[c] for c in ('subject_id', 'id', 'pid', 'user', 'userid', 'subject') if c in low)
dcol = next((low[c] for c in ('sleep_date', 'lifelog_date', 'date') if c in low), None)
labels = [c for c in ('Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4') if c in df.columns]
REF = {  # (춘자 ICC, LOV3 ICC, LOV3 lag-1, LOV3 positive rate)
    'Q1': (0.092, 0.11, 0.19, 0.50), 'Q2': (0.065, 0.08, 0.25, 0.56), 'Q3': (0.047, 0.06, 0.21, 0.60),
    'S1': (0.131, 0.14, 0.10, 0.68), 'S2': (0.179, 0.19, 0.01, 0.65), 'S3': (0.245, 0.25, 0.00, 0.66),
    'S4': (0.116, 0.14, 0.04, 0.56)}

if dcol:
    df[dcol] = pd.to_datetime(df[dcol])
    df = df.sort_values([sid, dcol])
rows = []
for c in labels:
    y = pd.to_numeric(df[c], errors='coerce')
    vals = sorted(y.dropna().unique().tolist())
    binary = set(vals) <= {0, 1}
    lag = np.nan
    if dcol:
        a, b = [], []
        for _, g in df.assign(y=y).dropna(subset=['y']).groupby(sid):
            nxt = (g[dcol].shift(-1) - g[dcol]).dt.days.values == 1
            a.append(g['y'].values[:-1][nxt[:-1]]); b.append(g['y'].values[1:][nxt[:-1]])
        a, b = np.concatenate(a), np.concatenate(b)
        lag = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan
    r = REF.get(c, (np.nan,) * 4)
    rows.append(dict(label=c, n=int(y.notna().sum()), n_values=len(vals), binary=binary,
                     pos_rate=round(float((y == 1).mean()), 3) if binary else np.nan,
                     icc_ours=round(icc1(y.values, df[sid].values), 3), icc_chunja=r[0], icc_lov3=r[1],
                     lag1_ours=round(lag, 3) if np.isfinite(lag) else np.nan, lag1_lov3=r[2], pos_rate_lov3=r[3]))
T = pd.DataFrame(rows)
T['icc_in_range'] = [(min(a, b) - 0.03) <= x <= (max(a, b) + 0.03) if np.isfinite(a) else np.nan
                     for x, a, b in zip(T.icc_ours, T.icc_chunja, T.icc_lov3)]
T.to_csv(os.path.join(outd, 'etri_icc_check.csv'), index=False)
pd.set_option('display.width', 200)
print(f'subjects={df[sid].nunique()} rows={len(df)} labels={labels} date_col={dcol}')
print(T.to_string(index=False))
print('rule: icc_in_range = ours within [min(paper values) - 0.03, max(paper values) + 0.03]')
