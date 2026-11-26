"""RQ3: can sensors keep the personal baseline on track when surveys become sparse?

Usage: python3 -I sparse_survey.py <daily.pkl> <out_dir>

Setting (fixed before running, see the experiment log):
- per person, labelled days in time order: first 50% = onboarding (daily survey, all labels visible),
  then the survey budget applies. Days >= cut + 7 are scored.
- subjective labels are re-binarized with the person's onboarding-period mean (no future information).
- budgets: k1 (every day), k3, k7 (one survey every k calendar days, answered only if the person actually
  answered that day), real (response-day pattern of a low-compliance participant, cycled).
- every method predicts day t before seeing y_t (one-step-ahead) and may use only labels revealed before t.
Methods:
  B1  fixed personal rate from onboarding (춘자 personal prior)
  B2  LightGBM on sensor features + personal rate, shrunk to the personal rate (춘자 ④); never updated by labels
  B3  label autocorrelation (LOV3): Gaussian-weighted mean of revealed labels, shrunk to the personal rate
  P1  local-level state-space on logit(rate), extended Kalman update on revealed labels (no sensors)
  P2  P1 + sensor term: p_t = sigmoid(m_t + beta . z_t), z = 7-day vs 28-day personal deviations; beta fit on onboarding
Hyperparameters of B3/P1/P2 are chosen per budget on the onboarding period only
(first 70% dense, last 30% under the same budget).
Outputs: rq3_table.csv, rq3_diffs.csv, rq3_summary.json, rq3_mnar.csv, rq3_preds.pkl (for external_check.py)
"""
import json, os, sys, warnings, zlib
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import targets, raw_of, feature_cols, z_cols, ll, GAP, holm  # noqa: E402

warnings.filterwarnings('ignore')
src, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)
df = pd.read_pickle(src).sort_values(['pid', 'date']).reset_index(drop=True)
BUDGETS = ['k1', 'k3', 'k7', 'real']
METHODS = ['B1', 'B2', 'B3', 'P1', 'P2']
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, random_state=0, n_jobs=2, verbose=-1)
H_GRID, W0_GRID = [3, 7, 14, 28], [0.5, 1.0, 2.0]
Q_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
rng = np.random.default_rng(0)


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


# ---------- budget masks ----------
resp_any = df.groupby('pid').apply(lambda g: g[[raw_of(t) or t for t in targets(df)][0]].notna().values, include_groups=False)
rate = resp_any.map(np.mean)
donors = list(rate[rate <= rate.quantile(0.25)].index)


def reveal_mask(n_days, budget, pid):
    """boolean array over the budget-period calendar days: True = survey asked that day (answer only if one exists)."""
    if budget == 'k1':
        return np.ones(n_days, bool)
    if budget in ('k3', 'k7'):
        k = int(budget[1]); return (np.arange(n_days) % k) == 0
    pool = [d for d in donors if d != pid] or donors
    dpat = resp_any[pool[zlib.crc32(pid.encode()) % len(pool)]]  # deterministic donor
    return np.resize(dpat, n_days)


# ---------- trackers (operate on one person's calendar arrays) ----------
def run_b3(y, vis, prior, h, w0):
    p = np.empty(len(y)); idx = np.where(vis)[0]
    for t in range(len(y)):
        j = idx[idx < t]
        if len(j) == 0:
            p[t] = prior; continue
        w = np.exp(-((t - j) ** 2) / (2 * h * h)); p[t] = (np.dot(w, y[j]) + w0 * prior) / (w.sum() + w0)
    return p


def run_ekf(y, vis, m0, q, zb=None):
    """local-level logit model; zb = beta.z_t per day (0 for P1). Returns one-step-ahead probabilities."""
    m, P = m0, 1.0; p = np.empty(len(y))
    zb = np.zeros(len(y)) if zb is None else zb
    for t in range(len(y)):
        P += q
        pt = sig(m + zb[t]); p[t] = pt
        if vis[t]:
            v = pt * (1 - pt)
            m = m + P * (y[t] - pt) / (1 + P * v); P = P / (1 + P * v)
    return p


def fit_beta(Z, y, off, lam=1.0, it=25):
    b = np.zeros(Z.shape[1])
    for _ in range(it):
        p = sig(off + Z @ b); W = p * (1 - p)
        H = Z.T @ (Z * W[:, None]) + lam * np.eye(Z.shape[1]); g = Z.T @ (y - p) - lam * b
        b = b + np.linalg.solve(H, g)
    return b


def lgb_shrunk(tr, ev, feats, t):
    in_tr, in_va = [], []
    for _, g in tr.groupby('pid'):
        c = int(len(g) * 0.7); in_tr.append(g.iloc[:c]); in_va.append(g.iloc[c:])
    in_tr, in_va = pd.concat(in_tr), pd.concat(in_va)
    X = feats + ['prior']
    m = lgb.LGBMClassifier(**LGB).fit(in_tr[X], in_tr['y'])
    pin = m.predict_proba(in_va[X])[:, 1]
    al = min(np.round(np.arange(0, 1.01, 0.1), 1), key=lambda a: ll(in_va['y'].values, (1 - a) * pin + a * in_va['prior'].values).mean())
    m = lgb.LGBMClassifier(**LGB).fit(tr[X], tr['y'])
    return (1 - al) * m.predict_proba(ev[X])[:, 1] + al * ev['prior'].values


# ---------- main loop ----------
rows, preds, mnar = [], [], []
SEL = [x for x in os.environ.get('RQ_TARGETS', '').split(',') if x]
for t in [x for x in targets(df) if not SEL or x in SEL]:
    raw = raw_of(t)
    feats = feature_cols(df, t); zc = z_cols(df, t)
    base = df[['pid', 'date'] + feats + zc + ([raw] if raw else [t])].copy()
    lab_col = raw if raw else t
    # per-person cut on labelled days (same rule as the reproduction)
    cuts = {}
    for pid, g in base[base[lab_col].notna()].groupby('pid'):
        cuts[pid] = g['date'].iloc[len(g) // 2]
    base = base[base['pid'].isin(cuts)]
    base['cut'] = base['pid'].map(cuts)
    base['phase'] = np.where(base['date'] < base['cut'], 'on', np.where(base['date'] < base['cut'] + pd.Timedelta(days=GAP), 'gap', 'eval'))
    if raw:
        thr = base[base.phase == 'on'].groupby('pid')[raw].mean()
        base['y'] = np.where(base[raw].notna(), (base[raw] > base['pid'].map(thr)).astype(float), np.nan)
    else:
        base['y'] = base[t]
    on = base[(base.phase == 'on') & base.y.notna()]
    prior_p = on.groupby('pid')['y'].mean(); glob_p = on['y'].mean()
    base['prior'] = base['pid'].map(prior_p).fillna(glob_p)
    base[zc] = base[zc].fillna(0.0).clip(-4, 4)

    # sensor-model B2 (static, never updated)
    tr = base[(base.phase == 'on') & base.y.notna()]; evr = base[(base.phase == 'eval') & base.y.notna()]
    b2 = pd.Series(lgb_shrunk(tr, evr, feats, t), index=evr.index)
    # beta for P2 (onboarding only, person offset = logit(personal rate))
    beta = fit_beta(tr[zc].values, tr['y'].values, logit(tr['prior'].values)) if zc else np.zeros(0)

    # MNAR check: sensors on answered vs unanswered days (within person, standardized)
    for c in [x for x in ['a_steps', 's_tst_min'] if x in base]:
        diffs = []
        for _, g in base.groupby('pid'):
            a, b = g.loc[g[lab_col].notna(), c].dropna(), g.loc[g[lab_col].isna(), c].dropna()
            sd = g[c].std()
            if len(a) >= 5 and len(b) >= 5 and sd > 0:
                diffs.append((a.mean() - b.mean()) / sd)
        mnar.append(dict(target=t, sensor=c, n_people=len(diffs), mean_std_diff=float(np.mean(diffs)) if diffs else np.nan))

    persons = {pid: g.reset_index() for pid, g in base.groupby('pid')}

    def person_arrays(g, budget, inner=False):
        """returns y, visible mask, scoring mask for one person under a budget (inner=True: within onboarding)."""
        y = g['y'].fillna(0).values; has = g['y'].notna().values
        if inner:
            on_idx = np.where(g.phase.values == 'on')[0]
            if len(on_idx) < 10:
                return None
            c = on_idx[int(len(on_idx) * 0.7)]
            keep = np.arange(len(g)) <= on_idx[-1]
            vis = has.copy(); vis[c:] = False
            m = reveal_mask(on_idx[-1] - c + 1, budget, g.pid.iloc[0]); vis[c:on_idx[-1] + 1] = has[c:on_idx[-1] + 1] & m
            score = keep & has & (np.arange(len(g)) >= c)
            return y, vis, score, keep
        start = np.where(g.phase.values != 'on')[0]
        vis = has & (g.phase.values == 'on')
        if len(start):
            s0 = start[0]; m = reveal_mask(len(g) - s0, budget, g.pid.iloc[0]); vis[s0:] = has[s0:] & m
        score = has & (g.phase.values == 'eval')
        return y, vis, score, np.ones(len(g), bool)

    def run(method, g, arr, hp):
        y, vis, _, keep = arr
        prior = g['prior'].iloc[0]
        if method == 'B3':
            return run_b3(y[keep], vis[keep], prior, *hp)
        zb = (g.loc[keep, zc].values @ beta) if (method == 'P2' and len(beta)) else None
        return run_ekf(y[keep], vis[keep], logit(glob_p), hp, zb)

    for budget in BUDGETS:
        # choose hyperparameters on the onboarding period only
        hp = {}
        for method, grid in [('B3', [(h, w) for h in H_GRID for w in W0_GRID]), ('P1', Q_GRID), ('P2', Q_GRID)]:
            best, best_ll = None, np.inf
            for cand in grid:
                tot, n = 0.0, 0
                for pid, g in persons.items():
                    arr = person_arrays(g, budget, inner=True)
                    if arr is None:
                        continue
                    p = run(method, g, arr, cand); s = arr[2][arr[3]]
                    tot += ll(arr[0][arr[3]][s], p[s]).sum(); n += s.sum()
                if n and tot / n < best_ll:
                    best, best_ll = cand, tot / n
            hp[method] = best
        for pid, g in persons.items():
            arr = person_arrays(g, budget)
            y, vis, score, _ = arr
            if score.sum() == 0:
                continue
            out = {'B1': np.full(len(g), g['prior'].iloc[0])}
            for method in ('B3', 'P1', 'P2'):
                out[method] = run(method, g, arr, hp[method])
            b2p = np.full(len(g), np.nan); ev_idx = g['index'].values
            sel = np.isin(ev_idx, b2.index); b2p[sel] = b2.loc[ev_idx[sel]].values
            out['B2'] = b2p
            for method in METHODS:
                p = out[method]
                preds.append(pd.DataFrame({'pid': pid, 'date': g['date'].values, 'target': t, 'budget': budget, 'method': method,
                                           'p': p, 'y': np.where(g['y'].notna(), g['y'], np.nan), 'scored': score,
                                           'visible': vis}))
                rows.append(dict(target=t, budget=budget, method=method, pid=pid, n=int(score.sum()),
                                 ll_sum=float(ll(y[score], p[score]).sum()), hp=str(hp.get(method, ''))))
        print(t, budget, {m: hp.get(m) for m in ('B3', 'P1', 'P2')}, 'beta', np.round(beta, 3).tolist(), flush=True)

R = pd.DataFrame(rows)
P = pd.concat(preds, ignore_index=True)
P.to_pickle(os.path.join(outd, 'rq3_preds.pkl'))
pd.DataFrame(mnar).to_csv(os.path.join(outd, 'rq3_mnar.csv'), index=False)

tab = R.groupby(['target', 'budget', 'method']).apply(lambda g: g.ll_sum.sum() / g.n.sum(), include_groups=False).unstack('method')[METHODS]
tab.round(4).to_csv(os.path.join(outd, 'rq3_table.csv'))

# paired person-bootstrap of differences (negative = first method better)
PAIRS = [('P2', 'B3'), ('P2', 'B1'), ('P2', 'P1'), ('P1', 'B3'), ('B3', 'B1')]
W = R.pivot_table(index=['target', 'budget', 'pid'], columns='method', values='ll_sum').join(
    R.pivot_table(index=['target', 'budget', 'pid'], columns='method', values='n')['B1'].rename('n'))
diffs = []
for (t, b), g in W.groupby(level=[0, 1]):
    g = g.droplevel([0, 1])
    n = g['n'].values.astype(float)
    idx = rng.integers(0, len(g), size=(2000, len(g)))
    nb = n[idx].sum(1)
    for a, c in PAIRS:
        d = (g[a].values - g[c].values)
        obs = d.sum() / n.sum()
        bs = d[idx].sum(1) / nb
        diffs.append(dict(target=t, budget=b, pair=f'{a}-{c}', diff=obs, lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975),
                          p_boot=float(min(1.0, 2 * min((bs >= 0).mean(), (bs <= 0).mean())))))
D = pd.DataFrame(diffs)
D['p_holm'] = np.nan
for (b, pr), idx in D.groupby(['budget', 'pair']).groups.items():
    D.loc[idx, 'p_holm'] = holm(D.loc[idx, 'p_boot'].values)
D.round(4).to_csv(os.path.join(outd, 'rq3_diffs.csv'), index=False)

main = D[(D.pair == 'P2-B3') & D.budget.isin(['k3', 'k7'])]
wins = main.assign(win=(main.hi < 0)).groupby('target')['win'].all()
summary = dict(criterion_targets=[str(x) for x in wins[wins].index], criterion=bool(wins.sum() >= 2),
               mean_ll=tab.groupby(level='budget').mean().round(4).to_dict(orient='index'), donors=donors)
json.dump(summary, open(os.path.join(outd, 'rq3_summary.json'), 'w'), indent=1, default=str)
pd.set_option('display.width', 200)
print(tab.round(3).to_string())
print(D[D.pair.isin(['P2-B3', 'P2-B1'])].round(3).to_string(index=False))
print(json.dumps(summary, indent=1, default=str))
