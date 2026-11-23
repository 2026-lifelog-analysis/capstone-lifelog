"""춘자 (State, not Trait) reproduction on a person-day table.

Usage: python3 -I repro_chunja.py <daily.pkl> <out_dir>
Protocol fixed in the reproduction log (절차 section) before running.
Outputs: chunja_table.csv (per target), chunja_boot.csv (bootstrap CIs), chunja_summary.json
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
src, outd = sys.argv[1], sys.argv[2]
os.makedirs(outd, exist_ok=True)
df = pd.read_pickle(src).sort_values(['pid', 'date']).reset_index(drop=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import targets, feature_cols, ll, icc1, GAP  # noqa: E402
TARGETS = targets(df)
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, random_state=0, n_jobs=2, verbose=-1)




def split_person(d):
    """first 50% of a person's labelled days -> train, last 50% -> eval, 7-day gap."""
    tr, ev = [], []
    for pid, g in d.groupby('pid'):
        g = g.sort_values('date')
        cut = g['date'].iloc[len(g) // 2]
        tr.append(g[g['date'] < cut]); ev.append(g[g['date'] >= cut + pd.Timedelta(days=GAP)])
    return pd.concat(tr), pd.concat(ev)




def fit_predict(Xtr, ytr, Xte):
    m = lgb.LGBMClassifier(**LGB)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


rows, per_person = [], []
for t in TARGETS:
    feats = feature_cols(df, t)
    d = df[df[t].notna()].copy()
    tr, ev = split_person(d)
    glob_p = tr[t].mean()
    pers = tr.groupby('pid')[t].mean()
    tr['prior'] = tr['pid'].map(pers); ev['prior'] = ev['pid'].map(pers).fillna(glob_p)
    Xcols = feats + ['prior']

    # alpha: inner temporal split inside train (last 30% of each person's train days)
    in_tr, in_va = [], []
    for pid, g in tr.groupby('pid'):
        g = g.sort_values('date'); c = int(len(g) * 0.7)
        in_tr.append(g.iloc[:c]); in_va.append(g.iloc[c:])
    in_tr, in_va = pd.concat(in_tr), pd.concat(in_va)
    pin = in_tr.groupby('pid')[t].mean()
    in_tr = in_tr.assign(prior=in_tr['pid'].map(pin)); in_va = in_va.assign(prior=in_va['pid'].map(pin).fillna(in_tr[t].mean()))
    p_in = fit_predict(in_tr[Xcols], in_tr[t], in_va[Xcols])
    alphas = np.round(np.arange(0, 1.01, 0.1), 1)
    a_ll = [ll(in_va[t].values, (1 - a) * p_in + a * in_va['prior'].values).mean() for a in alphas]
    alpha = float(alphas[int(np.argmin(a_ll))])

    p_model = fit_predict(tr[Xcols], tr[t], ev[Xcols])
    p_shrunk = (1 - alpha) * p_model + alpha * ev['prior'].values
    y = ev[t].values
    e = pd.DataFrame({'pid': ev['pid'].values, 'global': ll(y, np.full(len(y), glob_p)), 'personal': ll(y, ev['prior'].values),
                      'model': ll(y, p_model), 'shrunk': ll(y, p_shrunk)})
    e['target'] = t
    per_person.append(e)

    # sensor linkage lambda: ridge on sensor features only -> residual (y - personal prior)
    imp = SimpleImputer(strategy='median'); sc = StandardScaler()
    Xtr_s = sc.fit_transform(imp.fit_transform(tr[feats])); Xev_s = sc.transform(imp.transform(ev[feats]))
    r = Ridge(alpha=10.0).fit(Xtr_s, tr[t] - tr['prior'])
    res_hat = r.predict(Xev_s); res = y - ev['prior'].values
    lam = float(np.corrcoef(res_hat, res)[0, 1]) if np.std(res_hat) > 0 else 0.0

    rows.append(dict(target=t, n_train=len(tr), n_eval=len(ev), pos_rate=round(d[t].mean(), 3), icc=round(icc1(d[t].values, d['pid'].values), 3),
                     ll_global=e['global'].mean(), ll_personal=e['personal'].mean(), ll_model=e['model'].mean(), ll_shrunk=e['shrunk'].mean(),
                     alpha=alpha, sensor_gain=e['personal'].mean() - e['shrunk'].mean(), prior_gain=e['global'].mean() - e['personal'].mean(),
                     lam=round(lam, 3)))

tab = pd.DataFrame(rows)
E = pd.concat(per_person)

# person-level bootstrap (2000) for prior_gain and sensor_gain, per target and mean over targets
rng = np.random.default_rng(0)
pids = E['pid'].unique()
by = {(t, p): g for (t, p), g in E.groupby(['target', 'pid'])}
boot = []
for b in range(2000):
    s = rng.choice(pids, len(pids), replace=True)
    rec = {}
    for t in TARGETS:
        g = pd.concat([by[(t, p)] for p in s if (t, p) in by])
        rec[(t, 'prior_gain')] = g['global'].mean() - g['personal'].mean()
        rec[(t, 'sensor_gain')] = g['personal'].mean() - g['shrunk'].mean()
    rec[('mean', 'prior_gain')] = np.mean([rec[(t, 'prior_gain')] for t in TARGETS])
    rec[('mean', 'sensor_gain')] = np.mean([rec[(t, 'sensor_gain')] for t in TARGETS])
    boot.append(rec)
B = pd.DataFrame(boot)
B.columns = pd.MultiIndex.from_tuples(B.columns)
ci = B.quantile([0.025, 0.975]).T.reset_index(); ci.columns = ['target', 'quantity', 'lo', 'hi']
ci.to_csv(os.path.join(outd, 'chunja_boot.csv'), index=False)
for q in ('prior_gain', 'sensor_gain'):
    m = ci[ci.quantity == q].set_index('target')
    tab[q + '_ci'] = tab['target'].map(lambda t: f"[{m.loc[t,'lo']:+.3f}, {m.loc[t,'hi']:+.3f}]")

tab['law'] = (1 - tab['icc']) * tab['lam'].clip(lower=0)
law_r = float(np.corrcoef(tab['law'], tab['sensor_gain'])[0, 1])
icc_r = float(np.corrcoef(tab['icc'], tab['sensor_gain'])[0, 1])
mean = {k: float(tab[k].mean()) for k in ['ll_global', 'll_personal', 'll_model', 'll_shrunk', 'prior_gain', 'sensor_gain']}
mci = ci[ci.target == 'mean'].set_index('quantity')
summary = dict(mean=mean, mean_prior_gain_ci=[float(mci.loc['prior_gain', 'lo']), float(mci.loc['prior_gain', 'hi'])],
               mean_sensor_gain_ci=[float(mci.loc['sensor_gain', 'lo']), float(mci.loc['sensor_gain', 'hi'])],
               law_pearson=law_r, icc_pearson=icc_r, n_people=int(df.pid.nunique()))
summary['criterion_prior'] = summary['mean_prior_gain_ci'][0] > 0
summary['criterion_sensor_small'] = mean['sensor_gain'] < 0.02
tab.round(4).to_csv(os.path.join(outd, 'chunja_table.csv'), index=False)
json.dump(summary, open(os.path.join(outd, 'chunja_summary.json'), 'w'), indent=1)
pd.set_option('display.width', 220)
print(tab.round(3).to_string(index=False))
print(json.dumps(summary, indent=1))
