"""Build an isolated, development-only table using the approved fixed features."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snapshot'))
from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_split as fs
from experiments.seoyeon import baseline_blend as bb
from experiments.seungho import extended as ext
from experiments.seungho import sleep_delta as sd
from experiments.minjun.gap import features as mf

SOURCE = Path('/Users/angellove0112/종합설계/capstone-lifelog')
ART = ROOT / 'artifacts'
B_NAME = 'L4w_lag7_offbase28_term'
EXPECTED_SPLIT = 'de80b681465fa03846068800e2f707c204c57b850d94255c0eb6a7d8ab474d24'


def feature_spec():
    b = next(e for e in ext.window28_exps() if e.name == B_NAME)
    base = {t: fd.base_features(t) for t in fd.TARGETS}
    bcols = {t: b.features(t) for t in fd.TARGETS}
    ccols = {t: base[t] + [f'med28_{t}', f'sd28_{t}'] + mf.rec(t) for t in fd.TARGETS}
    both = {t: list(dict.fromkeys(bcols[t] + ccols[t])) for t in fd.TARGETS}
    return {'targets': fd.TARGETS, 'feature_sets': {'base': base, 'B': bcols, 'C': ccols, 'BC': both},
            'baseline_features': {t: bb.blend_features(t) for t in fd.TARGETS},
            'b_condition': B_NAME, 'seeds': [0, 1, 2], 'folds': list(fs.FOLDS),
            'a_definition': 'Train-only ShrinkBlend, residual regression, restore baseline at prediction',
            'final_test_scored': False,
            'source_commits': {'common_and_blend': '814aa1be5e2caf012a3827f7e62cfcf4b42f33c1',
                               'seungho': 'd3e96333e88d0a37386ee2bfe34c0044aa0b5779',
                               'minjun_gap': 'b06d8a8b1ac397baa7885e73d93fff6c3099f583',
                               'minjun_models': '022ad3a7476df2c4fbe6a31c9f0f463012a2fef1'}}


def selected_b_features(data, cal):
    """Extract only the approved upstream B equations; no surveys or daily activity."""
    table = data.table
    mats = sd.lag_matrices(table, data.nights, max_lag=28)
    cidx = cal.set_index('date')
    today, tomorrow = table.date, table.date + pd.Timedelta(days=1)
    cols = {}
    for c in ['h_class', 'h_finals', 'h_break', 'h_winter', 'h_summer', 'h_term_week', 'h_days_to_finals']:
        cols[f'x_{c}'] = cidx[c].reindex(tomorrow).to_numpy(dtype=np.float32)
    cols['x_h_fed_today'] = cidx.h_fed.reindex(today).to_numpy(dtype=np.float32)
    cols['x_h_fed_tomorrow'] = cidx.h_fed.reindex(tomorrow).to_numpy(dtype=np.float32)
    next_off = cidx.off.reindex(tomorrow).to_numpy(dtype=float)
    if not np.isfinite(next_off).all():
        raise ValueError('Calendar does not cover all evaluation dates')
    off = np.stack([cidx.off.reindex(today - pd.Timedelta(days=k - 1)).to_numpy(dtype=float)
                    for k in range(1, 29)], axis=1)
    for target, matrix in mats.items():
        holiday = np.where(off == 1, matrix, np.nan)
        work = np.where(off == 0, matrix, np.nan)
        hlevel, wlevel = ext._med(holiday), ext._med(work)
        hlevel[np.sum(~np.isnan(holiday), axis=1) < 3] = np.nan
        wlevel[np.sum(~np.isnan(work), axis=1) < 3] = np.nan
        same = np.where(next_off == 1, hlevel, wlevel)
        overall = sd.level(matrix, 28, 'med', 'in', 7)
        cols[f'x_{target}_Lsame'] = same.astype(np.float32)
        cols[f'x_{target}_gap_same'] = (same - overall).astype(np.float32)
        cols[f'x_{target}_Loffwork'] = (hlevel - wlevel).astype(np.float32)
        cols[f'x_{target}_dR7_Lsame'] = (ext._mean(matrix[:, :7]) - same).astype(np.float32)
    return pd.DataFrame(cols, index=table.index)


def verify_upstream_b(data, cal):
    indices = np.linspace(0, len(data.table) - 1, min(300, len(data.table)), dtype=int)
    sample = replace(data, table=data.table.iloc[indices].reset_index(drop=True))
    survey = pd.DataFrame({'pid': pd.Series([], dtype=str),
                           **{c: pd.Series([], dtype=float) for cs in ext.P_GROUPS.values() for c in cs}})
    full = ext.extended_features(sample, survey=survey, cal=cal,
                                 act=ext.raw_activity(SOURCE / 'data/nethealth'))
    selected = selected_b_features(sample, cal)
    for col in selected:
        np.testing.assert_allclose(selected[col], full[col], rtol=0, atol=0, equal_nan=True)
    return {'rows': len(sample.table), 'columns': len(selected.columns), 'exact_match': True}


def verify_old_targets(table):
    path = SOURCE / 'results/forecast/v3/runs/20261105-180221-seoyeon-blend/predictions.csv'
    if not path.exists():
        return {'available': False}
    old = pd.read_csv(path, dtype={'pid': str}, parse_dates=['date'])
    old = old[old.experiment == 'shrink_blend28']
    counts = {}
    for target, y in fd.TARGETS.items():
        a = table.loc[table[y].notna(), ['pid', 'date', y]]
        b = old[old.target == target][['pid', 'date', 'y_true']]
        match = a.merge(b, on=['pid', 'date'], how='outer', validate='one_to_one', indicator=True)
        if not (match._merge == 'both').all():
            raise ValueError('Previous and current evaluation nights differ')
        np.testing.assert_allclose(match[y], match.y_true, rtol=0, atol=0)
        counts[target] = len(match)
    return {'available': True, 'all_targets_exact': True, 'nights': counts}


def main():
    original = json.loads((ART / 'original_manifest.json').read_text())
    for name, expected in original['raw'].items():
        actual = hashlib.sha256((SOURCE / 'data/nethealth' / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Original raw data changed')
    split_path = SOURCE / 'results/forecast/v3'
    split, manifest = fs.load_split(split_path)
    assert fs.split_hash(split) == EXPECTED_SPLIT
    print('Loading unchanged original NetHealth data', flush=True)
    data = fd.load_forecast_data(path=SOURCE / 'data/nethealth')
    assert len(data.table) == 294243 and data.table.pid.nunique() == 621
    assert set(data.table.pid) == set(split.pid)
    fingerprint = fd.table_fingerprint(data.table)
    dev = set(split.loc[split.role == 'dev', 'pid'])
    data = replace(data, table=data.table[data.table.pid.isin(dev)].reset_index(drop=True),
                   nights=data.nights[data.nights.pid.isin(dev)].copy(),
                   episodes=data.episodes[data.episodes.pid.isin(dev)].copy(),
                   candidates=data.candidates[data.candidates.pid.isin(dev)].copy(),
                   activity=data.activity[data.activity.pid.isin(dev)].copy())
    assert data.table.pid.nunique() == 528 and len(data.table) == 247475
    cal = ext.calendar_days(path=SOURCE / 'data/nethealth')
    print('Checking selected B equations against unchanged upstream implementation', flush=True)
    b_check = verify_upstream_b(data, cal)
    print('Building A/B/C histories on fixed development participants', flush=True)
    table = fd.add_features(data, bb.feat_personal_history,
                            lambda d: selected_b_features(d, cal), mf.feat_long, mf.feat_recent, mf.feat_gap)
    for target in fd.TARGETS:
        np.testing.assert_allclose(table[bb.blend_features(target)[0]], table[f'med28_{target}'],
                                   rtol=0, atol=0, equal_nan=True)
    spec = feature_spec()
    for sets in spec['feature_sets'].values():
        for columns in sets.values():
            fd.check_features(columns, table)
            assert len(columns) == len(set(columns))
    old_check = verify_old_targets(table)
    keys = {t: fe.keys_hash(table.loc[table[y].notna()]) for t, y in fd.TARGETS.items()}
    table.to_parquet(ART / 'features.parquet', index=False)
    split.to_csv(ART / 'split.csv', index=False)
    spec.update({'split_sha256': EXPECTED_SPLIT, 'data_fingerprint_current': fingerprint,
                 'data_fingerprint_previous': manifest['data_fingerprint'], 'evaluation_key_hashes': keys,
                 'feature_file_sha256': hashlib.sha256((ART / 'features.parquet').read_bytes()).hexdigest(),
                 'n_dev': 528, 'n_final_excluded': 93, 'n_nights': len(table),
                 'b_upstream_verification': b_check, 'prior_target_verification': old_check,
                 'duplicate_median_definitions_exactly_equal': True,
                 'columns_count': len(table.columns)})
    (ART / 'feature_spec.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(json.dumps({'status': 'ready', 'people': 528, 'nights': len(table),
                       'fingerprint_matches_previous': fingerprint == manifest['data_fingerprint'],
                       'prior_target_check': old_check, 'upstream_b_check': b_check}), flush=True)


if __name__ == '__main__':
    main()
