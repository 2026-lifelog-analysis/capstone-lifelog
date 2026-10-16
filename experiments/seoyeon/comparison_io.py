"""Fixed comparison protocol, calendar boundaries, and private local artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT.parent / '모듈1_NetHealth'
MAIN_COMMIT = '07d88da81819845e4f03e79d57f64ab416c366ed'
RATES = (.05, 1 / 30)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_frame(path):
    frame = pd.read_csv(path, dtype={'pid': str, 'participant': str}, float_precision='round_trip')
    for col in ('date', 'baseline_start', 'baseline_end'):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def calendar_split(dates):
    dates = pd.DatetimeIndex(dates)
    start, end = dates.min(), dates.max()
    n = (end - start).days + 1
    first, second = int(np.floor(n * .4)), int(np.floor(n * .7))
    if not 0 < first < second < n:
        raise ValueError('Calendar span too short for 40/30/30')
    values = dict(start=start, development_end=start + pd.Timedelta(days=first - 1),
                  calibration_start=start + pd.Timedelta(days=first),
                  calibration_end=start + pd.Timedelta(days=second - 1),
                  test_start=start + pd.Timedelta(days=second), end=end)
    return {key: value.strftime('%Y-%m-%d') for key, value in values.items()}


def label_split(dates, split):
    dates = pd.DatetimeIndex(dates)
    if dates.isna().any() or (dates < split['start']).any() or (dates > split['end']).any():
        raise ValueError('Date outside fixed calendar span')
    return np.where(dates <= split['development_end'], 'development',
                    np.where(dates <= split['calibration_end'], 'calibration', 'test'))


def quantile_threshold(scores, calibration_mask, rate):
    values = np.asarray(scores, float)
    mask = np.asarray(calibration_mask, bool)
    selected = values[mask & np.isfinite(values)]
    if not len(selected):
        raise ValueError('No finite calibration scores')
    if not 0 < rate < 1:
        raise ValueError('Rate must be between zero and one')
    return float(np.quantile(selected, 1 - rate, method='higher'))


def preserved_inputs():
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    files = [ROOT / name for name in tracked if (ROOT / name).is_file()]
    files += list((ROOT / 'data/nethealth').glob('*.csv'))
    files += list((PRIOR / '02_코드').glob('*.py'))
    files += list((PRIOR / '02_코드').glob('*.json'))
    files += [PRIOR / '결과보고서.docx']
    return {str(path.relative_to(ROOT.parent)): sha256(path) for path in sorted(set(files))}


def assert_main_source():
    for path in (ROOT / 'src').glob('*.py'):
        expected = subprocess.check_output(['git', 'show', f'{MAIN_COMMIT}:src/{path.name}'], cwd=ROOT)
        if path.read_bytes() != expected:
            raise ValueError(f'Main source changed: {path.name}')


def protocol():
    return dict(version='main80-seoyeon-seungho-1', main_commit=MAIN_COMMIT,
                reference_commit='1806f0a5604ab08111a94d62c01ac874a2fcb91c',
                preprocessing=dict(wear=80, sleep_gap=90, min_days=42, period='all'),
                fractions=[.4, .3, .3], rates=list(RATES),
                quantile='higher', alarm='strict >',
                calibration='Common finite calibration person-days, never test scores',
                comparison='Common finite test person-days plus per-model available coverage',
                seoyeon='Prior five model formulae; midnight circular bedtime; no new 70% filtering',
                seungho='Four stages; continuous CUSUM, calibration h equals alarm h; same-day all-person reference',
                moment='Frozen local MOMENT-1-large, 3x512; fresh inference; within-run exact-input caching only',
                synthetic=dict(scenarios=['S1', 'S2', 'S3', 'S4', 'S5'], length=14,
                               repetitions=20, seeds=list(range(20)),
                               schedule='Same person-date segments for all scenarios and models; test only',
                               eligibility='At least 14 complete previous 28 days and 7 common finite test days in segment',
                               reference='Recompute same-day group median on perturbed population'),
                claims='Observed and synthetic deviations; not clinical accuracy or optimality')
