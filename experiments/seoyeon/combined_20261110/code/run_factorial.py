"""Run paired factorial comparisons without touching the original repository."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from model_adapters import (ROOT, CONDITIONS, ResidualModel, baseline_model,
                            condition_features, fixed_catboost)
from src import forecast_dataset as fd
from src import forecast_eval as fe
from src import forecast_split as fs

BASELINES = ('group_median', 'personal_mean7', 'personal_median28', 'shrink_blend28')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprints(root=ROOT):
    root = Path(root)
    artifacts = root / 'artifacts'
    hashes = {name: file_sha(artifacts / name)
              for name in ('feature_spec.json', 'features.parquet', 'split.csv')}
    hashes.update({name: file_sha(root / 'code' / name)
                   for name in ('model_adapters.py', 'run_factorial.py', 'run_models.py')})
    hashes.update({str(path.relative_to(root)): file_sha(path)
                   for path in sorted((root / 'snapshot').rglob('*.py'))})
    return hashes


def check_prepared_hashes(spec, feature_file_sha256, split):
    if spec['feature_file_sha256'] != feature_file_sha256:
        raise ValueError('Prepared feature file differs from feature_spec fingerprint')
    if spec['split_sha256'] != fs.split_hash(split):
        raise ValueError('Participant split differs from feature_spec fingerprint')


def load_inputs(root=ROOT):
    artifacts = Path(root) / 'artifacts'
    spec_path = artifacts / 'feature_spec.json'
    spec = json.loads(spec_path.read_text())
    hashes = input_fingerprints(root)
    data = pd.read_parquet(artifacts / 'features.parquet')
    split = pd.read_csv(artifacts / 'split.csv', dtype={'pid': str})
    check_prepared_hashes(spec, hashes['features.parquet'], split)
    data = data.assign(pid=data.pid.astype(str), date=pd.to_datetime(data.date),
                       forecast_at=pd.to_datetime(data.forecast_at))
    fs.check_split(split)
    if data.duplicated(['pid', 'date']).any():
        raise ValueError('Duplicate feature rows')
    final = fs.final_pids(split)
    if set(data.pid) & final:
        raise ValueError('Feature artifact must contain development people only')
    if set(data.pid) != set(split.loc[split.role == 'dev', 'pid']):
        raise ValueError('Feature artifact participant set differs from development split')
    return data, split, spec, hashes


def select_fold(data, split, target, fold):
    train_people, valid_people = fs.fold_pids(split, fold)
    final = fs.final_pids(split)
    if train_people & valid_people or (train_people | valid_people) & final:
        raise ValueError('Participant leakage')
    train = data.loc[data.pid.isin(train_people) & data[target].notna()].copy()
    valid = data.loc[data.pid.isin(valid_people) & data[target].notna()].copy()
    if train.empty or valid.empty:
        raise ValueError('Empty training or evaluation fold')
    return train, valid.sort_values(['pid', 'date']).reset_index(drop=True)


def prediction_inputs(frame, features, baseline_features=()):
    cols = list(dict.fromkeys(['pid', 'date', 'forecast_at', *features, *baseline_features]))
    forbidden = set(fd.NON_FEATURE_COLS) - {'pid', 'date', 'forecast_at'}
    if set(cols) & forbidden:
        raise ValueError('Target or target-derived metadata included in prediction inputs')
    return frame.loc[:, cols]


def validate_prediction(prediction, valid):
    if valid.duplicated(['pid', 'date']).any():
        raise ValueError('Duplicate evaluation rows')
    if prediction.shape != (len(valid),) or not np.isfinite(prediction).all():
        raise ValueError('Incomplete or nonfinite prediction; refusing to score')


def signature(hashes, identity, features, settings):
    payload = {'inputs': hashes, 'identity': identity,
               'features': features, 'settings': settings}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def output_paths(root, identity):
    family, condition, target, fold, seed = identity
    stem = f'{family}_{condition}_{target}_f{fold}_s{seed}'
    folder = Path(root) / 'artifacts' / 'predictions'
    return folder / f'{stem}.parquet', folder / f'{stem}.json'


def checkpoint_exists(root, identity, run_signature, valid, target_column):
    path, meta_path = output_paths(root, identity)
    if not path.exists() and not meta_path.exists():
        return False
    if not path.exists() or not meta_path.exists():
        raise ValueError(f'Incomplete checkpoint: {path.name}')
    metadata = json.loads(meta_path.read_text())
    if metadata['signature'] != run_signature:
        raise ValueError(f'Checkpoint configuration mismatch: {path.name}')
    cached = pd.read_parquet(path)
    for col in ('pid', 'date'):
        if not cached[col].reset_index(drop=True).equals(valid[col].reset_index(drop=True)):
            raise ValueError(f'Checkpoint evaluation rows mismatch: {path.name}')
    if not np.array_equal(cached.y_true.to_numpy(), valid[target_column].to_numpy()):
        raise ValueError('Checkpoint labels differ from current data')
    validate_prediction(cached.y_pred.to_numpy(), valid)
    return True


def save_prediction(root, identity, run_signature, valid, target_column,
                    prediction, metadata):
    validate_prediction(prediction, valid)
    family, condition, target, fold, seed = identity
    path, meta_path = output_paths(root, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or meta_path.exists():
        raise FileExistsError(f'Refusing to replace existing checkpoint: {path.name}')
    output = valid[['pid', 'date']].assign(
        target=target, fold=fold, seed=seed, family=family, condition=condition,
        y_true=valid[target_column].to_numpy(), y_pred=prediction)
    meta = {'signature': run_signature, 'identity': list(identity),
            'n_rows': len(valid), 'n_people': int(valid.pid.nunique()),
            'eval_keys_sha256': fe.keys_hash(valid),
            'mae': float(np.abs(prediction - valid[target_column].to_numpy()).mean()),
            'final_test_scored': False, **metadata}
    temporary = path.with_suffix('.partial.parquet')
    output.to_parquet(temporary, index=False)
    temporary.replace(path)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({'event': 'complete', 'identity': identity,
                      'mae': meta['mae'], 'seconds': metadata['fit_seconds']}), flush=True)


def run_one(root, data, split, spec, hashes, target, fold, seed, condition,
            family='catboost'):
    y = spec['targets'][target]
    train, valid = select_fold(data, split, y, fold)
    baseline_features = list(spec['baseline_features'][target])
    if family == 'baseline':
        features = list(spec['feature_sets']['base'][target])
        model = baseline_model(condition)
        if condition in ('personal_median28', 'shrink_blend28'):
            features += baseline_features
        settings = {'model': type(model).__name__}
    else:
        features = condition_features(spec, target, condition)
        model = ResidualModel(fixed_catboost(seed), use_blend='A' in condition)
        settings = {'params': {**fe.CATBOOST_PARAMS, 'random_seed': seed},
                    'use_blend': 'A' in condition}
    fd.check_features(features, data)
    identity = (family, condition, target, fold, seed)
    run_signature = signature(hashes, identity, features, settings)
    if checkpoint_exists(root, identity, run_signature, valid, y):
        print(json.dumps({'event': 'skip', 'identity': identity}), flush=True)
        return
    print(json.dumps({'event': 'start', 'identity': identity}), flush=True)
    started = time.perf_counter()
    if family == 'baseline':
        model.fit(train, y, features)
        metadata = {'blend': model.params_() if hasattr(model, 'params_') else None}
    else:
        model.fit(train, y, features, baseline_features)
        metadata = model.metadata()
    prediction = np.asarray(model.predict(prediction_inputs(valid, features, baseline_features)), float)
    metadata.update({'fit_seconds': time.perf_counter() - started,
                     'features': features, 'settings': settings,
                     'input_fingerprints': hashes,
                     'n_train_people': int(train.pid.nunique()),
                     'n_train_rows': len(train), 'training': 'four outer development folds'})
    save_prediction(root, identity, run_signature, valid, y, prediction, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--targets', nargs='+', default=['sleep', 'bed', 'wake'])
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--skip-baselines', action='store_true')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args(argv)
    data, split, spec, hashes = load_inputs(args.root)
    for target in args.targets:
        for fold in args.folds:
            if not args.skip_baselines:
                for name in BASELINES:
                    run_one(args.root, data, split, spec, hashes, target, fold, 0, name, 'baseline')
            if not args.baseline_only:
                for seed in args.seeds:
                    for condition in args.conditions:
                        run_one(args.root, data, split, spec, hashes, target, fold, seed, condition)


if __name__ == '__main__':
    main()
