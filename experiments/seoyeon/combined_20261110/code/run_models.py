"""Matched-participant model comparisons for basic and fully combined inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from model_adapters import (ROOT, comparison_config, condition_features,
                            fit_comparison, inner_people, require_device)
from run_factorial import (checkpoint_exists, load_inputs, prediction_inputs,
                           save_prediction, select_fold, signature)
from src import forecast_dataset as fd


def run_one(root, data, split, spec, hashes, family, condition, target, fold, seed, device='cpu'):
    y = spec['targets'][target]
    outer_train, valid = select_fold(data, split, y, fold)
    train_people, inner_valid_people = inner_people(set(outer_train.pid), seed, fold)
    train = outer_train.loc[outer_train.pid.isin(train_people)].copy()
    inner_valid = outer_train.loc[outer_train.pid.isin(inner_valid_people)].copy()
    if train.empty or inner_valid.empty:
        raise ValueError('Empty matched inner partition')
    if set(valid.pid) & (train_people | inner_valid_people):
        raise ValueError('Outer evaluation participant entered inner training')
    features = condition_features(spec, target, condition)
    fd.check_features(features, data)
    baseline_features = spec['baseline_features'][target]
    identity = (f'modelcheck_{family}', condition, target, fold, seed)
    settings = {'config': comparison_config(family), 'use_blend': condition == 'ABC',
                'inner_seed': 1000 * seed + fold, 'inner_valid_fraction': 0.1,
                'device': device if family in ('mlp', 'ftt') else 'cpu',
                'selection': 'first upstream configuration fixed before experiments'}
    run_signature = signature(hashes, identity, features, settings)
    if checkpoint_exists(root, identity, run_signature, valid, y):
        print(json.dumps({'event': 'skip', 'identity': identity}), flush=True)
        return
    print(json.dumps({'event': 'start', 'identity': identity}), flush=True)
    started = time.perf_counter()
    model, info = fit_comparison(family, seed, train, inner_valid, y, features,
                                baseline_features, use_blend=condition == 'ABC', device=device)
    prediction = np.asarray(model.predict(prediction_inputs(valid, features, baseline_features)), float)
    info.update({'fit_seconds': time.perf_counter() - started, 'settings': settings,
                 'input_fingerprints': hashes,
                 'n_train_people': len(train_people), 'n_inner_valid_people': len(inner_valid_people),
                 'n_train_rows': len(train), 'n_inner_valid_rows': len(inner_valid),
                 'training': 'same 90% of outer-training people for all four model families'})
    save_prediction(root, identity, run_signature, valid, y, prediction, info)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--families', nargs='+', default=['catboost', 'ridge', 'mlp', 'ftt'],
                        choices=['catboost', 'ridge', 'mlp', 'ftt'])
    parser.add_argument('--conditions', nargs='+', default=['base', 'ABC'], choices=['base', 'ABC'])
    parser.add_argument('--targets', nargs='+', default=['sleep', 'bed', 'wake'])
    parser.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    args = parser.parse_args(argv)
    if set(args.families) & {'mlp', 'ftt'}:
        require_device(args.device)
    data, split, spec, hashes = load_inputs(args.root)
    for target in args.targets:
        for fold in args.folds:
            for seed in args.seeds:
                for condition in args.conditions:
                    for family in args.families:
                        run_one(args.root, data, split, spec, hashes, family, condition,
                                target, fold, seed, device=args.device)


if __name__ == '__main__':
    main()
