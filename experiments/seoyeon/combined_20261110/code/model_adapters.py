"""Adapters that preserve upstream models and keep blending inside training folds."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'snapshot'))

from src import forecast_eval as fe
from experiments.seoyeon import baseline_blend as bb

CONDITIONS = ('base', 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC')


def condition_features(spec, target, condition):
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    key = ''.join(c for c in 'BC' if c in condition) or 'base'
    features = list(spec['feature_sets'][key][target])
    if len(features) != len(set(features)):
        raise ValueError('Duplicate model input columns')
    return features


def inner_people(people, seed, fold):
    ordered = np.asarray(sorted(people), dtype=str)
    if len(ordered) < 3:
        raise ValueError('At least three training participants required')
    shuffled = np.random.default_rng(1000 * seed + fold).permutation(ordered)
    n_valid = max(1, int(np.ceil(0.1 * len(ordered))))
    return set(shuffled[n_valid:]), set(shuffled[:n_valid])


class ResidualModel:
    """Fit the original shrinkage baseline, then a predictor of its residual."""

    def __init__(self, model, use_blend):
        self.model = model
        self.use_blend = use_blend
        self.base = None

    def fit(self, train, target, features, baseline_features):
        self.features = list(features)
        self.baseline_features = list(baseline_features)
        if self.use_blend:
            self.base = bb.ShrinkBlend().fit(train, target, self.baseline_features)
            residuals = train[target].to_numpy(float) - self.base.predict(train)
            self.model.fit(train.assign(**{target: residuals}), target, self.features)
        else:
            self.model.fit(train, target, self.features)
        return self

    def predict(self, frame):
        prediction = np.asarray(self.model.predict(frame), dtype=float)
        return prediction if self.base is None else prediction + self.base.predict(frame)

    def metadata(self):
        return {'blend': None if self.base is None else self.base.params_(),
                'model_params': getattr(self.model, 'params', {}),
                'features': self.features,
                'baseline_features': self.baseline_features if self.use_blend else []}


def fixed_catboost(seed):
    return fe.CatBoostModel(random_seed=seed)


def baseline_model(name):
    models = {'group_median': fe.GroupMedian,
              'personal_mean7': fe.PersonalMean,
              'personal_median28': bb.PersonalMedian,
              'shrink_blend28': bb.ShrinkBlend}
    return models[name]()


def comparison_config(family):
    from experiments.minjun.models import models as mm
    if family == 'catboost':
        return dict(fe.CATBOOST_PARAMS)
    return dict({'ridge': mm.RidgeM, 'mlp': mm.MLPM, 'ftt': mm.FTTM}[family].GRID[0])


def fit_comparison(family, seed, train, inner_valid, target, features,
                   baseline_features, use_blend, device='cpu'):
    """Every family receives exactly the same inner training participants."""
    from experiments.minjun.models import models as mm
    base = bb.ShrinkBlend().fit(train, target, baseline_features) if use_blend else None
    y_train = train[target].to_numpy(float)
    y_valid = inner_valid[target].to_numpy(float)
    if base is not None:
        y_train = y_train - base.predict(train)
        y_valid = y_valid - base.predict(inner_valid)
    config = comparison_config(family)
    if family == 'catboost':
        model = fixed_catboost(seed)
        model.fit(train.assign(**{target: y_train}), target, features)
    elif family == 'ridge':
        model = mm.RidgeM(config, seed).fit(train, y_train, inner_valid, y_valid, features)
    else:
        import torch
        torch.set_num_threads(4)
        cls = DeviceMLP if family == 'mlp' else BoundedFTT
        model = cls(config, seed, device=device).fit(train, y_train, inner_valid, y_valid, features)
    wrapper = ResidualModel(model, use_blend)
    wrapper.base = base
    wrapper.features = list(features)
    wrapper.baseline_features = list(baseline_features)
    info = {'config': config, 'fit_info': getattr(model, 'info', {}),
            'device': getattr(model, 'dev', 'cpu'),
            'requested_device': device,
            'prediction_batch': 1024 if family == 'ftt' else None,
            'selection': 'first upstream configuration, fixed before experiments',
            **wrapper.metadata()}
    return wrapper, info


def require_device(device):
    if device not in ('cpu', 'mps'):
        raise ValueError(f'Unsupported device: {device}')
    if device == 'mps':
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested but is not available; no CPU fallback is allowed')
    return device


class DeviceMixin:
    """Set the chosen device before the unchanged upstream fit creates tensors."""

    def __init__(self, config, seed, device='cpu'):
        self.requested_device = require_device(device)
        super().__init__(config, seed)

    def _tensors(self, frame):
        self.dev = self.requested_device
        return super()._tensors(frame)


# Smaller evaluation batches change memory use, not the fitted architecture.
from experiments.minjun.models.models import FTTM, MLPM


class DeviceMLP(DeviceMixin, MLPM):
    pass


class BoundedFTT(DeviceMixin, FTTM):
    def _pred_t(self, tensors, batch=1024):
        return super()._pred_t(tensors, batch=batch)
