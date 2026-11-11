"""Behavioral checks for participant isolation and residual reconstruction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'snapshot'))

from model_adapters import ResidualModel, condition_features, inner_people
from run_factorial import validate_prediction, select_fold


def example():
    return pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'], 'date': pd.to_datetime(['2020-01-01'] * 3),
        'forecast_at': pd.to_datetime(['2020-01-01 18:00'] * 3),
        'target_sleep_min': [400.0, 500.0, 600.0],
        'sleep_pmed28': [420.0, np.nan, 550.0], 'sleep_n28': [10, 0, 4],
        'sleep_lag1': [410.0, np.nan, 590.0], 'dow': [2, 2, 2],
    })


class MemorizedResidual:
    def fit(self, train, y, features):
        self.y = train[y].to_numpy().copy()
        self.features = list(features)
        return self

    def predict(self, frame):
        assert 'target_sleep_min' not in frame
        return self.y


def test_residual_restores_original_target_and_does_not_mutate_training():
    train = example()
    original = train.copy(deep=True)
    model = ResidualModel(MemorizedResidual(), use_blend=True)
    model.fit(train, 'target_sleep_min', ['sleep_lag1', 'dow'],
              ['sleep_pmed28', 'sleep_n28'])
    pred = model.predict(train.drop(columns='target_sleep_min'))
    np.testing.assert_allclose(pred, original.target_sleep_min)
    pd.testing.assert_frame_equal(train, original)
    assert model.base.group_ == 500.0
    assert model.model.features == ['sleep_lag1', 'dow']


def test_predictions_do_not_use_evaluation_targets():
    train = example()
    model = ResidualModel(MemorizedResidual(), use_blend=True)
    model.fit(train, 'target_sleep_min', ['sleep_lag1', 'dow'],
              ['sleep_pmed28', 'sleep_n28'])
    valid = train.drop(columns='target_sleep_min')
    result = model.predict(valid)
    assert np.isfinite(result).all()
    assert model.base.group_ == 500.0


def test_base_does_not_smuggle_personal_history_into_model_features():
    spec = {'feature_sets': {
        'base': {'sleep': ['sleep_lag1', 'dow']},
        'B': {'sleep': ['sleep_lag1', 'dow', 'offday']},
        'C': {'sleep': ['sleep_lag1', 'dow', 'rec1_sleep']},
        'BC': {'sleep': ['sleep_lag1', 'dow', 'offday', 'rec1_sleep']},
    }}
    assert condition_features(spec, 'sleep', 'A') == ['sleep_lag1', 'dow']
    assert condition_features(spec, 'sleep', 'ABC')[-2:] == ['offday', 'rec1_sleep']
    with pytest.raises(ValueError):
        condition_features(spec, 'sleep', 'invalid')


def test_inner_people_disjoint_reproducible_and_contained():
    people = {f'p{i:03d}' for i in range(100)}
    train, valid = inner_people(people, seed=1, fold=3)
    assert len(train) == 90 and len(valid) == 10
    assert train.isdisjoint(valid) and train | valid == people
    assert (train, valid) == inner_people(people, seed=1, fold=3)


def test_select_fold_keeps_all_valid_rows_and_excludes_final():
    data = example()
    split = pd.DataFrame({'pid': ['p1', 'p2', 'p3'],
                          'role': ['dev', 'dev', 'final'], 'fold': [1, 2, 0]})
    train, valid = select_fold(data, split, 'target_sleep_min', 1)
    assert train.pid.tolist() == ['p2']
    assert valid.pid.tolist() == ['p1']
    assert not (set(train.pid) | set(valid.pid)) & {'p3'}


@pytest.mark.parametrize('prediction', [[1.0], [1.0, np.nan, 3.0],
                                       [1.0, np.inf, 3.0]])
def test_rejects_missing_or_nonfinite_predictions(prediction):
    with pytest.raises(ValueError):
        validate_prediction(np.asarray(prediction), example())


def test_rejects_duplicate_evaluation_rows():
    data = example()
    duplicated = pd.concat([data.iloc[:1], data.iloc[:1]])
    with pytest.raises(ValueError):
        validate_prediction(np.array([1.0, 2.0]), duplicated)


def test_comparison_blend_is_fit_only_on_inner_training(monkeypatch):
    import model_adapters as adapters
    from experiments.minjun.models import models as upstream

    class CaptureRidge:
        def __init__(self, config, seed):
            self.info = {}

        def fit(self, train, y_train, valid, y_valid, features):
            self.y_train = y_train
            self.y_valid = y_valid
            return self

        def predict(self, frame):
            assert 'target_sleep_min' not in frame
            return np.zeros(len(frame))

    monkeypatch.setattr(upstream, 'RidgeM', CaptureRidge)
    monkeypatch.setattr(adapters, 'comparison_config', lambda _: {'alpha': 1.0})
    train = example()
    inner = example().assign(target_sleep_min=10000.0, pid=['v1', 'v2', 'v3'])
    model, _ = adapters.fit_comparison(
        'ridge', 0, train, inner, 'target_sleep_min', ['sleep_lag1', 'dow'],
        ['sleep_pmed28', 'sleep_n28'], True)
    assert model.base.group_ == 500.0
    np.testing.assert_allclose(model.model.y_train,
                               train.target_sleep_min - model.base.predict(train))
    np.testing.assert_allclose(model.model.y_valid,
                               inner.target_sleep_min - model.base.predict(inner))


def test_target_columns_cannot_be_passed_to_predict():
    from run_factorial import prediction_inputs
    with pytest.raises(ValueError):
        prediction_inputs(example(), ['target_sleep_min'])


def test_checkpoint_roundtrip_and_label_change_rejection(tmp_path):
    pytest.importorskip('pyarrow')
    from run_factorial import save_prediction, checkpoint_exists
    frame = example()
    identity = ('catboost', 'A', 'sleep', 1, 0)
    pred = np.array([410.0, 490.0, 590.0])
    save_prediction(tmp_path, identity, 'sig', frame, 'target_sleep_min', pred,
                    {'fit_seconds': 1.0})
    assert checkpoint_exists(tmp_path, identity, 'sig', frame, 'target_sleep_min')
    with pytest.raises(ValueError, match='configuration mismatch'):
        checkpoint_exists(tmp_path, identity, 'different', frame, 'target_sleep_min')
    with pytest.raises(ValueError, match='labels differ'):
        checkpoint_exists(tmp_path, identity, 'sig', frame.assign(target_sleep_min=1),
                          'target_sleep_min')
    with pytest.raises(FileExistsError):
        save_prediction(tmp_path, identity, 'sig', frame, 'target_sleep_min', pred,
                        {'fit_seconds': 1.0})


def test_incomplete_checkpoint_is_not_silently_reused(tmp_path):
    from run_factorial import checkpoint_exists, output_paths
    identity = ('catboost', 'ABC', 'sleep', 1, 0)
    path, _ = output_paths(tmp_path, identity)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        checkpoint_exists(tmp_path, identity, 'sig', example(), 'target_sleep_min')


def test_model_parameters_change_signature():
    from run_factorial import signature
    identity = ('catboost', 'base', 'sleep', 1, 0)
    a = signature({'data': 'a'}, identity, ['x'], {'depth': 6})
    assert a != signature({'data': 'a'}, identity, ['x'], {'depth': 7})
    assert a != signature({'data': 'b'}, identity, ['x'], {'depth': 6})


def test_real_catboost_residual_predicts_finite_values_without_targets():
    pytest.importorskip('catboost')
    from src.forecast_eval import CatBoostModel
    train = pd.concat([example().assign(date=lambda d: d.date + pd.Timedelta(days=i))
                       for i in range(5)], ignore_index=True)
    model = ResidualModel(CatBoostModel(iterations=3, thread_count=1), use_blend=True)
    model.fit(train, 'target_sleep_min', ['sleep_lag1', 'dow'],
              ['sleep_pmed28', 'sleep_n28'])
    pred = model.predict(example().drop(columns='target_sleep_min'))
    assert pred.shape == (3,) and np.isfinite(pred).all()


def test_prepared_artifacts_must_match_frozen_spec():
    from run_factorial import check_prepared_hashes
    from src.forecast_split import make_split, split_hash
    split = make_split([f'p{i:03d}' for i in range(20)])
    spec = {'feature_file_sha256': 'prepared-feature-hash', 'split_sha256': split_hash(split)}
    check_prepared_hashes(spec, 'prepared-feature-hash', split)
    with pytest.raises(ValueError, match='Prepared feature file differs'):
        check_prepared_hashes(spec, 'changed-feature-hash', split)
    changed_split = split.copy()
    changed_split.loc[0, 'pid'] = 'different-person'
    with pytest.raises(ValueError, match='Participant split differs'):
        check_prepared_hashes(spec, 'prepared-feature-hash', changed_split)


def test_unavailable_mps_fails_explicitly(monkeypatch):
    torch = pytest.importorskip('torch')
    from model_adapters import require_device
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='MPS was requested'):
        require_device('mps')
    assert require_device('cpu') == 'cpu'
    with pytest.raises(ValueError, match='Unsupported device'):
        require_device('cuda')


@pytest.mark.parametrize('family', ['mlp', 'ftt'])
def test_explicit_cpu_matches_upstream_training(family, monkeypatch):
    torch = pytest.importorskip('torch')
    import model_adapters as adapters
    from experiments.minjun.models import models as upstream
    torch.set_num_threads(1)
    monkeypatch.setattr(upstream._TorchM, 'MAX_EPOCHS', 2)
    rng = np.random.default_rng(9)
    data = pd.DataFrame({'sleep_lag1': rng.normal(400, 30, 48),
                         'dow': np.arange(48) % 7})
    y = data.sleep_lag1.to_numpy() + rng.normal(0, 10, 48)
    original_class = upstream.MLPM if family == 'mlp' else upstream.FTTM
    adapter_class = adapters.DeviceMLP if family == 'mlp' else adapters.BoundedFTT
    config = dict(original_class.GRID[0])
    args = (data.iloc[:32], y[:32], data.iloc[32:], y[32:], list(data))
    original = original_class(config, 4).fit(*args)
    adapted = adapter_class(config, 4, device='cpu').fit(*args)
    np.testing.assert_allclose(adapted.predict(data.iloc[32:]),
                               original.predict(data.iloc[32:]), atol=1e-7, rtol=1e-7)
    assert adapted.dev == 'cpu'


@pytest.fixture
def small_artifacts(tmp_path):
    import json
    import shutil
    from src.forecast_split import make_split, split_hash
    from run_factorial import file_sha
    pytest.importorskip('pyarrow')
    split = make_split([f'p{i:03d}' for i in range(20)])
    people = split.loc[split.role == 'dev', 'pid'].tolist()
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({'pid': np.repeat(people, 4),
                          'date': np.tile(pd.date_range('2020-01-01', periods=4), len(people))})
    frame = frame.assign(forecast_at=frame.date + pd.Timedelta(hours=18),
                         target_sleep_min=rng.normal(440, 35, len(frame)),
                         sleep_pmed28=rng.normal(440, 25, len(frame)),
                         sleep_n28=rng.integers(1, 29, len(frame)),
                         sleep_lag1=rng.normal(440, 25, len(frame)),
                         dow=frame.date.dt.dayofweek,
                         offday=0.0, rec1_sleep=rng.normal(440, 25, len(frame)))
    folder = tmp_path / 'artifacts'
    folder.mkdir()
    frame.to_parquet(folder / 'features.parquet', index=False)
    split.to_csv(folder / 'split.csv', index=False)
    basic = ['sleep_lag1', 'dow']
    spec = {'targets': {'sleep': 'target_sleep_min'},
            'feature_sets': {key: {'sleep': basic + extra} for key, extra in
                             [('base', []), ('B', ['offday']), ('C', ['rec1_sleep']),
                              ('BC', ['offday', 'rec1_sleep'])]},
            'baseline_features': {'sleep': ['sleep_pmed28', 'sleep_n28']},
            'feature_file_sha256': file_sha(folder / 'features.parquet'),
            'split_sha256': split_hash(split)}
    (folder / 'feature_spec.json').write_text(json.dumps(spec))
    (tmp_path / 'code').mkdir()
    for name in ('model_adapters.py', 'run_factorial.py', 'run_models.py'):
        shutil.copyfile(ROOT / 'code' / name, tmp_path / 'code' / name)
    return tmp_path


def test_real_factorial_cli_preserves_rows_and_reuses_checkpoints(small_artifacts):
    import json
    pytest.importorskip('catboost')
    from run_factorial import main, load_inputs
    args = ['--root', str(small_artifacts), '--targets', 'sleep', '--folds', '1',
            '--seeds', '0', '--conditions', 'base', 'A']
    main(args)
    folder = small_artifacts / 'artifacts' / 'predictions'
    paths = sorted(folder.glob('*.parquet'))
    assert len(paths) == 6
    keys = []
    for path in paths:
        pred = pd.read_parquet(path)
        keys.append(list(zip(pred.pid, pred.date)))
        assert np.isfinite(pred.y_pred).all()
        assert json.loads(path.with_suffix('.json').read_text())['final_test_scored'] is False
    assert all(key == keys[0] for key in keys)
    before = {p.name: p.stat().st_mtime_ns for p in paths}
    main(args)
    assert before == {p.name: p.stat().st_mtime_ns for p in paths}
    data, split, _, _ = load_inputs(small_artifacts)
    assert not set(data.pid) & set(split.loc[split.role == 'final', 'pid'])


def test_real_model_cli_uses_same_inner_people_and_restores_blend(small_artifacts, monkeypatch):
    import json
    pytest.importorskip('torch')
    pytest.importorskip('catboost')
    from experiments.minjun.models import models as upstream
    from run_models import main
    monkeypatch.setattr(upstream._TorchM, 'MAX_EPOCHS', 2)
    main(['--root', str(small_artifacts), '--targets', 'sleep', '--folds', '1',
          '--seeds', '0', '--device', 'cpu'])
    folder = small_artifacts / 'artifacts' / 'predictions'
    metadata = [json.loads(p.read_text()) for p in folder.glob('*.json')]
    assert len(metadata) == 8
    assert len({m['n_train_people'] for m in metadata}) == 1
    assert len({m['n_inner_valid_people'] for m in metadata}) == 1
    assert len({m['eval_keys_sha256'] for m in metadata}) == 1
    assert all(m['device'] == 'cpu' and m['settings']['device'] == 'cpu' for m in metadata)
    assert all(m['blend'] is not None for m in metadata if m['identity'][1] == 'ABC')
    assert all(m['blend'] is None for m in metadata if m['identity'][1] == 'base')
    assert all(np.isfinite(pd.read_parquet(p).y_pred).all() for p in folder.glob('*.parquet'))
