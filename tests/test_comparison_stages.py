"""Re-entry must never alter previously computed comparison artifacts."""
import pytest
import json
import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_run as run
from experiments.seoyeon.comparison_evaluate import finalize_clean
from experiments.seoyeon import comparison_io as io
from experiments.seoyeon import comparison_models as models
from experiments.seoyeon import comparison_seungho as seungho
from experiments.seoyeon import comparison_safety as safety


@pytest.mark.parametrize('stage,filename', [
    (run.score, 'seoyeon_scores.csv.gz'),
    (run.fit, 'clean_scores_r033.csv.gz'),
    (finalize_clean, 'alarm_summary.csv'),
])
def test_completed_or_partial_stage_is_rejected_before_any_read_or_write(tmp_path, stage, filename):
    artifact = tmp_path / filename
    artifact.write_bytes(b'existing-result')
    with pytest.raises(FileExistsError, match='stage output'):
        stage(tmp_path)
    assert artifact.read_bytes() == b'existing-result'
    assert [p.name for p in tmp_path.iterdir()] == [filename]


def test_source_guard_detects_changes(monkeypatch):
    before = safety.source_hashes(['comparison_safety.py'])
    assert safety.verify_sources(before)['source_preserved']
    monkeypatch.setattr(safety, 'source_hashes', lambda names: {'comparison_safety.py': 'changed'})
    with pytest.raises(ValueError, match='changed during'):
        safety.verify_sources(before)


def test_score_and_fit_small_fixture(tmp_path, monkeypatch):
    main = pd.DataFrame({'pid': ['001'] * 10, 'date': pd.date_range('2020-01-01', periods=10)})
    main.to_csv(tmp_path / 'main_daily.csv.gz', index=False)
    split = io.calendar_split(main.date)
    io.write_json(tmp_path / 'split.json', split)
    monkeypatch.setattr(models, 'adapt_main', lambda main: main.rename(columns={'pid': 'participant'}))

    def small_models(daily, development_end, calibration_start):
        frames = [daily.assign(model=name, score=np.arange(10, dtype=float), valid=True)
                  for name in ['fixed14', 'fixed28', 'rolling28', 'cusum28', 'group']]
        return pd.concat(frames), pd.DataFrame({'participant': ['001']})

    monkeypatch.setattr(models, 'score_models', small_models)
    run.score(tmp_path)
    record = json.loads((tmp_path / 'seoyeon_execution.json').read_text())
    assert record['rows'] == 50 and record['moment_requested_rows'] == 6
    assert record['source_preserved']
    names = ['sh_base', 'sh_dow', 'sh_common', 'sh_final']
    monkeypatch.setattr(seungho, 'directional_scores', lambda main: pd.DataFrame({n: np.arange(10.) for n in names}))
    monkeypatch.setattr(seungho, 'fit_h', lambda main, bad, mask, rate: {n: 6. for n in names})
    monkeypatch.setattr(seungho, 'score_with_h', lambda main, bad, h: bad.copy())
    run.fit(tmp_path)
    fitted = json.loads((tmp_path / 'fit_execution.json').read_text())
    assert fitted['calibration_days'] == 3 and fitted['test_days'] == 3
    settings = json.loads((tmp_path / 'clean_thresholds_without_moment.json').read_text())
    assert all(s['thresholds']['fixed28'] == 6. for s in settings)


def test_prepare_uses_main_and_preserves_originals(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(io, 'assert_main_source', lambda: None)
    monkeypatch.setattr(io, 'preserved_inputs', lambda: {'original': 'sha'})
    daily = pd.DataFrame({'pid': ['001'] * 10, 'date': pd.date_range('2020-01-01', periods=10)})
    monkeypatch.setattr(run.data_loader, 'load_nethealth', lambda: daily)
    monkeypatch.setattr(run, 'preparation', lambda daily: (pd.DataFrame({'rows': [10]}), {}))
    target = tmp_path / 'results/seoyeon_tuning/test'
    run.prepare(target)
    record = json.loads((target / 'preprocessing_execution.json').read_text())
    assert record['rows'] == 10 and record['people'] == 1
    assert record['original_source_preserved'] and record['source_preserved']
    with pytest.raises(FileExistsError):
        run.prepare(target)
    with pytest.raises(ValueError, match='new local'):
        run.prepare(tmp_path / 'wrong')
