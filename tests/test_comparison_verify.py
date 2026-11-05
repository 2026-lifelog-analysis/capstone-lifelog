import pandas as pd
import pytest
import json

from experiments.seoyeon.comparison_verify import compare_tables
from experiments.seoyeon import comparison_verify as verifier


def test_all_cells_and_identifier_types_are_compared(tmp_path):
    frame = pd.DataFrame({'participant': ['001', '002'], 'date': ['2020-01-01'] * 2,
                          'score': [1.0000000000000002, float('nan')]})
    first, second = tmp_path / 'first.csv.gz', tmp_path / 'second.csv.gz'
    frame.to_csv(first, index=False)
    frame.to_csv(second, index=False)
    result = compare_tables(first, second)
    assert result == {'rows': 2, 'columns': 3, 'all_cells_exact': True}
    frame.loc[0, 'score'] = 1.
    frame.to_csv(second, index=False)
    with pytest.raises(AssertionError):
        compare_tables(first, second)


@pytest.mark.parametrize('mismatch', [False, True])
def test_verifier_checks_actual_outputs_and_preserves_failure(tmp_path, monkeypatch, mismatch):
    run, replay = tmp_path / 'initial', tmp_path / 'replay'
    run.mkdir()
    (run / 'inputs_before.json').write_text('{}')
    pd.DataFrame({'score':[1.]}).to_csv(run / 'scores.csv', index=False)
    (run / 'settings.json').write_text('{"fixed":true}')
    monkeypatch.setattr(verifier, 'TABLES', ['scores.csv'])
    monkeypatch.setattr(verifier, 'RECORDS', ['settings.json'])
    monkeypatch.setattr(verifier.io, 'preserved_inputs', lambda: {})
    def prepare(path):
        path.mkdir()
        pd.DataFrame({'score':[2. if mismatch else 1.]}).to_csv(path / 'scores.csv', index=False)
        (path / 'settings.json').write_text('{"fixed":true}')
    monkeypatch.setattr(verifier.runner, 'prepare', prepare)
    monkeypatch.setattr(verifier.runner, 'score', lambda path: None)
    monkeypatch.setattr(verifier.runner, 'fit', lambda path: None)
    if mismatch:
        with pytest.raises(AssertionError):
            verifier.verify(run, replay)
        assert json.loads((replay / 'verification_failure.json').read_text())['status'] == 'failed'
    else:
        verifier.verify(run, replay)
        assert json.loads((replay / 'verification.json').read_text())['tables']['scores.csv']['all_cells_exact']
    with pytest.raises(FileExistsError):
        verifier.verify(run, replay)
