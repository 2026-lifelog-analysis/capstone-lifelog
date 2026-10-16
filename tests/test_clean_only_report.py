"""Publication must use verified clean results, never synthetic placeholders."""
import json
import ast
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon.clean_only_report import generate_clean_report


@pytest.fixture
def evidence(tmp_path):
    view = tmp_path / 'run_view'
    (view / 'moment_clean').mkdir(parents=True)
    (view / 'replay_verification').mkdir()
    rows = [dict(rate_id=r, model=m, split=s, scope='common', days=n,
                 people=p, alarm_days=100, alarm_rate=100/n, zero_people=1,
                 top_people=int(p*.1), top10_alarm_share=.25)
            for r in ['r05', 'r033'] for m in MODELS
            for s, n, p in [('calibration', 36260, 181), ('test', 23721, 135)]]
    pd.DataFrame(rows).to_csv(view / 'alarm_summary.csv', index=False)
    pd.DataFrame([dict(rate_id=r, model=m, split=s, finite_days=n+20, common_days=n)
                  for r in ['r05', 'r033'] for m in MODELS
                  for s, n in [('calibration', 36260), ('test', 23721)]]).to_csv(
        view / 'coverage.csv', index=False)
    pd.DataFrame([dict(rate_id=r, a=a, b=b, common_days=23721,
                      a_alarm_days=100, b_alarm_days=100, overlap_days=50,
                      jaccard=1/3, disagreement_days=100)
                  for r in ['r05', 'r033'] for a, b in combinations(MODELS, 2)]).to_csv(
        view / 'pairwise.csv', index=False)
    io.write_json(view / 'clean_thresholds.json', [])
    names = ['alarm_summary.csv', 'coverage.csv', 'pairwise.csv', 'clean_thresholds.json']
    io.write_json(view / 'clean_evaluation.json', dict(status='passed',
        outputs_sha256={n: io.sha256(view / n) for n in names}, threshold_test_access=False,
        all_requested_common_moment_finite=True, calibration_rows=36260, test_rows=23721))
    io.write_json(view / 'moment_clean/execution.json', dict(status='complete',
        requested_persondays=87886, completed_persondays=87886,
        new_inferred_persondays=9550, exact_six_field_cache_persondays=78336,
        cross_process_exact_six_field_cache_persondays=78336))
    io.write_json(view / 'split.json', dict(start='2015-07-24', development_end='2017-01-31',
        calibration_start='2017-02-01', calibration_end='2018-03-26',
        test_start='2018-03-27', end='2019-05-19'))
    io.write_json(view / 'protocol.json', io.protocol())
    io.write_json(view / 'model_port_validation.json', dict(status='passed', ast_function_count=13,
        sample_people=15, sample_person_days=10119, score_rows=50595,
        exact_all_cells_scores_and_baselines=True))
    io.write_json(view / 'replay_verification/verification.json', dict(status='passed',
        tables={str(i): {'all_cells_exact': True} for i in range(11)}))
    io.write_json(view.parent / 'cache_provenance.json', dict(verified_persondays=78336,
        fresh_replay_persondays=8, fresh_replay_max_abs=0,
        all_six_inputs_rebuilt=True, all_three_endpoint_errors_and_mse_recomputed=True,
        pre_interruption_chunk_hashes_available=False, model_runtime_identical=True))
    return view, tmp_path / 'report.md', tmp_path / 'figures'


def test_verified_clean_report_and_graph_hashes(evidence):
    view, report, figures = evidence
    result = generate_clean_report(view, report, figures)
    assert result['status'] == 'passed'
    assert result['report_sha256'] == io.sha256(report)
    assert len(result['outputs_sha256']) == 4
    for name, digest in result['outputs_sha256'].items():
        assert io.sha256(figures / name) == digest
    text = report.read_text()
    for required in ['합성 실험은 이번 공개 범위에서 제외', '미실시', '78,336', '9,550',
                     '23,721', '36,260', '정답 상태 라벨', '새 클론', '1.4826', 'Jaccard']:
        assert required in text
    assert '/Users/' not in text
    assert '20seed 평균' not in text
    with pytest.raises(FileExistsError):
        generate_clean_report(view, report, figures)


@pytest.mark.parametrize('field,value', [('status', 'running'), ('completed_persondays', 87885),
    ('new_inferred_persondays', 9551)])
def test_rejects_incomplete_or_inconsistent_inference(evidence, field, value):
    view, report, figures = evidence
    path = view / 'moment_clean/execution.json'
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, field: value}))
    with pytest.raises(ValueError):
        generate_clean_report(view, report, figures)
    assert not report.exists() and not figures.exists()


def test_rejects_changed_producer_file(evidence):
    view, report, figures = evidence
    with (view / 'alarm_summary.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        generate_clean_report(view, report, figures)


def test_rejects_invalid_recovery_proof(evidence):
    view, report, figures = evidence
    path = view.parent / 'cache_provenance.json'
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, 'all_six_inputs_rebuilt': False}))
    with pytest.raises(ValueError, match='recovery'):
        generate_clean_report(view, report, figures)


def test_rejects_different_protocol(evidence):
    view, report, figures = evidence
    path = view / 'protocol.json'
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, 'fractions': [.5, .2, .3]}))
    with pytest.raises(ValueError, match='protocol'):
        generate_clean_report(view, report, figures)


def test_notebook_final_completion_gate(tmp_path):
    path = Path(__file__).parents[1] / 'notebooks/03_서연_튜닝비교.ipynb'
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['outputs'] == [] and cell['execution_count'] is None for cell in code)
    assert all('subprocess' not in ''.join(cell['source']) for cell in code)
    config = next(cell for cell in code if cell['id'] == 'comparison-config')
    tree = ast.parse(''.join(config['source']))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {'json': json, 'sha256': io.sha256}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    verify = namespace['verify_completion']
    view = tmp_path / 'run_view'
    view.mkdir()
    artifact = view / 'proof.json'
    artifact.write_text('{}')
    record = dict(status='complete', scope='clean_only', clean_completed_claim=True,
                  full_comparison_completed_claim=False, synthetic_moment_jobs_completed=0,
                  report_sha256='report-hash', outputs_sha256={'run_view/proof.json': io.sha256(artifact)})
    destination = tmp_path / 'clean_only_completion.json'
    with pytest.raises(FileNotFoundError):
        verify(view, 'report-hash')
    destination.write_text(json.dumps(record))
    assert verify(view, 'report-hash')['status'] == 'complete'
    for key, wrong in [('status', 'failed'), ('scope', 'full'), ('clean_completed_claim', False),
                       ('full_comparison_completed_claim', True), ('synthetic_moment_jobs_completed', 1),
                       ('report_sha256', 'wrong'), ('outputs_sha256', {})]:
        destination.write_text(json.dumps({**record, key: wrong}))
        with pytest.raises(ValueError):
            verify(view, 'report-hash')
    destination.write_text(json.dumps(record))
    artifact.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='산출물'):
        verify(view, 'report-hash')
