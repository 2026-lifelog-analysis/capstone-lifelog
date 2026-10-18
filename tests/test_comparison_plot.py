import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from experiments.seoyeon.comparison_plot import plot
from experiments.seoyeon import comparison_io as io

from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon.comparison_plot import clean_figure, synthetic_figure


def test_clean_plot_marks_both_actual_rates_and_target():
    frame = pd.DataFrame([dict(rate_id=r, model=m, split=s, scope='common', alarm_rate=.04)
        for r in ['r05', 'r033'] for m in MODELS for s in ['calibration', 'test']])
    fig = clean_figure(frame)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 20
    assert fig.axes[0].patches[0].get_width() == 4.
    plt.close(fig)


def test_synthetic_plot_uses_seed_means_and_paired_difference():
    frame = pd.DataFrame([dict(rate_id=r, scenario=s, model=m, seed=seed,
                               detection_rate=.6, clean_alarm_rate=.2, delta_mean=.4)
        for r in ['r05', 'r033'] for s in ['S1', 'S2', 'S3', 'S4', 'S5']
        for m in MODELS for seed in range(20)])
    fig = synthetic_figure(frame, 'delta_mean')
    assert len(fig.axes) == 4
    assert np.allclose(fig.axes[0].images[0].get_array(), 40.)
    plt.close(fig)
    with pytest.raises(ValueError, match='Unknown'):
        synthetic_figure(frame, 'accuracy')


def test_plot_end_to_end_writes_six_files_only_from_verified_tables(tmp_path):
    run, agg = tmp_path / 'run', tmp_path / 'aggregate'
    run.mkdir()
    agg.mkdir()
    alarms = pd.DataFrame([dict(rate_id=r, model=m, split=s, scope='common', alarm_rate=.04)
        for r in ['r05','r033'] for m in MODELS for s in ['calibration','test']])
    alarms.to_csv(run / 'alarm_summary.csv', index=False)
    seeds = pd.DataFrame([dict(rate_id=r, scenario=s, model=m, seed=seed,
                               detection_rate=.6, clean_alarm_rate=.2, delta_mean=.4)
        for r in ['r05','r033'] for s in ['S1','S2','S3','S4','S5'] for m in MODELS for seed in range(20)])
    seeds.to_csv(agg / 'seed_summary.csv', index=False)
    io.write_json(run / 'clean_evaluation.json', dict(status='passed',
        outputs_sha256={'alarm_summary.csv': io.sha256(run / 'alarm_summary.csv')}))
    io.write_json(agg / 'aggregate_manifest.json', dict(status='passed',
        outputs_sha256={'seed_summary.csv': io.sha256(agg / 'seed_summary.csv')}))
    out = tmp_path / 'figures'
    plot(run, agg, out)
    assert len(list(out.glob('*.png'))) == 3 and len(list(out.glob('*.pdf'))) == 3
    assert (out / 'plot_manifest.json').exists()
    with pytest.raises(FileExistsError):
        plot(run, agg, out)
