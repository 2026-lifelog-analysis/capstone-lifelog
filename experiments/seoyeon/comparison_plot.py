"""Plots use observed alarm rates and artificial-change responses, not accuracy."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon.comparison_evaluate import MODELS
from experiments.seoyeon.comparison_safety import source_hashes, verify_sources, verified_outputs

LABELS = ['Fixed 14 d', 'Fixed 28 d', 'Rolling 28 d', 'Two-sided CUSUM',
          'Pooled development', 'Frozen MOMENT', 'SH base', 'SH weekday',
          'SH common effect', 'SH final']
RATES = [('r05', 5.), ('r033', 100 / 30)]


def clean_figure(frame):
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout='constrained')
    y = np.arange(len(MODELS))
    for axis, (rate_id, target) in zip(axes, RATES):
        part = frame.loc[frame.scope.eq('common') & frame.rate_id.eq(rate_id)]
        for split, offset, color in [('calibration', -.18, '#4477AA'), ('test', .18, '#EE7733')]:
            values = part.loc[part.split.eq(split)].set_index('model').loc[MODELS, 'alarm_rate'] * 100
            if not np.isfinite(values).all():
                raise ValueError('Missing clean plot values')
            axis.barh(y + offset, values, height=.34, color=color, label=split)
        axis.axvline(target, color='#444444', linestyle='--', label='calibration target')
        axis.set(yticks=y, yticklabels=LABELS, xlabel='Observed daily alarm rate (%)',
                 title=f'Target {target:.2f}% | common person-days')
        axis.grid(axis='x', alpha=.18)
        axis.legend(fontsize=8)
    axes[0].invert_yaxis()
    figure.suptitle('Frozen calibration thresholds; test alarm rates are not clinical error rates')
    return figure


def synthetic_figure(frame, metric):
    if metric not in {'detection_rate', 'delta_mean'}:
        raise ValueError('Unknown synthetic plot metric')
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    summary = frame.groupby(['rate_id', 'model', 'scenario'])[metric].mean()
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5']
    title = ('Injected segments with any alarm (%)' if metric == 'detection_rate'
             else 'Paired change: injected minus same-day unmodified alarms (percentage points)')
    for axis, (rate_id, target) in zip(axes, RATES):
        matrix = summary.loc[rate_id].unstack('scenario').reindex(index=MODELS, columns=scenarios) * 100
        if not np.isfinite(matrix).all().all():
            raise ValueError('Missing synthetic plot values')
        lo, hi, cmap = (0, 100, 'cividis') if metric == 'detection_rate' else (-100, 100, 'RdBu_r')
        view = axis.imshow(matrix, vmin=lo, vmax=hi, cmap=cmap, aspect='auto')
        for (row, col), value in np.ndenumerate(matrix.to_numpy()):
            color = ('white' if value < 50 else 'black') if metric == 'detection_rate' else 'black'
            axis.text(col, row, f'{value:.1f}', ha='center', va='center', color=color, fontsize=8)
        axis.set(xticks=range(5), xticklabels=scenarios, yticks=range(len(MODELS)),
                 yticklabels=LABELS, title=f'Target {target:.2f}% | mean over 20 seeds')
        figure.colorbar(view, ax=axis, shrink=.75)
    figure.suptitle(title + '\nArtificial changes only; no clinical ground truth')
    return figure


def plot(run, aggregate, output):
    run, aggregate, output = Path(run), Path(aggregate), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    verified_outputs(run, 'clean_evaluation.json', ['alarm_summary.csv'])
    verified_outputs(aggregate, 'aggregate_manifest.json', ['seed_summary.csv'])
    paths = [run / 'alarm_summary.csv', aggregate / 'seed_summary.csv']
    before = {str(path): io.sha256(path) for path in paths}
    source = source_hashes(['comparison_plot.py', 'comparison_evaluate.py', 'comparison_io.py'])
    alarms, seeds = (pd.read_csv(path) for path in paths)
    makers = [('daily_alarm_rates', lambda: clean_figure(alarms)),
              ('synthetic_detection', lambda: synthetic_figure(seeds, 'detection_rate')),
              ('synthetic_paired_delta', lambda: synthetic_figure(seeds, 'delta_mean'))]
    for name, make in makers:
        figure = make()
        for extension in ['png', 'pdf']:
            figure.savefig(output / f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(figure)
    if before != {str(path): io.sha256(path) for path in paths}:
        raise ValueError('Plot inputs changed')
    io.write_json(output / 'plot_manifest.json', dict(status='passed', inputs=before,
        **verify_sources(source), outputs_sha256={path.name: io.sha256(path) for path in output.iterdir()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['run', 'aggregate', 'output']:
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    plot(args.run, args.aggregate, args.output)
