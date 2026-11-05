import numpy as np
import pandas as pd
import pytest

from experiments.seoyeon.comparison_io import calendar_split, label_split, quantile_threshold


def test_inclusive_calendar_cumulative_floor_boundaries():
    dates = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-11']))
    split = calendar_split(dates)
    assert split['development_end'] == '2020-01-04'
    assert split['calibration_start'] == '2020-01-05'
    assert split['calibration_end'] == '2020-01-07'
    assert split['test_start'] == '2020-01-08'
    assert list(label_split(pd.date_range('2020-01-01', periods=11), split)).count('test') == 4


def test_threshold_uses_only_calibration_and_higher_quantile():
    score = np.array([1., 2., 3., 1000., np.nan])
    mask = np.array([True, True, True, False, False])
    assert quantile_threshold(score, mask, .34) == 3.
    score[3] = -1e9
    assert quantile_threshold(score, mask, .34) == 3.


def test_empty_calibration_is_failure_not_zero_alarm():
    with pytest.raises(ValueError, match='calibration'):
        quantile_threshold([np.nan], [True], .05)


def test_out_of_range_date_rejected():
    split = calendar_split(pd.Series(pd.date_range('2020-01-01', periods=100)))
    with pytest.raises(ValueError, match='outside'):
        label_split(pd.to_datetime(['2019-12-31']), split)
