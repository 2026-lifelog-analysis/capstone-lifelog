"""공통 표본의 경보 분모 및 승호 브랜치 경보선 대조를 검증합니다."""
import numpy as np
import pandas as pd

from experiments.seoyeon.audit import alarm_comparison, threshold_comparison


def test_comparison_uses_only_shared_finite_days():
    d = pd.DataFrame({'pid': ['a', 'a', 'b', 'b'],
                      'p': [0., 2., np.nan, 4.], 'g': [2., 0., 8., 4.]})
    r = alarm_comparison(d, 'p', 'g', rate=1/3)
    assert r['common_days'] == 3
    assert r['people'] == 2
    assert r['personal_alarm_days'] == r['group_alarm_days'] == 1
    assert r['overlap_over_personal'] == 1


def test_no_alarms_reports_undefined_overlap():
    d = pd.DataFrame({'pid': ['a', 'b'], 'p': [1., 1.], 'g': [1., 1.]})
    r = alarm_comparison(d, 'p', 'g', rate=.05)
    assert r['personal_alarm_days'] == 0
    assert np.isnan(r['overlap_over_personal'])
    assert r['personal_zero_people'] == 2


def test_thresholds_use_same_valid_denominator_and_strict_comparison():
    s = pd.Series([np.nan, 0., 1., 2., 3., 4.])
    r = threshold_comparison(s, reset_h=3., rate=.4)
    assert r['valid_days'] == 5
    assert r['reset_h_alarm_days'] == 1
    assert r['quantile_alarm_days'] == 2
    assert r['disagreement_days'] == 1
