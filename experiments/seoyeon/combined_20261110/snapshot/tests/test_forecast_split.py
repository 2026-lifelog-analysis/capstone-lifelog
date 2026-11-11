"""참가자 분할 테스트 (가짜 ID만 사용)."""
import pandas as pd
import pytest

from src import forecast_split as fs


def _pids(n):
    return [f'p{i:03d}' for i in range(n)]


def test_split_is_reproducible_and_independent_of_input_order_and_type():
    a = fs.make_split(_pids(100))
    b = fs.make_split(list(reversed(_pids(100))) + _pids(10))        # 순서 바뀜 + 중복
    pd.testing.assert_frame_equal(a, b)
    ints = fs.make_split(list(range(50)))
    strs = fs.make_split([str(i) for i in range(50)])
    pd.testing.assert_frame_equal(ints, strs)                         # 숫자 ID도 문자열로 통일
    assert not a.equals(fs.make_split(_pids(100), seed=1))


@pytest.mark.parametrize('n, n_final', [(100, 15), (621, 93), (10, 2), (103, 15), (110, 17)])
def test_final_size_rounding(n, n_final):
    s = fs.make_split(_pids(n))
    assert (s['role'] == 'final').sum() == n_final == fs.n_final_for(n)
    fs.check_split(s)


def test_roles_do_not_overlap_and_each_dev_validated_once():
    s = fs.make_split(_pids(100))
    final = fs.final_pids(s)
    dev = set(s.loc[s['role'] == 'dev', 'pid'])
    seen = []
    for f in fs.FOLDS:
        tr, va = fs.fold_pids(s, f)
        assert not tr & va and not (tr | va) & final
        assert tr | va == dev
        seen += sorted(va)
    assert sorted(seen) == sorted(dev)                               # 다섯 번 중 정확히 한 번
    # 매회 약 68 / 17 / 15
    tr, va = fs.fold_pids(s, 1)
    assert (len(tr), len(va), len(final)) == (68, 17, 15)


def test_saved_split_is_reused_and_mismatch_is_not_silently_rebuilt(tmp_path):
    t = pd.DataFrame({'pid': _pids(40), 'date': pd.Timestamp('2016-01-01')})
    s1, m1 = fs.load_or_make_split(t, tmp_path, data_fingerprint='abc')
    s2, m2 = fs.load_or_make_split(t, tmp_path, data_fingerprint='abc')
    pd.testing.assert_frame_equal(s1, s2)
    assert m1['split_sha256'] == m2['split_sha256'] == fs.split_hash(s1)

    with pytest.raises(fs.SplitMismatch):
        fs.load_or_make_split(t.iloc[:-1], tmp_path, data_fingerprint='abc')   # 참가자가 바뀜
    with pytest.raises(fs.SplitMismatch):
        fs.load_or_make_split(t, tmp_path, seed=1, data_fingerprint='abc')     # 설정이 바뀜
    with pytest.raises(fs.SplitMismatch):
        fs.load_or_make_split(t, tmp_path, data_fingerprint='other')           # 데이터가 바뀜
    edited = s1.copy()
    edited.loc[0, 'fold'] = 1 if edited.loc[0, 'fold'] != 1 else 2
    edited.to_csv(tmp_path / fs.SPLIT_FILE, index=False)
    with pytest.raises((fs.SplitMismatch, AssertionError)):
        fs.load_or_make_split(t, tmp_path, data_fingerprint='abc')             # 손으로 고침
    s3, _ = fs.load_or_make_split(t, tmp_path, data_fingerprint='abc', rebuild=True)
    pd.testing.assert_frame_equal(s3, s1)


def test_protocol_change_is_detected(tmp_path):
    t = pd.DataFrame({'pid': _pids(40), 'date': pd.Timestamp('2016-01-01')})
    fs.load_or_make_split(t, tmp_path, data_fingerprint='abc', protocol='v1')
    with pytest.raises(fs.SplitMismatch, match='프로토콜'):
        fs.load_or_make_split(t, tmp_path, data_fingerprint='abc', protocol='v2')
