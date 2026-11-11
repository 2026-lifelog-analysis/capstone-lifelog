"""학사 달력(Calendar.csv) 읽기 테스트."""
import pandas as pd
import pytest

from src import data_loader as dl

# NetHealth Calendar.csv의 2015~2016학년도 사건 (이름 표기 차이를 일부러 섞음)
EVENTS = {
    0: ('Orientation week', None, None),
    1: ('Class begins', 20150825, None),
    9: ('Mid-term break', 20151017, 20151025),
    14: ('Thanksgiving', 20151125, 20151129),
    17: ('Final exams', 20151214, 20151218),
    18: ('Christmas', None, None),
    21: ('Classes begin', 20160112, None),
    29: ('Mid-Term break', 20160305, 20160313),
    31: ('Easter', 20160325, 20160328),
    37: ('Final exam', 20160502, 20160506),
    52: ('Orientation Weeks', None, None),
}


def _num(t):
    return int(t.strftime('%Y%m%d'))


@pytest.fixture
def calendar_csv(tmp_path):
    """원본과 같은 형식의 작은 Calendar.csv (0~52주)를 만듦."""
    rows = []
    for w in range(53):
        b = pd.Timestamp('2015-08-16') + pd.Timedelta(weeks=w)
        ev, s, e = EVENTS.get(w, (None, None, None))
        rows.append({'Study Week': w, 'Begin Date': _num(b), 'End date': _num(b + pd.Timedelta(days=6)),
                     'Event': ev, 'Beginning event date': s, 'End event date': e})
    p = tmp_path / 'Calendar.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _tag(cal, day):
    r = cal[cal['date'] == pd.Timestamp(day)].iloc[0]
    return r['term_tag'], r['event_tag']


@pytest.mark.parametrize('day, term, event', [
    ('2015-10-20', '학기', '휴일'),      # 중간 방학 (Mid-term break)
    ('2015-12-15', '학기', '시험'),      # 기말고사
    ('2016-07-01', '방학', '없음'),      # 여름 방학
    ('2015-09-15', '학기', '없음'),      # 평범한 학기 중
    ('2015-08-24', '방학', '없음'),      # 개강 전날
    ('2015-08-25', '학기', '개강주'),    # 개강일
    ('2015-08-31', '학기', '개강주'),    # 개강일 + 6일
    ('2015-09-01', '학기', '없음'),      # 개강주 끝난 다음 날
    ('2015-12-19', '방학', '없음'),      # 기말고사 끝난 다음 날
    ('2015-12-22', '방학', '휴일'),      # Christmas (날짜 없음 → 그 주 전체)
    ('2016-01-12', '학기', '개강주'),    # 'Classes begin' 표기도 개강으로 인식
    ('2016-03-13', '학기', '휴일'),      # 'Mid-Term break' 표기, 사건 날짜가 그 주 범위를 넘어감
    ('2016-05-04', '학기', '시험'),      # 'Final exam' 단수 표기
    ('2016-08-16', '방학', '없음'),      # Orientation은 사건으로 표시 안 함
])
def test_calendar_tags(calendar_csv, day, term, event):
    assert _tag(dl.load_calendar(calendar_csv), day) == (term, event)


def test_calendar_shape(calendar_csv):
    cal = dl.load_calendar(calendar_csv)
    assert list(cal.columns) == ['date', 'study_week', 'term_tag', 'event_tag']
    assert len(cal) == 53 * 7 and cal['date'].is_unique
    assert cal.loc[cal['date'] == '2015-10-20', 'study_week'].iloc[0] == 9


def test_missing_calendar_warns_and_continues(tmp_path):
    d = pd.DataFrame({'pid': ['a'], 'date': pd.to_datetime(['2015-10-20'])})
    with pytest.warns(UserWarning, match='진행합니다'):
        out = dl.add_term_tag(d, tmp_path / 'none.csv')
    assert list(out.columns) == ['pid', 'date']


def test_add_term_tag_columns(calendar_csv):
    d = pd.DataFrame({'pid': ['a', 'a'], 'date': pd.to_datetime(['2015-10-20', '2014-01-01'])})
    out = dl.add_term_tag(d, calendar_csv)
    assert {'study_week', 'term_tag', 'event_tag'} <= set(out.columns)
    assert out['term_tag'].iloc[0] == '학기'
    assert out['term_tag'].isna().iloc[1]             # 달력 범위 밖 → 빈칸


@pytest.mark.skipif(not dl.PATHS['calendar'].exists(), reason='data/nethealth/Calendar.csv 없음')
@pytest.mark.parametrize('day, term, event', [
    ('2015-10-20', '학기', '휴일'), ('2015-12-15', '학기', '시험'),
    ('2016-07-01', '방학', '없음'), ('2015-09-15', '학기', '없음'),
])
def test_real_calendar(day, term, event):
    assert _tag(dl.load_calendar(), day) == (term, event)
