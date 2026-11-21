"""
split.py — 사람별 시간순 분할

사람마다 라벨이 있는 날을 시간순으로 세워 앞 절반을 학습으로 쓴다.
학습 마지막 날에서 GAP_DAYS일(달력 기준) 안에 드는 날은 버리고(gap), 그 뒤를 평가로 쓴다.
인접한 날이 학습과 평가 양쪽에 걸치지 않게 하려는 것이다.
"""
from __future__ import annotations

import pandas as pd

TRAIN_FRACTION = 0.5
GAP_DAYS = 7


def _split_person(dates: pd.Series) -> pd.Series:
    if not dates.is_monotonic_increasing:
        raise ValueError("날짜가 시간순으로 정렬돼 있지 않음")
    n_train = int(len(dates) * TRAIN_FRACTION)
    train_end = dates.iloc[n_train - 1]
    first_test_day = train_end + pd.Timedelta(days=GAP_DAYS + 1)
    position = pd.Series(range(len(dates)), index=dates.index)
    part = pd.Series("gap", index=dates.index)
    part = part.mask(dates >= first_test_day, "test")
    return part.mask(position < n_train, "train")


def assign_split(labels: pd.DataFrame) -> pd.DataFrame:
    """part 열(train / gap / test)을 붙인 새 표를 돌려준다. 입력은 바꾸지 않는다."""
    part = labels.groupby("subject_id", group_keys=False)["lifelog_date"].apply(_split_person)
    return labels.assign(part=part)
