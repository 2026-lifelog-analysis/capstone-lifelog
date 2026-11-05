"""
forecast_synthetic.py — 실제 NetHealth 없이 공통 흐름을 확인하기 위한 가짜 원본.

    A, S = make_raw(n_people=40, n_days=120, seed=0)
    data = fd.build_forecast_table(S, A)

FitbitSleep / FitbitActivity와 같은 열 이름·형식으로 만든다.
  - dataDate = 조각이 끝난 날, bedtimedur = 시계 차이 + 1분 (원본과 같은 관계)
  - 일부러 섞는 경우: 빠진 밤, 자정 이후 취침, 90분 안에 다시 잠든 두 조각, 낮잠, 18시 전 시작,
    같은 시작의 짧은 기록이 겹쳐 들어온 밤(겹침 제외 규칙 확인용)
pid는 900000번대(6자리)라 실제 egoid(5자리)와 겹치지 않음. 결과 숫자는 실제 성능이 아님.
"""
import numpy as np
import pandas as pd

PID_BASE = 900000


def _hms(ts):
    return ts.strftime('%H:%M:%S')


def _piece(pid, start, end, asleep):
    clock = int(round((end - start) / pd.Timedelta(minutes=1)))
    return {'egoid': pid, 'dataDate': end.strftime('%Y-%m-%d'), 'timetobed': _hms(start),
            'timeoutofbed': _hms(end), 'bedtimedur': clock + 1, 'minstofallasleep': 0,
            'minsafterwakeup': 0, 'minsasleep': int(asleep), 'minsawake': max(clock + 1 - int(asleep), 0),
            'Efficiency': min(asleep / (clock + 1), 1.0)}


def make_raw(n_people=40, n_days=120, seed=0, start='2016-01-04'):
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp(start)
    S, A = [], []
    for i in range(n_people):
        pid = PID_BASE + i
        bed_mu = rng.normal(23.5 * 60, 50)          # 사람별 평소 취침 (0시부터 분, 1440 넘으면 다음 날)
        dur_mu = rng.normal(430, 40)
        weekend_shift = rng.normal(45, 20)
        first = int(rng.integers(0, 20))             # 사람마다 시작일이 다름
        last = n_days - int(rng.integers(0, 20))
        for day in range(first, last):
            D = d0 + pd.Timedelta(days=day)
            A.append({'egoid': pid, 'datadate': D.strftime('%Y-%m-%d'),
                      'complypercent': int(rng.choice([95, 90, 85, 60])), 'meanrate': rng.normal(70, 5),
                      'sdrate': 10.0, 'steps': int(max(rng.normal(8000, 2500), 0)), 'floors': 5,
                      'sedentaryminutes': 700, 'lightlyactiveminutes': 150, 'fairlyactiveminutes': 20,
                      'veryactiveminutes': 15})
            if rng.random() < 0.12:                  # 빠진 밤
                continue
            wk = weekend_shift if D.dayofweek in (4, 5) else 0
            bed = D + pd.Timedelta(minutes=float(bed_mu + wk + rng.normal(0, 40)))
            if rng.random() < 0.03:                  # 18시 전에 시작한 주수면
                bed = D + pd.Timedelta(hours=17)
            dur = max(rng.normal(dur_mu, 50), 120)
            if rng.random() < 0.2:                   # 중간에 40분 깨서 두 조각
                cut = dur * 0.4
                e1 = bed + pd.Timedelta(minutes=cut)
                s2 = e1 + pd.Timedelta(minutes=40)
                e2 = s2 + pd.Timedelta(minutes=dur - cut)
                S.append(_piece(pid, bed, e1, cut * 0.93))
                S.append(_piece(pid, s2, e2, (dur - cut) * 0.93))
            else:
                S.append(_piece(pid, bed, bed + pd.Timedelta(minutes=dur), dur * 0.93))
            if rng.random() < 0.01:                  # 같은 시작의 짧은 기록이 겹쳐 들어옴 (원본에서 본 포함 관계 모양)
                S.append(_piece(pid, bed, bed + pd.Timedelta(minutes=90), 80))
            if rng.random() < 0.1:                   # 낮잠 (14시, 짧음)
                nap = D + pd.Timedelta(hours=14)
                S.append(_piece(pid, nap, nap + pd.Timedelta(minutes=45), 40))
    return pd.DataFrame(A), pd.DataFrame(S)
