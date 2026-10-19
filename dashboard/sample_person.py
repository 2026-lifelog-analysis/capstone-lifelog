# 작성 2026-10-20 17:30
"""
가상 예시 인물 만들기. NetHealth 데이터 없이도 대시보드를 빌드해 볼 수 있게 하기 위함이다.

export_person.py가 만드는 참가자 JSON과 같은 모양이지만, 값은 모두 정해진 규칙과 고정된 난수로 만든 가상 값이다.
실제 사람의 기록이 아니므로 레포에 올려도 된다. 화면에는 '가상 예시'라고 표시된다.

시나리오 (화면 구간 0~55일)
  -10~-6일 시험, -5~19일 방학(그중 -5~3일 휴일), 20~26일 개강주, 이후 학기
  방학에는 모두가 덜 걸음 → 출발점 방법은 경보, 공통효과를 뺀 최종 방법은 경보 없음
  34일부터 걸음이 서서히 줄어 40일 무렵 최종 방법 경보

탐지 계산은 레포 방법을 단순화한 것이다 (같은 날 공통효과 = 방학 계수로 대신). 경보선 h는 실제 579명에서 구한 값을 쓴다.

실행: python3 -I dashboard/sample_person.py <출력 JSON>
"""
import json
import sys
from pathlib import Path

import numpy as np

H = {'final': 5.7279, 'base': 2.8588, 'reg': 13.2269, 'group': 1.4382}   # export_person.py 실행 결과 (전체 579명)
START, VIEW, BASE_WIN = -84, 56, 28


def tags(t):
    term = '방학' if -5 <= t <= 19 else '학기'
    event = ('휴일' if -5 <= t <= 3 else '시험' if -10 <= t <= -6 else '개강주' if 20 <= t <= 26 else '없음')
    return term, event


def band(x, window=BASE_WIN, minp=14):
    med, sd = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - window):i]
        w = w[~np.isnan(w)]
        if len(w) >= minp:
            m = np.median(w)
            med[i], sd[i] = m, max(np.median(np.abs(w - m)) * 1.4826, 1e-6)
    return med, sd


def cusum(score, k, h):
    s, out, alarm = 0.0, [], []
    for v in score:
        if np.isnan(v):
            out.append(np.nan)
            alarm.append(False)
            continue
        s = max(0.0, s + v - k)
        out.append(s)
        alarm.append(s > h)
        if s > h:
            s = 0.0
    return np.array(out), np.array(alarm)


def roll_sd(x, k=7, minp=5):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= minp and not np.isnan(x[i]):
            out[i] = np.std(w, ddof=1)
    return out


def main():
    out_path = Path(sys.argv[1])
    rng = np.random.default_rng(2026)
    t = np.arange(START, VIEW)
    n = len(t)
    dow = (t + 2) % 7                       # 0일 = 수요일
    term_event = [tags(x) for x in t]
    vac = np.array([te[0] == '방학' for te in term_event])
    common = np.where(vac, 0.52, 1.0)       # 그날 모두에게 공통인 활동 수준

    weekend = np.isin(dow, [5, 6])
    steps = 10500 * common * np.where(weekend, 0.8, 1.0) * rng.lognormal(0, 0.28, n)
    decline = np.clip((t - 33) / 7, 0, 1) * 0.5
    steps *= 1 - decline * (t <= 44)
    steps[rng.random(n) < 0.06] = np.nan    # 착용하지 않은 날
    sleep = np.where(vac, 470, 425) + rng.normal(0, 40, n)
    sleep[(t >= 38) & (t <= 44)] -= 45
    bed = np.where(vac, 900, 830) + rng.normal(0, 35, n)
    sleep[rng.random(n) < 0.05] = np.nan
    bed[np.isnan(sleep)] = np.nan
    mid = bed + sleep / 2

    def z(x, med, sd):
        return np.clip((x - med) / sd, -8, 8)

    s_med, s_sd = band(steps)
    a_med, a_sd = band(steps / common)
    b_base = -z(steps, s_med, s_sd)
    b_final = -z(steps / common, a_med, a_sd)
    pop_med = 9000 * common
    b_group = -np.clip((steps - pop_med) / (0.45 * pop_med), -8, 8)
    S_base, a_base = cusum(b_base, 0.5, H['base'])
    S_final, a_final = cusum(b_final, 0.25, H['final'])
    a_group = b_group > H['group']
    mid_sd = roll_sd(mid)
    r_med, r_sd = band(mid_sd)
    b_reg = z(mid_sd, r_med, r_sd)
    S_reg, a_reg = cusum(b_reg, 0.5, H['reg'])
    sl_med, sl_sd = band(sleep)
    bd_med, bd_sd = band(bed)

    v = slice(n - VIEW - BASE_WIN, n)
    view_t = t[v]
    sem = [i for i in range(len(view_t)) if view_t[i] >= 30 and a_final[v][i]]
    alarm_day = int(view_t[sem[0]]) if sem else 42

    def col(x):
        return [None if (isinstance(a, float) and np.isnan(a)) else
                (bool(a) if isinstance(a, (bool, np.bool_)) else round(float(a), 3)) for a in np.asarray(x, dtype=object)]

    days = {
        'day': [int(x) for x in view_t], 'study_week': [int(20 + x // 7) for x in view_t],
        'term': [term_event[i][0] for i in range(n)][v], 'event': [term_event[i][1] for i in range(n)][v],
        'dow': [int(x) for x in dow[v]],
        'steps': col(steps[v]), 'steps_med': col(s_med[v]), 'steps_sd': col(s_sd[v]),
        'sleep_min': col(sleep[v]), 'sleep_med': col(sl_med[v]), 'sleep_sd': col(sl_sd[v]),
        'bed_min': col(bed[v]), 'bed_med': col(bd_med[v]), 'bed_sd': col(bd_sd[v]),
        'mid_sd': col(mid_sd[v]), 'mid_sd_med': col(r_med[v]), 'mid_sd_sd': col(r_sd[v]),
        'b_final': col(b_final[v]), 'S_final': col(S_final[v]), 'a_final': col(a_final[v]),
        'S_base': col(S_base[v]), 'a_base': col(a_base[v]), 'b_group': col(b_group[v]), 'a_group': col(a_group[v]),
        'b_reg': col(b_reg[v]), 'S_reg': col(S_reg[v]), 'a_reg': col(a_reg[v]),
    }
    payload = {'label': '예시 인물', 'sample': True,
               'rule': '실제 사람이 아닌 가상 예시 데이터 (dashboard/sample_person.py, 고정 난수)',
               'n_people': 0, 'n_eligible': 0, 'alarms_median': 0, 'alarms_chosen': int(a_final.sum()),
               'days_recorded': int(np.sum(~np.isnan(steps))), 'h': H, 'alarm_day': alarm_day, 'days': days}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    va = [int(x) for x, a in zip(view_t, a_final[v]) if a]
    vb = [int(x) for x, a in zip(view_t, a_base[v]) if a]
    print(f'최종 방법 경보 일차 {va}, 출발점 방법 경보 일차 {vb}, 데모 오늘 = {alarm_day}일차')
    print(f'저장: {out_path}')


if __name__ == "__main__":
    main()
