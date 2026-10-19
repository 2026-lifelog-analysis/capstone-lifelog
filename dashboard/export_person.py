# 작성 2026-10-20 09:40
"""
대시보드용 참가자 A 데이터 내보내기 (NetHealth 실제 참가자 1명, 식별 정보 제거).

계산은 레포 함수를 그대로 부른다. 대시보드가 탐지를 새로 구현하지 않기 위함이다.
  걸음 최종 방법 : experiments/final.py personal_bad(요일 보정 + 같은 날 공통효과 제거) → CUSUM k=0.25
  걸음 레포 기본 : bl.personal_z → CUSUM k=0.5
  수면 리듬      : experiments/minjun/regularity sleep_mid_sd → bl.personal_z → CUSUM k=0.5
  경보선 h       : 전체 579명 · 전체 날에서 하루 경보율 1/30 (bl.cusum_threshold)

참가자 선정 규칙 (결과가 좋아 보이는 사람을 고르지 않기 위함)
  1) 걸음 기록 300일 이상, 휴일 태그가 있는 날 1일 이상, 최종 방법 경보 1회 이상
  2) 그중 최종 방법 경보 횟수가 중앙값인 사람 (같으면 pid 순서로 첫 번째)
  3) 보여줄 구간: 학기 중 첫 경보 중, 앞뒤 84일(기준선 28 + 화면 56) 걸음 기록이 80% 이상인 경보
     → 화면 구간 = 그 경보 42일 전 ~ 13일 후 (56일), 기준선 확인용으로 앞 28일을 더 내보냄

식별 정보: pid와 날짜는 내보내지 않는다. 날짜 대신 구간 안 상대 일차, 연구 주차, 학기/방학·시험·휴일 태그만 남긴다.
출력 JSON은 개인 기록이므로 깃에 올리지 않는다 (레포는 공개 저장소).

실행: python3 -I dashboard/export_person.py <출력 JSON 경로>
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'experiments' / 'minjun' / 'regularity'))

import common as C                      # noqa: E402
import final as F                       # noqa: E402
import regularity as R                  # noqa: E402

bl, ev = C.bl, C.ev

MIN_DAYS = 300
VIEW = 56
BASE_WIN = 28
COVER = 0.8


def baseline_band(x, window=BASE_WIN, min_periods=14):
    """직전 달력 28일의 중앙값과 MAD×1.4826 (표시용, bl.personal_z와 같은 창). x는 빈 날 포함 하루 간격 배열."""
    med = np.full(len(x), np.nan)
    sd = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - window):i]
        w = w[~np.isnan(w)]
        if len(w) >= min_periods:
            m = np.median(w)
            med[i], sd[i] = m, np.median(np.abs(w - m)) * 1.4826
    return med, sd


def scores(nh):
    ref = nh.iloc[:0]
    out = pd.DataFrame(index=nh.index)
    out['b_final'] = F.personal_bad(nh, ref, dow=True, anchor=True)
    out['b_base'] = bl.bad_score(bl.personal_z(nh, 'steps'), 'steps')
    out['b_group'] = F.group_bad(nh, ref)
    nh = R.add_sleep_mid(nh)
    nh['mid_sd'] = R.sleep_mid_sd(nh)
    out['mid_sd'] = nh['mid_sd']
    out['b_reg'] = bl.personal_z(nh, 'mid_sd') * R.DIRECTION[R.REG]
    h = {}
    for name, col, k in [('final', 'b_final', 0.25), ('base', 'b_base', 0.5), ('reg', 'b_reg', 0.5)]:
        d = nh.assign(_b=out[col])
        h[name] = bl.cusum_threshold(d, '_b', k=k, rate=ev.ALARM_RATE)
        out['S_' + name] = bl.cusum(d, '_b', k=k, h=h[name])
        out['a_' + name] = out['S_' + name] > h[name]
    h['group'] = float(np.nanquantile(out['b_group'], 1 - ev.ALARM_RATE))
    out['a_group'] = out['b_group'] > h['group']
    return out, h


def pick(nh, sc):
    d = nh.assign(a=sc['a_final'], hol=nh['event_tag'].eq('휴일'))
    per = d.groupby('pid').agg(days=('steps', 'count'), hol=('hol', 'sum'), alarms=('a', 'sum'))
    elig = per[(per['days'] >= MIN_DAYS) & (per['hol'] >= 1) & (per['alarms'] >= 1)].sort_index()
    med = elig['alarms'].median()
    chosen = elig.iloc[(elig['alarms'] - med).abs().argsort(kind='stable')].index[0]
    return chosen, per, elig, med


def window(nh_p):
    s = nh_p.set_index('date')
    full = s.reindex(pd.date_range(s.index.min(), s.index.max()))
    for col, key in [('steps', 'steps'), ('sleep_min', 'sleep'), ('bed_min', 'bed'), ('mid_sd', 'mid_sd')]:
        full[key + '_med'], full[key + '_sd'] = baseline_band(full[col].to_numpy(float))
    for t in full.index[full['a'].fillna(False).astype(bool) & full['term_tag'].eq('학기')]:
        lo, hi = t - pd.Timedelta(days=42 + BASE_WIN), t + pd.Timedelta(days=13)
        seg = full.loc[lo:hi]
        if len(seg) == VIEW + BASE_WIN and seg['steps'].notna().mean() >= COVER:
            return seg, t
    raise SystemExit('조건에 맞는 구간 없음')


def main():
    out_path = Path(sys.argv[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nh = C.load()
        sc, h = scores(nh)
    chosen, per, elig, med = pick(nh, sc)
    nh_p = pd.concat([nh, sc], axis=1)[lambda x: x['pid'] == chosen].assign(a=lambda x: x['a_final'])
    seg, t_alarm = window(nh_p)

    x = seg['steps'].to_numpy(float)

    def col(v):
        return [None if (isinstance(a, float) and np.isnan(a)) or a is None or a is pd.NA else
                (bool(a) if isinstance(a, (bool, np.bool_)) else round(float(a), 3)) for a in v]

    days = {
        'day': list(range(-BASE_WIN, VIEW)),
        'study_week': col(seg['study_week']),
        'term': [None if pd.isna(a) else a for a in seg['term_tag']],
        'event': [None if pd.isna(a) else a for a in seg['event_tag']],
        'dow': [int(i.dayofweek) for i in seg.index],
        'steps': col(x), 'steps_med': col(seg['steps_med']), 'steps_sd': col(seg['steps_sd']),
        'sleep_min': col(seg['sleep_min']), 'sleep_med': col(seg['sleep_med']), 'sleep_sd': col(seg['sleep_sd']),
        'bed_min': col(seg['bed_min']), 'bed_med': col(seg['bed_med']), 'bed_sd': col(seg['bed_sd']),
        'mid_sd': col(seg['mid_sd']), 'mid_sd_med': col(seg['mid_sd_med']), 'mid_sd_sd': col(seg['mid_sd_sd']),
    }
    for k in ['b_final', 'S_final', 'a_final', 'S_base', 'a_base', 'b_group', 'a_group', 'b_reg', 'S_reg', 'a_reg']:
        days[k] = col(seg[k])

    payload = {
        'label': '참가자 A',
        'rule': ('걸음 기록 300일 이상 · 휴일 포함 · 최종 방법 경보 1회 이상인 사람 중 경보 횟수가 중앙값인 사람. '
                 '구간은 학기 중 첫 경보 중 앞뒤 84일 기록이 80% 이상인 경보의 42일 전 ~ 13일 후.'),
        'n_people': int(per.shape[0]), 'n_eligible': int(elig.shape[0]),
        'alarms_median': float(med), 'alarms_chosen': int(per.loc[chosen, 'alarms']),
        'days_recorded': int(per.loc[chosen, 'days']),
        'h': {k: round(float(v), 4) for k, v in h.items()},
        'alarm_day': int((t_alarm - seg.index[BASE_WIN]).days),
        'days': days,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False))
    a = np.array(days['a_final'][BASE_WIN:], dtype=object)
    print(f"전체 {payload['n_people']}명 중 조건 충족 {payload['n_eligible']}명, 경보 횟수 중앙값 {med:.0f}회")
    print(f"참가자 A: 기록 {payload['days_recorded']}일, 경보 {payload['alarms_chosen']}회, "
          f"화면 구간 경보 {int(sum(1 for v in a if v))}일, 경보선 h {payload['h']}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
