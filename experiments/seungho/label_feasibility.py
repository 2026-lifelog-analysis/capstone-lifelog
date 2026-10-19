# 작성 2026-10-19 18:40
"""NetHealth 러프 라벨 후보의 발생 빈도 확인.

의학 기준(Watson 2015, Lichstein 2003, Ohayon 2017, Huang 2020)을 Fitbit 수면에,
활동 기준(WHO 2020 주 150분, Ding 2025 5,000보 꺾임점)을 Fitbit 활동에,
검증된 절단점(PSQI>5, CES-D>=16)을 설문에 적용해
"사람-주 양성 비율"과 "직전 4주 정상 -> 이번 주 위반(발생)" 건수를 센다.

실행: python3 -I label_feasibility.py <data/nethealth 폴더> <출력 CSV>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_NIGHTS = 4      # 주 안에 기록된 밤이 이만큼 있어야 그 주를 판정
BAD_NIGHTS = 3      # 위반한 밤이 주 3회 이상이면 양성 (Lichstein 2003 빈도 기준)
PRIOR_WEEKS = 4     # 발생 = 직전 4주 모두 판정 가능하고 음성
NIGHT_COLS = ["short7", "short6", "waso31", "waso41", "sol31", "sol46", "eff85", "eff64"]


def load_sleep(folder):
    f = next(Path(folder).glob("FitbitSleep*.csv"))
    d = pd.read_csv(f)
    d["date"] = pd.to_datetime(d["dataDate"])
    # 하루 여러 조각: 총 수면은 합산, 나머지는 가장 긴 조각 기준
    d = d.sort_values(["egoid", "date", "minsasleep"])
    main = d.groupby(["egoid", "date"]).tail(1).set_index(["egoid", "date"])
    tot = d.groupby(["egoid", "date"])["minsasleep"].sum()
    n = main[["minstofallasleep", "minsawake", "Efficiency", "timetobed"]].copy()
    n["total_sleep"] = tot
    t = pd.to_datetime(n["timetobed"], format="%H:%M:%S", errors="coerce")
    m = t.dt.hour * 60 + t.dt.minute
    n["onset_noon"] = (m - 720) % 1440          # 정오부터 분, 자정 넘김 연속
    return n.reset_index()


def night_flags(n):
    return pd.DataFrame({
        "egoid": n["egoid"], "date": n["date"],
        "short7": n["total_sleep"] < 420,        # Watson 2015: 7시간 미만
        "short6": n["total_sleep"] <= 360,       # 6시간 이하
        "waso31": n["minsawake"] >= 31,          # Lichstein 2003
        "sol31": n["minstofallasleep"] >= 31,    # Lichstein 2003
        "eff85": n["Efficiency"] < 0.85,         # Ohayon 2017 (좋은 수면 >=85%)
        "sol46": n["minstofallasleep"] >= 46,    # Ohayon 2017 나쁨 (18-64세)
        "waso41": n["minsawake"] >= 41,          # Ohayon 2017 나쁨 (18-64세)
        "eff64": n["Efficiency"] <= 0.64,        # Ohayon 2017 나쁨 (18-25세)
        "total_sleep": n["total_sleep"], "onset_noon": n["onset_noon"],
    })


def weekly(fl):
    fl = fl.copy()
    fl["week"] = fl["date"].dt.to_period("W-SUN")
    g = fl.groupby(["egoid", "week"])
    w = g.size().rename("nights").to_frame()
    for c in NIGHT_COLS:
        w[c] = g[c].sum() >= BAD_NIGHTS
    # Huang 2020: 7일 SD 수면시간 >120분, 잠든 시각 >90분
    w["sd_dur120"] = g["total_sleep"].std() > 120
    w["sd_onset90"] = g["onset_noon"].std() > 90
    w = w[w["nights"] >= MIN_NIGHTS]
    return w.reset_index()


def load_activity(folder, wear=80):
    """착용 80% 이상인 날만. 걸음 5,000 미만인 날이 주 3회 이상, 주 중강도 이상 활동 150분 미만."""
    f = next(Path(folder).glob("FitbitActivity*.csv"))
    a = pd.read_csv(f)
    a = a[a["complypercent"] >= wear].drop_duplicates(["egoid", "datadate"])
    a["date"] = pd.to_datetime(a["datadate"])
    a["week"] = a["date"].dt.to_period("W-SUN")
    a["low5000"] = a["steps"] < 5000
    a["mvpa"] = a["fairlyactiveminutes"] + a["veryactiveminutes"]
    g = a.groupby(["egoid", "week"])
    w = g.size().rename("nights").to_frame()
    w["steps5000"] = g["low5000"].sum() >= BAD_NIGHTS
    w["mvpa150"] = g["mvpa"].mean() * 7 < 150   # 기록일 평균을 주 단위로 환산
    return w[w["nights"] >= MIN_NIGHTS].reset_index()


def onsets(w, col):
    """직전 4주(연속 주, 모두 판정 가능)가 전부 음성이고 이번 주 양성."""
    out = 0
    people = 0
    for _, p in w.groupby("egoid"):
        s = p.set_index("week")[col].astype(float)
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="W-SUN"))
        prior = s.shift(1).rolling(PRIOR_WEEKS, min_periods=PRIOR_WEEKS).max()
        hit = ((s == 1) & (prior == 0)).sum()
        out += int(hit)
        people += int(hit > 0)
    return out, people


def survey(folder):
    f = next(Path(folder).glob("BasicSurvey*.csv"))
    s = pd.read_csv(f, low_memory=False, encoding="latin1")
    rows = []
    for name, prefix, cut, waves in [("PSQI>5", "PSQIGlobal_", 5, [1, 2, 3, 4, 5]),
                                     ("CES-D>=16", "CESDOverall_", 15, [1, 2, 4, 6, 8])]:
        prev = None
        for wv in waves:
            v = pd.to_numeric(s[f"{prefix}{wv}"], errors="coerce")
            pos = v > cut
            r = {"label": name, "wave": wv, "n": int(v.notna().sum()),
                 "pos_rate": round(float(pos[v.notna()].mean()), 3)}
            if prev is not None:
                both = v.notna() & prev.notna()
                r["onset_n"] = int((both & (prev <= cut) & (v > cut)).sum())
                r["pairs"] = int(both.sum())
            rows.append(r)
            prev = v
    return pd.DataFrame(rows)


def main():
    folder, out = sys.argv[1], sys.argv[2]
    n = load_sleep(folder)
    fl = night_flags(n)
    print(f"밤 {len(n):,}개, {n['egoid'].nunique()}명")
    print(f"minstofallasleep=0 비율 {100 * (n['minstofallasleep'] == 0).mean():.1f}%, "
          f"minsawake=0 비율 {100 * (n['minsawake'] == 0).mean():.1f}%")
    night_rate = fl[NIGHT_COLS].mean().round(3)
    print("밤 단위 위반 비율\n", night_rate.to_string())
    w = weekly(fl)
    print(f"판정 가능한 사람-주 {len(w):,}개 ({w['egoid'].nunique()}명, 주당 밤 {MIN_NIGHTS}개 이상)")
    aw = load_activity(folder)
    print(f"활동 판정 가능한 사람-주 {len(aw):,}개 ({aw['egoid'].nunique()}명)")
    rows = []
    for c, w in [(c, w) for c in NIGHT_COLS + ["sd_dur120", "sd_onset90"]] + [(c, aw) for c in ["steps5000", "mvpa150"]]:
        k, p = onsets(w, c)
        pp = w.groupby("egoid")[c].mean()
        rows.append({"label": c, "week_pos_rate": round(float(w[c].mean()), 3),
                     "onset_n": k, "people_with_onset": p,
                     "people_always_pos(>=80%wk)": int((pp >= 0.8).sum()),
                     "people_never_pos": int((pp == 0).sum())})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    sv = survey(folder)
    print(sv.to_string(index=False))
    pd.concat([res.assign(kind="sensor"), sv.assign(kind="survey")]).to_csv(out, index=False)


if __name__ == "__main__":
    main()
