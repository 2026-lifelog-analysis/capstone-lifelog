"""
run_table7.py — Wang 외 2014 표 7(사후 UCLA 외로움과 센서 특징의 상관 6개)을 다시 계산한다

    python run_table7.py <StudentLife dataset 폴더> [--out results]

저장하는 파일 (사람별 표는 저장하지 않음)
  table7_compare.csv             : 기본 설정의 r, p와 논문 값 비교
  table7_sensitivity_runs.csv    : 논문에 정의가 없는 선택을 바꿔 가며 계산한 r
  table7_sensitivity_summary.csv : 특징별 r 범위

기본 설정은 결과를 보기 전에 정했다 (DEFAULTS). 논문에 값이 없는 선택이라 근거는 아래 주석뿐이다.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as ft   # noqa: E402
import scoring          # noqa: E402
import table7 as t7     # noqa: E402

DEFAULTS = {
    "threshold": 0.5,          # 10분 구간의 과반이 비정지 추론일 때 "움직임"
    "include_unknown": True,   # "알 수 없음" 추론도 분모에 넣음
    "agg": "mean",             # 사람 단위 요약은 일별 값의 평균
    "min_bins": 72,            # 하루 144개 구간 중 절반 이상 기록된 날만 사용
    "max_accuracy": None,      # GPS 정확도 필터 없음
    "max_gap": None,           # GPS 간격 한도 없음
    "near_as_in": False,       # 'near[...]'는 건물 안으로 보지 않음
    "indoor_method": "ratio",  # 실내 구간의 걷기 추론 비율 × 10분
}
TOLERANCE = 0.10
GRID_ACTIVITY = {"threshold": (0.1, 0.2, 0.3, 0.5), "include_unknown": (True, False),
                 "agg": t7.AGGREGATIONS, "min_bins": (0, 72)}
GRID_DISTANCE = {"max_accuracy": (None, 100.0), "max_gap": (None, 3600.0), "agg": t7.AGGREGATIONS}
GRID_INDOOR = {"near_as_in": (False, True), "indoor_method": ft.INDOOR_METHODS, "agg": t7.AGGREGATIONS}
ACT_COLS, DIST_COLS, INDOOR_COLS = ["act_all", "act_day", "act_evening"], ["dist_all", "dist_day"], ["indoor_day"]


def load_user(data: Path, uid: str) -> dict[str, pd.DataFrame]:
    """한 사람의 센서 파일을 읽어 구간 표(활동·와이파이)와 GPS 표로 만든다."""
    act = pd.read_csv(data / "sensing/activity" / f"activity_{uid}.csv")
    act.columns = ["timestamp", "inference"]
    gps = pd.read_csv(data / "sensing/gps" / f"gps_{uid}.csv", index_col=False)
    wifi = pd.read_csv(data / "sensing/wifi_location" / f"wifi_location_{uid}.csv", index_col=False, usecols=[0, 1])
    wifi.columns = ["time", "location"]
    return {"act_bins": ft.activity_bins(act), "wifi_bins": ft.wifi_bins(wifi),
            "gps": gps[["time", "latitude", "longitude", "accuracy"]]}


def _stack(users: dict[str, dict], make) -> pd.DataFrame:
    return pd.concat([make(u).assign(uid=uid) for uid, u in users.items()], ignore_index=True)


def activity_features(users: dict, cfg: dict) -> pd.DataFrame:
    daily = _stack(users, lambda u: ft.daily_activity(u["act_bins"], cfg["threshold"], cfg["include_unknown"]))
    return t7.person_summary(daily, ACT_COLS, agg=cfg["agg"], min_bins=cfg["min_bins"])


def distance_features(users: dict, cfg: dict) -> pd.DataFrame:
    daily = _stack(users, lambda u: ft.daily_distance(u["gps"], cfg["max_accuracy"], cfg["max_gap"]))
    return t7.person_summary(daily, DIST_COLS, agg=cfg["agg"])


def indoor_features(users: dict, cfg: dict) -> pd.DataFrame:
    daily = _stack(users, lambda u: ft.daily_indoor_mobility(
        u["act_bins"], u["wifi_bins"], near_as_in=cfg["near_as_in"], method=cfg["indoor_method"],
        threshold=cfg["threshold"], include_unknown=cfg["include_unknown"]))
    return t7.person_summary(daily, INDOOR_COLS, agg=cfg["agg"], min_bins=cfg["min_bins"])


def sensitivity(users: dict, scores: pd.Series, base: dict) -> pd.DataFrame:
    """정의가 없는 선택을 하나씩 조합해 r을 다시 계산한다. 나머지 선택은 base 설정."""
    runs = []
    for grid, build in ((GRID_ACTIVITY, activity_features), (GRID_DISTANCE, distance_features),
                        (GRID_INDOOR, indoor_features)):
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            cfg = {**base, **dict(zip(keys, values))}
            corr = t7.correlate(build(users, cfg), scores)
            runs.append(corr.assign(setting=", ".join(f"{k}={v}" for k, v in zip(keys, values))))
    return pd.concat(runs, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="StudentLife 표 7 재현")
    ap.add_argument("data", type=Path, help="StudentLife dataset 폴더")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                    help="움직임 구간으로 보는 비정지 비율 문턱값 (기본 %(default)s)")
    args = ap.parse_args(argv)
    base = {**DEFAULTS, "threshold": args.threshold}
    print(f"[설정] 문턱값={base['threshold']} (기본값 {DEFAULTS['threshold']})")

    survey = scoring.ucla_scores(pd.read_csv(args.data / "survey/LonelinessScale.csv"))
    post = survey[survey["type"] == "post"].dropna(subset=["score"]).set_index("uid")["score"]
    print(f"[사후 UCLA] n={len(post)} 평균={post.mean():.1f} 표준편차={post.std():.1f} (논문 표 4: n=37, 40.9, 10.5)")

    users = {uid: load_user(args.data, uid) for uid in post.index}
    feats = pd.concat([activity_features(users, base), distance_features(users, base),
                       indoor_features(users, base)], axis=1)
    print("\n[사람 단위 특징의 분포: 분 또는 m]")
    print(feats.describe().loc[["count", "mean", "50%", "min", "max"]].round(1).to_string())

    compare = t7.compare_with_paper(t7.correlate(feats, post), tolerance=TOLERANCE)
    runs = sensitivity(users, post, base)
    summary = t7.sensitivity_summary(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    compare.to_csv(args.out / "table7_compare.csv", index=False)
    runs.to_csv(args.out / "table7_sensitivity_runs.csv", index=False)
    summary.to_csv(args.out / "table7_sensitivity_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n[기본 설정 결과와 논문 표 7]")
    print(compare.round(3).to_string(index=False))
    print(f"\n부호 일치 {int(compare['same_sign'].sum())}/6, ±{TOLERANCE} 안 {int(compare['within_tol'].sum())}/6")
    print("\n[민감도: 설정 조합별 r 범위]")
    print(summary.round(3).to_string(index=False))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
