"""
trend_share.py — 평소 비율의 움직임에서 한 방향 추세가 차지하는 몫

usage: python trend_share.py <lifelog-bench 저장소 경로> <결과 폴더>
결과: <결과 폴더>/trend_share_summary.csv, <결과 폴더>/trend_slopes.csv. 집계만 쓴다(줄 값·한 사람의 값 없음).

측정은 승호님 repro_lov3.py(RQ2)와 같은 두 가지다.
- 28일 블록 변화: 사람마다 연속한 28일 블록 간 양성 비율의 평균 절대 변화 (줄이 MIN_ROWS개 미만인 블록은 뺌)
- 전반→후반 변화: 사람마다 라벨 있는 날 앞 절반과 뒤 절반의 양성 비율 차이의 절댓값
관측값을 두 기준과 비교한다.
- 섞기 기준: 사람 안에서 날짜 순서를 섞은 200회. 구조가 전혀 없는 경우 (승호님 방식)
- 추세만 기준: 사람마다 라벨을 날짜에 대한 로지스틱 직선으로 맞추고, 그 확률로 라벨을 새로 뽑은 200회.
  한 방향 추세만 있고 그 밖의 움직임은 없는 경우
추세의 몫 = (추세만 기준 평균 − 섞기 기준 평균) ÷ (관측 − 섞기 기준 평균)
보조로 원점수(이진화 전)의 블록 평균 변화를, 사람별 직선을 뺀 잔차와 함께 같은 방식으로 본다.

자료: lifelog-bench의 정답 원천(data/targets/v1/, Human Rhythms 개발용 109명). 틀의 bench.sealed로 읽는다.
라벨은 틀의 회고 라벨과 같은 정의(그 사람의 전체 평균보다 높으면 1)를 원점수에서 다시 만든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offset_logit import fit_offset_logistic, predict_offset_logistic  # noqa: E402

KEY = ["participant_key", "reference_day"]
RAW = {"sleep_quality": "q_sleep_quality_next", "physical_energy": "q_physical_energy",
       "emotional_energy": "q_emotional_energy", "hurried": "q_hurried"}
BLOCK_DAYS = 28
MIN_ROWS = 7          # 근거 없음: 블록 평균이 몇 줄로 흔들리지 않게 둔 값
N_SIM = 200           # 승호님 repro_lov3.py와 같은 횟수
N_BOOT = 2000
SEED = 20261207       # lifelog-bench 계약 18절의 부트스트랩 시드
TREND_L2 = 0.1        # 근거 없음: 사람별 추세 직선의 벌점 세기
MEASURES = ("28일 블록 변화", "전반→후반 변화")


def block_change(values: np.ndarray, block: np.ndarray) -> float:
    """연속한 블록 간 평균의 평균 절대 변화. 쓸 수 있는 블록이 둘 미만이면 NaN."""
    _, inv, cnt = np.unique(block, return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=values) / cnt
    mean = mean[cnt >= MIN_ROWS]
    return np.nan if len(mean) < 2 else float(np.abs(np.diff(mean)).mean())


def half_change(values: np.ndarray) -> float:
    """앞 절반과 뒤 절반 평균의 차이의 절댓값."""
    h = len(values) // 2
    return abs(values[:h].mean() - values[h:].mean())


def stat(people: list[dict], values: list[np.ndarray]) -> tuple[float, float]:
    """(28일 블록 변화, 전반→후반 변화)의 사람 균등 평균."""
    return (float(np.nanmean([block_change(v, p["block"]) for v, p in zip(values, people)])),
            float(np.mean([half_change(v) for v in values])))


def null(people: list[dict], draw, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """draw로 N_SIM번 다시 만든 값의 (평균, 95% 분위)."""
    out = np.array([stat(people, [draw(p, rng) for p in people]) for _ in range(N_SIM)])
    return out.mean(0), np.quantile(out, .95, axis=0)


def person_arrays(x: np.ndarray, day: np.ndarray) -> dict:
    """한 사람의 원점수 x와 첫날부터의 날 수 day로 비교에 필요한 배열을 만든다."""
    y = (x * len(x) > x.sum()).astype(float)
    z = ((day - day.mean()) / day.std()).reshape(-1, 1)
    model = fit_offset_logistic(z, y, np.zeros(len(y)), TREND_L2)
    slope, icpt = np.polyfit(day, x, 1)
    return {"block": (day // BLOCK_DAYS).astype(int), "y": y, "x": x,
            "p": predict_offset_logistic(model, z, np.zeros(len(y))),
            "resid": x - (icpt + slope * day), "slope": slope * BLOCK_DAYS}


def shuffle(key: str):
    return lambda p, rng: rng.permutation(p[key])


def trend_only(p: dict, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(len(p["p"])) < p["p"]).astype(float)


def compare(people: list[dict], rng: np.random.Generator) -> list[dict]:
    rows = []
    obs = stat(people, [p["y"] for p in people])
    sh_m, sh_q = null(people, shuffle("y"), rng)
    tr_m, tr_q = null(people, trend_only, rng)
    for i, name in enumerate(MEASURES):
        rows.append(dict(scale="라벨", measure=name, observed=obs[i], shuffle_mean=sh_m[i], shuffle_q95=sh_q[i],
                         trend_mean=tr_m[i], trend_q95=tr_q[i],
                         trend_share=(tr_m[i] - sh_m[i]) / (obs[i] - sh_m[i]),
                         beyond_trend=bool(obs[i] > tr_q[i])))
    obs_x = stat(people, [p["x"] for p in people])
    shx_m, shx_q = null(people, shuffle("x"), rng)
    obs_r = stat(people, [p["resid"] for p in people])
    shr_m, shr_q = null(people, shuffle("resid"), rng)
    for i, name in enumerate(MEASURES):
        rows.append(dict(scale="원점수", measure=name, observed=obs_x[i], shuffle_mean=shx_m[i], shuffle_q95=shx_q[i],
                         excess=obs_x[i] - shx_m[i], detrended_observed=obs_r[i], detrended_shuffle_mean=shr_m[i],
                         detrended_shuffle_q95=shr_q[i], detrended_excess=obs_r[i] - shr_m[i]))
    return rows


def slope_summary(people: list[dict], rng: np.random.Generator) -> dict:
    slopes = np.array([p["slope"] for p in people])
    boots = slopes[rng.integers(0, len(slopes), (N_BOOT, len(slopes)))].mean(1)
    lo, hi = np.quantile(boots, [.025, .975])
    return dict(slope_28d=float(slopes.mean()), lo=float(lo), hi=float(hi), share_down=float((slopes < 0).mean()))


def load_people(table: pd.DataFrame, target: str) -> list[dict]:
    people = []
    for _, g in table[table[f"valid_{target}"]].groupby("participant_key", sort=True):
        x = g[RAW[target]].to_numpy(float)
        d = pd.to_datetime(g["reference_day"])
        day = (d - d.iloc[0]).dt.days.to_numpy(float)
        if len(x) < 2 * MIN_ROWS or day.max() == 0:
            continue
        people.append(person_arrays(x, day))
    return people


def main(bench: Path, out: Path) -> None:
    sys.path.insert(0, str(bench))
    from bench.sealed import read_sealed
    labels = read_sealed(bench / "data/targets/v1/labels.sealed")
    sources = read_sealed(bench / "data/targets/v1/sources.sealed")
    table = labels.merge(sources, on=KEY, validate="one_to_one").sort_values(KEY)
    rng = np.random.default_rng(SEED)
    rows, slopes = [], []
    for target in RAW:
        people = load_people(table, target)
        rows += [dict(target=target, n_people=len(people), **r) for r in compare(people, rng)]
        slopes.append(dict(target=target, n_people=len(people), **slope_summary(people, np.random.default_rng(SEED))))
    out.mkdir(parents=True, exist_ok=True)
    summary, slope_table = pd.DataFrame(rows), pd.DataFrame(slopes)
    summary.to_csv(out / "trend_share_summary.csv", index=False)
    slope_table.to_csv(out / "trend_slopes.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(slope_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
