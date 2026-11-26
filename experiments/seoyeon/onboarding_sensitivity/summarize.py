"""조건별 RQ3 결과를 한 표로 묶는다. 집계만 출력한다.

usage: python summarize.py <results 폴더>
- 방법별 평균 log loss: 다섯 목표의 평균 (승호님 README의 "평균 log loss"와 같은 집계)
- P1-B1, P1-B3, P2-P1: 목표마다 사람 단위 짝 부트스트랩 2,000회(승호님 sparse_survey.py와 같은 날 합산 방식),
  구간이 0 밖에서 음수(앞쪽이 나음)인 목표 수와 양수(앞쪽이 나쁨)인 목표 수
- 온보딩 길이: 사람별 온보딩 동안 공개된 라벨 수의 중앙값
결과: <results>/summary_conditions.csv, <results>/summary_pairs.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT, SEED = 2000, 0
PAIRS = [("P1", "B1"), ("P1", "B3"), ("P2", "P1"), ("B3", "B1")]
METHODS = ["B1", "B2", "B3", "P1", "P2"]
ORDER = ["half_onboard", "d28_fixed", "d28_onboard", "d14_fixed", "d14_onboard",
         "d28_onboard_skip7", "d28_onboard_shrink", "d14_onboard_skip7", "d14_onboard_shrink"]


def paired(per_person: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> tuple[float, float, float]:
    d = (per_person[a] - per_person[b]).to_numpy(); n = per_person["n"].to_numpy(float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].sum(1) / n[idx].sum(1)
    return float(d.sum() / n.sum()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def summarize_condition(folder: Path) -> tuple[list[dict], list[dict]]:
    cond = json.loads((folder / "rq3_summary.json").read_text())["conditions"]
    preds = pd.read_pickle(folder / "rq3_preds.pkl")
    scored = preds[preds["scored"]].copy()
    p = scored["p"].clip(1e-6, 1 - 1e-6)
    scored["ll"] = -(scored["y"] * np.log(p) + (1 - scored["y"]) * np.log(1 - p))
    per = scored.pivot_table(index=["target", "budget", "pid"], columns="method", values="ll", aggfunc="sum")
    per["n"] = scored[scored["method"] == "B1"].groupby(["target", "budget", "pid"]).size()
    rng = np.random.default_rng(SEED)
    onboard = preds[(preds["method"] == "P1") & (preds["budget"] == "k1")]
    pairs, conds = [], []
    for (t, b), g in per.groupby(level=[0, 1]):
        for a, c in PAIRS:
            diff, lo, hi = paired(g, a, c, rng)
            pairs.append(dict(condition=folder.name, target=t, budget=b, pair=f"{a}-{c}", diff=diff, lo=lo, hi=hi))
    table = per.groupby(level=[0, 1]).sum()
    mean_ll = table[METHODS].div(table["n"], axis=0).groupby(level="budget").mean()
    for b, row in mean_ll.iterrows():
        conds.append(dict(condition=folder.name, **cond, budget=b, **{m: row[m] for m in METHODS}))
    return conds, pairs


def main(root: Path) -> None:
    all_conds, all_pairs = [], []
    for name in ORDER:
        folder = root / name
        if not (folder / "rq3_summary.json").exists():
            continue
        c, p = summarize_condition(folder)
        all_conds += c; all_pairs += p
    C = pd.DataFrame(all_conds); P = pd.DataFrame(all_pairs)
    C.to_csv(root / "summary_conditions.csv", index=False); P.to_csv(root / "summary_pairs.csv", index=False)
    P["better"] = P["hi"] < 0; P["worse"] = P["lo"] > 0
    counts = P.groupby(["condition", "budget", "pair"]).agg(mean_diff=("diff", "mean"), better=("better", "sum"),
                                                          worse=("worse", "sum"), n=("diff", "size"))
    pd.set_option("display.width", 220)
    print(C.set_index(["condition", "budget"])[METHODS].round(4).to_string())
    print(counts.round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
