"""동일 시험 구간에 다섯 합성 변화를 주입하기 위한 배치와 고정 평가 분모.

실제 임상 정답을 만들지 않습니다. 주입 여부만 알려진 구성적 검증입니다.
단위는 달력일이며 점수 유효성은 주입 전 확정한 common_keys를 사용합니다.
"""
from __future__ import annotations

from collections.abc import Iterable
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.seoyeon import comparison_io as io
from experiments.seoyeon import comparison_models as models
from experiments.seoyeon.comparison_seungho import K_VALUES, directional_scores, score_with_h

FEATURES = ("steps", "sleep_min", "bed_min")
SCENARIOS = {
    "S1": {"steps": -.30, "gradual": False},
    "S2": {"steps": -.30, "gradual": True},
    "S3": {"steps": -.15, "gradual": True},
    "S4": {"bed_min": 120., "gradual": False},
    "S5": {"steps": -.20, "sleep_min": -.10, "bed_min": 60., "gradual": True},
}
SEGMENT_COLUMNS = ["pid", "start", "end", "seed"]
KEY_COLUMNS = ["participant", "date", "start", "day_offset", "seed"]


def _main_keys(main: pd.DataFrame) -> pd.DataFrame:
    required = {"pid", "date", *FEATURES}
    if not required.issubset(main):
        raise ValueError(f"Missing main columns: {sorted(required - set(main))}")
    if main.pid.isna().any():
        raise ValueError("Missing participant identifiers")
    dates = pd.to_datetime(main.date, errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("Missing observation dates")
    keys = pd.DataFrame({"participant": main.pid.astype(str), "date": dates})
    if keys.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate main participant/date")
    return keys


def _common_keys(common_keys: pd.DataFrame, main_keys: pd.DataFrame) -> pd.DataFrame:
    identifier = "participant" if "participant" in common_keys else "pid"
    if identifier not in common_keys or "date" not in common_keys:
        raise ValueError("Common keys require participant (or pid) and date")
    frame = common_keys[[identifier, "date"]].rename(columns={identifier: "participant"}).copy()
    if frame.participant.isna().any():
        raise ValueError("Missing common participant identifiers")
    frame["participant"] = frame.participant.astype(str)
    frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.normalize()
    if frame.date.isna().any():
        raise ValueError("Missing common dates")
    if frame.duplicated(["participant", "date"]).any():
        raise ValueError("Duplicate common participant/date")
    return frame.merge(main_keys, on=["participant", "date"], how="inner", validate="one_to_one")


def _segments(segments: pd.DataFrame) -> pd.DataFrame:
    if not set(SEGMENT_COLUMNS).issubset(segments):
        raise ValueError("Segments require pid, start, end, seed")
    frame = segments[SEGMENT_COLUMNS].copy()
    if frame.seed.nunique(dropna=False) > 1:
        raise ValueError("Each injection/evaluation must use a single seed")
    if frame.pid.isna().any() or frame.seed.isna().any():
        raise ValueError("Missing segment participant or seed")
    frame["pid"] = frame.pid.astype(str)
    frame["start"] = pd.to_datetime(frame.start, errors="raise").dt.normalize()
    frame["end"] = pd.to_datetime(frame.end, errors="raise").dt.normalize()
    if frame[["start", "end"]].isna().any().any():
        raise ValueError("Missing segment dates")
    if not frame.end.sub(frame.start).dt.days.eq(13).all():
        raise ValueError("Each segment must contain exactly 14 calendar days")
    if frame.pid.duplicated().any():
        raise ValueError("Only one segment per participant is allowed within a seed")
    return frame


def schedule(main: pd.DataFrame, common_keys: pd.DataFrame, split: dict,
             seeds: Iterable[int] = range(20)) -> pd.DataFrame:
    """시나리오에 무관한 동일 20회 구간을 선택합니다.

    시작 후보는 시험 기간 전체 달력일입니다. 직전 28일의 완전 관측 >=14,
    주입 14일 내 공통 유효 인일 >=7, 주입 끝은 시험 마지막 날 이하입니다.
    사람 식별자를 문자열 정렬하고 각 seed의 default_rng로 균등 선택합니다.
    """
    keys = _main_keys(main)
    common = _common_keys(common_keys, keys)
    test_start = pd.Timestamp(split["test_start"]).normalize()
    test_end = pd.Timestamp(split["end"]).normalize()
    if pd.isna(test_start) or pd.isna(test_end) or test_end < test_start:
        raise ValueError("Invalid test interval")
    candidates = pd.date_range(test_start, test_end - pd.Timedelta(days=13))
    candidate_days = candidates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    complete = np.isfinite(main[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    observed = keys.assign(complete=complete)
    common_by_pid = {str(pid): group.date.to_numpy(dtype="datetime64[D]").astype(np.int64)
                     for pid, group in common.sort_values("date").groupby("participant", sort=True)}
    eligible = {}
    for pid, person in observed.sort_values("date").groupby("participant", sort=True):
        past = person.loc[person.complete, "date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
        evaluation = common_by_pid.get(str(pid), np.array([], dtype=np.int64))
        history_n = np.searchsorted(past, candidate_days) - np.searchsorted(past, candidate_days - 28)
        scored_n = np.searchsorted(evaluation, candidate_days + 14) - np.searchsorted(evaluation, candidate_days)
        valid = candidates[(history_n >= 14) & (scored_n >= 7)]
        if len(valid):
            eligible[str(pid)] = valid
    seed_values = list(seeds)
    if len(set(seed_values)) != len(seed_values) or any(not isinstance(s, (int, np.integer)) or s < 0 for s in seed_values):
        raise ValueError("Seeds must be distinct nonnegative integers")
    records = []
    for seed in seed_values:
        rng = np.random.default_rng(seed)
        for pid in sorted(eligible):
            valid = eligible[pid]
            start = valid[rng.integers(len(valid))]
            records.append({"pid": pid, "start": start, "end": start + pd.Timedelta(days=13), "seed": int(seed)})
    return pd.DataFrame(records, columns=SEGMENT_COLUMNS)


def injection(main: pd.DataFrame, segments: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """원본을 바꾸지 않고 단일 반복의 S1~S5 변화를 관측값에 주입합니다.

    점진 변화 비율은 (달력 일차 + 1)/14입니다. bed_min에 분을 선형으로 더하고
    여기서는 나머지 연산하지 않습니다. 서연 모델 연결에서만 원형 변환합니다.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    keys = _main_keys(main)
    selected = _segments(segments)
    unknown = set(selected.pid) - set(keys.participant)
    if unknown:
        raise ValueError("Segment contains participants absent from main input")
    frame = main.copy(deep=True)
    if selected.empty:
        return frame
    starts = keys.participant.map(selected.set_index("pid").start)
    offsets = keys.date.sub(starts).dt.days
    included = offsets.between(0, 13).to_numpy()
    config = SCENARIOS[scenario]
    fractions = (offsets.to_numpy(dtype=float) + 1.) / 14. if config["gradual"] else np.ones(len(main))
    for feature in FEATURES:
        if feature not in config:
            continue
        values = main[feature].to_numpy(dtype=float, copy=True)
        change = config[feature] * fractions[included]
        values[included] = values[included] + change if feature == "bed_min" else values[included] * (1. + change)
        frame[feature] = values
    return frame


def segment_keys(main: pd.DataFrame, segments: pd.DataFrame,
                 common_keys: pd.DataFrame) -> pd.DataFrame:
    """주입 전 확정된 공통 분모 중 주입 구간의 키와 0부터 시작하는 일차를 반환."""
    keys = _main_keys(main)
    common = _common_keys(common_keys, keys)
    selected = _segments(segments).rename(columns={"pid": "participant"})
    joined = common.merge(selected, on="participant", how="inner", validate="many_to_one")
    joined = joined[joined.date.between(joined.start, joined.end)].copy()
    joined["day_offset"] = joined.date.sub(joined.start).dt.days
    return joined[KEY_COLUMNS].sort_values(["participant", "date"]).reset_index(drop=True)


def _statistical_scores(changed: pd.DataFrame, participants: list[str], split: dict,
                        baselines: pd.DataFrame) -> pd.DataFrame:
    """개인별 독립 계산만 부분 실행하고 개발 집단 기준선은 전체 자료 값으로 고정."""
    reference = baselines[baselines.model.eq("group") & baselines.participant.eq("__pooled__")]
    if len(reference) != 1:
        raise ValueError("Exactly one clean pooled group baseline is required")
    row = reference.iloc[0]
    config = {
        "group_reference": {feature: {"center": float(row[f"center_{feature}"]),
                                       "scale": float(row[f"scale_{feature}"])}
                            for feature in models.FEATURES},
        "group_reference_n": int(row.baseline_n),
    }
    selected = changed[changed.pid.astype(str).isin(participants)]
    scores, _ = models.score_models(models.adapt_main(selected), split["development_end"],
                                    split["calibration_start"], config)
    return scores.pivot(index=["participant", "date"], columns="model", values="score").reset_index()


def _job(main: pd.DataFrame, keys: pd.DataFrame, segments: pd.DataFrame, scenario: str,
         split: dict, baselines: pd.DataFrame, thresholds: list[dict], output: Path,
         input_hashes: dict) -> dict:
    """하나의 시나리오·반복을 실제 재계산하며 경보선을 다시 학습하지 않습니다."""
    seed = int(segments.seed.iloc[0])
    job_dir = output / "synthetic" / f"{scenario}_seed{seed:02d}"
    job_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        changed = injection(main, segments, scenario)
        sample = segment_keys(main, segments, keys)
        if sample.empty:
            raise ValueError("No common finite sample days in selected injection segments")
        segments.to_csv(job_dir / "segments.csv", index=False)
        canonical = models.adapt_main(changed[changed.pid.astype(str).isin(segments.pid)])
        canonical.to_csv(job_dir / "canonical_daily.csv.gz", index=False)
        sample.assign(split="test").to_csv(job_dir / "moment_keys.csv.gz", index=False)
        stats_started = time.monotonic()
        own = _statistical_scores(changed, segments.pid.tolist(), split, baselines)
        own_sample = sample.merge(own, on=["participant", "date"], how="left", validate="one_to_one")
        statistics_seconds = time.monotonic() - stats_started
        sh_started = time.monotonic()
        directional = directional_scores(changed)
        direction_seconds = time.monotonic() - sh_started
        records = []
        for setting in thresholds:
            h = {name: setting["thresholds"][name] for name in K_VALUES}
            sh = score_with_h(changed, directional, h)
            sh_keys = changed[["pid", "date"]].rename(columns={"pid": "participant"})
            combined = own_sample.merge(pd.concat([sh_keys, sh], axis=1),
                                        on=["participant", "date"], how="left", validate="one_to_one")
            names = [*models.MODELS, *K_VALUES]
            combined.to_csv(job_dir / f"scores_{setting['id']}.csv.gz", index=False)
            invalid = {name: int((~np.isfinite(combined[name])).sum()) for name in names}
            records.append({"id": setting["id"], "rate": setting["rate"],
                            "thresholds": setting["thresholds"], "invalid_days": invalid})
            if any(invalid.values()):
                raise ValueError(f"Synthetic scores became nonfinite on fixed common keys: {invalid}")
        artifact_hashes = {path.name: io.sha256(path) for path in sorted(job_dir.iterdir()) if path.is_file()}
        result = {"status": "passed", "scenario": scenario, "seed": seed,
                  "participants": len(segments), "sample_days": len(sample),
                  "canonical_rows": len(canonical), "seconds": time.monotonic() - started,
                  "statistics_seconds": statistics_seconds, "direction_seconds": direction_seconds,
                  "input_hashes": input_hashes, "artifact_hashes": artifact_hashes,
                  "settings": records, "moment_status": "awaiting_new_inference",
                  "group_reference": "Clean full-development pooled reference; test-only perturbation",
                  "seungho_population": "Full perturbed population recomputed before sampling",
                  "thresholds_refitted": False}
        io.write_json(job_dir / "job.json", result)
        print(json.dumps({k: result[k] for k in ["status", "scenario", "seed", "participants", "sample_days", "seconds"]}), flush=True)
        return result
    except Exception as error:
        io.write_json(job_dir / "job_failure.json", {
            "status": "failed", "scenario": scenario, "seed": seed,
            "seconds": time.monotonic() - started, "error": f"{type(error).__name__}: {error}",
            "input_hashes": input_hashes,
        })
        raise


def run_jobs(output: str | Path, seeds: Iterable[int] = range(20),
             scenarios: Iterable[str] = tuple(SCENARIOS)) -> list[dict]:
    """한 작업자에서 순차 실행합니다. 기존 작업 디렉터리는 덮어쓰지 않습니다."""
    output = Path(output)
    selected_scenarios = list(scenarios)
    if len(set(selected_scenarios)) != len(selected_scenarios) or set(selected_scenarios) - set(SCENARIOS):
        raise ValueError("Scenarios must be distinct S1 through S5")
    files = ["main_daily.csv.gz", "common_candidate_keys.csv.gz", "split.json",
             "seoyeon_baselines.csv", "clean_thresholds_without_moment.json"]
    input_hashes = {name: io.sha256(output / name) for name in files}
    main = io.read_frame(output / "main_daily.csv.gz").sort_values(["pid", "date"]).reset_index(drop=True)
    keys = io.read_frame(output / "common_candidate_keys.csv.gz")
    keys = keys[keys.split.eq("test")].copy()
    split = json.loads((output / "split.json").read_text())
    baselines = io.read_frame(output / "seoyeon_baselines.csv")
    thresholds = json.loads((output / "clean_thresholds_without_moment.json").read_text())
    if {entry["id"] for entry in thresholds} != {"r05", "r033"}:
        raise ValueError("Both precomputed rate settings are required")
    planned = schedule(main, keys, split, seeds)
    if planned.empty:
        raise ValueError("No eligible injection participants")
    results = []
    for seed, segments in planned.groupby("seed", sort=True):
        for scenario in selected_scenarios:
            result = _job(main, keys, segments, scenario, split, baselines, thresholds, output, input_hashes)
            results.append(result)
    if input_hashes != {name: io.sha256(output / name) for name in files}:
        raise ValueError("An input file changed during synthetic execution")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=20)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    arguments = parser.parse_args()
    run_jobs(arguments.output, range(arguments.seed_start, arguments.seed_stop), arguments.scenarios)
