"""Analyze complete paired prediction sets without treating seeds as an ensemble.

Usage: python -B code/analyze_results.py --root /path/to/experiment
Only new CSV/JSON files under artifacts/analysis are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from statistics_helpers import (aggregate_errors, bootstrap_contrasts, canonical_keys,
                                holm_adjust, metrics_from_errors, validate_group)

CONDITIONS = ("base", "A", "B", "C", "AB", "AC", "BC", "ABC")
BASELINES = ("group_median", "personal_mean7", "personal_median28", "shrink_blend28")
TARGETS = {"sleep": "target_sleep_min", "bed": "target_bedtime", "wake": "target_waketime"}
PREFIXES = {"sleep": "sleep", "bed": "bedtime", "wake": "waketime"}
MODEL_FAMILIES = tuple(f"modelcheck_{m}" for m in ("catboost", "ridge", "mlp", "ftt"))
FAMILY_PLAN = {"catboost": CONDITIONS, "baseline": BASELINES,
               **{family: ("base", "ABC") for family in MODEL_FAMILIES}}
GROUP_COLUMNS = ["family", "condition", "target"]
SEEDS = (0, 1, 2)
FOLDS = (1, 2, 3, 4, 5)


def expected_nights(features: pd.DataFrame, split: pd.DataFrame, target: str,
                    target_column: str) -> pd.DataFrame:
    f = canonical_keys(features)
    if f.duplicated(["pid", "date"]).any() or split["pid"].astype(str).duplicated().any():
        raise ValueError("Features or participant split contain duplicate keys")
    dev = split.loc[split["role"] == "dev", ["pid", "fold"]].copy()
    dev["pid"] = dev["pid"].astype(str)
    expected = f[["pid", "date", target_column]].merge(dev, on="pid", validate="many_to_one")
    expected = expected.loc[expected[target_column].notna()].rename(columns={target_column: "y_true"})
    if expected.empty:
        raise ValueError(f"No valid evaluation targets for {target}")
    return expected.sort_values(["pid", "date"]).reset_index(drop=True)


def read_group(folder: Path, family: str, condition: str, target: str,
               fingerprints: dict) -> pd.DataFrame:
    from run_factorial import signature
    from src.forecast_eval import keys_hash
    pattern = re.compile(rf"{re.escape(family)}_{re.escape(condition)}_{re.escape(target)}_f(\d+)_s(\d+)\.parquet")
    files = [p for p in sorted(folder.glob(f"{family}_{condition}_{target}_f*_s*.parquet"))
             if pattern.fullmatch(p.name)]
    if not files:
        return pd.DataFrame()
    parts = []
    for path in files:
        match = pattern.fullmatch(path.name)
        identity = [family, condition, target, int(match[1]), int(match[2])]
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            raise ValueError(f"Checkpoint metadata is missing: {path.name}")
        metadata = json.loads(meta_path.read_text())
        if metadata.get("identity") != identity or metadata.get("final_test_scored") is not False:
            raise ValueError(f"Checkpoint identity or holdout policy mismatch: {path.name}")
        if metadata.get("input_fingerprints") != fingerprints:
            raise ValueError(f"Checkpoint input or implementation changed: {path.name}")
        reconstructed = signature(fingerprints, identity, metadata["features"], metadata["settings"])
        if reconstructed != metadata.get("signature"):
            raise ValueError(f"Checkpoint signature is invalid: {path.name}")
        part = canonical_keys(pd.read_parquet(path))
        for column, value in zip(GROUP_COLUMNS + ["fold", "seed"], identity):
            if column not in part or not (part[column] == value).all():
                raise ValueError(f"Prediction filename and {column} labels disagree")
        if (len(part) != metadata.get("n_rows") or part["pid"].nunique() != metadata.get("n_people")
                or keys_hash(part) != metadata.get("eval_keys_sha256")):
            raise ValueError(f"Checkpoint row metadata differs from predictions: {path.name}")
        actual_mae = float((part["y_pred"] - part["y_true"]).abs().mean())
        if not np.isclose(actual_mae, metadata.get("mae", np.nan), rtol=0, atol=1e-9):
            raise ValueError(f"Checkpoint MAE metadata differs from predictions: {path.name}")
        parts.append(part)
    group = pd.concat(parts, ignore_index=True)
    return canonical_keys(group)


def summarize_group(group: pd.DataFrame, expected: pd.DataFrame, family: str,
                    condition: str, target: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    seeds = (0,) if family == "baseline" else SEEDS
    identity = dict(zip(GROUP_COLUMNS, (family, condition, target)))
    status = validate_group(group, expected, seeds, FOLDS)
    row = {**identity, **status, "n_required_seeds": len(seeds)}
    if status["status"] != "complete":
        return row, pd.DataFrame(), pd.DataFrame()
    nights = aggregate_errors(group)
    residual = group["y_pred"] - group["y_true"]
    err = group.assign(abs_error=residual.abs(), squared_error=residual ** 2,
                       within30=(residual.abs() <= 30).astype(float))
    details = []
    for (seed, fold), part in err.groupby(["seed", "fold"], sort=True):
        details.append({**identity, "seed": seed, "fold": fold, **metrics_from_errors(part)})
    seed_maes = err.groupby("seed")["abs_error"].mean()
    fold_maes = nights.groupby("fold")["abs_error"].mean()
    row.update({**metrics_from_errors(nights), "fold_mae_mean": float(fold_maes.mean()),
                "fold_mae_std": float(fold_maes.std(ddof=1)),
                "seed_mae_mean": float(seed_maes.mean()),
                "seed_mae_std": float(seed_maes.std(ddof=1)) if len(seed_maes) > 1 else np.nan})
    return row, nights, pd.DataFrame(details)


def paired_wide(nights_by_condition: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = None
    for condition, nights in nights_by_condition.items():
        current = nights[["pid", "date", "abs_error"]].rename(columns={"abs_error": condition})
        if result is None:
            result = current.copy()
        else:
            before = len(result)
            result = result.merge(current, on=["pid", "date"], how="outer", validate="one_to_one",
                                  indicator=True)
            if len(result) != before or not (result["_merge"] == "both").all():
                raise ValueError("Conditions have different paired evaluation nights")
            result = result.drop(columns="_merge")
    return result if result is not None else pd.DataFrame()


def factorial_contrasts() -> tuple[dict, dict]:
    primary = {f"{c}_vs_base": {"base": 1., c: -1.} for c in CONDITIONS if c != "base"}
    primary.update({f"ABC_vs_{c}": {c: 1., "ABC": -1.} for c in ("AB", "AC", "BC", "A", "B", "C")})
    interactions = {
        "ABC_gain_minus_sum_of_single_gains": {"base": -2., "A": 1., "B": 1., "C": 1., "ABC": -1.},
        **{f"{a}{b}_interaction": {"base": -1., a: 1., b: 1., a + b: -1.}
           for a, b in (("A", "B"), ("A", "C"), ("B", "C"))},
    }
    return primary, interactions


def subgroup_metrics(nights: pd.DataFrame, features: pd.DataFrame, target: str,
                     family: str, condition: str) -> pd.DataFrame:
    count_col = f"{PREFIXES[target]}_n28"
    required = ["pid", "date", count_col, "days_since_last"]
    if not set(required) <= set(features.columns):
        raise ValueError("History-count and gap features are required for subgroup analysis")
    merged = nights.merge(canonical_keys(features)[required], on=["pid", "date"], how="left",
                          validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("Subgroup metadata is missing evaluation nights")
    count = merged[count_col]
    if count.isna().any() or (count < 0).any() or (count > 28).any() or (count % 1 != 0).any():
        raise ValueError("Invalid 28-night history counts")
    gap = merged["days_since_last"]
    if ((gap < 1) | np.isinf(gap) | (gap.notna() & (gap % 1 != 0))).any():
        raise ValueError("Last-record distance must be a positive whole number or missing")
    history = pd.cut(count, [-.5, .5, 3.5, 7.5, 14.5, 21.5, 28.5],
                     labels=["0", "1-3", "4-7", "8-14", "15-21", "22-28"])
    gap_bin = pd.Series(np.select([gap.isna(), gap == 1, gap.between(2, 7), gap >= 8],
                                  ["no_history", "continuous", "missing_1-6", "missing_7+"],
                                  default="invalid"), index=merged.index)
    rows = []
    for dimension, bins in (("history", history), ("gap", gap_bin)):
        for label, part in merged.groupby(bins, observed=True, sort=False):
            rows.append({"family": family, "condition": condition, "target": target,
                         "dimension": dimension, "bin": str(label), **metrics_from_errors(part)})
    return pd.DataFrame(rows)


def _adjust_all(parts: list[pd.DataFrame], family: str, planned_tests: int) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    # Missing planned comparisons count as p=1, so partial runs never weaken correction.
    observed = len(frame)
    if observed > planned_tests:
        raise ValueError("More observed comparisons than the fixed multiplicity family")
    pvalues = np.r_[frame["p_bootstrap_sign_approx"].to_numpy(), np.ones(planned_tests - observed)]
    frame["p_holm"] = holm_adjust(pvalues)[:observed]
    frame["multiplicity_family"] = family
    frame["multiplicity_n_tests"] = planned_tests
    frame["multiplicity_observed_tests"] = observed
    return frame


def run(root: Path, families: tuple[str, ...], repetitions: int) -> dict:
    from run_factorial import input_fingerprints
    artifacts = root / "artifacts"
    fingerprints = input_fingerprints(root)
    features = pd.read_parquet(artifacts / "features.parquet")
    split = pd.read_csv(artifacts / "split.csv", dtype={"pid": str})
    spec = json.loads((artifacts / "feature_spec.json").read_text())
    targets = spec.get("targets", TARGETS)
    summaries, detail_parts, subgroups, primary, interactions, models, additive = [], [], [], [], [], [], []
    for target, target_column in targets.items():
        expected = expected_nights(features, split, target, target_column)
        caches = {}
        for family in families:
            for condition in FAMILY_PLAN[family]:
                group = read_group(artifacts / "predictions", family, condition, target, fingerprints)
                row, nights, detail = summarize_group(group, expected, family, condition, target)
                summaries.append(row)
                if row["status"] != "complete":
                    continue
                caches[(family, condition)] = nights
                detail_parts.append(detail)
                subgroups.append(subgroup_metrics(nights, features, target, family, condition))
        if all(("catboost", c) in caches for c in CONDITIONS):
            wide = paired_wide({c: caches[("catboost", c)] for c in CONDITIONS})
            p, i = factorial_contrasts()
            primary.append(bootstrap_contrasts(wide, p, repetitions).assign(target=target))
            interactions.append(bootstrap_contrasts(wide, i, repetitions).assign(target=target))
            means = wide[list(CONDITIONS)].mean()
            single_sum = sum(means["base"] - means[c] for c in ("A", "B", "C"))
            observed = means["base"] - means["ABC"]
            additive.append({"target": target, "base_mae": means["base"], "ABC_mae": means["ABC"],
                             "single_gain_sum": single_sum, "observed_combined_gain": observed,
                             "interaction": observed - single_sum,
                             "additive_implied_mae": means["base"] - single_sum})
        if all((family, c) in caches for family in MODEL_FAMILIES for c in ("base", "ABC")):
            wide = paired_wide({f"{family}:{c}": caches[(family, c)]
                                for family in MODEL_FAMILIES for c in ("base", "ABC")})
            contrasts = {f"{f}_ABC_vs_base": {f"{f}:base": 1., f"{f}:ABC": -1.} for f in MODEL_FAMILIES}
            contrasts.update({f"{f}_{c}_vs_catboost": {f"modelcheck_catboost:{c}": 1., f"{f}:{c}": -1.}
                              for f in MODEL_FAMILIES[1:] for c in ("base", "ABC")})
            models.append(bootstrap_contrasts(wide, contrasts, repetitions).assign(target=target))
    output = artifacts / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    tables = {"fold_seed_metrics": detail_parts, "subgroups": subgroups}
    for name, parts in tables.items():
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(output / f"{name}.csv", index=False)
    _adjust_all(primary, "catboost_primary_all_targets", 13 * len(targets)).to_csv(output / "comparisons.csv", index=False)
    _adjust_all(interactions, "catboost_interactions_all_targets", 4 * len(targets)).to_csv(output / "interactions.csv", index=False)
    _adjust_all(models, "model_comparisons_all_targets", 10 * len(targets)).to_csv(output / "model_comparisons.csv", index=False)
    pd.DataFrame(additive).to_csv(output / "additivity.csv", index=False)
    metadata = {
        "complete": bool(summary["status"].eq("complete").all()),
        "group_status_counts": {str(k): int(v) for k, v in summary["status"].value_counts().items()},
        "families": list(families), "targets": list(targets), "seeds": list(SEEDS), "folds": list(FOLDS),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": 20261110,
        "confidence_intervals": "Individual unadjusted 95% percentile intervals, not simultaneous intervals.",
        "estimand": "Mean seed-wise absolute error over evaluation nights; not ensemble-prediction error.",
        "rmse": "Square root of mean squared errors over nights and seeds.",
        "uncertainty": "Paired resampling of participants and all their nights; fixed fitted models. "
                       "Does not include retraining, feature selection, or independent final-holdout uncertainty.",
        "p_values": "Approximate two-sided bootstrap sign-tail probabilities; +1 finite simulation correction.",
        "holm": "Fixed families: 39 primary, 12 interaction, 30 model tests for 3 targets. "
                "Missing planned comparisons count as p=1; observed and planned counts are recorded.",
        "subgroups": "Last-record distance 1=continuous; 2..7=1..6 missing nights; >=8=7+ missing nights.",
        "final_holdout_scored": False,
        "verified_input_fingerprints": fingerprints,
    }
    (output / "analysis_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--families", nargs="+", choices=list(FAMILY_PLAN), default=list(FAMILY_PLAN))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    metadata = run(args.root, tuple(args.families), args.bootstrap)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
