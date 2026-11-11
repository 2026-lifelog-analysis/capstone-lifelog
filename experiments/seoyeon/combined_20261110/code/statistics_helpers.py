"""Paired evaluation and participant-clustered uncertainty for fixed predictions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

KEY = ["pid", "date"]


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pid"] = result["pid"].astype(str)
    result["date"] = pd.to_datetime(result["date"])
    return result


def validate_group(predictions: pd.DataFrame, expected: pd.DataFrame,
                   seeds: Sequence[int] = (0, 1, 2),
                   folds: Sequence[int] = (1, 2, 3, 4, 5)) -> dict:
    """Reject changed evaluation identities; never score an incomplete prediction set."""
    truth = canonical_keys(expected)
    if truth.duplicated(KEY).any() or not np.isfinite(truth["y_true"]).all():
        raise ValueError("Expected evaluation keys or targets are invalid")
    if set(truth["fold"]) != set(folds):
        raise ValueError("Expected evaluation data does not cover the required folds")
    n_expected = len(truth) * len(seeds)
    empty = {"status": "partial", "n_predictions": 0, "n_expected_predictions": n_expected,
             "n_missing": n_expected, "n_nonfinite": 0, "missing_seed_folds": "all"}
    if predictions.empty:
        return empty
    p = canonical_keys(predictions)
    if p.duplicated(KEY + ["seed"]).any():
        raise ValueError("Duplicate participant/date/seed predictions")
    if not set(p["seed"]) <= set(seeds) or not set(p["fold"]) <= set(folds):
        raise ValueError("Unexpected seed or fold in predictions")
    merged = p.merge(truth, on=KEY, how="left", suffixes=("", "_expected"),
                     validate="many_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Predictions include excluded or unknown evaluation nights")
    if not (merged["fold"] == merged["fold_expected"]).all():
        raise ValueError("Participant fold assignment differs from the fixed split")
    if not np.allclose(merged["y_true"], merged["y_true_expected"], rtol=0, atol=1e-9):
        raise ValueError("Recorded targets differ from the fixed evaluation truth")
    nonfinite = int((~np.isfinite(p["y_pred"])).sum())
    missing = n_expected - len(p)
    absent = [(s, f) for s in seeds for f in folds
              if not ((p["seed"] == s) & (p["fold"] == f)).any()]
    status = "incomplete" if nonfinite else "partial" if missing else "complete"
    return {"status": status, "n_predictions": len(p), "n_expected_predictions": n_expected,
            "n_missing": missing, "n_nonfinite": nonfinite,
            "missing_seed_folds": ";".join(f"{s}:{f}" for s, f in absent)}


def aggregate_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average seed-wise errors for each night, not the seed-wise predictions."""
    p = canonical_keys(predictions)
    residual = p["y_pred"].to_numpy() - p["y_true"].to_numpy()
    errors = p[KEY + ["fold"]].assign(abs_error=np.abs(residual), squared_error=residual ** 2,
                                      within30=(np.abs(residual) <= 30).astype(float))
    return errors.groupby(KEY + ["fold"], sort=True, as_index=False).mean()


def metrics_from_errors(nights: pd.DataFrame) -> dict:
    return {"n_people": int(nights["pid"].nunique()), "n_nights": len(nights),
            "mae": float(nights["abs_error"].mean()),
            "rmse": float(np.sqrt(nights["squared_error"].mean())),
            "person_mae_mean": float(nights.groupby("pid")["abs_error"].mean().mean()),
            "within30_percent": float(100 * nights["within30"].mean())}


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if ((p[finite] < 0) | (p[finite] > 1)).any():
        raise ValueError("P values must be in [0,1]")
    output = np.full(len(p), np.nan)
    order = finite[np.argsort(p[finite], kind="stable")]
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * p[order])
    output[order] = np.minimum(adjusted, 1.)
    return output


def bootstrap_contrasts(nights: pd.DataFrame,
                        contrasts: Mapping[str, Mapping[str, float]],
                        repetitions: int = 2000, seed: int = 20261110) -> pd.DataFrame:
    """Resample people, retaining all their nights and the night-weighted denominator.

    Intervals condition on the fitted models and observed development cohort. They
    do not include uncertainty from retraining or reselecting feature definitions.
    Bootstrap sign-tail p values are approximate, not exact permutation-test p values.
    """
    if repetitions < 100 or nights.empty:
        raise ValueError("Require nonempty paired nights and at least 100 bootstrap draws")
    columns = list(dict.fromkeys(c for weights in contrasts.values() for c in weights))
    if not columns or not np.isfinite(nights[columns].to_numpy()).all():
        raise ValueError("Contrasts require complete finite paired errors")
    grouped = nights.groupby("pid", sort=True)
    sums = grouped[columns].sum().to_numpy(dtype=float)
    counts = grouped.size().to_numpy(dtype=float)
    n_people = len(counts)
    coefficients = np.array([[weights.get(c, 0.) for c in columns]
                             for weights in contrasts.values()])
    if not np.allclose(coefficients.sum(axis=1), 0.):
        raise ValueError("An error-difference contrast must have coefficients summing to zero")
    contrast_sums = sums @ coefficients.T
    point = contrast_sums.sum(axis=0) / counts.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, len(contrasts)))
    for start in range(0, repetitions, 128):
        stop = min(start + 128, repetitions)
        weights = rng.multinomial(n_people, np.full(n_people, 1. / n_people), size=stop - start)
        draws[start:stop] = (weights @ contrast_sums) / (weights @ counts)[:, None]
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    lower_tail = (np.sum(draws <= 0, axis=0) + 1) / (repetitions + 1)
    upper_tail = (np.sum(draws >= 0, axis=0) + 1) / (repetitions + 1)
    pvalues = np.minimum(1., 2 * np.minimum(lower_tail, upper_tail))
    return pd.DataFrame({"contrast": list(contrasts), "estimate": point,
                         "ci_low": low, "ci_high": high, "p_bootstrap_sign_approx": pvalues,
                         "n_people": n_people, "n_nights": int(counts.sum()),
                         "bootstrap_repetitions": repetitions, "bootstrap_seed": seed})
