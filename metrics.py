"""Point-forecast errors and chronological window summaries."""
from numbers import Integral

import numpy as np
import pandas as pd


METRIC_NAMES = ("WAPE_percent", "MAE", "RMSE")


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def paired_arrays(actual, prediction):
    """Validate aligned finite vectors without dropping or broadcasting rows."""
    if isinstance(actual, pd.Series) and isinstance(prediction, pd.Series):
        if not actual.index.equals(prediction.index):
            raise ValueError("Actual and prediction Series indexes must match in order.")
    try:
        y, p = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Actuals and predictions must be numeric vectors.") from exc
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or not y.size:
        raise ValueError("Actuals and predictions must be nonempty, equal-length 1D vectors.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Actuals and predictions must be finite; missing rows are not dropped.")
    return y, p


def metrics(actual, prediction):
    """Return WAPE in percent, and MAE/RMSE in the target's original units."""
    y, p = paired_arrays(actual, prediction)
    error = p - y
    denominator = np.abs(y).sum()
    return {
        "WAPE_percent": float(100 * np.abs(error).sum() / denominator) if denominator > 0 else None,
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def scored_frame(predictions, names, score_start=42, window_size=21):
    """Select a common, consecutive set of scored daily origins for all models."""
    positive_integer(score_start, "score_start", minimum=0)
    positive_integer(window_size, "window_size")
    if isinstance(names, str):
        raise ValueError("names must be a sequence of model column names.")
    names = list(names)
    reserved = {"actual", "origin_index", "date", "context_end"}
    if not names or any(not isinstance(n, str) or n in reserved for n in names):
        raise ValueError("Provide prediction column names, excluding metadata and actuals.")
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique.")
    if not isinstance(predictions, pd.DataFrame) or predictions.columns.has_duplicates:
        raise ValueError("predictions must be a DataFrame with unique columns.")
    missing = {"origin_index", "actual", *names} - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    origins = predictions.origin_index.to_numpy(dtype=float)
    if (not len(origins) or not np.isfinite(origins).all() or (origins < 0).any()
            or (origins != np.floor(origins)).any() or (np.diff(origins) != 1).any()):
        raise ValueError("origin_index must contain consecutive, increasing nonnegative integers.")
    if "date" in predictions:
        try:
            dates = pd.to_datetime(predictions.date, errors="raise")
            valid_dates = (not dates.isna().any() and dates.dt.tz is None
                           and dates.equals(dates.dt.normalize())
                           and dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all())
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("date must contain consecutive daily dates without a timezone.") from exc
        if not valid_dates:
            raise ValueError("date must contain consecutive daily dates without a timezone.")
    frame = predictions.loc[origins >= score_start].copy()
    if frame.empty:
        raise ValueError("No scored targets: provide more history or a smaller score_start.")
    for name in names:
        y, p = paired_arrays(frame.actual, frame[name])
        frame["actual"], frame[name] = y, p
    return frame


def score(predictions, names, score_start=42, window_size=21):
    """Pooled (ALL) and consecutive window (W1, W2, ...) point metrics."""
    names = list(names) if not isinstance(names, str) else names
    frame = scored_frame(predictions, names, score_start, window_size)
    periods = [("ALL", frame)] + [
        (f"W{i}", frame.iloc[start:start + window_size])
        for i, start in enumerate(range(0, len(frame), window_size), 1)
    ]
    return pd.DataFrame([
        {"period": period, "model": name, "n": len(part), **metrics(part.actual, part[name])}
        for period, part in periods for name in names
    ])


def window_summary(scores):
    """Equal-weight window means and sample SDs (ddof=1), with pooled metrics."""
    rows = []
    for name, frame in scores.groupby("model", sort=False):
        pooled = frame.loc[frame.period == "ALL"].iloc[0]
        windows = frame.loc[frame.period != "ALL"]
        row = {"model": name, "n": int(pooled.n), "n_windows": len(windows)}
        row.update({key: pooled[key] for key in METRIC_NAMES})
        for key in METRIC_NAMES:
            values = windows[key].to_numpy(dtype=float)
            complete = len(values) > 0 and np.isfinite(values).all()
            prefix = "WAPE" if key == "WAPE_percent" else key
            unit = "_percent" if prefix == "WAPE" else ""
            sd_unit = "_pp" if prefix == "WAPE" else ""
            row[f"{prefix}_valid_windows"] = int(np.isfinite(values).sum())
            row[f"{prefix}_macro{unit}"] = float(values.mean()) if complete else None
            row[f"{prefix}_window_sample_std{sd_unit}"] = (
                float(values.std(ddof=1)) if complete and len(values) > 1 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def relative_improvements(scores, baseline):
    """Candidate-minus-baseline deltas; positive relative reduction is better."""
    if baseline not in set(scores.model):
        raise ValueError("baseline must be one of the scored model columns.")
    columns = ["period", "model", "baseline", "n", "delta_WAPE_pp",
               "relative_WAPE_reduction_percent", "delta_MAE", "relative_MAE_reduction_percent",
               "delta_RMSE", "relative_RMSE_reduction_percent"]
    rows = []
    for period, frame in scores.groupby("period", sort=False):
        base = frame.loc[frame.model == baseline].iloc[0]
        for _, model in frame.loc[frame.model != baseline].iterrows():
            row = {"period": period, "model": model.model, "baseline": baseline, "n": int(model.n)}
            for key in METRIC_NAMES:
                a, b = model[key], base[key]
                valid = pd.notna(a) and pd.notna(b)
                label = "WAPE" if key == "WAPE_percent" else key
                row[f"delta_{label}" + ("_pp" if label == "WAPE" else "")] = float(a-b) if valid else None
                row[f"relative_{label}_reduction_percent"] = float(100*(b-a)/b) if valid and b > 0 else None
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)
