"""Sequential evaluation with delayed target updates."""
from dataclasses import replace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import CALIBRATION_START, validate_cache, validate_daily
from .model import Configuration, RRAForecaster


def configurations(ablations=False):
    base = Configuration()
    controls = {
        "Without_operations": replace(base, operational_features=False),
        "Without_quantile_features": replace(base, quantile_features=False),
        "Uniform_weights": replace(base, half_life=None),
        "Expanding_archive": replace(base, window=None),
    }
    return {"RRA": base, **(controls if ablations else {})}


def replay(data, cache, configs=None, stop=None):
    data = validate_daily(data)
    cache = validate_cache(data, cache.reset_index() if cache.index.name == "origin_index" else cache)
    models = {name: RRAForecaster(c) for name, c in (configs or configurations()).items()}
    rows = []
    with threadpool_limits(limits=1):
        for t in range(CALIBRATION_START, len(data) if stop is None else stop):
            row = {"origin_index": t, "date": str(data.date.iloc[t].date()),
                   "context_end": str(data.date.iloc[t - 1].date()),
                   "Chronos2": float(cache.at[t, "Chronos2"])}
            for name, model in models.items():
                row[name] = model.predict(data.iloc[:t], data.date.iloc[t], cache.loc[t])
            actual = float(data.sales.iloc[t])
            for model in models.values():
                model.observe(actual)
            rows.append({**row, "actual": actual})
    return pd.DataFrame(rows)


def metrics(actual, prediction):
    y, p = np.asarray(actual, float), np.asarray(prediction, float)
    error = p - y
    denominator = np.abs(y).sum()
    return {
        "WAPE_percent": float(100 * np.abs(error).sum() / denominator) if denominator > 0 else None,
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def score(predictions, names, score_start=42, window_size=21):
    if score_start < CALIBRATION_START or window_size < 1:
        raise ValueError("Invalid scoring start or window size.")
    scored = predictions[predictions.origin_index >= score_start]
    if scored.empty:
        raise ValueError("No scored targets: provide more history or a smaller score start.")
    periods = [("ALL", scored)]
    for i, start in enumerate(range(0, len(scored), window_size), 1):
        periods.append((f"W{i}", scored.iloc[start:start + window_size]))
    rows = []
    for period, frame in periods:
        for name in names:
            rows.append({"period": period, "model": name, "n": len(frame),
                         **metrics(frame.actual, frame[name])})
    return pd.DataFrame(rows)
