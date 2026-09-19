"""Validate daily inputs and bind cached forecasts to their source data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_COLUMNS = ["date", "sales", "order_count", "quantity", "active_products", "avg_order_value"]
QUANTILES = [f"q{i}" for i in range(10, 100, 10)]
CALIBRATION_START = 14


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_daily(data):
    missing = set(DATA_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing daily columns: {sorted(missing)}")
    data = data.loc[:, DATA_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    dates = data["date"]
    if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Use nonmissing, timezone-free calendar dates.")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Daily dates must be unique and in increasing order.")
    if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily dates must be consecutive; missing dates are not zero-filled.")
    values = data[DATA_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("All daily measurements must be finite and nonnegative.")
    data[DATA_COLUMNS[1:]] = values
    return data.reset_index(drop=True)


def read_daily(path):
    return validate_daily(pd.read_csv(path, usecols=DATA_COLUMNS))


def validate_cache(data, cache):
    required = ["origin_index", "date", "context_end", "Chronos2"] + QUANTILES
    if not set(required).issubset(cache.columns):
        raise ValueError(f"Forecast cache requires columns: {required}")
    cache = cache.loc[:, required].copy()
    if cache.origin_index.tolist() != list(range(CALIBRATION_START, len(data))):
        raise ValueError("Cache must cover every origin from index 14 to the last data row.")
    cache["date"] = pd.to_datetime(cache["date"], errors="raise")
    cache["context_end"] = pd.to_datetime(cache["context_end"], errors="raise")
    for row in cache.itertuples():
        t = int(row.origin_index)
        if row.date != data.date.iloc[t] or row.context_end != data.date.iloc[t - 1]:
            raise ValueError(f"Cache dates do not match the daily series at origin {t}.")
    values = cache[["Chronos2"] + QUANTILES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Base forecasts and quantiles must be finite and nonnegative.")
    return cache.set_index("origin_index")


def load_inputs(data_path, cache_path):
    data = read_daily(data_path)
    cache_path = Path(cache_path)
    with cache_path.with_suffix(".json").open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    if provenance.get("data_sha256") != sha256(data_path):
        raise ValueError("Data differ from the source of this cache; regenerate forecasts.")
    if provenance.get("cache_sha256") != sha256(cache_path):
        raise ValueError("Cache checksum mismatch; regenerate forecasts.")
    return data, validate_cache(data, pd.read_csv(cache_path)), provenance
