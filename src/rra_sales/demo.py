"""Synthetic daily inputs and a trailing-mean forecast for installation checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CALIBRATION_START, sha256


def make_inputs(days=105):
    rng = np.random.default_rng(20260919)
    t = np.arange(days)
    orders = np.maximum(1, np.round(90 + 0.7 * t + 18 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 8, days)))
    aov = 25 + 2 * np.cos(t / 11) + rng.uniform(0, 2, days)
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=days),
                         "sales": orders * aov, "order_count": orders,
                         "quantity": orders * 2, "active_products": 12 + t % 4,
                         "avg_order_value": aov})
    rows = []
    for i in range(CALIBRATION_START, days):
        base = float(data.sales.iloc[i - 7:i].mean())
        row = {"origin_index": i, "date": data.date.iloc[i],
               "context_end": data.date.iloc[i - 1], "Chronos2": base}
        row.update({f"q{q}": base * (0.6 + q / 125) for q in range(10, 100, 10)})
        rows.append(row)
    return data, pd.DataFrame(rows)


def write_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    data, cache = make_inputs()
    data_path, cache_path = directory / "synthetic_daily.csv", directory / "synthetic_base.csv"
    data.to_csv(data_path, index=False)
    cache.to_csv(cache_path, index=False)
    metadata = {"base_model": "synthetic_trailing_mean", "synthetic": True,
                "description": "Installation example; base values are not Chronos-2 outputs.",
                "data_sha256": sha256(data_path), "cache_sha256": sha256(cache_path)}
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return data_path, cache_path
