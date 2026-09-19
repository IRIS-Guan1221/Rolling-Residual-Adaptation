"""Synthetic tests for chronology, delayed updates, and cache validation."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from rra_sales import Configuration, RRAForecaster
from rra_sales.data import DATA_COLUMNS, load_inputs, validate_cache, validate_daily
from rra_sales.demo import make_inputs, write_demo
from rra_sales.evaluate import configurations, metrics, replay


class RRATests(unittest.TestCase):
    def setUp(self):
        self.data, raw_cache = make_inputs(days=64)
        self.cache = validate_cache(self.data, raw_cache)

    def test_delayed_update_and_feature_shape(self):
        model = RRAForecaster()
        t = 14
        forecast = model.predict(self.data.iloc[:t], self.data.date.iloc[t], self.cache.loc[t])
        self.assertAlmostEqual(forecast, self.cache.at[t, "Chronos2"])
        self.assertEqual(len(model.pending["features"]), 22)
        self.assertEqual(model.pending["features"][-2], 0.0)
        self.assertTrue((np.abs(model.pending["features"]) <= 10).all())
        with self.assertRaises(RuntimeError):
            model.predict(self.data.iloc[:t], self.data.date.iloc[t], self.cache.loc[t])
        with self.assertRaises(ValueError):
            model.observe(float("nan"))
        model.observe(self.data.sales.iloc[t])
        with self.assertRaises(RuntimeError):
            model.observe(self.data.sales.iloc[t])

    def test_future_measurements_do_not_change_issued_forecasts(self):
        baseline = replay(self.data, self.cache, stop=44)
        changed = self.data.copy()
        changed.loc[42:, DATA_COLUMNS[1:]] *= 17
        candidate = replay(changed, self.cache, stop=43)
        np.testing.assert_array_equal(candidate.RRA, baseline.RRA.iloc[:len(candidate)])

    def test_future_cache_does_not_change_earlier_forecasts(self):
        baseline = replay(self.data, self.cache, stop=43)
        changed = self.cache.copy()
        names = ["Chronos2"] + [f"q{i}" for i in range(10, 100, 10)]
        changed.loc[43:, names] *= 17
        candidate = replay(self.data, changed, stop=43)
        np.testing.assert_array_equal(candidate.RRA, baseline.RRA)

    def test_seed_repetition_and_bounds(self):
        frame = replay(self.data, self.cache, {
            "RRA": Configuration(), "Repeat": replace(Configuration(), seed=20260831),
        })
        np.testing.assert_array_equal(frame.RRA, frame.Repeat)
        ratio = (1 + frame.RRA) / (1 + frame.Chronos2)
        self.assertTrue((ratio >= 1 / 3 - 1e-12).all())
        self.assertTrue((ratio <= 3 + 1e-12).all())

    def test_invalid_dates_and_cache_binding(self):
        with self.assertRaises(ValueError):
            validate_daily(self.data.drop(index=3))
        bad_cache = self.cache.reset_index()
        bad_cache.loc[0, "context_end"] = bad_cache.loc[0, "date"]
        with self.assertRaises(ValueError):
            validate_cache(self.data, bad_cache)
        with tempfile.TemporaryDirectory() as directory:
            data_path, cache_path = write_demo(Path(directory) / "demo")
            load_inputs(data_path, cache_path)
            data_path.write_bytes(data_path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                load_inputs(data_path, cache_path)

    def test_zero_sales_and_ablation_shapes(self):
        self.assertIsNone(metrics([0, 0], [1, 2])["WAPE_percent"])
        for name, config in configurations(True).items():
            model = RRAForecaster(config)
            model.predict(self.data.iloc[:14], self.data.date.iloc[14], self.cache.loc[14])
            expected = 14 if name == "Without_operations" else 19 if name == "Without_quantile_features" else 22
            self.assertEqual(len(model.pending["features"]), expected)


if __name__ == "__main__":
    unittest.main()
