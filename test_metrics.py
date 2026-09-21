"""Numerical and input-contract tests using synthetic forecasts only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from rra_sales.cli import score_file
from rra_sales.metrics import metrics, relative_improvements, score, window_summary
from rra_sales.statistics import block_interval, hac_test, paired_statistics


def forecasts(n=63):
    x = np.arange(n)
    return pd.DataFrame({"origin_index": x+42, "date": pd.date_range("2020-01-01", periods=n),
                         "actual": 100.+x, "Base": 90.+x+5*np.sin(x),
                         "Candidate": 95.+x+3*np.cos(x)})


class MetricTests(unittest.TestCase):
    def test_hand_computed_errors(self):
        result = metrics([100, 0, 200], [90, 10, 240])
        self.assertEqual(result["WAPE_percent"], 20.)
        self.assertEqual(result["MAE"], 20.)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(600))
        self.assertEqual(metrics([-10, 10], [0, 0])["WAPE_percent"], 100.)

    def test_undefined_wape_still_reports_absolute_errors(self):
        self.assertEqual(metrics([0, 0], [3, 4]),
                         {"WAPE_percent": None, "MAE": 3.5, "RMSE": np.sqrt(12.5)})

    def test_invalid_arrays_are_not_dropped_or_broadcast(self):
        for actual, predicted in [([], []), ([1, 2], [1]), ([[1]], [[1]]),
                                  ([1, np.nan], [1, 2]), ([1], [np.inf]), (["bad"], [1])]:
            with self.subTest(actual=actual, predicted=predicted), self.assertRaises(ValueError):
                metrics(actual, predicted)
        with self.assertRaises(ValueError):
            metrics(pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[1, 0]))

    def test_macro_and_pooled_rankings_can_differ(self):
        frame = pd.DataFrame({"origin_index": range(4), "actual": [10, 10, 100, 100],
                              "Base": [0, 0, 100, 100], "Candidate": [5, 5, 90, 90]})
        scores = score(frame, ["Base", "Candidate"], score_start=0, window_size=2)
        summary = window_summary(scores).set_index("model")
        self.assertAlmostEqual(summary.loc["Base", "WAPE_percent"], 100*20/220)
        self.assertEqual(summary.loc["Base", "WAPE_macro_percent"], 50.)
        self.assertEqual(summary.loc["Candidate", "WAPE_macro_percent"], 30.)
        self.assertAlmostEqual(summary.loc["Candidate", "WAPE_window_sample_std_pp"], np.sqrt(800))
        changes = relative_improvements(scores, "Base").set_index("period")
        self.assertAlmostEqual(changes.loc["ALL", "delta_WAPE_pp"], 100*10/220)
        self.assertAlmostEqual(changes.loc["ALL", "relative_WAPE_reduction_percent"], -50.)
        self.assertEqual(changes.loc["W1", "relative_WAPE_reduction_percent"], 50.)
        self.assertTrue(pd.isna(changes.loc["W2", "relative_WAPE_reduction_percent"]))

    def test_partial_and_zero_volume_windows_are_explicit(self):
        frame = pd.DataFrame({"origin_index": range(5), "actual": [10, 10, 0, 0, 10], "Base": [0]*5})
        scores = score(frame, ["Base"], score_start=0, window_size=2)
        self.assertEqual(scores.n.tolist(), [5, 2, 2, 1])
        row = window_summary(scores).iloc[0]
        self.assertEqual(row.WAPE_valid_windows, 2)
        self.assertTrue(pd.isna(row.WAPE_macro_percent))
        self.assertEqual(row.MAE_valid_windows, 3)
        single = window_summary(score(frame, ["Base"], score_start=0, window_size=5)).iloc[0]
        self.assertTrue(pd.isna(single.WAPE_window_sample_std_pp))

    def test_invalid_forecast_alignment_and_names(self):
        original = forecasts()
        bad_frames = [original.iloc[::-1], original.drop(index=2),
                      pd.concat([original.iloc[:1], original]), original.assign(Candidate=np.nan)]
        gap = original.copy()
        gap.loc[2, "date"] += pd.Timedelta(days=1)
        bad_frames.append(gap)
        for frame in bad_frames:
            with self.subTest(frame=frame.shape), self.assertRaises(ValueError):
                score(frame, ["Base", "Candidate"])
        for names in ["Base", [], ["Missing"], ["Base", "Base"], ["actual"]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                score(original, names)
        for start, size in [(-1, 21), (42, 0), (42, 1.5), (200, 21)]:
            with self.assertRaises(ValueError):
                score(original, ["Base"], start, size)


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_reduction_and_scale(self):
        for scale in [1., 1000.]:
            result = block_interval(np.full(63, 100.*scale), np.full(63, -20.*scale), n_resamples=300)
            self.assertEqual(result["delta_WAPE_pp"], -20.)
            self.assertEqual(result["CI_lower_pp"], -20.)
            self.assertEqual(result["CI_upper_pp"], -20.)
            self.assertFalse(result["CI_includes_zero"])

    def test_paired_orientation_and_reproducibility(self):
        frame = forecasts()
        result = paired_statistics(frame, ["Base", "Candidate"], "Base", n_resamples=500)
        repeat = paired_statistics(frame, ["Base", "Candidate"], "Base", n_resamples=500)
        reverse = paired_statistics(frame, ["Base", "Candidate"], "Candidate", n_resamples=500)
        pd.testing.assert_frame_equal(result, repeat)
        np.testing.assert_allclose(result.CI_lower_pp, -reverse.CI_upper_pp)
        np.testing.assert_allclose(result.CI_upper_pp, -reverse.CI_lower_pp)
        np.testing.assert_allclose(result.HAC_p_two_sided, reverse.HAC_p_two_sided)
        for row in result.itertuples():
            self.assertLess(row.delta_WAPE_pp, 0)
            self.assertEqual(row.bootstrap_status, "ok")
        scores = score(frame, ["Base", "Candidate"])
        delta = relative_improvements(scores, "Base").iloc[0].delta_WAPE_pp
        self.assertAlmostEqual(delta, result.delta_WAPE_pp.iloc[0])

    def test_identical_forecasts(self):
        frame = forecasts().assign(Candidate=lambda x: x.Base)
        result = paired_statistics(frame, ["Base", "Candidate"], "Base", n_resamples=100)
        self.assertTrue((result.CI_lower_pp == 0).all())
        self.assertTrue(result.CI_includes_zero.all())
        self.assertTrue((result.HAC_p_two_sided == 1).all())
        self.assertTrue((result.HAC_status == "identical_losses").all())

    def test_hac_hand_computed_bartlett_variance(self):
        # Centered differences [-1, 0, 1, -1, 0, 1]: gamma0=4/6, gamma1=-1/6.
        result = hac_test([0, 1, 2, 0, 1, 2], lags=1)
        self.assertAlmostEqual(result["HAC_standard_error"], np.sqrt(0.1))
        self.assertAlmostEqual(result["HAC_z"], np.sqrt(10))
        self.assertAlmostEqual(result["HAC_p_two_sided"], 2*norm.sf(np.sqrt(10)))
        self.assertEqual(hac_test([1]*20)["HAC_status"], "zero_long_run_variance")
        self.assertIsNone(hac_test([1]*20)["HAC_p_two_sided"])
        self.assertEqual(hac_test([1]*7)["HAC_status"], "insufficient_targets")

    def test_short_windows_and_zero_denominators(self):
        short = block_interval([10]*23, [-1]*23, block_length=3, n_resamples=100)
        self.assertEqual(short["bootstrap_status"], "insufficient_window_length")
        self.assertIsNone(short["CI_lower_pp"])
        zero = block_interval([0]*63, [1]*63, n_resamples=100)
        self.assertEqual(zero["bootstrap_status"], "zero_actual_denominator")
        mixed = block_interval([0, 1], [1, 1], window_size=2, block_length=1, n_resamples=1000)
        self.assertEqual(mixed["bootstrap_status"], "zero_bootstrap_denominator")
        self.assertLess(mixed["valid_resamples"], 1000)
        self.assertIsNone(mixed["CI_lower_pp"])

    def test_partial_window_bootstrap_keeps_target_count(self):
        result = block_interval([100]*31, [-10]*31, n_resamples=100)
        self.assertEqual(result["bootstrap_status"], "ok")
        self.assertEqual(result["CI_lower_pp"], -10.)
        self.assertEqual(result["CI_upper_pp"], -10.)

    def test_invalid_statistical_settings(self):
        for options in [{"block_length": 0}, {"n_resamples": 1}, {"confidence": 1.},
                        {"confidence": float("nan")}, {"seed": -1}, {"window_size": 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                block_interval([100]*63, [-1]*63, **options)
        with self.assertRaises(ValueError):
            paired_statistics(forecasts(), ["Base", "Candidate"], "Absent")
        with self.assertRaises(ValueError):
            paired_statistics(forecasts(), ["Base", "Candidate"], "Base", block_lengths=[])

    def test_score_csv_reports_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            forecasts().to_csv(source, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                score_file(source, ["Base", "Candidate"], "Base", root / "report", n_resamples=100)
            expected = {"predictions.csv", "metrics.csv", "summary.csv", "improvements.csv",
                        "paired_statistics.csv", "run.json"}
            self.assertEqual({p.name for p in (root / "report").iterdir()}, expected)
            metadata = json.loads((root / "report" / "run.json").read_text())
            self.assertEqual(metadata["evaluation"]["n_scored"], 63)
            self.assertEqual(metadata["evaluation"]["baseline"], "Base")
            self.assertEqual(len(metadata["predictions_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                score_file(source, ["Base", "Candidate"], "Base", root / "report")


if __name__ == "__main__":
    unittest.main()
