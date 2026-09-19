"""Bounded, recency-weighted log-residual regression around a fixed median."""
from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import DATA_COLUMNS, QUANTILES, CALIBRATION_START, validate_daily


@dataclass(frozen=True)
class Configuration:
    window: int | None = 42
    half_life: float | None = 21.0
    operational_features: bool = True
    quantile_features: bool = True
    seed: int = 20260827

    def __post_init__(self):
        if self.window is not None and (not isinstance(self.window, int) or self.window < 10):
            raise ValueError("window must be at least 10, or None for expanding history.")
        if self.half_life is not None and (not math.isfinite(self.half_life) or self.half_life <= 0):
            raise ValueError("half_life must be positive, or None for uniform weights.")


class RRAForecaster:
    """Call predict with observed history, then observe after the target completes."""

    def __init__(self, config=Configuration()):
        self.config = config
        self.history = []
        self.pending = None

    def _features(self, prefix, date, c, foundation):
        y = prefix.sales.to_numpy(float)
        scale = max(1.0, float(np.mean(y[-7:])))
        errors = (np.array([(h["actual"] - h["forecast"]) / h["scale"] for h in self.history])
                  if self.history else np.zeros(1))
        basic = [c / scale, y[-1] / scale, y[-2] / scale, np.mean(y[-3:]) / scale,
                 np.mean(y[-14:]) / scale, np.std(y[-7:]) / scale,
                 np.log(scale / max(1.0, np.mean(y[-28:]))),
                 np.clip(errors[-1], -3, 3), np.clip(np.median(errors[-7:]), -3, 3)]
        features = list(basic)
        if self.config.operational_features:
            for col in DATA_COLUMNS[2:]:
                values = prefix[col].to_numpy(float)
                norm = max(1.0, np.mean(values[-7:]))
                features.extend([values[-1] / norm, np.mean(values[-3:]) / norm])
        features.extend([np.sin(2 * np.pi * date.dayofweek / 7),
                         np.cos(2 * np.pi * date.dayofweek / 7)])
        if self.config.quantile_features:
            q = np.asarray([foundation[name] for name in QUANTILES], dtype=float)
            if not np.isfinite(q).all() or (q < 0).any():
                raise ValueError("Quantiles must be finite and nonnegative.")
            # Preserve the native point forecast when ordering the two tails.
            q = np.r_[np.sort(np.minimum(q[:4], c)), c, np.sort(np.maximum(q[5:], c))]
            features.extend((q[[0, 4, 8]] - c) / scale)
        return np.clip(features, -10, 10), scale

    def predict(self, prefix, target_date, foundation):
        if self.pending is not None:
            raise RuntimeError("Observe the outstanding target before predicting again.")
        prefix = validate_daily(prefix)
        target_date = pd.Timestamp(target_date)
        if len(prefix) < CALIBRATION_START:
            raise ValueError("At least 14 observed daily records are required.")
        if target_date != prefix.date.iloc[-1] + pd.Timedelta(days=1):
            raise ValueError("The target must be the day immediately after the observed prefix.")
        if self.history:
            previous = self.history[-1]
            if previous["date"] != prefix.date.iloc[-1] or previous["actual"] != float(prefix.sales.iloc[-1]):
                raise ValueError("History must extend the last observed target without changing its sales.")
        c = float(foundation["Chronos2"])
        if not np.isfinite(c) or c < 0:
            raise ValueError("The base median must be finite and nonnegative.")
        x, scale = self._features(prefix, target_date, c, foundation)
        h = self.history if self.config.window is None else self.history[-self.config.window:]
        raw = 0.0
        if len(h) >= 10:
            X = np.array([r["features"] for r in h])
            actual = np.array([r["actual"] for r in h])
            base = np.array([r["forecast"] for r in h])
            response = np.log1p(actual) - np.log1p(base)
            weights = (np.exp2(-np.arange(len(h) - 1, -1, -1) / self.config.half_life)
                       if self.config.half_life is not None else np.ones(len(h)))
            weights /= weights.mean()
            regressor = LGBMRegressor(
                objective="regression", n_estimators=80, num_leaves=4, max_depth=2,
                learning_rate=0.03, min_child_samples=5, reg_lambda=10.0,
                colsample_bytree=1.0, random_state=self.config.seed,
                n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
            )
            regressor.fit(X, response, sample_weight=weights)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                raw = float(regressor.predict(x[None, :])[0])
        prediction = max(0.0, np.expm1(np.log1p(c) + np.clip(raw, -np.log(3), np.log(3))))
        self.pending = {"features": x, "forecast": c, "scale": scale, "date": target_date}
        return float(prediction)

    def observe(self, actual):
        if self.pending is None:
            raise RuntimeError("Issue a prediction before observing its target.")
        actual = float(actual)
        if not np.isfinite(actual) or actual < 0:
            raise ValueError("Observed sales must be finite and nonnegative.")
        self.history.append({**self.pending, "actual": actual})
        self.pending = None

    def configuration(self):
        return asdict(self.config)
