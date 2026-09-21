"""Paired, time-dependent uncertainty for fixed sets of daily forecasts."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .metrics import paired_arrays, positive_integer, scored_frame


def hac_test(loss_difference, lags=7):
    """Two-sided normal test of mean loss difference with Bartlett HAC and HC1."""
    d, _ = paired_arrays(loss_difference, loss_difference)
    positive_integer(lags, "lags", minimum=0)
    result = {"HAC_lags": lags, "mean_absolute_loss_difference": float(d.mean()),
              "HAC_standard_error": None, "HAC_z": None, "HAC_p_two_sided": None}
    if len(d) <= lags + 1:
        return {**result, "HAC_status": "insufficient_targets"}
    centered = d - d.mean()
    longvar = np.dot(centered, centered) / len(d)
    for lag in range(1, lags + 1):
        longvar += 2 * (1 - lag / (lags + 1)) * np.dot(centered[lag:], centered[:-lag]) / len(d)
    se = float(np.sqrt(max(0., longvar / (len(d) - 1))))
    result["HAC_standard_error"] = se
    if np.all(d == 0):
        return {**result, "HAC_z": 0., "HAC_p_two_sided": 1., "HAC_status": "identical_losses"}
    if se == 0:
        return {**result, "HAC_status": "zero_long_run_variance"}
    z = float(d.mean() / se)
    return {**result, "HAC_z": z, "HAC_p_two_sided": float(2 * norm.sf(abs(z))), "HAC_status": "ok"}


def block_interval(actual, difference, window_size=21, block_length=7,
                   n_resamples=10000, confidence=0.95, seed=20260919):
    """Percentile CI for delta WAPE; noncircular moving blocks within windows."""
    y, d = paired_arrays(actual, difference)
    positive_integer(window_size, "window_size")
    positive_integer(block_length, "block_length")
    positive_integer(n_resamples, "n_resamples", minimum=2)
    positive_integer(seed, "seed", minimum=0)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    effective_seed = seed + block_length
    denominator = np.abs(y).sum()
    result = {"delta_WAPE_pp": float(100*d.sum()/denominator) if denominator > 0 else None,
              "block_length": block_length, "window_size": window_size,
              "n_resamples": n_resamples, "confidence": confidence, "bootstrap_seed": effective_seed,
              "CI_lower_pp": None, "CI_upper_pp": None, "CI_includes_zero": None,
              "valid_resamples": 0}
    if denominator == 0:
        return {**result, "bootstrap_status": "zero_actual_denominator"}
    lengths = [min(window_size, len(y)-s) for s in range(0, len(y), window_size)]
    if min(lengths) < block_length:
        return {**result, "bootstrap_status": "insufficient_window_length"}
    rng = np.random.default_rng(effective_seed)
    offsets = np.cumsum([0] + lengths[:-1])
    if len(set(lengths)) == 1:
        # Keep the manuscript's replicate/window/block draw order for equal windows.
        size = lengths[0]
        starts = rng.integers(0, size-block_length+1,
                              size=(n_resamples, len(lengths), int(np.ceil(size/block_length))))
        ids = (starts[..., None]+np.arange(block_length)).reshape(n_resamples, len(lengths), -1)[:, :, :size]
        ids = (ids+offsets[None, :, None]).reshape(n_resamples, len(y))
    else:
        pieces = []
        for size, offset in zip(lengths, offsets):
            starts = rng.integers(0, size-block_length+1,
                                  size=(n_resamples, int(np.ceil(size/block_length))))
            pieces.append((starts[..., None]+np.arange(block_length)).reshape(n_resamples, -1)[:, :size]+offset)
        ids = np.concatenate(pieces, axis=1)
    totals = np.abs(y[ids]).sum(axis=1)
    valid = totals > 0
    result["valid_resamples"] = int(valid.sum())
    if not valid.all():
        return {**result, "bootstrap_status": "zero_bootstrap_denominator"}
    samples = 100*d[ids].sum(axis=1)/totals
    alpha = (1-confidence)/2
    lo, hi = np.quantile(samples, [alpha, 1-alpha])
    return {**result, "CI_lower_pp": float(lo), "CI_upper_pp": float(hi),
            "CI_includes_zero": bool(lo <= 0 <= hi), "bootstrap_status": "ok"}


def paired_statistics(predictions, names, baseline, score_start=42, window_size=21,
                      block_lengths=(3, 7), n_resamples=10000, confidence=0.95,
                      seed=20260919, hac_lags=7):
    """Compare every candidate with the same baseline on exactly the same dates."""
    names = list(names) if not isinstance(names, str) else names
    frame = scored_frame(predictions, names, score_start, window_size)
    if baseline not in names or len(names) < 2:
        raise ValueError("Paired statistics require a baseline and at least one candidate.")
    block_lengths = tuple(block_lengths)
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("Provide at least one unique block length.")
    actual = frame.actual.to_numpy(dtype=float)
    base_loss = np.abs(frame[baseline].to_numpy(dtype=float)-actual)
    rows = []
    for name in names:
        if name == baseline:
            continue
        difference = np.abs(frame[name].to_numpy(dtype=float)-actual)-base_loss
        hac = hac_test(difference, hac_lags)
        for block in block_lengths:
            interval = block_interval(actual, difference, window_size, block, n_resamples, confidence, seed)
            rows.append({"model": name, "baseline": baseline, "n": len(frame), **interval, **hac})
    return pd.DataFrame(rows)
