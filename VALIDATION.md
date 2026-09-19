# Release validation

Version 1.0.0 was validated on 2026-09-19 with Python 3.10.14 on macOS ARM64.

- The wheel installed in a new virtual environment with its declared dependencies.
- `pip check` reported no broken requirements.
- All six synthetic unit tests passed.
- The installed `rra-sales demo` command generated the synthetic inputs,
  predictions, metrics, and configuration record successfully.
- Chronos-2 inference regenerated 91 daily prefixes from the fixed checkpoint.
  All cached medians and quantiles matched the historical reference exactly.
- The installed RRA package reproduced 91 adapter predictions, including
  calibration, with maximum absolute difference 5.82e-11 monetary units.
- All 24 metric rows (base, RRA, and four ablations across pooled and three
  window summaries) matched the reference with rtol=1e-12 and atol=1e-8.

The historical pooled WAPE values were 43.62633929957006% for RRA and
45.93189460650352% for Chronos-2. Real-data validation used confidential local
records, which are excluded from this code distribution. The synthetic example
has its own inputs and metrics.

The optional Chronos inference path was run in the existing reference inference
environment with chronos-forecasting 2.3.2, PyTorch 2.12.1, and NumPy 2.2.6.
The fresh adapter environment used the exact versions in `requirements.txt`.
