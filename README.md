# Rolling Residual Adaptation (RRA)

Implementation of **Rolling Residual Adaptation of Pretrained Forecasts for
Short-History Sales Prediction**, by Jing Guan.

RRA uses a fixed Chronos-2 median forecast and a shallow LightGBM adapter trained
on completed log-domain forecast errors. This release includes the method,
Chronos-2 inference, sequential evaluation, four matched ablations, and a synthetic
installation example. Store records and pretrained weights are supplied separately.

## Installation

Use Python 3.10, 3.11, or 3.12. Python 3.10 is the reference environment.

```bash
python -m venv .venv
```

Activate on macOS/Linux with `source .venv/bin/activate`, or in Windows PowerShell
with `.venv\Scripts\Activate.ps1`. Then install from this directory:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

The upload ZIP also contains an installable wheel:

```bash
python -m pip install dist/rra_sales-1.0.0-py3-none-any.whl
```

Dependencies are downloaded by pip. This is a source/wheel distribution, not an
offline bundle of third-party libraries. On macOS, install the OpenMP runtime
with `brew install libomp` if LightGBM reports a missing `libomp` library.

## Installation example

```bash
rra-sales demo --output demo_run
python -m unittest discover -s tests -v
```

The demo generates 105 synthetic days and a trailing-mean base forecast. It runs
the actual RRA adapter and four ablations. Its metrics demonstrate execution
only; the base predictor is not Chronos-2 and these are not manuscript results.
Output columns call this predictor `Synthetic_base`.

## Input data

Provide one UTF-8 CSV with the following header:

```text
date,sales,order_count,quantity,active_products,avg_order_value
```

- `date`: consecutive, unique daily dates in increasing order (`YYYY-MM-DD`).
- `sales`: nonnegative daily sales in consistent monetary units.
- `order_count`, `quantity`, `active_products`: completed daily operational counts.
- `avg_order_value`: completed daily average order value in the same currency.

All measurements must be finite and nonnegative. Missing days are rejected rather
than imputed. Each forecast uses the rows strictly before its target date.
Operational measurements for the target day are used only after that day finishes.

## Chronos-2 and full evaluation

Install the optional inference dependencies:

```bash
python -m pip install ".[chronos]"
hf download amazon/chronos-2 --revision 29ec3766d36d6f73f0696f85560a422f50e8498c --local-dir checkpoints/chronos-2
rra-sales infer --data daily_sales.csv --checkpoint checkpoints/chronos-2 --output chronos_quantiles.csv
rra-sales evaluate --data daily_sales.csv --cache chronos_quantiles.csv --output evaluation --ablations
```

Inference uses CPU float32, four threads, one-step predictions, the complete
observed prefix up to a 2,048-step context limit, and fixed pretrained parameters.
The expected weights and configuration checksums are embedded in `chronos.py`.
Download the pinned checkpoint once; inference then uses local files.

The cache contains `origin_index,date,context_end,Chronos2,q10,...,q90`.
Here `Chronos2` is the native point prediction. The generated JSON sidecar records
the checkpoint and the SHA256 hashes of the data and cache. Evaluation requires
that sidecar and rejects mismatched hashes or dates. Regenerate the cache when
the input file changes. Hashes and dates identify the input artifact; prefix-only
generation is implemented in the inference loop.

Calibration starts at zero-based index 14. Scoring starts at index 42 by default.
All subsequent rows are scored; consecutive 21-day windows are also summarized.
For 105 input days, these are the manuscript's 63 targets and three windows.
Other lengths are supported, with a shorter final window when necessary:

```bash
rra-sales evaluate --data daily_sales.csv --cache chronos_quantiles.csv --output evaluation_long --score-start 42 --window-size 21
```

The output directory contains:

- `predictions.csv`: calibration and scoring predictions, actuals, and origin dates.
- `metrics.csv`: pooled and window WAPE (%), MAE, RMSE, and target counts.
- `run.json`: configurations, dependency versions, and input/cache provenance.

WAPE is `100 * sum(abs(actual - prediction)) / sum(abs(actual))`. WAPE is blank
when all actual sales in a scored segment are zero; MAE and RMSE are still reported.
The command requires a new output directory to preserve earlier results.

## Method and Python API

At each forecast origin, the adapter fits the latest at most 42 completed errors:

```text
residual_j = log(1 + sales_j) - log(1 + base_j)
weight_j = 2 ** (-age_j / 21), normalized to mean 1
correction = clip(adapter(features_t), -log(3), log(3))
forecast_t = max(0, exp(log(1 + base_t) + correction) - 1)
```

The adapter uses L2 regression, 80 trees, four leaves per tree, maximum depth two,
learning rate 0.03, minimum child size five, and leaf L2 regularization 10.
It uses one training thread and no random subsampling. Before ten completed
forecast records exist, the output equals the base median.

The 22-entry vector includes nine sales/error features, eight operational
features, two weekday encodings, and three quantile deviations. The middle
quantile deviation is zero. Lower and upper tails are ordered around the native
median, and features are clipped to [-10, 10]. See `model.py` for the exact order.

```python
import pandas as pd
from rra_sales import Configuration, RRAForecaster

daily = pd.read_csv("daily_sales.csv", parse_dates=["date"])
cache = pd.read_csv("chronos_quantiles.csv").set_index("origin_index")
model = RRAForecaster(Configuration())
for t in range(14, len(daily)):
    forecast = model.predict(daily.iloc[:t], daily.date.iloc[t], cache.loc[t])
    # In deployment, wait until the target day's sales are observed.
    model.observe(daily.sales.iloc[t])
```

Process days consecutively from calibration start to initialize the error archive.
Call `observe` once after each issued prediction. Calling `predict` again before
that update is rejected. The current release keeps the adapter history in memory.

The `--ablations` flag removes operational features, removes quantile features,
sets uniform sample weights, and uses an expanding archive. Each changes only
the named component. The pooled scores on the confidential historical dataset
were 43.626339% WAPE for RRA and 45.931895% for Chronos-2. The RRA configuration
was selected using that historical period; the seven-day-block paired 95%
interval was [-6.323195, 2.660013] percentage points. Reproducing those scores
requires the original store data and pinned checkpoint.

## Files and attribution

`src/rra_sales/model.py` contains the method; `evaluate.py` provides chronological
replay; `chronos.py` generates the base forecasts; `data.py` validates inputs;
`cli.py` exposes the commands. `tests/` checks delayed updates, future-input
invariance, feature bounds, cache binding, and input validation using synthetic data.

See `THIRD_PARTY.md` for the underlying methods and `CITATION.cff` for the software
citation. Chronos-2 and LightGBM retain their original attribution and licenses.

Contact: Jing Guan, Hangzhou Yuanxiang Wansheng Technology Co., Ltd.,
guanjing@biumdigits.com.
