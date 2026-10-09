# Cryptocurrency Machine Learning Trade Filter

## Download the Complete Runnable Package

[Download the full project ZIP](https://github.com/yifaruhun12/crypto-ml-trade-filter/releases/download/v1.0.0/Morgan_Ruiming_Li_Crypto_ML_Backtest.zip) from [Releases](https://github.com/yifaruhun12/crypto-ml-trade-filter/releases/tag/v1.0.0). It contains the source code, historical data, and reference results. Extract the ZIP and follow the commands below from the extracted project folder.

This repository presents the source code, English documentation, and compact result tables for browsing. The large market-data CSV files are supplied in the release ZIP.

This research package extracts the machine learning component of `quant-backtests-257`. It compares five classifiers for filtering candidate trades from a four-hour cryptocurrency mean-reversion strategy. It includes the original backtest engine, the ML pipeline, the required historical OHLCV data, and archived model comparison results.

## What the Pipeline Does

1. Generate long and short candidate trades using lagged Stochastic and Williams %R signals.
2. Construct 21 features from returns, volatility, volume, technical signals, market breadth and dispersion, and BTC market context.
3. Label candidate trades by their simulated net return after fees, slippage, and short carry costs, using an 84-hour maximum label horizon.
4. Compare logistic regression, Random Forest, histogram gradient boosting, XGBoost, and LightGBM.
5. Use four expanding-window out-of-sample test folds. Training observations end 84 hours before each test fold begins.
6. Compare the unfiltered strategy with each ML filter using a predicted win-probability threshold of 0.55. Report portfolio return, Sharpe ratio, maximum drawdown, trade count, win rate, ROC AUC, average precision, and Brier score.

## Installation and Execution

The package was checked with Python 3.12. Install the dependencies from the extracted project folder:

```bash
python -m pip install -r requirements.txt
```

Check the included data and library imports:

```bash
python run_backtest.py --validate-only
```

Run the complete model comparison:

```bash
python run_backtest.py
```

To use a different dataset or output directory:

```bash
python run_backtest.py --data path/to/market_data.csv --output my_results
```

Training and portfolio simulation can take several minutes. The programs operate on local CSV files and do not connect to an exchange or submit orders.

## Package Contents

```text
README.md                          English setup and methodology guide
run_backtest.py                    Portable command-line entry point
requirements.txt                   Python dependencies
SOURCE.md                          Source repository and packaging changes
src/backtest_engine.py             Original four-hour backtest engine
src/ml_trade_filter.py             Original multi-model ML filtering pipeline
data/top100_4h_730d.csv             Historical market data used by the engine
data/DATA_MANIFEST.json            Data schema, coverage, and SHA-256 checksum
reference_results/                 Archived research outputs
verified_results/                  Outputs from the packaging validation run
VALIDATION.md                      Validation performed for this package
outputs/                           New outputs, created when the pipeline runs
```

Both the archived outputs and the newly verified outputs are available immediately for review. A new run writes to `outputs/` and leaves `reference_results/` intact. The CSV files contain:

- `ml_model_comparison.csv`: aggregate portfolio and classifier metrics for the baseline and five models
- `ml_fold_metrics.csv`: training/test boundaries, sample counts, and classifier metrics for each model and fold
- `ml_oos_predictions.csv`: out-of-sample candidate-trade scores and labels
- `ml_trade_samples.csv`: candidate-trade labels and feature values

## Data and Backtest Assumptions

The data covers 100 USDT-quoted cryptocurrency symbols at four-hour intervals. Required columns are `datetime_utc`, `symbol`, `open`, `high`, `low`, `close`, and `volume`. Timestamps are interpreted as UTC. The loader checks missing values, duplicate symbol/timestamp keys, OHLCV validity, interval continuity, and aligned ending timestamps.

The engine uses 300,000 units of initial equity, a 7.5% equity-based notional size per new position, a one-way fee of 0.08%, one-way slippage of 0.05%, and 10% annualized short carry cost. Position sizing refers to notional exposure, rather than a leveraged margin allocation. The portfolio holding limit is 84 hours. This package contains the four-hour research version; it is separate from later 15-minute live-aligned strategy configurations.

## Reading the Metrics

`total_return`, `max_drawdown`, and `win_rate` are stored as decimal fractions. For example, `0.1459` represents 14.59%. `total_return` is the cumulative return over the evaluated period, not an annualized return. Maximum drawdown is reported as a positive loss magnitude. The Sharpe ratio is a ratio, not a percentage. The classifier metrics in the aggregate comparison are averages across test folds.

Archived results can differ from a new run if library versions, data, or parameters differ. These are research backtests, not live trading results. The market universe was selected using volume over the data collection window, so the experiment does not establish a point-in-time investment universe or remove universe-selection bias.
