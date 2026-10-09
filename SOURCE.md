# Source and Packaging Notes

Source repository: https://github.com/yifaruhun12/quant-backtests-257

The original files were reviewed on the `main` branch at commit `5efaf5c008633f6740816767bdc189b6f1675805`:

- `cyp入门/机器学习过滤回测.py`, packaged as `src/ml_trade_filter.py`
- `cyp入门/回测震荡.py`, packaged as `src/backtest_engine.py`

The ML file was checked against the version displayed in the repository. The engine was downloaded from the private repository. The market data and archived ML outputs were copied from the matching local research materials.

Packaging changes are limited to portable English filenames, removal of the local `.ml_packages` lookup, and the addition of a command-line launcher that separates data, archived results, and generated outputs. The model definitions, feature calculations, labels, time splits, trading rules, and cost assumptions were retained.

Original ML file SHA-256: `eb3da864857f629099782a5fed1d90939956b9554604a4c8132e260b132441cb`

Packaged engine SHA-256: `a564c28921383052852cf16f00dcc2c539381c08612cf6a9af9c2f2deb0904e1`

The bundle contains market research materials only. It excludes account statements, personal documents, exchange credentials, live configuration, live order logs, installed libraries, and serialized live models.
