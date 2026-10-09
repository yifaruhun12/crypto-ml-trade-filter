# Package Validation

The extracted command-line workflow completed on October 8, 2026 using Python 3.12 and the dependency versions in `requirements.txt`.

- The bundled market-data SHA-256 checksum matched its manifest.
- Input validation passed for 401,787 OHLCV rows and 100 symbols at four-hour intervals.
- All five classifiers completed four expanding-window test folds each.
- Every fold used an 84-hour interval between the training cutoff and the test start.
- Portfolio comparison completed for the unfiltered baseline and all five model filters.
- Python source syntax and package integrity checks passed.

`verified_results/` contains outputs generated during this packaging check. `reference_results/` contains the previously archived outputs. Differences between them are not hidden or substituted: each set is retained separately for review. Successful execution does not establish economic validity or guarantee future performance.
