# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [0.2.0] - 2025

### Added
- **New risk metrics**: `sortino_ratio`, `calmar_ratio`, `annualized_return`
  (`cagr` alias), `annualized_volatility`, and `exposure` (time in market).
- **Position sizing** module (`sizing`): `fixed_fraction` and an inverse-vol
  `volatility_target` sizer that scales exposure to a target annualised vol.
- **Bundled strategies** module (`strategies`): `buy_and_hold`, `momentum`,
  `rsi_reversion`, `bollinger_breakout`, and `donchian_breakout`, plus a
  `STRATEGIES` registry keyed by CLI name.
- **Portfolio backtesting** (`portfolio.run_portfolio_backtest`): run one signal
  across many assets and blend them into a single weighted equity curve, with a
  `PortfolioResult` reporting blended and per-asset metrics.
- **Transaction costs**: `run_backtest` now accepts `cost_per_turnover` and
  reports `total_costs`.
- **Fractional positions**: the engine now accepts target weights in `[0, 1]`
  (not just `{0, 1}`), enabling sized strategies.
- **Export helpers** (`export`): `result_to_frame`, `write_equity_csv`,
  `summary_dict`, and `write_summary_csv` for CSV/pandas output.
- **CLI subcommands**: `run` (named strategy + sizing + costs + CSV export),
  `portfolio` (multi-asset), and `strategies` (list the registry), alongside the
  existing `demo` and `version`.

### Changed
- `BacktestResult` now also reports `sortino_ratio`, `calmar_ratio`, `cagr`,
  `annual_volatility`, `exposure`, and `total_costs`.
- Added `pandas` as a runtime dependency for the export helpers.

## [0.1.0] - 2025

### Added
- Initial release of backtesting-python: Vectorized backtesting engine for trading strategies in Python.
