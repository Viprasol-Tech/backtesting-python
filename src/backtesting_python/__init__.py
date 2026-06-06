"""Backtesting Python — a vectorized backtesting engine for trading strategies.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from backtesting_python.engine import BacktestResult, SignalFn, run_backtest
from backtesting_python.export import (
    result_to_frame,
    summary_dict,
    write_equity_csv,
    write_summary_csv,
)
from backtesting_python.metrics import (
    annualized_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    equity_curve,
    exposure,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from backtesting_python.portfolio import PortfolioResult, run_portfolio_backtest
from backtesting_python.signals import moving_average_crossover
from backtesting_python.sizing import fixed_fraction, volatility_target
from backtesting_python.strategies import (
    STRATEGIES,
    bollinger_breakout,
    buy_and_hold,
    donchian_breakout,
    momentum,
    rsi_reversion,
)

__version__ = "0.2.0"

__all__ = [
    "STRATEGIES",
    "BacktestResult",
    "PortfolioResult",
    "SignalFn",
    "__version__",
    "annualized_return",
    "annualized_volatility",
    "bollinger_breakout",
    "buy_and_hold",
    "cagr",
    "calmar_ratio",
    "donchian_breakout",
    "equity_curve",
    "exposure",
    "fixed_fraction",
    "max_drawdown",
    "momentum",
    "moving_average_crossover",
    "result_to_frame",
    "rsi_reversion",
    "run_backtest",
    "run_portfolio_backtest",
    "sharpe_ratio",
    "sortino_ratio",
    "summary_dict",
    "total_return",
    "volatility_target",
    "write_equity_csv",
    "write_summary_csv",
]
