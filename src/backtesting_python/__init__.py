"""Backtesting Python — a vectorized backtesting engine for trading strategies.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from backtesting_python.engine import BacktestResult, SignalFn, run_backtest
from backtesting_python.metrics import (
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from backtesting_python.signals import moving_average_crossover

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "SignalFn",
    "__version__",
    "equity_curve",
    "max_drawdown",
    "moving_average_crossover",
    "run_backtest",
    "sharpe_ratio",
    "total_return",
]
