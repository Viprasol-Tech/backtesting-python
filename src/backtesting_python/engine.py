"""Vectorized backtesting engine.

The engine takes a price series and a signal function that maps prices to a
sequence of *target positions* in ``{0, 1}`` (flat or long). Positions are
shifted by one bar so that a signal computed at the close of bar ``t`` is acted
on at bar ``t + 1`` — this avoids look-ahead bias. Strategy returns are then
compared against a passive buy-and-hold benchmark.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from backtesting_python.metrics import (
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

__all__ = ["BacktestResult", "SignalFn", "run_backtest"]

#: A signal function maps a price series to target positions in ``{0, 1}``.
SignalFn = Callable[[Sequence[float]], Sequence[int]]


class BacktestResult(BaseModel):
    """Outcome of a single backtest run.

    Attributes:
        strategy_returns: Per-bar simple returns of the strategy.
        strategy_equity: Strategy equity curve (starts at ``starting_equity``).
        buy_hold_equity: Buy-and-hold equity curve over the same prices.
        strategy_total_return: Compounded total return of the strategy.
        buy_hold_total_return: Compounded total return of buy-and-hold.
        strategy_final_equity: Final value of the strategy equity curve.
        buy_hold_final_equity: Final value of the buy-and-hold equity curve.
        sharpe_ratio: Annualised Sharpe ratio of the strategy.
        max_drawdown: Maximum drawdown of the strategy equity curve.
        n_trades: Number of position changes (entries plus exits).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    strategy_returns: NDArray[np.float64]
    strategy_equity: NDArray[np.float64]
    buy_hold_equity: NDArray[np.float64]
    strategy_total_return: float
    buy_hold_total_return: float
    strategy_final_equity: float
    buy_hold_final_equity: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int


def run_backtest(
    prices: Sequence[float],
    signal_fn: SignalFn,
    starting_equity: float = 10_000.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a long/flat vectorized backtest.

    The ``signal_fn`` is called once with the full price series and must return
    a target position in ``{0, 1}`` for each price. Positions are lagged by one
    bar before being multiplied by the per-bar asset returns, so a signal seen
    at bar ``t`` takes effect on the return from ``t`` to ``t + 1``.

    Args:
        prices: The asset price series (length ``n >= 2``).
        signal_fn: Maps the price series to ``n`` target positions in ``{0, 1}``.
        starting_equity: Initial capital for both strategy and benchmark.
        periods_per_year: Periods per year used to annualise the Sharpe ratio.

    Returns:
        A :class:`BacktestResult` with returns, equity curves, and metrics.

    Raises:
        ValueError: If fewer than two prices are given, if the number of
            positions does not match the number of prices, or if any position
            is outside ``{0, 1}``.
    """
    price_arr = np.asarray(prices, dtype=np.float64).ravel()
    if price_arr.size < 2:
        raise ValueError("need at least two prices to run a backtest")

    positions = np.asarray(list(signal_fn(prices)), dtype=np.float64).ravel()
    if positions.size != price_arr.size:
        raise ValueError(
            f"signal_fn returned {positions.size} positions for {price_arr.size} prices"
        )
    if not np.all((positions == 0.0) | (positions == 1.0)):
        raise ValueError("positions must all be 0 or 1")

    asset_returns = price_arr[1:] / price_arr[:-1] - 1.0
    # Lag positions by one bar: act on bar t's signal at bar t+1.
    held = positions[:-1]
    strategy_returns = held * asset_returns

    strat_equity = equity_curve(strategy_returns, starting_equity)
    bh_equity = equity_curve(asset_returns, starting_equity)
    n_trades = int(np.count_nonzero(np.diff(positions) != 0.0))

    return BacktestResult(
        strategy_returns=strategy_returns,
        strategy_equity=strat_equity,
        buy_hold_equity=bh_equity,
        strategy_total_return=total_return(strategy_returns),
        buy_hold_total_return=total_return(asset_returns),
        strategy_final_equity=float(strat_equity[-1]),
        buy_hold_final_equity=float(bh_equity[-1]),
        sharpe_ratio=sharpe_ratio(strategy_returns, periods_per_year),
        max_drawdown=max_drawdown(strat_equity),
        n_trades=n_trades,
    )
