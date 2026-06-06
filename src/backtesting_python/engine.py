"""Vectorized backtesting engine.

The engine takes a price series and a signal function that maps prices to a
sequence of *target positions* in ``[0, 1]`` (a flat-to-long fractional
weight; binary ``{0, 1}`` signals are the common special case). Positions are
shifted by one bar so that a signal computed at the close of bar ``t`` is acted
on at bar ``t + 1`` — this avoids look-ahead bias. Per-bar transaction costs
can be charged on turnover, and strategy returns are compared against a passive
buy-and-hold benchmark.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from backtesting_python.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    equity_curve,
    exposure,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

__all__ = ["BacktestResult", "Positions", "SignalFn", "run_backtest"]

#: Target positions: any sequence of floats or a NumPy float array.
Positions = Sequence[float] | NDArray[np.float64]

#: A signal function maps a price series to target positions in ``[0, 1]``.
SignalFn = Callable[[Sequence[float]], Positions]


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
        sortino_ratio: Annualised Sortino ratio of the strategy.
        calmar_ratio: Annualised return divided by max drawdown.
        cagr: Compound annual growth rate of the strategy.
        annual_volatility: Annualised volatility of strategy returns.
        max_drawdown: Maximum drawdown of the strategy equity curve.
        exposure: Fraction of bars with a nonzero position (time in market).
        total_costs: Total transaction costs charged over the run.
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
    sortino_ratio: float
    calmar_ratio: float
    cagr: float
    annual_volatility: float
    max_drawdown: float
    exposure: float
    total_costs: float
    n_trades: int


def run_backtest(
    prices: Sequence[float],
    signal_fn: SignalFn,
    starting_equity: float = 10_000.0,
    periods_per_year: int = 252,
    cost_per_turnover: float = 0.0,
) -> BacktestResult:
    """Run a long/flat (optionally fractional) vectorized backtest.

    The ``signal_fn`` is called once with the full price series and must return
    a target position in ``[0, 1]`` for each price (binary ``{0, 1}`` is the
    usual case; fractional weights from a sizer are also allowed). Positions are
    lagged by one bar before being multiplied by the per-bar asset returns, so a
    signal seen at bar ``t`` takes effect on the return from ``t`` to ``t + 1``.

    Transaction costs, if any, are charged on the absolute change in position
    (turnover) at each bar: ``cost = cost_per_turnover * |Δposition|``.

    Args:
        prices: The asset price series (length ``n >= 2``).
        signal_fn: Maps the price series to ``n`` target positions in ``[0, 1]``.
        starting_equity: Initial capital for both strategy and benchmark.
        periods_per_year: Periods per year used to annualise ratio metrics.
        cost_per_turnover: Proportional cost charged per unit of turnover (e.g.
            ``0.001`` for 10 bps round-trip on a full entry/exit).

    Returns:
        A :class:`BacktestResult` with returns, equity curves, and metrics.

    Raises:
        ValueError: If fewer than two prices are given, if the number of
            positions does not match the number of prices, if any position is
            outside ``[0, 1]``, or if ``cost_per_turnover`` is negative.
    """
    if cost_per_turnover < 0.0:
        raise ValueError("cost_per_turnover must be non-negative")

    price_arr = np.asarray(prices, dtype=np.float64).ravel()
    if price_arr.size < 2:
        raise ValueError("need at least two prices to run a backtest")

    positions = np.asarray(list(signal_fn(prices)), dtype=np.float64).ravel()
    if positions.size != price_arr.size:
        raise ValueError(
            f"signal_fn returned {positions.size} positions for {price_arr.size} prices"
        )
    if not np.all((positions >= 0.0) & (positions <= 1.0)):
        raise ValueError("positions must all be within [0, 1]")

    asset_returns = price_arr[1:] / price_arr[:-1] - 1.0
    # Lag positions by one bar: act on bar t's signal at bar t+1.
    held = positions[:-1]
    gross_returns = held * asset_returns

    # Turnover at each traded bar = |position[t] - position[t-1]|.
    turnover = np.abs(np.diff(positions))
    costs = cost_per_turnover * turnover  # length n - 1, aligned to held bars
    strategy_returns = gross_returns - costs
    total_costs = float(np.sum(costs)) * float(starting_equity)

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
        sortino_ratio=sortino_ratio(strategy_returns, periods_per_year),
        calmar_ratio=calmar_ratio(strategy_returns, periods_per_year),
        cagr=annualized_return(strategy_returns, periods_per_year),
        annual_volatility=annualized_volatility(strategy_returns, periods_per_year),
        max_drawdown=max_drawdown(strat_equity),
        exposure=exposure(positions),
        total_costs=total_costs,
        n_trades=n_trades,
    )
