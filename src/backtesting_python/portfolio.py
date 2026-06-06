"""Multi-asset (portfolio) backtesting.

Runs the same signal logic across several assets at once and combines them into
a single portfolio equity curve. Capital is split across assets by a weight
vector (equal-weight by default), each asset is rebalanced to its target weight
each bar, and the blended per-bar return drives the portfolio metrics.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from backtesting_python.engine import SignalFn
from backtesting_python.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

__all__ = ["PortfolioResult", "run_portfolio_backtest"]


class PortfolioResult(BaseModel):
    """Outcome of a multi-asset portfolio backtest.

    Attributes:
        symbols: Asset symbols in column order.
        weights: Capital weight assigned to each asset (sums to 1).
        portfolio_returns: Per-bar blended portfolio returns.
        portfolio_equity: Portfolio equity curve from ``starting_equity``.
        total_return: Compounded total return of the portfolio.
        final_equity: Final value of the portfolio equity curve.
        cagr: Compound annual growth rate of the portfolio.
        sharpe_ratio: Annualised Sharpe ratio of the portfolio.
        sortino_ratio: Annualised Sortino ratio of the portfolio.
        calmar_ratio: Annualised return over max drawdown.
        annual_volatility: Annualised portfolio volatility.
        max_drawdown: Maximum drawdown of the portfolio equity curve.
        per_asset_total_return: Each asset's standalone strategy total return.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    symbols: list[str]
    weights: NDArray[np.float64]
    portfolio_returns: NDArray[np.float64]
    portfolio_equity: NDArray[np.float64]
    total_return: float
    final_equity: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annual_volatility: float
    max_drawdown: float
    per_asset_total_return: dict[str, float]


def run_portfolio_backtest(
    price_data: Mapping[str, Sequence[float]],
    signal_fn: SignalFn,
    weights: Sequence[float] | None = None,
    starting_equity: float = 10_000.0,
    periods_per_year: int = 252,
    cost_per_turnover: float = 0.0,
) -> PortfolioResult:
    """Backtest one signal across several assets and blend into a portfolio.

    Each asset's positions are produced by ``signal_fn``, lagged one bar, and
    multiplied by that asset's per-bar returns (less turnover costs). The
    per-asset return streams are then combined with ``weights`` into a single
    portfolio return stream.

    Args:
        price_data: Mapping of symbol -> price series. All series must share the
            same length ``n >= 2``.
        signal_fn: Maps a price series to ``n`` target positions in ``[0, 1]``.
        weights: Capital weight per asset (in ``price_data`` insertion order).
            Defaults to equal weight. Need not sum to 1 — it is normalised.
        starting_equity: Initial portfolio capital.
        periods_per_year: Periods per year used to annualise ratio metrics.
        cost_per_turnover: Proportional cost charged per unit of turnover.

    Returns:
        A :class:`PortfolioResult` with blended returns, equity, and metrics.

    Raises:
        ValueError: If ``price_data`` is empty, series lengths differ or are too
            short, weights length mismatches, weights are negative or sum to
            zero, or ``cost_per_turnover`` is negative.
    """
    if not price_data:
        raise ValueError("price_data must contain at least one asset")
    if cost_per_turnover < 0.0:
        raise ValueError("cost_per_turnover must be non-negative")

    symbols = list(price_data.keys())
    arrays = [np.asarray(price_data[s], dtype=np.float64).ravel() for s in symbols]
    n = arrays[0].size
    if n < 2:
        raise ValueError("each price series needs at least two prices")
    if any(a.size != n for a in arrays):
        raise ValueError("all price series must have the same length")

    if weights is None:
        weight_arr = np.full(len(symbols), 1.0 / len(symbols), dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64).ravel()
        if weight_arr.size != len(symbols):
            raise ValueError("weights length must match number of assets")
        if np.any(weight_arr < 0.0):
            raise ValueError("weights must be non-negative")
        total_weight = float(np.sum(weight_arr))
        if total_weight == 0.0:
            raise ValueError("weights must not sum to zero")
        weight_arr = weight_arr / total_weight

    per_asset_returns: list[NDArray[np.float64]] = []
    per_asset_total: dict[str, float] = {}
    for sym, prices in zip(symbols, arrays, strict=True):
        positions = np.asarray(list(signal_fn(prices.tolist())), dtype=np.float64).ravel()
        if positions.size != n:
            raise ValueError(
                f"signal_fn returned {positions.size} positions for {n} prices on {sym}"
            )
        if not np.all((positions >= 0.0) & (positions <= 1.0)):
            raise ValueError(f"positions for {sym} must be within [0, 1]")
        asset_returns = prices[1:] / prices[:-1] - 1.0
        held = positions[:-1]
        costs = cost_per_turnover * np.abs(np.diff(positions))
        strat_returns = held * asset_returns - costs
        per_asset_returns.append(strat_returns)
        per_asset_total[sym] = total_return(strat_returns)

    # Blend per-asset return streams by capital weight (rebalanced each bar).
    stacked = np.vstack(per_asset_returns)  # shape (assets, n - 1)
    portfolio_returns = weight_arr @ stacked
    port_equity = equity_curve(portfolio_returns, starting_equity)

    return PortfolioResult(
        symbols=symbols,
        weights=weight_arr,
        portfolio_returns=portfolio_returns,
        portfolio_equity=port_equity,
        total_return=total_return(portfolio_returns),
        final_equity=float(port_equity[-1]),
        cagr=annualized_return(portfolio_returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(portfolio_returns, periods_per_year),
        sortino_ratio=sortino_ratio(portfolio_returns, periods_per_year),
        calmar_ratio=calmar_ratio(portfolio_returns, periods_per_year),
        annual_volatility=annualized_volatility(portfolio_returns, periods_per_year),
        max_drawdown=max_drawdown(port_equity),
        per_asset_total_return=per_asset_total,
    )
