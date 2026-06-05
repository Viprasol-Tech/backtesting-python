"""Pure performance metrics for backtests.

Every function here is a pure, side-effect-free transformation of a returns or
equity series. They accept any sequence of floats (lists, tuples, or NumPy
arrays) and return plain Python floats so they compose cleanly with the engine.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "equity_curve",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
]


def _as_array(values: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``values`` as a 1-D float64 NumPy array.

    Args:
        values: Any sequence of numbers (list, tuple, or NumPy array).

    Returns:
        A contiguous 1-D ``float64`` array view/copy of ``values``.
    """
    return np.asarray(values, dtype=np.float64).ravel()


def total_return(returns: Sequence[float] | NDArray[np.float64]) -> float:
    """Compound a series of period returns into a single total return.

    Given per-period simple returns ``r_t``, the total return is
    ``prod(1 + r_t) - 1``.

    Args:
        returns: Per-period simple returns (e.g. ``0.01`` for +1%).

    Returns:
        The compounded total return over the whole series. Returns ``0.0`` for
        an empty input.
    """
    arr = _as_array(returns)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def equity_curve(
    returns: Sequence[float] | NDArray[np.float64],
    starting_equity: float = 1.0,
) -> NDArray[np.float64]:
    """Build a cumulative equity curve from period returns.

    Args:
        returns: Per-period simple returns.
        starting_equity: The equity value before the first period.

    Returns:
        An array of length ``len(returns) + 1`` whose first element is
        ``starting_equity`` and whose subsequent elements compound each return.
    """
    arr = _as_array(returns)
    growth = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
    return growth * float(starting_equity)


def sharpe_ratio(
    returns: Sequence[float] | NDArray[np.float64],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualised Sharpe ratio of a returns series.

    The Sharpe ratio is the mean excess return divided by its standard
    deviation, scaled by ``sqrt(periods_per_year)`` to annualise it.

    Args:
        returns: Per-period simple returns.
        periods_per_year: Number of return periods in a year (252 for daily
            trading data, 12 for monthly, etc.). Used to annualise.
        risk_free_rate: Per-period risk-free rate to subtract from each return.

    Returns:
        The annualised Sharpe ratio. Returns ``0.0`` when there are fewer than
        two observations or when the return volatility is zero.
    """
    arr = _as_array(returns) - float(risk_free_rate)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    mean = float(np.mean(arr))
    return mean / std * float(np.sqrt(periods_per_year))


def max_drawdown(equity: Sequence[float] | NDArray[np.float64]) -> float:
    """Compute the maximum drawdown of an equity curve.

    The maximum drawdown is the largest peak-to-trough decline, expressed as a
    non-positive fraction of the running peak.

    Args:
        equity: An equity (cumulative value) curve.

    Returns:
        The maximum drawdown as a value in ``[-1.0, 0.0]``. A monotonically
        non-decreasing curve yields ``0.0``. Returns ``0.0`` for an empty input.
    """
    arr = _as_array(equity)
    if arr.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    drawdowns = arr / running_peak - 1.0
    return float(np.min(drawdowns))
