"""Position-sizing helpers.

A *signal* produces target positions in ``{0, 1}`` (in or out of the market).
A *sizer* turns those binary signals into fractional exposures (e.g. ``0.5`` for
half capital) so the engine can scale risk. Every sizer maps a price series and
a binary-position series to a same-length array of fractional weights, where
``0.0`` means flat and ``1.0`` means fully invested.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "fixed_fraction",
    "volatility_target",
]


def fixed_fraction(
    positions: Sequence[int] | Sequence[float] | NDArray[np.float64],
    fraction: float = 1.0,
) -> NDArray[np.float64]:
    """Scale every in-market bar to a constant fraction of capital.

    Args:
        positions: Binary (or fractional) target positions.
        fraction: Capital fraction to allocate when in market, in ``[0, 1]``.

    Returns:
        An array of fractional weights, ``positions * fraction``.

    Raises:
        ValueError: If ``fraction`` is outside ``[0.0, 1.0]``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    arr = np.asarray(positions, dtype=np.float64).ravel()
    return arr * float(fraction)


def volatility_target(
    prices: Sequence[float],
    positions: Sequence[int] | Sequence[float] | NDArray[np.float64],
    target_vol: float = 0.15,
    lookback: int = 20,
    periods_per_year: int = 252,
    max_leverage: float = 1.0,
) -> NDArray[np.float64]:
    """Scale exposure inversely to recent realised volatility.

    For each bar, the weight is ``target_vol / realised_vol`` (annualised),
    capped at ``max_leverage`` and zeroed when flat. Realised volatility uses a
    trailing window of simple returns; before enough history exists, the weight
    falls back to the raw position.

    Args:
        prices: The asset price series.
        positions: Binary target positions aligned to ``prices``.
        target_vol: Desired annualised volatility (e.g. ``0.15`` for 15%).
        lookback: Trailing window of returns used to estimate volatility.
        periods_per_year: Periods per year used to annualise volatility.
        max_leverage: Upper bound on the per-bar weight.

    Returns:
        An array of fractional weights aligned to ``prices``.

    Raises:
        ValueError: If ``target_vol`` is not positive, ``lookback`` is less than
            two, ``max_leverage`` is not positive, or lengths differ.
    """
    if target_vol <= 0.0:
        raise ValueError("target_vol must be positive")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if max_leverage <= 0.0:
        raise ValueError("max_leverage must be positive")

    price_arr = np.asarray(prices, dtype=np.float64).ravel()
    pos_arr = np.asarray(positions, dtype=np.float64).ravel()
    if price_arr.size != pos_arr.size:
        raise ValueError("prices and positions must have the same length")

    n = price_arr.size
    weights = pos_arr.copy()
    if n < 2:
        return weights

    rets = price_arr[1:] / price_arr[:-1] - 1.0  # length n - 1
    for i in range(lookback, n):
        # Use returns strictly before bar i to keep the rule causal.
        window = rets[i - lookback : i]
        vol = float(np.std(window, ddof=1) * np.sqrt(periods_per_year))
        scale = max_leverage if vol == 0.0 else min(target_vol / vol, max_leverage)
        weights[i] = pos_arr[i] * scale
    return weights
