"""Bundled example strategies (signal functions).

Each strategy maps a price series to a list of target positions in ``{0, 1}``
(long or flat). They are intentionally dependency-light and well-documented so
they double as worked examples for writing your own. They sit alongside the
canonical :func:`backtesting_python.signals.moving_average_crossover`.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "STRATEGIES",
    "bollinger_breakout",
    "buy_and_hold",
    "donchian_breakout",
    "momentum",
    "rsi_reversion",
]


def buy_and_hold(prices: Sequence[float]) -> list[int]:
    """Always long. A trivial baseline strategy.

    Args:
        prices: The asset price series.

    Returns:
        A list of ``1`` positions, one per price.
    """
    return [1] * len(list(prices))


def momentum(prices: Sequence[float], lookback: int = 20) -> list[int]:
    """Long when price is above its value ``lookback`` bars ago, else flat.

    A classic time-series momentum rule: ride the trend while it persists.
    The position is flat until enough history exists.

    Args:
        prices: The asset price series.
        lookback: Number of bars to look back when measuring momentum.

    Returns:
        A list of target positions in ``{0, 1}``, one per price.

    Raises:
        ValueError: If ``lookback`` is not positive.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    arr = np.asarray(prices, dtype=np.float64).ravel()
    n = arr.size
    positions = [0] * n
    for i in range(lookback, n):
        positions[i] = 1 if arr[i] > arr[i - lookback] else 0
    return positions


def rsi_reversion(
    prices: Sequence[float],
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> list[int]:
    """Mean-reversion on Wilder's RSI: buy oversold, exit overbought.

    Goes long when RSI falls at or below ``oversold`` and holds until RSI rises
    at or above ``overbought``, at which point it goes flat. Between thresholds
    the previous position is carried forward.

    Args:
        prices: The asset price series.
        period: RSI averaging period.
        oversold: RSI level at or below which to enter long.
        overbought: RSI level at or above which to exit to flat.

    Returns:
        A list of target positions in ``{0, 1}``, one per price.

    Raises:
        ValueError: If ``period`` is not positive or thresholds are inverted.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if not 0.0 <= oversold < overbought <= 100.0:
        raise ValueError("require 0 <= oversold < overbought <= 100")

    arr = np.asarray(prices, dtype=np.float64).ravel()
    n = arr.size
    positions = [0] * n
    if n < period + 1:
        return positions

    deltas = np.diff(arr)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    pos = 0
    for i in range(period, n):
        # deltas[i - 1] is the change into bar i.
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        if rsi <= oversold:
            pos = 1
        elif rsi >= overbought:
            pos = 0
        positions[i] = pos
    return positions


def bollinger_breakout(
    prices: Sequence[float],
    window: int = 20,
    num_std: float = 2.0,
) -> list[int]:
    """Long on an upper-band breakout, exit back inside the middle band.

    Computes a rolling mean and standard deviation. Goes long when price closes
    above the upper band (mean + ``num_std`` * std) and exits to flat when price
    falls back below the middle band (the rolling mean).

    Args:
        prices: The asset price series.
        window: Rolling window for the mean and standard deviation.
        num_std: Number of standard deviations for the upper band.

    Returns:
        A list of target positions in ``{0, 1}``, one per price.

    Raises:
        ValueError: If ``window`` is not positive or ``num_std`` is negative.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if num_std < 0.0:
        raise ValueError("num_std must be non-negative")

    arr = np.asarray(prices, dtype=np.float64).ravel()
    n = arr.size
    positions = [0] * n
    pos = 0
    for i in range(window - 1, n):
        win = arr[i - window + 1 : i + 1]
        mean = float(np.mean(win))
        std = float(np.std(win, ddof=0))
        upper = mean + num_std * std
        if arr[i] > upper:
            pos = 1
        elif arr[i] < mean:
            pos = 0
        positions[i] = pos
    return positions


def donchian_breakout(
    prices: Sequence[float],
    entry: int = 20,
    exit_: int = 10,
) -> list[int]:
    """Turtle-style breakout: long on an N-bar high, exit on an M-bar low.

    Goes long when the price makes a new high over the trailing ``entry`` bars
    and exits to flat when it makes a new low over the trailing ``exit_`` bars.
    Both channels use prior bars only, so the rule is causal.

    Args:
        prices: The asset price series.
        entry: Lookback window for the breakout (entry) high.
        exit_: Lookback window for the breakdown (exit) low.

    Returns:
        A list of target positions in ``{0, 1}``, one per price.

    Raises:
        ValueError: If ``entry`` or ``exit_`` is not positive.
    """
    if entry <= 0 or exit_ <= 0:
        raise ValueError("entry and exit_ windows must be positive")

    arr = np.asarray(prices, dtype=np.float64).ravel()
    n = arr.size
    positions = [0] * n
    pos = 0
    start = max(entry, exit_)
    for i in range(start, n):
        entry_high = float(np.max(arr[i - entry : i]))
        exit_low = float(np.min(arr[i - exit_ : i]))
        if arr[i] > entry_high:
            pos = 1
        elif arr[i] < exit_low:
            pos = 0
        positions[i] = pos
    return positions


#: Registry of bundled zero-argument-friendly strategies keyed by CLI name.
STRATEGIES: dict[str, object] = {
    "buy-and-hold": buy_and_hold,
    "momentum": momentum,
    "rsi": rsi_reversion,
    "bollinger": bollinger_breakout,
    "donchian": donchian_breakout,
}
