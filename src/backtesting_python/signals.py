"""Reusable signal functions for the backtesting engine.

A signal function maps a price series to a list of target positions in
``{0, 1}``. These helpers are intentionally simple and dependency-light so they
double as worked examples for writing your own.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["moving_average_crossover"]


def moving_average_crossover(
    prices: Sequence[float],
    fast: int = 10,
    slow: int = 30,
) -> list[int]:
    """Long when the fast moving average is above the slow one, else flat.

    Both moving averages are simple (equal-weighted) trailing averages. Until
    enough bars exist to compute the slow average, the position is flat (``0``).

    Args:
        prices: The asset price series.
        fast: Window length of the fast moving average.
        slow: Window length of the slow moving average (must exceed ``fast``).

    Returns:
        A list of target positions in ``{0, 1}``, one per price.

    Raises:
        ValueError: If ``fast`` or ``slow`` is not positive, or ``fast >= slow``.
    """
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow windows must be positive")
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")

    arr = np.asarray(prices, dtype=np.float64).ravel()
    n = arr.size
    positions = [0] * n
    for i in range(slow - 1, n):
        fast_ma = float(np.mean(arr[i - fast + 1 : i + 1]))
        slow_ma = float(np.mean(arr[i - slow + 1 : i + 1]))
        positions[i] = 1 if fast_ma > slow_ma else 0
    return positions
