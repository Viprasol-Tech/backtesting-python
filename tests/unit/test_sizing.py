"""Unit tests for the position-sizing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from backtesting_python.engine import run_backtest
from backtesting_python.sizing import fixed_fraction, volatility_target


def test_fixed_fraction_scales_positions() -> None:
    weights = fixed_fraction([1, 0, 1, 1], fraction=0.25)
    assert np.allclose(weights, [0.25, 0.0, 0.25, 0.25])


def test_fixed_fraction_full_is_identity() -> None:
    weights = fixed_fraction([1, 0, 1], fraction=1.0)
    assert np.allclose(weights, [1.0, 0.0, 1.0])


def test_fixed_fraction_validates_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fixed_fraction([1, 0], fraction=1.5)


def test_fixed_fraction_halves_strategy_return() -> None:
    prices = [100.0, 110.0, 121.0]
    full = run_backtest(prices, lambda p: [1, 1, 1])
    half = run_backtest(prices, lambda p: fixed_fraction([1, 1, 1], 0.5))
    # Half exposure earns +5% per bar instead of +10%.
    assert half.strategy_total_return < full.strategy_total_return
    assert half.strategy_total_return == pytest.approx(1.05 * 1.05 - 1.0)


def test_volatility_target_returns_aligned_weights() -> None:
    prices = [100.0 + i for i in range(50)]
    positions = [1] * 50
    weights = volatility_target(prices, positions, target_vol=0.10, lookback=10)
    assert weights.shape == (50,)
    assert np.all(weights >= 0.0)


def test_volatility_target_caps_leverage() -> None:
    # A near-flat series has tiny vol => unscaled weight would explode; cap holds.
    prices = [100.0 + 0.0001 * i for i in range(60)]
    positions = [1] * 60
    weights = volatility_target(prices, positions, target_vol=0.20, lookback=20, max_leverage=1.0)
    assert float(np.max(weights)) <= 1.0 + 1e-9


def test_volatility_target_zeroes_flat_bars() -> None:
    prices = [100.0 + (i % 5) for i in range(40)]
    positions = [0] * 40
    weights = volatility_target(prices, positions, lookback=10)
    assert np.allclose(weights, 0.0)


def test_volatility_target_higher_vol_means_lower_weight() -> None:
    calm = [100.0 + 0.01 * i for i in range(60)]
    wild = [100.0 + 5.0 * (-1) ** i for i in range(60)]
    pos = [1] * 60
    w_calm = volatility_target(calm, pos, target_vol=0.15, lookback=20, max_leverage=10.0)
    w_wild = volatility_target(wild, pos, target_vol=0.15, lookback=20, max_leverage=10.0)
    assert float(np.mean(w_calm[20:])) > float(np.mean(w_wild[20:]))


def test_volatility_target_validates_args() -> None:
    prices = [100.0, 101.0, 102.0]
    with pytest.raises(ValueError, match="target_vol"):
        volatility_target(prices, [1, 1, 1], target_vol=0.0)
    with pytest.raises(ValueError, match="lookback"):
        volatility_target(prices, [1, 1, 1], lookback=1)
    with pytest.raises(ValueError, match="length"):
        volatility_target(prices, [1, 1])
