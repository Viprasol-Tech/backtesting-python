"""Unit tests for the bundled example strategies."""

from __future__ import annotations

import pytest

from backtesting_python.engine import run_backtest
from backtesting_python.strategies import (
    STRATEGIES,
    bollinger_breakout,
    buy_and_hold,
    donchian_breakout,
    momentum,
    rsi_reversion,
)


def test_buy_and_hold_is_always_long() -> None:
    positions = buy_and_hold([1.0, 2.0, 3.0, 4.0])
    assert positions == [1, 1, 1, 1]


def test_momentum_goes_long_in_uptrend() -> None:
    prices = [float(i) for i in range(1, 60)]
    positions = momentum(prices, lookback=10)
    assert len(positions) == len(prices)
    assert all(p == 0 for p in positions[:10])
    # A steady uptrend keeps price above its lagged value.
    assert positions[-1] == 1


def test_momentum_flat_in_downtrend() -> None:
    prices = [float(i) for i in range(60, 1, -1)]
    positions = momentum(prices, lookback=10)
    assert positions[-1] == 0


def test_momentum_validates_lookback() -> None:
    with pytest.raises(ValueError, match="positive"):
        momentum([1.0, 2.0, 3.0], lookback=0)


def test_rsi_reversion_returns_binary_aligned() -> None:
    prices = [100.0 + (i % 7) - 3 for i in range(80)]
    positions = rsi_reversion(prices, period=14)
    assert len(positions) == len(prices)
    assert all(p in (0, 1) for p in positions)


def test_rsi_reversion_enters_after_selloff() -> None:
    # A long monotonic decline drives RSI to oversold, triggering a long.
    prices = [200.0 - i for i in range(60)]
    positions = rsi_reversion(prices, period=14, oversold=30.0, overbought=70.0)
    assert 1 in positions


def test_rsi_reversion_validates_args() -> None:
    with pytest.raises(ValueError, match="positive"):
        rsi_reversion([1.0, 2.0, 3.0], period=0)
    with pytest.raises(ValueError, match="oversold"):
        rsi_reversion([1.0, 2.0, 3.0], oversold=80.0, overbought=20.0)


def test_rsi_short_series_is_all_flat() -> None:
    assert rsi_reversion([1.0, 2.0], period=14) == [0, 0]


def test_bollinger_breakout_triggers_on_spike() -> None:
    prices = [100.0] * 25 + [130.0]  # a sharp breakout above the band
    positions = bollinger_breakout(prices, window=20, num_std=2.0)
    assert positions[-1] == 1


def test_bollinger_breakout_flat_when_quiet() -> None:
    prices = [100.0] * 30  # no movement, never breaks out
    positions = bollinger_breakout(prices, window=20)
    assert all(p == 0 for p in positions)


def test_bollinger_validates_args() -> None:
    with pytest.raises(ValueError, match="window"):
        bollinger_breakout([1.0, 2.0], window=0)
    with pytest.raises(ValueError, match="num_std"):
        bollinger_breakout([1.0, 2.0], num_std=-1.0)


def test_donchian_breakout_enters_on_new_high() -> None:
    prices = [float(i) for i in range(1, 60)]
    positions = donchian_breakout(prices, entry=20, exit_=10)
    assert positions[-1] == 1


def test_donchian_exits_on_new_low() -> None:
    prices = [float(i) for i in range(60, 1, -1)]
    positions = donchian_breakout(prices, entry=20, exit_=10)
    assert positions[-1] == 0


def test_donchian_validates_args() -> None:
    with pytest.raises(ValueError, match="positive"):
        donchian_breakout([1.0, 2.0, 3.0], entry=0)


def test_registry_strategies_are_runnable() -> None:
    prices = [100.0 + i * 0.5 for i in range(80)]
    for name, fn in STRATEGIES.items():
        positions = fn(prices)  # type: ignore[operator]
        assert len(positions) == len(prices), name
        result = run_backtest(prices, fn)  # type: ignore[arg-type]
        assert result.strategy_equity.size == len(prices)
