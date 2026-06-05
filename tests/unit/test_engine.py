"""Unit tests for the backtesting engine and signal helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from backtesting_python.engine import run_backtest
from backtesting_python.signals import moving_average_crossover


def _always_long(prices: Sequence[float]) -> list[int]:
    return [1] * len(prices)


def _always_flat(prices: Sequence[float]) -> list[int]:
    return [0] * len(prices)


def test_always_long_matches_buy_and_hold() -> None:
    prices = [100.0, 110.0, 121.0, 133.1]
    result = run_backtest(prices, _always_long, starting_equity=1000.0)
    # Each bar is +10%, three bars => 1.331x.
    assert result.strategy_total_return == pytest.approx(0.331)
    assert result.strategy_final_equity == pytest.approx(1331.0)
    assert result.buy_hold_final_equity == pytest.approx(1331.0)
    assert result.n_trades == 0


def test_always_flat_has_no_pnl() -> None:
    prices = [100.0, 90.0, 120.0, 80.0]
    result = run_backtest(prices, _always_flat, starting_equity=5000.0)
    assert result.strategy_total_return == 0.0
    assert result.strategy_final_equity == pytest.approx(5000.0)
    assert result.max_drawdown == 0.0


def test_positions_are_lagged_one_bar() -> None:
    # Signal goes long only at the last bar, so it never captures any move.
    prices = [100.0, 200.0, 100.0]

    def late_entry(p: Sequence[float]) -> list[int]:
        return [0, 0, 1]

    result = run_backtest(prices, late_entry, starting_equity=1000.0)
    assert result.strategy_total_return == 0.0
    assert result.strategy_final_equity == pytest.approx(1000.0)


def test_lagged_signal_captures_next_bar_move() -> None:
    # Long signal at bar 0 captures the +100% move into bar 1.
    prices = [100.0, 200.0, 200.0]

    def early_entry(p: Sequence[float]) -> list[int]:
        return [1, 0, 0]

    result = run_backtest(prices, early_entry, starting_equity=1000.0)
    assert result.strategy_total_return == pytest.approx(1.0)
    assert result.strategy_final_equity == pytest.approx(2000.0)


def test_trade_count_counts_position_changes() -> None:
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]

    def toggle(p: Sequence[float]) -> list[int]:
        return [0, 1, 1, 0, 1]

    result = run_backtest(prices, toggle)
    # Changes at indices 1, 3, 4 => 3 trades.
    assert result.n_trades == 3


def test_run_backtest_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least two prices"):
        run_backtest([100.0], _always_long)


def test_run_backtest_rejects_wrong_length_positions() -> None:
    with pytest.raises(ValueError, match="positions"):
        run_backtest([1.0, 2.0, 3.0], lambda p: [1, 0])


def test_run_backtest_rejects_invalid_positions() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        run_backtest([1.0, 2.0], lambda p: [2, 0])


def test_moving_average_crossover_is_flat_before_warmup() -> None:
    prices = list(range(1, 50))
    positions = moving_average_crossover(prices, fast=5, slow=20)
    assert len(positions) == len(prices)
    assert all(p == 0 for p in positions[:19])
    assert all(p in (0, 1) for p in positions)


def test_moving_average_crossover_long_in_uptrend() -> None:
    prices = [float(i) for i in range(1, 60)]
    positions = moving_average_crossover(prices, fast=5, slow=20)
    # A steady uptrend keeps the fast MA above the slow MA after warmup.
    assert positions[-1] == 1


def test_moving_average_crossover_validates_windows() -> None:
    with pytest.raises(ValueError, match="shorter"):
        moving_average_crossover([1.0, 2.0, 3.0], fast=30, slow=10)
    with pytest.raises(ValueError, match="positive"):
        moving_average_crossover([1.0, 2.0, 3.0], fast=0, slow=10)
