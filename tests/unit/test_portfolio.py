"""Unit tests for the multi-asset portfolio backtest."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from backtesting_python.portfolio import run_portfolio_backtest


def _always_long(prices: Sequence[float]) -> list[int]:
    return [1] * len(list(prices))


def test_equal_weight_two_assets_averages_returns() -> None:
    # Asset A: +10% each bar; Asset B: 0% each bar. Equal weight => +5% each bar.
    data = {
        "A": [100.0, 110.0, 121.0],
        "B": [100.0, 100.0, 100.0],
    }
    result = run_portfolio_backtest(data, _always_long, starting_equity=1000.0)
    # (1.05)(1.05) - 1 = 0.1025
    assert result.total_return == pytest.approx(0.1025)
    assert result.final_equity == pytest.approx(1000.0 * 1.05 * 1.05)
    assert result.symbols == ["A", "B"]


def test_weights_are_normalised() -> None:
    data = {"A": [100.0, 110.0], "B": [100.0, 100.0]}
    result = run_portfolio_backtest(data, _always_long, weights=[3.0, 1.0])
    assert np.allclose(result.weights, [0.75, 0.25])
    # 75% of a +10% move = +7.5%.
    assert result.total_return == pytest.approx(0.075)


def test_per_asset_returns_reported() -> None:
    data = {"A": [100.0, 120.0], "B": [100.0, 90.0]}
    result = run_portfolio_backtest(data, _always_long)
    assert result.per_asset_total_return["A"] == pytest.approx(0.20)
    assert result.per_asset_total_return["B"] == pytest.approx(-0.10)


def test_portfolio_diversification_lowers_volatility() -> None:
    # Two anti-correlated assets => the blend is far less volatile than either.
    a = [100.0 + 5.0 * (-1) ** i for i in range(40)]
    b = [100.0 - 5.0 * (-1) ** i for i in range(40)]
    result = run_portfolio_backtest({"A": a, "B": b}, _always_long)
    assert result.annual_volatility < 0.05


def test_portfolio_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        run_portfolio_backtest({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0]}, _always_long)


def test_portfolio_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        run_portfolio_backtest({}, _always_long)


def test_portfolio_rejects_bad_weight_length() -> None:
    data = {"A": [1.0, 2.0], "B": [1.0, 2.0]}
    with pytest.raises(ValueError, match="weights length"):
        run_portfolio_backtest(data, _always_long, weights=[1.0])


def test_portfolio_rejects_zero_weight_sum() -> None:
    data = {"A": [1.0, 2.0], "B": [1.0, 2.0]}
    with pytest.raises(ValueError, match="sum to zero"):
        run_portfolio_backtest(data, _always_long, weights=[0.0, 0.0])


def test_portfolio_rejects_negative_weights() -> None:
    data = {"A": [1.0, 2.0], "B": [1.0, 2.0]}
    with pytest.raises(ValueError, match="non-negative"):
        run_portfolio_backtest(data, _always_long, weights=[1.0, -1.0])


def test_portfolio_costs_reduce_return() -> None:
    data = {"A": [100.0, 110.0, 110.0], "B": [100.0, 110.0, 110.0]}

    def toggle(p: Sequence[float]) -> list[int]:
        return [1, 0, 1]

    free = run_portfolio_backtest(data, toggle)
    costed = run_portfolio_backtest(data, toggle, cost_per_turnover=0.01)
    assert costed.total_return < free.total_return
