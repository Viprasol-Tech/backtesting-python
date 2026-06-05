"""Unit tests for the pure metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting_python.metrics import (
    equity_curve,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


def test_total_return_known_series() -> None:
    # (1.1)(0.9)(1.2) - 1 = 0.188
    result = total_return([0.1, -0.1, 0.2])
    assert result == pytest.approx(0.188, abs=1e-12)


def test_total_return_empty_is_zero() -> None:
    assert total_return([]) == 0.0


def test_total_return_doubling() -> None:
    # Two +100% periods compound to 4x => +300%.
    assert total_return([1.0, 1.0]) == pytest.approx(3.0)


def test_equity_curve_compounds_from_start() -> None:
    curve = equity_curve([0.1, -0.5, 0.2], starting_equity=1000.0)
    assert curve[0] == pytest.approx(1000.0)
    assert curve[-1] == pytest.approx(1000.0 * 1.1 * 0.5 * 1.2)
    assert len(curve) == 4


def test_sharpe_of_constant_positive_returns_is_large() -> None:
    # Zero volatility but a tiny jitter -> very large positive Sharpe.
    returns = [0.01, 0.0100001, 0.0099999, 0.01, 0.0100001]
    sharpe = sharpe_ratio(returns, periods_per_year=252)
    assert sharpe > 100.0


def test_sharpe_sign_follows_mean() -> None:
    pos = sharpe_ratio([0.02, -0.01, 0.03, 0.01], periods_per_year=252)
    neg = sharpe_ratio([-0.02, 0.01, -0.03, -0.01], periods_per_year=252)
    assert pos > 0.0
    assert neg < 0.0


def test_sharpe_zero_volatility_returns_zero() -> None:
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_too_few_points_returns_zero() -> None:
    assert sharpe_ratio([0.05]) == 0.0


def test_sharpe_annualisation_scales_with_sqrt_periods() -> None:
    returns = [0.01, -0.005, 0.02, 0.003, -0.002]
    base = sharpe_ratio(returns, periods_per_year=1)
    annual = sharpe_ratio(returns, periods_per_year=252)
    assert annual == pytest.approx(base * math.sqrt(252))


def test_max_drawdown_of_rising_curve_is_zero() -> None:
    assert max_drawdown([100.0, 101.0, 102.0, 110.0, 200.0]) == 0.0


def test_max_drawdown_known_value() -> None:
    # Peak 100 -> trough 50 is a -50% drawdown.
    assert max_drawdown([100.0, 120.0, 60.0, 90.0]) == pytest.approx(-0.5)


def test_max_drawdown_empty_is_zero() -> None:
    assert max_drawdown([]) == 0.0


def test_metrics_accept_numpy_arrays() -> None:
    arr = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    assert total_return(arr) == pytest.approx(0.188, abs=1e-12)
