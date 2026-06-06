"""Unit tests for the pure metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtesting_python.metrics import (
    annualized_return,
    annualized_volatility,
    cagr,
    calmar_ratio,
    equity_curve,
    exposure,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
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


def test_sortino_ignores_upside_volatility() -> None:
    # All-positive returns => no downside deviation => zero by convention.
    assert sortino_ratio([0.01, 0.02, 0.03]) == 0.0


def test_sortino_penalises_downside_only() -> None:
    mixed = [0.05, -0.02, 0.04, -0.01, 0.03]
    sortino = sortino_ratio(mixed, periods_per_year=252)
    sharpe = sharpe_ratio(mixed, periods_per_year=252)
    # With strong upside, downside deviation is smaller than total stdev,
    # so Sortino exceeds Sharpe for the same positive-mean series.
    assert sortino > sharpe > 0.0


def test_sortino_too_few_points_is_zero() -> None:
    assert sortino_ratio([0.01]) == 0.0


def test_sortino_sign_follows_mean() -> None:
    neg = sortino_ratio([-0.03, 0.01, -0.02, -0.01], periods_per_year=12)
    assert neg < 0.0


def test_annualized_return_compounds_per_year() -> None:
    # One year of 252 days at +0.1%/day compounds to (1.001)^252 - 1.
    daily = [0.001] * 252
    expected = (1.001) ** 252 - 1.0
    assert annualized_return(daily, periods_per_year=252) == pytest.approx(expected)


def test_annualized_return_half_year_extrapolates() -> None:
    # 126 days (half a year) that double should annualise to roughly 4x growth.
    rets = [0.005488] * 126  # ~ doubles over 126 bars
    ann = annualized_return(rets, periods_per_year=252)
    assert ann == pytest.approx(3.0, rel=0.05)


def test_cagr_is_alias_of_annualized_return() -> None:
    rets = [0.01, -0.02, 0.03, 0.0]
    assert cagr(rets, periods_per_year=12) == annualized_return(rets, periods_per_year=12)


def test_annualized_return_empty_is_zero() -> None:
    assert annualized_return([]) == 0.0


def test_annualized_return_total_wipeout() -> None:
    assert annualized_return([-1.0, 0.5]) == -1.0


def test_annualized_volatility_scales_with_sqrt_periods() -> None:
    rets = [0.01, -0.02, 0.015, -0.005, 0.02]
    per_period = float(np.std(rets, ddof=1))
    assert annualized_volatility(rets, 252) == pytest.approx(per_period * math.sqrt(252))


def test_annualized_volatility_too_few_points_is_zero() -> None:
    assert annualized_volatility([0.01]) == 0.0


def test_calmar_zero_when_no_drawdown() -> None:
    # Monotonically rising returns => no drawdown => zero by convention.
    assert calmar_ratio([0.01, 0.02, 0.01]) == 0.0


def test_calmar_positive_for_profitable_drawdown_series() -> None:
    # Net positive but with a dip, so there is a finite drawdown.
    rets = [0.1, -0.05, 0.1, -0.03, 0.08]
    calmar = calmar_ratio(rets, periods_per_year=252)
    assert calmar > 0.0


def test_calmar_empty_is_zero() -> None:
    assert calmar_ratio([]) == 0.0


def test_exposure_counts_nonzero_fraction() -> None:
    assert exposure([0, 1, 1, 0, 1]) == pytest.approx(3 / 5)


def test_exposure_fractional_positions_count_as_in_market() -> None:
    assert exposure([0.0, 0.5, 0.0, 0.25]) == pytest.approx(0.5)


def test_exposure_empty_is_zero() -> None:
    assert exposure([]) == 0.0
