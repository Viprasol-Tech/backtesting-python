"""Export helpers for backtest results.

Turn a :class:`~backtesting_python.engine.BacktestResult` into a tidy pandas
DataFrame, write its equity curves to CSV, or dump the scalar summary metrics
to CSV — handy for plotting, reporting, or feeding downstream tooling.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting_python.engine import BacktestResult

__all__ = [
    "result_to_frame",
    "summary_dict",
    "write_equity_csv",
    "write_summary_csv",
]


def result_to_frame(result: BacktestResult) -> pd.DataFrame:
    """Build a per-bar DataFrame of strategy vs buy-and-hold equity.

    The strategy and buy-and-hold equity curves both have length ``n`` (one
    point per price), so they align bar-for-bar in the frame.

    Args:
        result: A completed backtest result.

    Returns:
        A DataFrame indexed by integer ``bar`` with ``strategy_equity`` and
        ``buy_hold_equity`` columns.
    """
    frame = pd.DataFrame(
        {
            "strategy_equity": result.strategy_equity,
            "buy_hold_equity": result.buy_hold_equity,
        }
    )
    frame.index.name = "bar"
    return frame


def write_equity_csv(result: BacktestResult, path: str | Path) -> Path:
    """Write the strategy and buy-and-hold equity curves to a CSV file.

    Args:
        result: A completed backtest result.
        path: Destination CSV path. Parent directories are created as needed.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_to_frame(result).to_csv(out)
    return out


def summary_dict(result: BacktestResult) -> dict[str, float | int]:
    """Collect the scalar summary metrics of a result into a flat dict.

    Args:
        result: A completed backtest result.

    Returns:
        An ordered mapping of metric name to value, suitable for CSV/JSON.
    """
    return {
        "strategy_total_return": result.strategy_total_return,
        "buy_hold_total_return": result.buy_hold_total_return,
        "strategy_final_equity": result.strategy_final_equity,
        "buy_hold_final_equity": result.buy_hold_final_equity,
        "cagr": result.cagr,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "calmar_ratio": result.calmar_ratio,
        "annual_volatility": result.annual_volatility,
        "max_drawdown": result.max_drawdown,
        "exposure": result.exposure,
        "total_costs": result.total_costs,
        "n_trades": result.n_trades,
    }


def write_summary_csv(result: BacktestResult, path: str | Path) -> Path:
    """Write the scalar summary metrics of a result to a one-row CSV file.

    Args:
        result: A completed backtest result.
        path: Destination CSV path. Parent directories are created as needed.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_dict(result)]).to_csv(out, index=False)
    return out
