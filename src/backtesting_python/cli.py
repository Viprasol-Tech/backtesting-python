"""Command-line interface for Backtesting Python.

``backtesting-python demo`` runs a moving-average-crossover strategy on a
synthetic price series and prints the strategy metrics next to buy-and-hold.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math

import typer
from rich.console import Console
from rich.table import Table

from backtesting_python import __version__
from backtesting_python.engine import run_backtest
from backtesting_python.signals import moving_average_crossover

app = typer.Typer(add_completion=False, help="Backtesting Python — by Viprasol Tech.")
console = Console()


def _synthetic_prices(n: int = 300, base: float = 100.0) -> list[float]:
    """Build a deterministic trending-plus-cyclical synthetic price series.

    Args:
        n: Number of price points to generate.
        base: Baseline price level.

    Returns:
        A list of ``n`` synthetic prices with an upward drift and a sine cycle.
    """
    return [base + 0.10 * i + 12.0 * math.sin(i / 18.0) for i in range(n)]


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"backtesting-python [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo(
    fast: int = typer.Option(10, help="Fast moving-average window."),
    slow: int = typer.Option(30, help="Slow moving-average window."),
) -> None:
    """Backtest a moving-average crossover on synthetic data."""
    prices = _synthetic_prices()
    result = run_backtest(
        prices,
        lambda p: moving_average_crossover(p, fast=fast, slow=slow),
    )

    table = Table(title=f"MA crossover ({fast}/{slow}) vs buy-and-hold")
    table.add_column("Metric", style="bold")
    table.add_column("Strategy", justify="right")
    table.add_column("Buy & hold", justify="right")
    table.add_row(
        "Total return",
        f"{result.strategy_total_return:+.2%}",
        f"{result.buy_hold_total_return:+.2%}",
    )
    table.add_row(
        "Final equity",
        f"${result.strategy_final_equity:,.2f}",
        f"${result.buy_hold_final_equity:,.2f}",
    )
    table.add_row("Sharpe ratio", f"{result.sharpe_ratio:.2f}", "-")
    table.add_row("Max drawdown", f"{result.max_drawdown:.2%}", "-")
    table.add_row("Trades", str(result.n_trades), "-")
    console.print(table)


if __name__ == "__main__":
    app()
