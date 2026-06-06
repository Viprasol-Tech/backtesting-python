"""Command-line interface for Backtesting Python.

Subcommands:

* ``version``   — print the installed version.
* ``demo``      — moving-average crossover on synthetic data.
* ``run``       — backtest a named bundled strategy, with optional sizing,
  transaction costs, and CSV export.
* ``strategies``— list the bundled strategies.
* ``portfolio`` — run a strategy across several synthetic assets and blend them.

Part of Backtesting Python by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from backtesting_python import __version__
from backtesting_python.engine import BacktestResult, SignalFn, run_backtest
from backtesting_python.export import write_equity_csv, write_summary_csv
from backtesting_python.portfolio import run_portfolio_backtest
from backtesting_python.signals import moving_average_crossover
from backtesting_python.sizing import fixed_fraction, volatility_target
from backtesting_python.strategies import STRATEGIES

app = typer.Typer(add_completion=False, help="Backtesting Python — by Viprasol Tech.")
console = Console()


def _synthetic_prices(n: int = 300, base: float = 100.0, phase: float = 0.0) -> list[float]:
    """Build a deterministic trending-plus-cyclical synthetic price series.

    Args:
        n: Number of price points to generate.
        base: Baseline price level.
        phase: Phase offset for the sine cycle (de-correlates multiple assets).

    Returns:
        A list of ``n`` synthetic prices with an upward drift and a sine cycle.
    """
    return [base + 0.10 * i + 12.0 * math.sin(i / 18.0 + phase) for i in range(n)]


def _named_strategy(name: str) -> Callable[[Sequence[float]], list[int]]:
    """Look up a bundled strategy by CLI name or exit with an error.

    Args:
        name: A key from :data:`backtesting_python.strategies.STRATEGIES`.

    Returns:
        The matching signal function.
    """
    fn = STRATEGIES.get(name)
    if fn is None:
        available = ", ".join(sorted(STRATEGIES))
        console.print(f"[red]Unknown strategy '{name}'.[/] Available: {available}")
        raise typer.Exit(code=1)
    return fn  # type: ignore[return-value]


def _result_table(title: str, result: BacktestResult) -> Table:
    """Render a backtest result as a rich comparison table.

    Args:
        title: Table title.
        result: A completed backtest result.

    Returns:
        A populated :class:`rich.table.Table`.
    """
    table = Table(title=title)
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
    table.add_row("CAGR", f"{result.cagr:+.2%}", "-")
    table.add_row("Sharpe ratio", f"{result.sharpe_ratio:.2f}", "-")
    table.add_row("Sortino ratio", f"{result.sortino_ratio:.2f}", "-")
    table.add_row("Calmar ratio", f"{result.calmar_ratio:.2f}", "-")
    table.add_row("Annual volatility", f"{result.annual_volatility:.2%}", "-")
    table.add_row("Max drawdown", f"{result.max_drawdown:.2%}", "-")
    table.add_row("Exposure", f"{result.exposure:.1%}", "-")
    table.add_row("Costs", f"${result.total_costs:,.2f}", "-")
    table.add_row("Trades", str(result.n_trades), "-")
    return table


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"backtesting-python [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def strategies() -> None:
    """List the bundled strategies usable with ``run``."""
    table = Table(title="Bundled strategies")
    table.add_column("Name", style="bold cyan")
    table.add_column("Function")
    for name in sorted(STRATEGIES):
        table.add_row(name, getattr(STRATEGIES[name], "__name__", str(STRATEGIES[name])))
    console.print(table)


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
    console.print(_result_table(f"MA crossover ({fast}/{slow}) vs buy-and-hold", result))


@app.command()
def run(
    strategy: str = typer.Argument("momentum", help="Bundled strategy name."),
    cost: float = typer.Option(0.0, help="Proportional cost per unit turnover."),
    sizing: str = typer.Option(
        "full",
        help="Position sizing: 'full', 'half', or 'vol-target'.",
    ),
    csv: Path | None = typer.Option(None, help="Write the equity curve to this CSV."),
    summary_csv: Path | None = typer.Option(None, help="Write the summary metrics to this CSV."),
) -> None:
    """Backtest a named bundled strategy on synthetic data."""
    prices = _synthetic_prices()
    base_signal = _named_strategy(strategy)

    if sizing == "full":
        signal_fn: SignalFn = base_signal
    elif sizing == "half":
        signal_fn = lambda p: fixed_fraction(base_signal(p), 0.5)  # noqa: E731
    elif sizing == "vol-target":
        signal_fn = lambda p: volatility_target(p, base_signal(p))  # noqa: E731
    else:
        console.print(f"[red]Unknown sizing '{sizing}'.[/] Use full, half, or vol-target.")
        raise typer.Exit(code=1)

    result = run_backtest(prices, signal_fn, cost_per_turnover=cost)
    console.print(_result_table(f"{strategy} ({sizing}) vs buy-and-hold", result))

    if csv is not None:
        written = write_equity_csv(result, csv)
        console.print(f"[green]Equity curve written to[/] {written}")
    if summary_csv is not None:
        written = write_summary_csv(result, summary_csv)
        console.print(f"[green]Summary written to[/] {written}")


@app.command()
def portfolio(
    strategy: str = typer.Argument("momentum", help="Bundled strategy name."),
    assets: int = typer.Option(3, min=1, help="Number of synthetic assets."),
    cost: float = typer.Option(0.0, help="Proportional cost per unit turnover."),
) -> None:
    """Run a strategy across several synthetic assets and blend equally."""
    signal_fn = _named_strategy(strategy)
    price_data = {f"ASSET{i + 1}": _synthetic_prices(phase=i * 1.3) for i in range(assets)}
    result = run_portfolio_backtest(price_data, signal_fn, cost_per_turnover=cost)

    table = Table(title=f"Portfolio: {strategy} across {assets} assets (equal weight)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total return", f"{result.total_return:+.2%}")
    table.add_row("Final equity", f"${result.final_equity:,.2f}")
    table.add_row("CAGR", f"{result.cagr:+.2%}")
    table.add_row("Sharpe ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Sortino ratio", f"{result.sortino_ratio:.2f}")
    table.add_row("Calmar ratio", f"{result.calmar_ratio:.2f}")
    table.add_row("Annual volatility", f"{result.annual_volatility:.2%}")
    table.add_row("Max drawdown", f"{result.max_drawdown:.2%}")
    console.print(table)

    per_asset = Table(title="Per-asset strategy total return")
    per_asset.add_column("Asset", style="bold cyan")
    per_asset.add_column("Total return", justify="right")
    for sym, ret in result.per_asset_total_return.items():
        per_asset.add_row(sym, f"{ret:+.2%}")
    console.print(per_asset)


if __name__ == "__main__":
    app()
