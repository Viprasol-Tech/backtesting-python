"""Smoke tests for the Typer CLI subcommands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from backtesting_python import __version__
from backtesting_python.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_strategies_command_lists_registry() -> None:
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    assert "momentum" in result.stdout


def test_demo_runs() -> None:
    result = runner.invoke(app, ["demo", "--fast", "5", "--slow", "20"])
    assert result.exit_code == 0
    assert "Sharpe" in result.stdout


def test_run_with_unknown_strategy_errors() -> None:
    result = runner.invoke(app, ["run", "does-not-exist"])
    assert result.exit_code == 1


def test_run_with_sizing_and_export(tmp_path: Path) -> None:
    csv = tmp_path / "equity.csv"
    summary = tmp_path / "summary.csv"
    result = runner.invoke(
        app,
        [
            "run",
            "momentum",
            "--sizing",
            "vol-target",
            "--cost",
            "0.0005",
            "--csv",
            str(csv),
            "--summary-csv",
            str(summary),
        ],
    )
    assert result.exit_code == 0
    assert csv.exists()
    assert summary.exists()


def test_run_with_bad_sizing_errors() -> None:
    result = runner.invoke(app, ["run", "momentum", "--sizing", "nope"])
    assert result.exit_code == 1


def test_portfolio_command_runs() -> None:
    result = runner.invoke(app, ["portfolio", "momentum", "--assets", "3"])
    assert result.exit_code == 0
    assert "Portfolio" in result.stdout
