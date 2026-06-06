"""Unit tests for the export helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from backtesting_python.engine import run_backtest
from backtesting_python.export import (
    result_to_frame,
    summary_dict,
    write_equity_csv,
    write_summary_csv,
)


def _always_long(prices: Sequence[float]) -> list[int]:
    return [1] * len(list(prices))


def _result() -> object:
    return run_backtest([100.0, 110.0, 99.0, 120.0], _always_long, starting_equity=1000.0)


def test_result_to_frame_has_aligned_columns() -> None:
    frame = result_to_frame(_result())  # type: ignore[arg-type]
    assert list(frame.columns) == ["strategy_equity", "buy_hold_equity"]
    assert len(frame) == 4  # one row per price
    assert frame.index.name == "bar"


def test_write_equity_csv_roundtrips(tmp_path: Path) -> None:
    out = write_equity_csv(_result(), tmp_path / "equity.csv")  # type: ignore[arg-type]
    assert out.exists()
    loaded = pd.read_csv(out)
    assert "strategy_equity" in loaded.columns
    assert len(loaded) == 4
    assert loaded["strategy_equity"].iloc[0] == 1000.0


def test_write_equity_csv_creates_parent_dirs(tmp_path: Path) -> None:
    out = write_equity_csv(_result(), tmp_path / "nested" / "deep" / "e.csv")  # type: ignore[arg-type]
    assert out.exists()


def test_summary_dict_contains_key_metrics() -> None:
    summary = summary_dict(_result())  # type: ignore[arg-type]
    for key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "exposure"):
        assert key in summary
    assert summary["n_trades"] == 0


def test_write_summary_csv_is_single_row(tmp_path: Path) -> None:
    out = write_summary_csv(_result(), tmp_path / "summary.csv")  # type: ignore[arg-type]
    loaded = pd.read_csv(out)
    assert len(loaded) == 1
    assert "cagr" in loaded.columns
