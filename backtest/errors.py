"""Typed failures from deterministic historical evaluation."""


class BacktestError(Exception):
    """Raised when a backtest cannot produce an honest, complete result."""
