"""Typed local-data API for Truffle's solver and walk-forward backtester."""

from pydantic import ValidationError

from backtest.config import BacktestConfig
from backtest.errors import BacktestError
from backtest.tearsheet import Tearsheet
from core.exceptions import (
    CompilationError,
    DiagnosisError,
    InfeasibleError,
    SolverError,
    TruffleError,
    UnboundedError,
)
from core.ir import PortfolioSpec
from core.report import ConflictReport, SolutionReport
from core.report_semantics import ObjectiveDecomposition, ObjectiveTerm, PortfolioMetric
from core.sensitivity import SensitivityCoverage, SensitivityRecord
from truffle.api import SpecInput, run_walk_forward_backtest, solve_portfolio

__all__ = [
    "BacktestConfig",
    "BacktestError",
    "CompilationError",
    "ConflictReport",
    "DiagnosisError",
    "InfeasibleError",
    "ObjectiveDecomposition",
    "ObjectiveTerm",
    "PortfolioSpec",
    "PortfolioMetric",
    "SensitivityCoverage",
    "SensitivityRecord",
    "SolutionReport",
    "SolverError",
    "SpecInput",
    "Tearsheet",
    "TruffleError",
    "UnboundedError",
    "ValidationError",
    "run_walk_forward_backtest",
    "solve_portfolio",
]
