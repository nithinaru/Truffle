"""Contract tests for the small local-data Python facade."""

from __future__ import annotations

import subprocess
import sys
import typing
from collections.abc import Mapping
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import truffle
import truffle.api as public_api
from backtest import BacktestConfig, BacktestError, run_backtest
from core.exceptions import InfeasibleError
from core.ir import PortfolioSpec
from core.report import SolutionReport
from core.solve import solve_spec


def _spec_mapping() -> dict[str, object]:
    return {
        "universe": ["A"],
        "objective": {"kind": "min_variance"},
        "constraints": [
            {"kind": "budget", "id": "budget", "total": 1.0},
            {"kind": "long_only", "id": "long"},
        ],
    }


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [100.0, 101.0, 100.5, 102.0, 103.0]},
        index=pd.to_datetime(
            ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
        ),
    )


def test_public_all_is_deliberate_and_resolves() -> None:
    expected = {
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
    }

    assert set(truffle.__all__) == expected
    assert all(hasattr(truffle, name) for name in truffle.__all__)
    assert truffle.ValidationError is ValidationError
    hints = typing.get_type_hints(SolutionReport)
    assert hints["sensitivities"] == tuple[truffle.SensitivityRecord, ...]


def test_solve_accepts_mapping_validates_it_and_matches_core() -> None:
    prices = _prices()
    mapping = _spec_mapping()
    validated = PortfolioSpec.model_validate(mapping)

    _, core_report = solve_spec(validated, prices)
    report = truffle.solve_portfolio(mapping, prices)

    assert isinstance(report, SolutionReport)
    assert report.objective_kind == core_report.objective_kind
    assert report.status == core_report.status
    assert report.solver == core_report.solver
    assert report.n_assets == core_report.n_assets
    assert report.schema_version == "2.0"
    assert report.problem_class == "convex"
    assert isinstance(report.objective_decomposition, truffle.ObjectiveDecomposition)
    assert all(isinstance(metric, truffle.PortfolioMetric) for metric in report.metrics)
    assert all(
        isinstance(record, truffle.SensitivityRecord) for record in report.sensitivities
    )
    np.testing.assert_allclose(report.objective_value, core_report.objective_value)
    np.testing.assert_allclose(
        list(report.weights.values()),
        list(core_report.weights.values()),
    )


def test_backtest_accepts_mapping_and_matches_underlying_engine() -> None:
    prices = _prices()
    mapping = _spec_mapping()
    validated = PortfolioSpec.model_validate(mapping)
    config = BacktestConfig(lookback_returns=2, rebalance_frequency="monthly")

    expected = run_backtest(validated, prices, config=config)
    actual = truffle.run_walk_forward_backtest(mapping, prices, config=config)

    assert actual == expected
    assert actual.rebalances[0].signal_date.isoformat() == "2024-01-31"
    assert actual.rebalances[0].fill_date.isoformat() == "2024-02-01"


def test_mapping_errors_are_pydantic_validation_errors() -> None:
    invalid = _spec_mapping()
    invalid["universe"] = ["A", "A"]

    with pytest.raises(ValidationError, match="Duplicate ticker"):
        truffle.solve_portfolio(invalid, _prices())


def test_existing_specs_are_revalidated_instead_of_blindly_trusted() -> None:
    spec = PortfolioSpec.model_validate(_spec_mapping())
    spec.universe.append("A")  # frozen models still contain mutable nested lists

    with pytest.raises(ValidationError, match="Duplicate ticker"):
        truffle.solve_portfolio(spec, _prices())


def test_natural_language_is_rejected_before_any_solver_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_solve(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("solver must not be called")

    monkeypatch.setattr(public_api, "_solve_spec", unexpected_solve)

    with pytest.raises(TypeError, match="PortfolioSpec or a mapping"):
        truffle.solve_portfolio("put everything in A", _prices())  # type: ignore[arg-type]
    assert called is False


@pytest.mark.parametrize("entrypoint", ["solve", "backtest"])
def test_typed_engine_failures_propagate_unchanged(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PortfolioSpec.model_validate(_spec_mapping())
    error: Exception
    if entrypoint == "solve":
        error = InfeasibleError("deliberate infeasibility")

        def fail(*_args: object, **_kwargs: object) -> object:
            raise error

        monkeypatch.setattr(public_api, "_solve_spec", fail)
        call = partial(truffle.solve_portfolio, spec, _prices())
    else:
        error = BacktestError("deliberate backtest failure")

        def fail(*_args: object, **_kwargs: object) -> object:
            raise error

        monkeypatch.setattr(public_api, "_run_backtest", fail)
        call = partial(truffle.run_walk_forward_backtest, spec, _prices())

    with pytest.raises(type(error)) as caught:
        call()
    assert caught.value is error


def test_import_has_no_agent_or_network_client_imports_or_connections() -> None:
    script = r"""
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError("public facade attempted a network connection during import")

socket.create_connection = forbidden
socket.socket.connect = forbidden
import truffle

blocked = ("agent", "anthropic", "httpx", "requests")
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in blocked)
)
if loaded:
    raise AssertionError(f"unexpected modules imported: {loaded}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_spec_input_alias_accepts_mapping_at_runtime() -> None:
    # Keep the documented input contract honest without relying on a type checker.
    assert isinstance(_spec_mapping(), Mapping)
