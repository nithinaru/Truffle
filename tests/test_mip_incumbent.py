"""Fail-closed handling for mixed-integer termination and incumbents."""

from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from core.compiler import compile_spec
from core.exceptions import SolverError
from core.ir import Budget, Cardinality, LongOnly, MinVariance, PortfolioSpec
from core.solve import (
    _mip_termination,
    _MipTermination,
    _optimality_gap,
    _run_solver,
    _validate_mip_incumbent,
    _validate_time_limit,
    solve_spec,
)


class _ScipModel:
    def __init__(self, *, status: str = "timelimit", gap: object = 0.25) -> None:
        self._status = status
        self._gap = gap

    def getStatus(self) -> str:  # noqa: N802 - mirrors PySCIPOpt
        return self._status

    def getGap(self) -> object:  # noqa: N802 - mirrors PySCIPOpt
        return self._gap

    def infinity(self) -> float:
        return 1e20


def _problem(status: str, extra_stats: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        solver_stats=SimpleNamespace(extra_stats=extra_stats),
    )


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_time_limit_must_be_a_finite_positive_real(value: object) -> None:
    with pytest.raises(SolverError, match="finite positive"):
        _validate_time_limit(value)  # type: ignore[arg-type]
    assert _validate_time_limit(0.25) == 0.25


def test_run_solver_forwards_backend_specific_time_limit() -> None:
    class _Problem:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def solve(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    highs = _Problem()
    _run_solver(highs, cp.HIGHS, "HiGHS", time_limit_s=1.25)  # type: ignore[arg-type]
    assert highs.calls == [{"solver": cp.HIGHS, "time_limit": 1.25}]

    scip = _Problem()
    _run_solver(scip, cp.SCIP, "SCIP", time_limit_s=2.5)  # type: ignore[arg-type]
    assert scip.calls == [
        {"solver": cp.SCIP, "scip_params": {"limits/time": 2.5}}
    ]


def test_backend_verified_time_limit_mapping() -> None:
    highs = _problem("user_limit", SimpleNamespace(mip_gap=0.4))
    stop = _mip_termination(highs, cp_solver=cp.HIGHS, time_limit_requested=True)
    assert stop.reason == "time_limit"
    assert stop.optimality_proven is False

    scip = _problem(
        "optimal",
        {"scip_status": "timelimit", "model": _ScipModel()},
    )
    stop = _mip_termination(scip, cp_solver=cp.SCIP, time_limit_requested=True)
    assert stop.reason == "time_limit"
    assert stop.optimality_proven is False


@pytest.mark.parametrize(
    ("problem", "solver", "requested"),
    [
        (_problem("user_limit", SimpleNamespace(mip_gap=0.4)), cp.HIGHS, False),
        (_problem("optimal_inaccurate", {}), cp.SCIP, True),
        (
            _problem("optimal_inaccurate", {"scip_status": "gaplimit"}),
            cp.SCIP,
            True,
        ),
        (_problem("optimal", {"scip_status": "gaplimit"}), cp.SCIP, True),
    ],
)
def test_unverified_mip_stop_is_rejected(
    problem: SimpleNamespace, solver: str, requested: bool
) -> None:
    with pytest.raises(SolverError, match="unverified termination"):
        _mip_termination(
            problem,  # type: ignore[arg-type]
            cp_solver=solver,
            time_limit_requested=requested,
        )


def test_gap_is_actual_or_inferred_only_from_proof() -> None:
    limited = _MipTermination(reason="time_limit", optimality_proven=False)
    proven = _MipTermination(reason="optimal", optimality_proven=True)
    assert _optimality_gap(
        _problem("user_limit", SimpleNamespace(mip_gap=0.375)),  # type: ignore[arg-type]
        limited,
    ) == pytest.approx(0.375)
    assert _optimality_gap(
        _problem("optimal", None),  # type: ignore[arg-type]
        proven,
    ) == 0.0
    with pytest.raises(SolverError, match="no finite backend optimality gap"):
        _optimality_gap(
            _problem("user_limit", None),  # type: ignore[arg-type]
            limited,
        )
    with pytest.raises(SolverError, match="invalid relative"):
        _optimality_gap(
            _problem("user_limit", SimpleNamespace(mip_gap=float("inf"))),  # type: ignore[arg-type]
            limited,
        )
    with pytest.raises(SolverError, match="infinity sentinel"):
        _optimality_gap(
            _problem(
                "optimal_inaccurate",
                {"scip_status": "timelimit", "model": _ScipModel(gap=1e20)},
            ),  # type: ignore[arg-type]
            limited,
        )


def _compiled_incumbent() -> object:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Cardinality(max_names=1)],
    )
    compiled = compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2))
    compiled.weights.value = np.array([1.0, 0.0])
    compiled.extra_vars["y"].value = np.array([1.0, 0.0])
    compiled.problem._status = "optimal"
    compiled.problem._value = float(compiled.problem.objective.value)
    return compiled


def test_complete_feasible_mip_incumbent_is_validated() -> None:
    compiled = _compiled_incumbent()
    validated = _validate_mip_incumbent(compiled)  # type: ignore[arg-type]
    np.testing.assert_array_equal(validated.weights, [1.0, 0.0])
    assert validated.selected_indices == (0,)


def test_fractional_or_constraint_violating_mip_primal_is_rejected() -> None:
    fractional = _compiled_incumbent()
    # Solver adapters populate leaf values below CVXPY's public assignment
    # validation, so emulate a malformed backend primal through save_value.
    fractional.extra_vars["y"].save_value(np.array([0.75, 0.25]))
    with pytest.raises(SolverError, match="domain|binary"):
        _validate_mip_incumbent(fractional)

    infeasible = _compiled_incumbent()
    infeasible.weights.value = np.array([0.99, 0.0])
    infeasible.problem._value = float(infeasible.problem.objective.value)
    with pytest.raises(SolverError, match="without a feasible incumbent"):
        _validate_mip_incumbent(infeasible)


def test_missing_variable_or_nonfinite_objective_is_rejected() -> None:
    missing = _compiled_incumbent()
    missing.extra_vars["y"].value = None
    with pytest.raises(SolverError, match="did not populate variable"):
        _validate_mip_incumbent(missing)

    nonfinite = _compiled_incumbent()
    nonfinite.problem._value = float("nan")
    with pytest.raises(SolverError, match="non-finite incumbent objective"):
        _validate_mip_incumbent(nonfinite)

    inconsistent = _compiled_incumbent()
    inconsistent.problem._value = 123.0
    with pytest.raises(SolverError, match="inconsistent with the returned primal"):
        _validate_mip_incumbent(inconsistent)


def test_solve_spec_reports_only_a_validated_time_limit_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 99.0, 98.0, 97.0],
        },
        index=pd.date_range("2026-01-01", periods=4),
    )
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Cardinality(max_names=1)],
    )

    def fake_time_limited_solve(
        problem: cp.Problem,
        cp_solver: str,
        name: str,
        *,
        time_limit_s: float | None = None,
    ) -> float:
        assert cp_solver == cp.SCIP
        assert name == "SCIP"
        assert time_limit_s == 0.25
        for variable in problem.variables():
            if variable.name() == "w":
                variable.value = np.array([1.0, 0.0])
            elif variable.name() == "y":
                variable.value = np.array([1.0, 0.0])
        problem._status = "optimal_inaccurate"
        problem._value = float(problem.objective.value)
        problem._solver_stats = SimpleNamespace(
            extra_stats={
                "scip_status": "timelimit",
                "model": _ScipModel(gap=0.25),
            }
        )
        return 250.0

    monkeypatch.setattr("core.solve._run_solver", fake_time_limited_solve)

    _, report = solve_spec(spec, prices, time_limit_s=0.25)

    assert report.termination_reason == "time_limit"
    assert report.problem_class == "mip"
    assert report.duals_conditional is False
    assert report.optimality_proven is False
    assert report.incumbent_validated is True
    assert report.optimality_gap == pytest.approx(0.25)
    assert report.weights == {"A": 1.0, "B": 0.0}
    assert report.sensitivities == ()
    assert report.sensitivity_note is not None
    assert "time-limited incumbent" in report.sensitivity_note
