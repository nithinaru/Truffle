"""Row, sign, scale, and unit contracts for typed sensitivities."""

from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from core.compiler import compile_spec
from core.exceptions import DualsUnavailableError
from core.ir import (
    Box,
    Budget,
    Cardinality,
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    MaxSharpe,
    MeanVariance,
    MinVariance,
    PortfolioSpec,
    RiskParity,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)
from core.sensitivity import (
    harvest_sensitivities,
    sensitivity_coverage,
    sensitivity_dependency_reasons,
)
from core.solve import solve_spec


class _Dual:
    def __init__(self, value: object) -> None:
        self.dual_value = value


def _compiled(
    spec: PortfolioSpec,
    duals: dict[str, object],
    *,
    status: str | None = "optimal",
    raw_weights: np.ndarray | None = None,
) -> object:
    return SimpleNamespace(
        spec=spec,
        problem=SimpleNamespace(status=status),
        constraint_objs={key: _Dual(value) for key, value in duals.items()},
        weights=SimpleNamespace(value=raw_weights),
    )


def test_rows_preserve_solver_order_sign_units_and_user_primal_values() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.10, upper=0.80, tickers=["B", "A"]),
            GroupCap(id="group", group="Tech", min_weight=0.20, max_weight=0.50),
            FactorExposure(id="factor", factor="value", min_exposure=-0.50, max_exposure=0.30),
            TurnoverCap(id="turnover", max_turnover=0.30),
            CVaRLimit(id="cvar", alpha=0.50, max_cvar=0.20),
            TrackingErrorCap(id="te", benchmark="bench", max_te=0.50),
        ],
    )
    compiled = _compiled(
        spec,
        {
            "budget": -2.0,
            "long": np.array([0.0, 0.1]),
            # Standard Box compiler order: every lower, then every upper.
            "box": np.array([1.0, 2.0, 3.0, 4.0]),
            # Two-sided GroupCap / FactorExposure: upper, then lower.
            "group": np.array([5.0, 6.0]),
            "factor": np.array([7.0, 8.0]),
            "turnover": 9.0,
            "cvar": 10.0,
            "te": 11.0,
        },
    )
    weights = np.array([0.40, 0.60])
    records = harvest_sensitivities(
        compiled,
        weights,
        np.array([0.50, 0.50]),
        np.eye(2),
        scenarios=np.array([[-0.10, -0.20], [0.10, 0.20]]),
        sectors={"A": "Tech", "B": "Energy"},
        benchmarks={"bench": np.array([0.50, 0.50])},
        factors={"value": np.array([1.0, -1.0])},
    )

    by_id: dict[str, list] = {}
    for record in records:
        by_id.setdefault(record.constraint_id, []).append(record)

    assert [(r.row_label, r.side) for r in by_id["box"]] == [
        ("B", "lower"),
        ("A", "lower"),
        ("B", "upper"),
        ("A", "upper"),
    ]
    assert [r.raw_solver_dual for r in by_id["box"]] == [1.0, 2.0, 3.0, 4.0]
    assert [r.objective_derivative_per_bound_unit for r in by_id["box"]] == [
        1.0,
        2.0,
        -3.0,
        -4.0,
    ]
    assert [(r.side, r.objective_derivative_per_bound_unit) for r in by_id["group"]] == [
        ("upper", -5.0),
        ("lower", 6.0),
    ]
    assert [(r.side, r.objective_derivative_per_bound_unit) for r in by_id["factor"]] == [
        ("upper", -7.0),
        ("lower", 8.0),
    ]
    assert by_id["budget"][0].objective_derivative_per_bound_unit == 2.0
    assert by_id["budget"][0].bound_unit == "portfolio_weight_fraction"
    assert by_id["turnover"][0].bound_unit == "l1_weight_fraction"
    assert by_id["turnover"][0].primal_value == pytest.approx(0.20)
    assert by_id["cvar"][0].bound_unit == "scenario_loss_fraction"
    assert by_id["cvar"][0].primal_value == pytest.approx(0.16)
    assert by_id["te"][0].bound_unit == "annualized_volatility_fraction"
    assert by_id["te"][0].primal_value == pytest.approx(np.sqrt(0.02))
    assert all(record.objective_unit == "annualized_variance" for record in records)


def test_binding_is_primal_slack_not_nonzero_dual() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Box(id="box", lower=0.0, upper=0.8)],
    )
    records = harvest_sensitivities(
        _compiled(spec, {"box": np.array([99.0, 0.0, 0.0, 0.0])}),
        np.array([0.4, 0.8]),
        np.zeros(2),
        np.eye(2),
    )

    # A's lower row has a huge synthetic dual but 40% primal slack.
    assert records[0].raw_solver_dual == 99.0
    assert records[0].is_binding is False
    # B's upper row is exactly at its bound despite a zero dual.
    assert records[3].raw_solver_dual == 0.0
    assert records[3].slack == pytest.approx(0.0)
    assert records[3].is_binding is True
    assert records[0].shadow_price == 99.0


def test_max_sharpe_uses_transformed_order_and_kappa_scale() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MaxSharpe(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.10, upper=0.70),
        ],
    )
    compiled = _compiled(
        spec,
        {
            # Budget is implicit and intentionally absent.
            "long": np.array([1.0, 2.0]),
            # Max-Sharpe compiler order: upper A/B, then lower A/B.
            "box": np.array([3.0, 4.0, 5.0, 6.0]),
        },
        raw_weights=np.array([2.0, 3.0]),
    )
    records = harvest_sensitivities(
        compiled,
        np.array([0.40, 0.60]),
        np.zeros(2),
        np.eye(2),
    )
    box = [record for record in records if record.constraint_id == "box"]

    assert [(r.row_label, r.side) for r in box] == [
        ("A", "upper"),
        ("B", "upper"),
        ("A", "lower"),
        ("B", "lower"),
    ]
    assert [r.parameter_scale for r in box] == [5.0] * 4
    assert [r.objective_derivative_per_bound_unit for r in box] == [
        -15.0,
        -20.0,
        25.0,
        30.0,
    ]
    assert [r.primal_value for r in box] == [0.40, 0.60, 0.40, 0.60]
    assert all(r.objective_unit == "inverse_sharpe_squared" for r in records)
    long_only = [record for record in records if record.constraint_id == "long"]
    assert [record.parameter_scale for record in long_only] == [5.0, 5.0]


@pytest.mark.parametrize(
    ("dual", "message"),
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "dual shape"),
        (np.array([1.0, 2.0, 3.0, np.nan]), "non-finite"),
    ],
)
def test_dual_shape_and_finiteness_are_strict(dual: np.ndarray, message: str) -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Box(id="box", lower=0.0, upper=1.0)],
    )
    with pytest.raises(DualsUnavailableError, match=message):
        harvest_sensitivities(
            _compiled(spec, {"box": dual}),
            np.array([0.5, 0.5]),
            np.zeros(2),
            np.eye(2),
        )


def test_unsolved_or_nonoptimal_problem_is_rejected() -> None:
    spec = PortfolioSpec(
        universe=["A"],
        objective=MinVariance(),
        constraints=[LongOnly(id="long")],
    )
    with pytest.raises(DualsUnavailableError, match="not been solved"):
        harvest_sensitivities(
            _compiled(spec, {"long": np.array([0.0])}, status=None),
            np.ones(1),
            np.zeros(1),
            np.eye(1),
        )
    with pytest.raises(DualsUnavailableError, match="optimal continuous solve"):
        harvest_sensitivities(
            _compiled(spec, {"long": np.array([0.0])}, status="user_limit"),
            np.ones(1),
            np.zeros(1),
            np.eye(1),
        )
    with pytest.raises(DualsUnavailableError, match="require solver status 'optimal'"):
        harvest_sensitivities(
            _compiled(spec, {"long": np.array([0.0])}, status="optimal_inaccurate"),
            np.ones(1),
            np.zeros(1),
            np.eye(1),
            conditional=True,
        )


def test_dependent_active_rows_are_suppressed_as_a_complete_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LongOnly and Box(lower=0) must not expose an arbitrary multiplier split."""

    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MeanVariance(risk_aversion=15.0),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=2.0),
        ],
    )
    mu = np.array([0.20, 0.0])
    sigma = np.eye(2)
    compiled = compile_spec(spec, mu, sigma, w_prev=np.zeros(2))
    compiled.problem.solve(solver=cp.CLARABEL)

    # B's two lower rows are identical and Clarabel splits their multiplier.
    assert compiled.constraint_objs["long"].dual_value[1] > 0.1
    assert compiled.constraint_objs["box"].dual_value[1] > 0.1
    reasons = sensitivity_dependency_reasons(compiled)
    assert set(reasons) == {"long", "box"}
    assert all("non-identifiable" in reason for reason in reasons.values())

    records = harvest_sensitivities(
        compiled,
        compiled.recovered_weights(),
        np.zeros(2),
        sigma,
    )
    assert {record.constraint_id for record in records} == {"budget"}

    prices = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [100.0, 100.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    monkeypatch.setattr(
        "core.solve.estimate_moments",
        lambda panel, periods_per_year=252: (mu, sigma),
    )
    _, report = solve_spec(spec, prices)
    assert report.sensitivity_coverage["budget"].availability == "available"
    for constraint_id in ("long", "box"):
        item = report.sensitivity_coverage[constraint_id]
        assert item.availability == "unavailable"
        assert "non-identifiable" in (item.reason or "")


def test_dependency_check_uses_max_sharpe_transformed_variable_gradients() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MaxSharpe(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=2.0),
        ],
    )
    compiled = compile_spec(spec, np.array([0.20, -0.10]), np.eye(2))
    compiled.problem.solve(solver=cp.CLARABEL)

    # The duplicate B lower rows are represented in homogenized y-space. The
    # Jacobian detector reads those compiled expressions instead of pretending
    # their normals live in recovered portfolio-weight space.
    assert set(sensitivity_dependency_reasons(compiled)) == {"long", "box"}
    assert harvest_sensitivities(
        compiled,
        compiled.recovered_weights(),
        np.zeros(2),
        np.eye(2),
    ) == ()


def test_optimal_inaccurate_continuous_report_withholds_all_sensitivities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(id="budget"), LongOnly(id="long")],
    )
    prices = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [100.0, 99.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    monkeypatch.setattr(
        "core.solve.estimate_moments",
        lambda panel, periods_per_year=252: (np.zeros(2), np.eye(2)),
    )

    def inaccurate_solve(
        problem: cp.Problem,
        cp_solver: str,
        name: str,
        *,
        time_limit_s: float | None = None,
    ) -> float:
        del name, time_limit_s
        problem.solve(solver=cp_solver)
        problem._status = "optimal_inaccurate"
        return 1.0

    monkeypatch.setattr("core.solve._run_solver", inaccurate_solve)

    _, report = solve_spec(spec, prices)

    assert report.status == "optimal_inaccurate"
    assert report.termination_reason == "optimal_inaccurate"
    assert report.optimality_proven is False
    assert report.sensitivities == ()
    assert report.sensitivity_note is not None
    assert "exact 'optimal' status" in report.sensitivity_note
    assert all(
        item.availability == "unavailable"
        and "exact 'optimal' status" in (item.reason or "")
        for item in report.sensitivity_coverage.values()
    )


def test_coverage_names_penalties_mip_rows_and_transformed_implicit_rows() -> None:
    mip_spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            TransactionCost(id="cost", bps=5.0),
            Cardinality(id="names", max_names=1),
        ],
    )
    records = harvest_sensitivities(
        _compiled(mip_spec, {"budget": 0.0, "long": np.zeros(2)}),
        np.array([0.5, 0.5]),
        np.zeros(2),
        np.eye(2),
        conditional=True,
    )
    coverage = sensitivity_coverage(mip_spec, records, conditional=True)

    assert coverage["budget"].availability == "conditional"
    assert "fixed selected name set" in (coverage["budget"].reason or "")
    assert coverage["cost"].availability == "unavailable"
    assert "objective penalty" in (coverage["cost"].reason or "")
    assert coverage["names"].availability == "unavailable"
    assert "mixed-integer" in (coverage["names"].reason or "")

    sharpe_spec = PortfolioSpec(
        universe=["A"],
        objective=MaxSharpe(),
        constraints=[Budget(id="budget"), LongOnly(id="long")],
    )
    sharpe_records = harvest_sensitivities(
        _compiled(
            sharpe_spec,
            {"long": np.zeros(1)},
            raw_weights=np.ones(1),
        ),
        np.ones(1),
        np.zeros(1),
        np.eye(1),
    )
    sharpe_coverage = sensitivity_coverage(sharpe_spec, sharpe_records, conditional=False)
    assert sharpe_coverage["budget"].availability == "unavailable"
    assert "implicit" in (sharpe_coverage["budget"].reason or "")

    parity_spec = PortfolioSpec(
        universe=["A"],
        objective=RiskParity(),
        constraints=[Budget(id="budget"), LongOnly(id="long")],
    )
    parity_coverage = sensitivity_coverage(parity_spec, (), conditional=False)
    assert parity_coverage["budget"].availability == "unavailable"
    assert parity_coverage["long"].availability == "unavailable"

    implicit_parity = PortfolioSpec(universe=["A"], objective=RiskParity())
    implicit_coverage = sensitivity_coverage(implicit_parity, (), conditional=False)
    assert implicit_coverage["__implicit_unit_budget__"].availability == "unavailable"
    assert implicit_coverage["__implicit_positive_domain__"].availability == "unavailable"


def test_transaction_cost_marks_composite_objective_units() -> None:
    spec = PortfolioSpec(
        universe=["A"],
        objective=MinVariance(),
        constraints=[LongOnly(id="long"), TransactionCost(id="cost", bps=1.0)],
    )
    records = harvest_sensitivities(
        _compiled(spec, {"long": np.zeros(1)}),
        np.ones(1),
        np.zeros(1),
        np.eye(1),
    )
    assert records[0].objective_unit == "composite_objective_score"
