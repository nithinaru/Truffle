"""Sprint 5 deterministic infeasibility diagnosis tests."""

from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

import core.diagnose as diagnosis
from core.diagnose import DiagnosisData, diagnose, elastic_solve, find_iis
from core.exceptions import DiagnosisError
from core.ir import (
    Box,
    Budget,
    Cardinality,
    GroupCap,
    LongOnly,
    MinVariance,
    PortfolioSpec,
)
from core.patch import apply_patch


def _data(n: int, *, sectors: dict[str, str] | None = None) -> DiagnosisData:
    return DiagnosisData(
        mu=np.zeros(n),
        sigma=np.eye(n),
        scenarios=None,
        w_prev=np.zeros(n),
        sectors=sectors,
        benchmark_weights=None,
        factor_loadings=None,
    )


def test_exact_iis_includes_structural_budget_and_verified_repair() -> None:
    universe = list("ABCDE")
    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=0.10),
        ],
    )
    data = _data(5)

    elastic = elastic_solve(spec, data)
    assert elastic.candidate_constraint_ids == ("box",)
    box_slack = next(s for s in elastic.slacks if s.constraint_id == "box")
    assert box_slack.raw_slack == pytest.approx(0.10, abs=1e-5)

    iis = find_iis(spec, data, elastic.candidate_constraint_ids)
    assert iis.verified
    assert set(iis.constraint_ids) == {"budget", "box"}

    report = diagnose(spec, data)
    assert report.minimality_status == "verified_iis"
    assert report.conflict_scope == "mixed"
    assert report.repairs
    repaired = apply_patch(spec, report.repairs[0].patch)
    replacement = next(c for c in repaired.constraints if c.id == "box")
    assert isinstance(replacement, Box)
    assert replacement.upper == pytest.approx(0.20)


def test_normalized_elasticity_exposes_red_herring_then_iis_removes_it() -> None:
    universe = [f"A{i}" for i in range(10)]
    sectors = {ticker: ("G" if i < 2 else "Other") for i, ticker in enumerate(universe)}
    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=0.0005),
            GroupCap(id="group", group="G", max_weight=0.01),
        ],
    )
    data = _data(10, sectors=sectors)

    elastic = elastic_solve(spec, data)
    assert set(elastic.candidate_constraint_ids) == {"box", "group"}
    assert elastic.total_relative_slack == pytest.approx(218.0, abs=2e-2)

    iis = find_iis(spec, data, elastic.candidate_constraint_ids)
    assert iis.verified
    assert set(iis.constraint_ids) == {"budget", "box"}


def test_safe_direction_rounding_produces_a_feasible_cap() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=0.30),
        ],
    )
    report = diagnose(spec, _data(3))
    repair = report.repairs[0]
    change = repair.changes[0]
    assert change.solver_required_value == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert change.applied_value == pytest.approx(0.34)

    repaired = apply_patch(spec, repair.patch)
    replacement = next(c for c in repaired.constraints if c.id == "box")
    assert isinstance(replacement, Box)
    assert replacement.upper == pytest.approx(0.34)


def test_group_cap_repair_snaps_solver_dust_to_domain_endpoint() -> None:
    node = GroupCap(id="group", group="G", max_weight=0.40)
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), node],
    )
    data = _data(2, sectors={"A": "G", "B": "G"})
    changed = diagnosis._changes_for_constraint(  # noqa: SLF001
        node,
        raw_slack=0.60,
        spec=spec,
        data=data,
        weights={"A": 0.5000000000000001, "B": 0.5000000000000001},
    )

    assert changed is not None
    replacement, changes, _ = changed
    assert isinstance(replacement, GroupCap)
    assert replacement.max_weight == 1.0
    assert changes[0].solver_required_value == 1.0
    assert changes[0].applied_value == 1.0


def test_user_locked_constraint_returns_hard_only_conflict() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=0.40, elastic=False),
        ],
    )
    report = diagnose(spec, _data(2))
    assert report.elastic.kind == "hard_infeasible"
    assert report.conflict_scope == "hard_only"
    assert set(member.constraint_id for member in report.conflict_set) == {"budget", "box"}
    assert report.repairs == ()


def test_cardinality_lower_rows_are_reported_as_structural() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Cardinality(
                id="card",
                max_names=2,
                min_names=2,
                min_position=0.60,
                elastic=False,
            ),
        ],
    )
    report = diagnose(spec, _data(3))
    card = next(member for member in report.conflict_set if member.constraint_id == "card")
    assert card.relaxability == "structural"
    assert report.repairs == ()


@pytest.mark.skipif(
    cp.SCIP not in cp.installed_solvers(), reason="SCIP is needed to verify MIQP repairs"
)
def test_mip_cardinality_group_conflict_has_verified_repairs() -> None:
    universe = ["A", "B", "C"]
    sectors = {ticker: ticker for ticker in universe}
    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            GroupCap(id="g_a", group="A", max_weight=0.40),
            GroupCap(id="g_b", group="B", max_weight=0.40),
            GroupCap(id="g_c", group="C", max_weight=0.40),
            Cardinality(id="card", max_names=2),
        ],
    )
    report = diagnose(spec, _data(3, sectors=sectors))
    assert report.minimality_status == "verified_iis"
    assert set(member.constraint_id for member in report.conflict_set) == {
        "budget",
        "g_a",
        "g_b",
        "g_c",
        "card",
    }
    assert report.elastic.solver == "HiGHS"
    assert report.repairs
    assert any(
        any(
            change.constraint_id == "card" and change.applied_value == 3
            for change in repair.changes
        )
        for repair in report.repairs
    )


def test_elastic_solve_rejects_a_feasible_spec() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Box(lower=0.0, upper=0.60)],
    )
    with pytest.raises(DiagnosisError, match="zero violation"):
        elastic_solve(spec, _data(2))


def test_budget_box_evidence_describes_lower_floor_excess() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.60, upper=1.0),
        ],
    )
    report = diagnose(spec, _data(2))
    evidence = next(item for item in report.evidence if "minimum total" in item.text)
    assert "120%" in evidence.text
    assert "100% budget" in evidence.text
    assert {value.key for value in evidence.values} == {
        "covered_names",
        "position_floor",
        "minimum_total",
        "budget_total",
    }


def test_optimal_inaccurate_is_not_feasibility_or_repair_proof(monkeypatch) -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly()],
    )
    fake_problem = SimpleNamespace(status="optimal_inaccurate", solve=lambda **_kwargs: None)
    fake_built = SimpleNamespace(
        problem=fake_problem,
        spec=spec,
        cardinality_domain_valid=True,
    )
    monkeypatch.setattr(diagnosis, "_build_diagnostic_problem", lambda *_a, **_k: fake_built)
    monkeypatch.setattr(diagnosis, "_run", lambda *_a, **_k: ("Clarabel", 0.0))
    assert diagnosis._feasibility_status(spec, _data(2), []) == "unknown"  # noqa: SLF001

    monkeypatch.setattr(
        diagnosis,
        "_compile_original",
        lambda *_a, **_k: SimpleNamespace(problem=fake_problem),
    )
    monkeypatch.setattr(
        diagnosis,
        "select_solver",
        lambda *_a, **_k: SimpleNamespace(cp_solver=cp.CLARABEL),
    )
    assert diagnosis._verified_feasible(spec, _data(2)) is False  # noqa: SLF001
