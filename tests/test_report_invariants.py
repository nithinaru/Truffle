"""Cross-field invariants for the versioned solution-report schema."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from core.report import BindingConstraint, SolutionReport
from core.sensitivity import SensitivityRecord


def _sensitivity(*, conditional: bool) -> SensitivityRecord:
    return SensitivityRecord(
        constraint_id="budget",
        kind="budget",
        row_index=0,
        row_label="portfolio",
        side="equality",
        bound_value=1.0,
        bound_unit="portfolio_weight_fraction",
        raw_solver_dual=0.02,
        parameter_scale=1.0,
        objective_derivative_per_bound_unit=-0.02,
        objective_unit="annualized_variance",
        primal_value=1.0,
        slack=0.0,
        is_binding=True,
        conditional=conditional,
    )


def _report(**overrides: object) -> SolutionReport:
    values: dict[str, object] = {
        "weights": {"AAA": 0.6, "BBB": 0.4},
        "objective_kind": "min_variance",
        "objective_value": 0.02,
        "solver": "Clarabel",
        "solve_time_ms": 4.0,
        "status": "optimal",
        "n_assets": 2,
        "nonzero_names": 2,
        "sensitivities": (_sensitivity(conditional=False),),
    }
    values.update(overrides)
    return SolutionReport(**values)  # type: ignore[arg-type]


def _mip_values(**overrides: object) -> Mapping[str, object]:
    values: dict[str, object] = {
        "solver": "SCIP",
        "problem_class": "mip",
        "selected_names": ["AAA", "BBB"],
        "optimality_gap": 0.0,
        "incumbent_validated": True,
        "duals_conditional": True,
        "sensitivities": (_sensitivity(conditional=True),),
    }
    values.update(overrides)
    return values


def test_valid_continuous_and_mip_report_states() -> None:
    continuous = _report()
    inaccurate = _report(
        status="optimal_inaccurate",
        termination_reason="optimal_inaccurate",
        optimality_proven=False,
    )
    mip = _report(**_mip_values())
    mip_without_sensitivities = _report(
        **_mip_values(duals_conditional=False, sensitivities=())
    )
    time_limited = _report(
        **_mip_values(
            status="user_limit",
            optimality_gap=0.15,
            duals_conditional=False,
            sensitivities=(),
            termination_reason="time_limit",
            optimality_proven=False,
        )
    )

    assert continuous.problem_class == "convex"
    assert inaccurate.termination_reason == "optimal_inaccurate"
    assert mip.duals_conditional
    assert not mip_without_sensitivities.duals_conditional
    assert time_limited.incumbent_validated


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"selected_names": ["AAA"]}, "convex report cannot contain selected_names"),
        ({"optimality_gap": 0.1}, "convex report cannot contain a MIP optimality_gap"),
        (
            {
                "duals_conditional": True,
                "sensitivities": (_sensitivity(conditional=True),),
            },
            "convex report cannot contain conditional sensitivities",
        ),
        (
            {
                "sensitivities": (
                    _sensitivity(conditional=False),
                    _sensitivity(conditional=True),
                ),
            },
            "cannot mix conditional and unconditional",
        ),
        (
            {
                "status": "user_limit",
                "sensitivities": (),
                "termination_reason": "time_limit",
                "optimality_proven": False,
            },
            "time_limit termination is supported only for MIP",
        ),
        (
            _mip_values(selected_names=None),
            "MIP report requires a non-empty selected_names",
        ),
        (_mip_values(optimality_gap=None), "MIP report requires a finite relative"),
        (_mip_values(incumbent_validated=False), "MIP report requires a validated"),
        (
            _mip_values(
                duals_conditional=False,
                sensitivities=(_sensitivity(conditional=False),),
            ),
            "MIP sensitivity records must be conditional",
        ),
        (
            _mip_values(
                binding=[BindingConstraint("budget", "the budget", 0.02)],
                duals_conditional=False,
                sensitivities=(),
            ),
            "cannot contain binding summaries without authoritative",
        ),
        (
            _mip_values(duals_conditional=False),
            "duals_conditional must be true exactly",
        ),
        (
            _mip_values(sensitivities=()),
            "duals_conditional must be true exactly",
        ),
        (
            _mip_values(
                termination_reason="optimal_inaccurate",
                optimality_proven=False,
            ),
            "not a supported MIP termination reason",
        ),
        (
            _mip_values(
                status="user_limit",
                termination_reason="time_limit",
                optimality_proven=False,
            ),
            "time-limit report cannot contain sensitivity data",
        ),
    ],
)
def test_incoherent_report_states_are_rejected(
    overrides: Mapping[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _report(**overrides)


@pytest.mark.parametrize(
    "selected_names",
    [[], ["AAA", "AAA"], ["AAA", "UNKNOWN"]],
)
def test_selected_names_must_be_a_nonempty_unique_subset(
    selected_names: list[str],
) -> None:
    with pytest.raises(ValueError, match="selected_names"):
        _report(**_mip_values(selected_names=selected_names))


@pytest.mark.parametrize(
    ("termination_reason", "optimality_proven"),
    [("optimal", False), ("optimal_inaccurate", True)],
)
def test_termination_reason_and_proof_flag_must_agree(
    termination_reason: str,
    optimality_proven: bool,
) -> None:
    with pytest.raises(ValueError, match="optimal"):
        _report(
            termination_reason=termination_reason,
            optimality_proven=optimality_proven,
        )
