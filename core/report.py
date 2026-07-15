"""Versioned, typed solver output ready for deterministic or LLM narration.

``SolutionReport`` is the explanation layer's complete numeric trust surface.
Schema 2.0 deliberately separates the raw solver score, financial metrics, and
row-aware constraint sensitivities:

* ``objective_decomposition`` records the mathematical terms that form the
  minimized score, including transforms and transaction-cost penalties;
* ``metrics`` carries financial definitions, units, horizons, and objective-
  appropriate values; and
* ``sensitivities`` preserves constraint row, side, dual sign, transform scale,
  primal slack, conditionality, and compound units.

``objective_value``, ``var``, and ``binding`` remain compatibility fields.
``binding`` is now derived only from rows that are both primally active and
materially sensitive; it must not replace the authoritative row records.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.patch import SpecPatch
from core.report_semantics import ObjectiveDecomposition, PortfolioMetric
from core.sensitivity import SensitivityCoverage, SensitivityRecord

BINDING_THRESHOLD = 1e-6


class _DiagnosticModel(BaseModel):
    """Frozen, JSON-serializable base for infeasibility reports."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GroundValue(_DiagnosticModel):
    """One trusted numeric fact available to conflict narration."""

    key: str
    value: float
    unit: Literal["raw", "fraction", "bps", "count", "milliseconds"] = "raw"
    source: Literal["solver", "spec", "derived"] = "solver"


class ConflictEvidence(_DiagnosticModel):
    """Deterministic explanation fragment and the values that support it."""

    text: str
    values: tuple[GroundValue, ...] = ()


class ConstraintSlack(_DiagnosticModel):
    """Normalized elastic violation for one IR constraint."""

    constraint_id: str
    human_name: str
    raw_slack: float
    slack_scale: float
    relative_slack: float


class ElasticResult(_DiagnosticModel):
    """Result of the normalized elastic feasibility pass."""

    kind: Literal["soft_repair", "hard_infeasible"]
    status: str
    solver: str
    solve_time_ms: float
    total_relative_slack: float | None = None
    slacks: tuple[ConstraintSlack, ...] = ()
    candidate_constraint_ids: tuple[str, ...] = ()
    repaired_weights: dict[str, float] | None = None


class IISResult(_DiagnosticModel):
    """A node-level irreducible infeasible subsystem search result."""

    constraint_ids: tuple[str, ...]
    verified: bool
    checks: int
    fallback_reason: str | None = None


class ConflictMember(_DiagnosticModel):
    """One constraint named in a verified or candidate conflict set."""

    constraint_id: str
    constraint_kind: str
    human_name: str
    relaxability: Literal["relaxable", "structural", "user_locked"]
    required_slack: float | None = None
    slack_scale: float | None = None
    relative_slack: float | None = None
    parameters: tuple[GroundValue, ...] = ()


class RepairChange(_DiagnosticModel):
    """One solver-derived field change applied by a repair patch."""

    constraint_id: str
    field: str
    direction: Literal["raise", "lower"]
    old_value: float
    solver_required_value: float
    applied_value: float
    required_change: float
    normalized_change: float
    unit: Literal["raw", "fraction", "bps", "count"]


class Repair(_DiagnosticModel):
    """A deterministic, verified amendment that restores feasibility."""

    repair_id: str
    description: str
    patch: SpecPatch
    changes: tuple[RepairChange, ...]
    required_change: float | None = None
    relative_change: float
    kind: Literal["single_lever", "joint"]
    rank: int = Field(ge=1)
    verified: Literal[True] = True


class ConflictReport(_DiagnosticModel):
    """The only numeric surface the infeasibility explainer may read."""

    kind: Literal["conflict_report"] = "conflict_report"
    solver_status: str
    n_assets: int = Field(ge=1)
    minimality_status: Literal["verified_iis", "unverified_candidate"]
    conflict_scope: Literal["soft_only", "mixed", "hard_only"]
    candidate_constraint_ids: tuple[str, ...]
    conflict_set: tuple[ConflictMember, ...]
    elastic: ElasticResult
    evidence: tuple[ConflictEvidence, ...] = ()
    repairs: tuple[Repair, ...] = ()


@dataclass(frozen=True)
class BindingConstraint:
    """Deprecated impact summary retained for API compatibility.

    Row-aware consumers should use :attr:`SolutionReport.sensitivities`. The
    optional row fields prevent this adapter from erasing which side and unit
    produced the summarized shadow price.
    """

    constraint_id: str
    human_name: str
    shadow_price: float
    row_label: str | None = None
    side: Literal["lower", "upper", "equality"] | None = None
    sensitivity_unit: str | None = None


@dataclass(frozen=True)
class SolutionReport:
    """Everything the narration layer is allowed to reference.

    Attributes:
        weights: ``{ticker -> weight}`` in canonical universe order.
        objective_kind: The IR objective ``kind`` discriminator
            (``"min_variance"`` / ``"mean_variance"`` / ``"min_cvar"``).
        objective_value: Compatibility field containing the raw minimized
            solver score. Use ``objective_decomposition`` and ``metrics`` for
            meaning; this value can include penalties or a transformed score.
        var: Optimal ``t`` for ``min_cvar``; ``None`` otherwise. Reported
            so the explanation can say "VaR(α=0.95) = X" without re-running
            the solver.
        solver: Solver name (e.g. ``"Clarabel"``).
        solve_time_ms: Wall-clock solve time in milliseconds.
        status: CVXPY status string (``"optimal"`` /
            ``"optimal_inaccurate"`` after a successful run).
        binding: Compatibility summary of materially sensitive, primally active
            rows, sorted by derivative magnitude. ``sensitivities`` is the
            authoritative signed and unit-aware surface.
        n_assets: Universe size; useful for sanity-checking "k of n names".
        nonzero_names: Tickers with weight magnitude above a small floor. This
            is a position count, not the mixed-integer selected-set count.
        duals_conditional: Compatibility flag set only when reported
            sensitivities come from a MIP fix-and-resolve restriction. Use
            ``problem_class`` to identify MIPs that have no sensitivities.
        selected_names: For a mixed-integer (cardinality) solve, the names the
            integer program selected. ``None`` on the continuous path.
        optimality_gap: Backend-reported relative MIP gap. Zero is inferred only
            from a proven optimum when the backend omits the statistic.
        objective_decomposition: Typed reconstruction of the solver score.
        metrics: Objective-appropriate portfolio statistics with units.
        sensitivities: One signed record per named constraint row.
        sensitivity_coverage: Per-constraint availability and reason.
        termination_reason: Proven optimum or verified time-limit stop.
        optimality_proven: Whether the MIP backend proved optimality.
        incumbent_validated: Whether a complete MIP primal passed validation.
        problem_class: Continuous-convex or mixed-integer solve path.
    """

    weights: dict[str, float]
    objective_kind: str
    objective_value: float
    solver: str
    solve_time_ms: float
    status: str
    n_assets: int
    nonzero_names: int
    var: float | None = None
    binding: list[BindingConstraint] = field(default_factory=list)
    duals_conditional: bool = False
    selected_names: list[str] | None = None
    optimality_gap: float | None = None
    objective_decomposition: ObjectiveDecomposition | None = None
    metrics: tuple[PortfolioMetric, ...] = ()
    sensitivities: tuple[SensitivityRecord, ...] = ()
    sensitivity_coverage: dict[str, SensitivityCoverage] = field(default_factory=dict)
    sensitivity_note: str | None = None
    termination_reason: Literal["optimal", "optimal_inaccurate", "time_limit"] = "optimal"
    optimality_proven: bool = True
    incumbent_validated: bool = False
    problem_class: Literal["convex", "mip"] = "convex"
    schema_version: Literal["2.0"] = "2.0"

    def __post_init__(self) -> None:
        if self.optimality_gap is not None and (
            not math.isfinite(self.optimality_gap) or self.optimality_gap < 0.0
        ):
            raise ValueError("optimality_gap must be a finite non-negative relative fraction.")

        if self.selected_names is not None:
            if not self.selected_names:
                raise ValueError("selected_names must be non-empty when reported.")
            if len(set(self.selected_names)) != len(self.selected_names):
                raise ValueError("selected_names must not contain duplicates.")
            unknown_names = set(self.selected_names).difference(self.weights)
            if unknown_names:
                raise ValueError(
                    "selected_names must be drawn from the report's weight universe."
                )

        sensitivity_modes = {record.conditional for record in self.sensitivities}
        if len(sensitivity_modes) > 1:
            raise ValueError(
                "A report cannot mix conditional and unconditional sensitivity records."
            )
        records_are_conditional = sensitivity_modes == {True}
        if self.duals_conditional != records_are_conditional:
            raise ValueError(
                "duals_conditional must be true exactly when the report contains "
                "conditional sensitivity records."
            )

        if self.termination_reason == "optimal" and not self.optimality_proven:
            raise ValueError("An optimal report must have optimality_proven=True.")
        if self.termination_reason != "optimal" and self.optimality_proven:
            raise ValueError(
                "optimal_inaccurate and time_limit reports require optimality_proven=False."
            )

        if self.problem_class == "convex":
            if self.selected_names is not None:
                raise ValueError("A convex report cannot contain selected_names.")
            if self.optimality_gap is not None:
                raise ValueError("A convex report cannot contain a MIP optimality_gap.")
            if self.incumbent_validated:
                raise ValueError("incumbent_validated applies only to MIP reports.")
            if records_are_conditional:
                raise ValueError("A convex report cannot contain conditional sensitivities.")
            if self.termination_reason == "time_limit":
                raise ValueError("time_limit termination is supported only for MIP reports.")
            return

        if self.selected_names is None:
            raise ValueError("A MIP report requires a non-empty selected_names set.")
        if self.optimality_gap is None:
            raise ValueError("A MIP report requires a finite relative optimality_gap.")
        if not self.incumbent_validated:
            raise ValueError("A MIP report requires a validated incumbent.")
        if self.termination_reason == "optimal_inaccurate":
            raise ValueError("optimal_inaccurate is not a supported MIP termination reason.")
        if self.sensitivities and not records_are_conditional:
            raise ValueError("MIP sensitivity records must be conditional.")
        if self.binding and not self.sensitivities:
            raise ValueError(
                "A MIP report cannot contain binding summaries without authoritative "
                "conditional sensitivity records."
            )
        if self.termination_reason == "time_limit" and (
            self.sensitivities or self.binding
        ):
            raise ValueError(
                "A time-limit report cannot contain sensitivity data because "
                "fix-and-resolve could optimize a different portfolio."
            )

    def metric(self, key: str) -> PortfolioMetric | None:
        """Return one typed portfolio metric by stable key."""

        return next((metric for metric in self.metrics if metric.key == key), None)

    def to_dict(self) -> dict[str, object]:
        """Return the complete deterministic, JSON-serializable report."""

        payload = asdict(self)
        payload["field_units"] = {
            "weights": "portfolio_weight_fraction",
            "objective_value": "objective_score",
            "solve_time_ms": "milliseconds",
            "var": (
                "fraction_per_scenario_period" if self.var is not None else None
            ),
            "optimality_gap": (
                "relative_fraction" if self.optimality_gap is not None else None
            ),
            "n_assets": "count",
            "nonzero_names": "count",
        }
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the versioned report without a custom encoder."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_report(
    *,
    weights: dict[str, float],
    objective_kind: str,
    objective_value: float,
    solver: str,
    solve_time_ms: float,
    status: str,
    duals: dict[str, float],
    constraint_human_names: dict[str, str],
    var: float | None = None,
    nonzero_floor: float = 1e-4,
    duals_conditional: bool = False,
    selected_names: list[str] | None = None,
    optimality_gap: float | None = None,
    objective_decomposition: ObjectiveDecomposition | None = None,
    metrics: tuple[PortfolioMetric, ...] = (),
    sensitivities: tuple[SensitivityRecord, ...] = (),
    sensitivity_coverage: dict[str, SensitivityCoverage] | None = None,
    sensitivity_note: str | None = None,
    termination_reason: Literal["optimal", "optimal_inaccurate", "time_limit"] = "optimal",
    optimality_proven: bool = True,
    incumbent_validated: bool = False,
    problem_class: Literal["convex", "mip"] = "convex",
) -> SolutionReport:
    """Assemble a ``SolutionReport`` from solver outputs and the IR id map.

    Args:
        weights: Final weights keyed by ticker.
        objective_kind: One of ``"min_variance"`` / ``"mean_variance"`` /
            ``"min_cvar"``.
        objective_value: Solver objective value.
        solver: e.g. ``"Clarabel"``.
        solve_time_ms: Wall-clock solve time in milliseconds.
        status: CVXPY status string.
        duals: Output of :func:`core.duals.harvest_duals`, mapping IR
            constraint id to a scalar shadow-price magnitude.
        constraint_human_names: ``{ir_id -> human-readable phrase}``,
            e.g. ``{"cap_aaa": "the AAA position cap"}``. Used for
            narration. If an id is missing here, the id itself is used.
        var: Optimal ``t`` for ``min_cvar``, else ``None``.
        nonzero_floor: Weights below this magnitude are treated as zero
            for the ``nonzero_names`` count.
    """
    if sensitivities:
        binding = []
        for record in sensitivities:
            shadow_price = abs(record.objective_derivative_per_bound_unit)
            if not record.is_binding or shadow_price <= BINDING_THRESHOLD:
                continue
            sensitivity_unit = f"{record.objective_unit}_per_{record.bound_unit}"
            binding.append(
                BindingConstraint(
                    constraint_id=record.constraint_id,
                    human_name=constraint_human_names.get(
                        record.constraint_id, record.constraint_id
                    ),
                    shadow_price=shadow_price,
                    row_label=record.row_label,
                    side=record.side,
                    sensitivity_unit=sensitivity_unit,
                )
            )
    else:
        binding = [
            BindingConstraint(
                constraint_id=cid,
                human_name=constraint_human_names.get(cid, cid),
                shadow_price=val,
            )
            for cid, val in duals.items()
            if abs(val) > BINDING_THRESHOLD
        ]
    binding.sort(key=lambda b: -abs(b.shadow_price))
    nonzero_names = sum(1 for w in weights.values() if abs(w) > nonzero_floor)
    return SolutionReport(
        weights=dict(weights),
        objective_kind=objective_kind,
        objective_value=float(objective_value),
        var=None if var is None else float(var),
        solver=solver,
        solve_time_ms=float(solve_time_ms),
        status=status,
        binding=binding,
        n_assets=len(weights),
        nonzero_names=nonzero_names,
        duals_conditional=duals_conditional,
        selected_names=selected_names,
        optimality_gap=optimality_gap,
        objective_decomposition=objective_decomposition,
        metrics=tuple(metrics),
        sensitivities=tuple(sensitivities),
        sensitivity_coverage=dict(sensitivity_coverage or {}),
        sensitivity_note=sensitivity_note,
        termination_reason=termination_reason,
        optimality_proven=optimality_proven,
        incumbent_validated=incumbent_validated,
        problem_class=problem_class,
    )
