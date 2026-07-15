"""Semantic normalization for parse-result comparison.

Constraint IDs are compiler/reporting handles generated independently across
parses, so they are excluded.  Constraint ordering is also immaterial and is
canonicalized.  All other public fields remain in the comparison, including
universe order, objective parameters, patch operations, clarification text,
and current weights.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agent.schema import Clarification, FreshSpec, ParseResult, SpecPatch
from core.ir import Constraint, PortfolioSpec


def _json_key(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _constraint(constraint: Constraint) -> dict[str, Any]:
    return constraint.model_dump(mode="json", exclude={"id"})


def _constraints(constraints: list[Constraint]) -> list[dict[str, Any]]:
    return sorted((_constraint(constraint) for constraint in constraints), key=_json_key)


def _spec(spec: PortfolioSpec) -> dict[str, Any]:
    return {
        "universe": list(spec.universe),
        "objective": spec.objective.model_dump(mode="json"),
        "constraints": _constraints(spec.constraints),
        "current_weights": spec.current_weights,
    }


def _removed_constraints(
    patch: SpecPatch,
    current_spec: PortfolioSpec | None,
) -> list[dict[str, Any]]:
    """Resolve remove IDs to semantic constraints when context is available.

    This makes two generated IDs equivalent only when they identify the same
    current constraint.  An unresolved ID is retained explicitly and cannot
    accidentally compare equal to a resolved operation.
    """
    current_by_id = (
        {constraint.id: constraint for constraint in current_spec.constraints}
        if current_spec is not None
        else {}
    )
    removed: list[dict[str, Any]] = []
    for constraint_id in patch.remove_constraint_ids:
        constraint = current_by_id.get(constraint_id)
        if constraint is None:
            removed.append({"unresolved_id": constraint_id})
        else:
            removed.append({"constraint": _constraint(constraint)})
    return sorted(removed, key=_json_key)


def normalize_parse_result(
    result: ParseResult,
    *,
    current_spec: PortfolioSpec | None = None,
) -> dict[str, Any]:
    """Return the canonical semantic representation used for exact match."""
    if isinstance(result, FreshSpec):
        return {"kind": result.kind, "spec": _spec(result.spec)}
    if isinstance(result, SpecPatch):
        return {
            "kind": result.kind,
            "add_constraints": _constraints(result.add_constraints),
            "remove_constraints": _removed_constraints(result, current_spec),
            "replace_objective": (
                result.replace_objective.model_dump(mode="json")
                if result.replace_objective is not None
                else None
            ),
            "set_universe": result.set_universe,
            "estimation_overrides": result.estimation_overrides,
            "backtest_overrides": result.backtest_overrides,
        }
    if isinstance(result, Clarification):
        return {
            "kind": result.kind,
            "question": result.question,
            "partial_spec": _spec(result.partial_spec) if result.partial_spec is not None else None,
            "reason": result.reason,
        }
    raise TypeError(f"Unsupported ParseResult type: {type(result).__name__}.")


def constraint_multiset(
    result: ParseResult,
    *,
    current_spec: PortfolioSpec | None = None,
) -> Counter[str]:
    """Return constraint-operation fingerprints for micro P/R/F1.

    Fresh constraints are tagged ``present``; patch additions and removals are
    tagged ``add`` and ``remove``.  The tags prevent an opposite operation
    from being scored as a match.  Clarification partial specs are excluded
    because they are explicitly non-final.
    """
    items: list[dict[str, Any]] = []
    if isinstance(result, FreshSpec):
        items = [{"operation": "present", "constraint": item} for item in _constraints(result.spec.constraints)]
    elif isinstance(result, SpecPatch):
        items.extend(
            {"operation": "add", "constraint": item}
            for item in _constraints(result.add_constraints)
        )
        items.extend(
            {"operation": "remove", **item}
            for item in _removed_constraints(result, current_spec)
        )
    return Counter(_json_key(item) for item in items)
