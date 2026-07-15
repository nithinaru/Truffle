"""Slice 1 tests: the Cardinality node validates and compiles to a MIP.

These cover the node's typed validation (the bad cases the prompt calls out)
and that compiling a spec with a Cardinality produces a mixed-integer problem
with a binary selection vector and a working big-M linking constraint. The
"reduces nonzeros end to end" and routing assertions live in Slice 5.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from core.compiler import compile_spec
from core.exceptions import CompilationError
from core.ir import Box, Budget, Cardinality, LongOnly, MinVariance, PortfolioSpec


def _spec(constraints) -> PortfolioSpec:
    return PortfolioSpec(
        universe=["A", "B", "C", "D", "E"],
        objective=MinVariance(),
        constraints=constraints,
    )


def test_cardinality_forces_mip_problem_class() -> None:
    spec = _spec([Budget(), LongOnly(), Cardinality(max_names=2)])
    assert spec.problem_class == "mip"


def test_max_names_above_universe_size_raises() -> None:
    with pytest.raises(ValueError, match="exceeds the universe size"):
        _spec([Budget(), LongOnly(), Cardinality(max_names=6)])


def test_min_names_above_max_names_raises() -> None:
    with pytest.raises(ValueError, match="min_names"):
        Cardinality(max_names=3, min_names=4)


def test_min_position_above_universe_cap_raises() -> None:
    # Universe-wide cap of 10% but a 20% floor on held names is contradictory.
    with pytest.raises(ValueError, match="min_position"):
        _spec(
            [
                Budget(),
                LongOnly(),
                Box(lower=0.0, upper=0.10),
                Cardinality(max_names=3, min_position=0.20),
            ]
        )


def test_more_than_one_cardinality_raises() -> None:
    with pytest.raises(ValueError, match="at most one"):
        _spec([Budget(), Cardinality(max_names=2), Cardinality(max_names=3)])


def test_compile_builds_binary_selection_and_links_weights() -> None:
    spec = _spec([Budget(), LongOnly(), Cardinality(max_names=2)])
    compiled = compile_spec(spec, mu=np.zeros(5), sigma=np.eye(5) * 0.04)

    # Binary selection vector is exposed for the fix-and-resolve dual path.
    y = compiled.extra_vars["y"]
    assert y.attributes["boolean"] is True
    assert y.shape == (5,)
    # The count cap is named for elastic diagnosis. It still has no usable MIP
    # dual; production reporting obtains conditional duals after dropping the
    # Cardinality node in the continuous fix-and-resolve pass.
    assert spec.constraints[-1].id in compiled.constraint_objs

    compiled.problem.solve(solver=cp.SCIP if "SCIP" in cp.installed_solvers() else cp.HIGHS)
    assert compiled.problem.status == "optimal"
    nonzero = int(np.sum(np.abs(compiled.weights.value) > 1e-4))
    assert nonzero <= 2


@pytest.mark.parametrize(
    "constraints",
    [
        [Budget(), Cardinality(max_names=2)],
        [LongOnly(), Cardinality(max_names=2)],
        [Budget(total=2.0), LongOnly(), Cardinality(max_names=2)],
    ],
)
def test_compile_requires_long_only_unit_budget_for_cardinality(constraints) -> None:
    spec = _spec(constraints)
    with pytest.raises(CompilationError, match=r"LongOnly and Budget\(total=1\.0\)"):
        compile_spec(spec, mu=np.zeros(5), sigma=np.eye(5))


def test_min_position_links_held_weights() -> None:
    # With a 30% floor and max 2 names, the two held names must each be >= 0.30.
    spec = _spec([Budget(), LongOnly(), Cardinality(max_names=2, min_position=0.30)])
    compiled = compile_spec(spec, mu=np.zeros(5), sigma=np.eye(5) * 0.04)
    solver = cp.SCIP if "SCIP" in cp.installed_solvers() else cp.HIGHS
    compiled.problem.solve(solver=solver)
    assert compiled.problem.status == "optimal"
    held = compiled.weights.value[np.abs(compiled.weights.value) > 1e-4]
    assert np.all(held >= 0.30 - 1e-6)
