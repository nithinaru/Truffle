# Contributing to Truffle

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the suite and the linter before every commit:

```bash
pytest
ruff check .
```

The default `pytest` run makes **no live API calls** and requires **no paid
solver license**. The natural-language layer is exercised with a fake LLM
client; the math layer is exercised with the bundled sample CSVs.

## Solvers

Truffle routes each compiled problem to a solver by its character (see
`core/routing.py`):

| Problem | Solver | Install |
|---|---|---|
| Continuous convex (QP / SOCP / LP) | Clarabel | ships with cvxpy |
| MILP (cardinality + min-CVaR) | HiGHS | ships with cvxpy |
| MIQP (cardinality + min-variance / mean-variance) | SCIP | `pip install pyscipopt` |

SCIP (Apache-2.0, free) is reached through cvxpy's PySCIPOpt interface. The
`pyscipopt` wheel bundles the SCIP binaries on common platforms; if it cannot
build a wheel, install SCIP from <https://www.scipopt.org/> first.

CI note: if SCIP cannot be installed in a given environment, the MIQP-routing
test skips with a clear reason rather than failing the suite. Everything else —
including the MILP (HiGHS) path — still runs.
