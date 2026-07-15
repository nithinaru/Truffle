# Contributing to Truffle

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The Anthropic SDK is optional and needed only for an interactive chat session:

```bash
pip install -e ".[agent]"
```

Run the suite and the linter before every commit:

```bash
pytest
ruff check .
```

The default `pytest` run makes **no live API calls** and requires **no paid
solver license**. The natural-language layer is exercised with a fake LLM
client; the math layer is exercised with the bundled sample CSVs.

## Deterministic evaluation contracts

- Infeasibility diagnosis is opt-in for the YAML CLI (`--diagnose`). Keep
  elastic violations in natural units, distinguish a proof-verified IIS from
  an unverified candidate, and re-solve every repair before exposing it.
- A backtest signal may use data only through its signal close. With close-only
  inputs, the target fills at the next observed close and first earns the next
  interval. Preserve this timing in fixtures and add a future-data perturbation
  test for any scheduling change.
- Treat a supplied backtest index as a complete trading calendar. Reject data
  outages and failed scheduled solves; never skip them and shorten the sample.
- Bootstrap tests must pass an explicit seed. Document whether each scenario
  row is a one-period return or a compounded multi-period block.
- Deterministic results must not include wall-clock timestamps, random IDs, or
  machine-specific paths.
- The parse benchmark is offline by default. Score supplied prediction JSONL
  with `python -m evaluation.run --predictions ...`; never add an implicit live
  client path or publish a metric from the 12-case starter corpus as if it were
  a research benchmark.

## Paper-testing policy

Paper testing is staged behind local, inspectable components. Historical replay,
the exact `$100` fake-money ledger, conservative local fills, atomic risk gates,
four-arm replay reports, durable live-shadow journaling, and a GET-only Alpaca
data boundary now ship. A broker-paper adapter comes only after this stage has
an incident-free operational record. Do not add live market-data or broker calls
to the default test suite. Network adapters must sit behind a narrow interface
and have deterministic fakes so accounting and risk behavior remain testable
offline.

A confirmed live-shadow signal must be durably queued before waiting for the
later execution snapshot. Never permit same-close activation, more than one
activation per signal, an execution record without its risk-approved manifest,
or a healthy-session streak inferred from missing incident rows. Readiness must
compare explicit session closures with an injected official market calendar.

Do not describe a local shadow result as a broker fill or imply that positive
returns are guaranteed. The historical backtester currently models a
next-observed-close fill and proportional cost only; broker-specific execution
belongs in the later paper adapter.

## Solvers

Truffle routes each compiled problem to a solver by its character (see
`core/routing.py`):

| Problem | Solver | Install |
|---|---|---|
| Continuous convex (QP / SOCP / LP) | Clarabel | ships with cvxpy |
| MILP (cardinality + min-CVaR) | HiGHS | ships with cvxpy |
| MIQP (cardinality + min-variance / mean-variance) | SCIP | `pip install pyscipopt` |
| MISOCP (cardinality + tracking-error cap) | SCIP | `pip install pyscipopt` |

SCIP (Apache-2.0, free) is reached through cvxpy's PySCIPOpt interface. The
`pyscipopt` wheel bundles the SCIP binaries on common platforms; if it cannot
build a wheel, install SCIP from <https://www.scipopt.org/> first.

CI note: if SCIP cannot be installed in a given environment, the MIQP/MISOCP-routing
test skips with a clear reason rather than failing the suite. Everything else —
including the MILP (HiGHS) path — still runs.
