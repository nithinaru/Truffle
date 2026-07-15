# Truffle

**Describe your portfolio in plain English. Truffle unearths the optimal allocation — and tells you exactly what each constraint costs you.**

Current product requirements and the staged delivery plan live in [PRD.md](PRD.md).

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Tell it *"$50k, long-only, no stock over 8%, cap tech at 30%, minimize my downside risk, keep it to 15 names"* — and Truffle formulates the optimization rigorously, solves it, diagnoses conflicts, and explains the result in plain language. The language model never writes the math. A typed compiler does.

---

## What this is

Truffle is a natural-language portfolio optimizer. You state your objective and constraints however you'd say them out loud; Truffle translates that into a verified mathematical program, solves it with a real convex/mixed-integer solver, and hands you back an allocation **plus** the things most tools hide:

- **What's binding** — which of your constraints are actually limiting you, with shadow prices ("your 30% tech cap is costing ~11bps of expected annual return").
- **What conflicts** — when your constraints are mutually infeasible, Truffle can run
  deterministic conflict analysis and offer only repairs that it has re-solved successfully.
- **What it would do out-of-sample** — a local walk-forward backtest uses trailing
  data, fills one observed close later, charges transaction costs, and compares modeled
  risk with realized results.

The core design principle: **the LLM only ever emits a typed intermediate representation; a deterministic compiler owns all the mathematics.** Language model for language, math for math. That's what makes it reliable enough to trust.

## Why "Truffle"

The solver underneath ([Clarabel](https://github.com/oxfordcontrol/Clarabel.rs)) is an *interior-point method* — it doesn't crawl the edges of the feasible region, it tunnels through the interior along the central path toward the optimum. A truffle grows buried underground and is found by following the scent down to the hidden prize. Truffle digs through the interior of your feasible set to unearth the buried optimal allocation. The objective is the scent; the optimum is the truffle.

## Demo

The repository includes a synthetic five-asset panel, so both deterministic
paths can be exercised without an API key or network access:

```bash
# A single allocation; --diagnose is opt-in if the spec proves infeasible.
truffle solve examples/spec_minvar.yaml \
  --prices examples/prices_sample.csv \
  --diagnose

# Monthly walk-forward evaluation with a next-observed-close fill.
truffle backtest examples/spec_minvar.yaml \
  --prices examples/prices_sample.csv \
  --lookback 252 \
  --rebalance monthly \
  --cost-bps 10 \
  --json-out /tmp/truffle-tearsheet.json
```

## Features

- Natural-language → typed problem spec, with a confirmation step before every solve
- Objectives: minimum-variance, mean-variance, max-Sharpe, **minimum-CVaR** (Rockafellar–Uryasev), risk parity, min-tracking-error
- Constraints: budget, long-only, position bounds, sector/group caps, cardinality, turnover, transaction costs, factor exposures, CVaR limits, tracking-error caps
- Proper covariance estimation (Ledoit–Wolf shrinkage) and historical CVaR scenarios
- Seeded IID and contiguous-block bootstrap scenario APIs, with explicit horizon units
- Shadow-price explanations grounded in real solver duals — every number is verified against solver output
- Opt-in infeasibility diagnosis via normalized elastic relaxation and deletion filtering
- Verified repairs: every suggested patch is re-solved before it is offered
- No-lookahead walk-forward backtesting with drifted holdings, proportional costs,
  equal-weight comparison, optional local market comparison, and deterministic JSON
- A typed local-data Python facade for solving and backtesting without importing
  the agent or constructing a network client
- An exact `$100` local paper replay with conservative fills, atomic risk gates,
  normalized ideal ledgers, and Truffle/equal-weight/market/cash arms
- A restart-safe, non-submitting live-shadow workflow with a durable SQLite
  journal, strictly delayed activation, and an operational 30-session gate
- An optional read-only Alpaca IEX adapter that retains per-symbol source-time
  provenance and fails closed on stale, incomplete, or out-of-session data

## Install

Truffle is not published to PyPI yet. Install the current development version
from source:

```bash
git clone https://github.com/nithinaru/Truffle.git
cd Truffle
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The deterministic solver and backtester do not install or call an LLM. For the
optional natural-language chat layer:

```bash
pip install -e ".[agent]"
export ANTHROPIC_API_KEY=...
```

For explicitly invoked Alpaca market-data reads (never broker orders):

```bash
pip install -e ".[alpaca-data]"
cp .env.example .env  # fill locally; .env files are ignored by git
```

### Mixed-integer problems (cardinality limits)

A "max N names" (cardinality) limit turns the problem mixed-integer. Two backends
cover those, both free:

- **MILP** (cardinality + min-CVaR) routes to **HiGHS**, which ships with cvxpy —
  nothing extra to install.
- **MIQP/MISOCP** (cardinality + a variance objective or tracking-error cone)
  routes to **SCIP** (Apache-2.0) via PySCIPOpt. Install it with:

  ```bash
  pip install -e ".[mip]"   # or: pip install pyscipopt
  ```

  The `pyscipopt` wheel bundles the SCIP binaries on common platforms, so a
  separate system SCIP install is usually unnecessary. If the wheel can't build,
  install SCIP from <https://www.scipopt.org/> first, then `pip install pyscipopt`.
  If SCIP is missing when you solve a MIQP/MISOCP, Truffle raises a clear error
  telling you which solver to install — it never silently falls back to a wrong backend.

## Quickstart

```bash
# Solve from a YAML spec (no LLM needed — the deterministic core)
truffle solve examples/spec_minvar.yaml --prices examples/prices_sample.csv

# Run the local historical backtester and retain its complete JSON result
truffle backtest examples/spec_minvar.yaml \
  --prices examples/prices_sample.csv \
  --lookback 252 --rebalance monthly --json-out /tmp/truffle-tearsheet.json

# Or talk to it (an API key and a price panel are required)
truffle chat --prices examples/prices_sample.csv
```

The Python facade is deterministic and accepts only a typed spec (or a mapping)
plus caller-supplied local data:

```python
import pandas as pd
from truffle import BacktestConfig, run_walk_forward_backtest, solve_portfolio

prices = pd.read_csv("prices.csv", parse_dates=[0], index_col=0)
spec = {
    "universe": ["AAA", "BBB", "CCC"],
    "objective": {"kind": "min_variance"},
    "constraints": [{"kind": "budget"}, {"kind": "long_only"}],
}
report = solve_portfolio(spec, prices)
tearsheet = run_walk_forward_backtest(
    spec,
    prices,
    config=BacktestConfig(lookback_returns=252, rebalance_frequency="monthly"),
)
```

This facade does not interpret natural language, fetch data, create an LLM
client, or hide typed solver failures. The confirmed chat workflow remains the
only natural-language entrypoint.

## How it works

```
  natural language
        |
        v
  [ Agent ]  -- Claude emits a typed spec, asks clarifying questions
        |        (never writes solver code)
        v
  [ Validator ]  -- Pydantic types + semantic checks
        |
        v
  [ Compiler ]  -- spec -> CVXPY model (deterministic, unit-tested)
        |
        v
  [ Solver ]  -- Clarabel (convex) / HiGHS or SCIP (mixed-integer)
        |
        +--> duals & shadow prices --> [ Explainer ] (grounded narration)
        +--> infeasible? ----------> [ Diagnoser ] (elastic relaxation)
        +--> [ Backtester ] (walk-forward, delayed fills, transaction costs)
```

Everything below the agent is deterministic, typed, and tested. The language model's only jobs are turning your words into a spec, asking when you're ambiguous, and narrating numbers the solver actually produced.

## Backtest contract

`truffle backtest` and `backtest.run_backtest` operate only on caller-supplied
local data. At a scheduled period-end close, the engine supplies exactly the
configured trailing return window to the solver. Existing holdings earn the
return to the next observed close; the new target fills at that close, pays the
configured proportional cost, and first earns the following interval. Drifted
pre-trade weights are carried into every later solve.

Every scheduled rebalance is mandatory. A failed solve raises a dated error
instead of silently skipping a period. The price index is treated as the known,
complete trading calendar, so callers must distinguish real non-trading days
from data outages before running the engine. Missing values, duplicate dates,
and non-positive prices are rejected.

The frozen tearsheet records normalized net and gross curves, every signal and
fill date, training bounds, target and pre-trade weights, decision and realized
L1 turnover, costs, annualized metrics, drawdown, Sharpe ratio, and modeled and
realized empirical CVaR. An equal-weight net baseline is always included; an
exactly aligned local market-price series can be supplied with `--market-prices`.
Serialization is deterministic and contains no wall-clock fields.

## Local `$100` paper replay

The paper layer is an offline experiment, not a broker connection. It replays
complete local snapshots through four fully isolated arms: the confirmed
Truffle targets, equal weight, a selected market symbol, and cash. Each arm has
an exact `Decimal("100.00")` shadow ledger and a separate frictionless unit-NAV
ledger, so rounding, spread, slippage, and fees cannot contaminate the ideal
comparison.

At every scheduled timestamp, Truffle fresh-marks all holdings, creates
sell-first orders, rounds quantities down, scales buys to available cash, and
simulates buys at ask plus adverse slippage and sells at bid minus adverse
slippage. Every event persists the exact snapshot, price/fee ticks, and
assumptions; the ledger recomputes those economics before append. All three
traded arms must pass the atomic shadow-only risk gate before any arm executes.
A rejected or missing dated step aborts the replay instead of being skipped.

`paper.run_paper_replay` accepts a `LocalReplayProvider`, a caller-supplied
schedule of confirmed `TargetAllocation` objects, explicit execution/planning/
risk settings, and an explicit evaluation-time policy. Its deterministic report
contains exact and ideal curves, fills, IDs, fees, positions, and turnover
defined as full executed notional divided by fresh pre-trade equity—there is no
`1/2` convention. It never reads a key, clock, broker, or network service.

## Non-submitting live shadow

The next stage is a library workflow, not an unattended trading daemon. A
confirmed signal is first written to `SQLiteShadowJournal` while it is waiting
for a future execution snapshot. If the process stops after a completed close,
the target, strategy version, source snapshot, confirmation time, and
`execute_not_before` boundary are already durable.

`paper.run_live_shadow_step` activates that queued signal only against a
strictly later snapshot. It fresh-marks every held symbol, plans sell-first
orders, invokes the mandatory `mode="shadow"` risk gate, and atomically records
the plan plus either conservative local fills or an incident-backed rejection.
One signal can activate exactly once. Byte-identical retries recover the prior
result before replanning; changed content under an existing ID is a collision.
There is no clock, scheduler, credential, broker client, or order-submission
method in this workflow—all times, snapshots, and configurations are injected.

The SQLite journal uses WAL mode, full synchronization, append-only triggers,
canonical JSON, stable semantic IDs, typed cross-record validation, and a
chained SHA-256 digest. It reconstructs the exact `$100` ledger after restart
and detects accidental or external mutation. The digest is an integrity chain,
not a signed or externally anchored proof against an attacker who can rewrite
the whole database.

Operational readiness is separate from strategy performance. Each official
market session must be explicitly closed as healthy, incident, or incomplete.
`paper.evaluate_journal_operational_readiness` compares those closures with an
injected official exchange calendar, so missing days break the streak. The
default gate requires 30 consecutive healthy sessions and always reports
`real_money_authorized=False`; it does not replace separately declared
performance and drawdown criteria.

## Read-only Alpaca data boundary

`paper.AlpacaDataProvider` is an optional IEX-only first adapter. Its concrete
transport can issue only fixed-host HTTPS `GET` requests for stock snapshots,
adjusted bars, asset metadata, and the official market calendar. It has no
generic URL or HTTP-method entrypoint and contains no account, position, or
order endpoint. Tests use an injected fake transport and never contact Alpaca.

Latest captures retain the Alpaca request ID, nanosecond quote/trade source
times, exchange and condition metadata, capture time, and official session
window in an `ObservedMarketSnapshot`. The normalized execution snapshot uses
the oldest required source timestamp, and capture fails closed for stale or
future data, missing symbols, crossed markets, or pre/after-hours observations.
Historical analytics use paginated `adjustment=all` daily bars and relabel them
to official session closes; raw latest quotes/trades remain the execution input.
The latest-data API deliberately does not pretend to implement replay's
`snapshot_at(exact_time)` contract.

The initial adapter defaults to Alpaca's real-time IEX feed, which is useful for
a small operational experiment but is not full-market NBBO data. Personal paper
keys are not documented as read-only, so use paper-only credentials and restrict
process egress even though Truffle's adapter has no write operation. See Alpaca's
[market-data plans](https://docs.alpaca.markets/us/docs/about-market-data-api),
[stock snapshots](https://docs.alpaca.markets/us/reference/stocksnapshots-1),
and [paper-trading limitations](https://docs.alpaca.markets/us/v1.4.2/docs/paper-trading).

The scenario helpers are library APIs:

```python
from data.scenarios import block_bootstrap_scenarios, iid_bootstrap_scenarios

iid = iid_bootstrap_scenarios(prices, 1_000, seed=7)
five_period = block_bootstrap_scenarios(
    prices, 1_000, block_length=5, seed=7
)
```

IID rows are one-period simple returns. Each block row compounds a contiguous
historical block, so its financial horizon is `block_length` periods and must
match the horizon of any CVaR limit or interpretation.

## Infeasibility diagnosis

Diagnosis is deliberately opt-in for deterministic YAML solves:

```bash
truffle solve spec.yaml --prices prices.csv --diagnose
```

The diagnostic path ranks normalized elastic violations, uses proof-checked
deletion trials to isolate a conflict, and verifies every proposed repair by
solving the patched spec. When irreducibility cannot be proved, the report says
so rather than presenting the candidate as a verified IIS. In chat mode, choosing
a server-owned repair creates a new spec echo and requires confirmation before
another solve; a repair is never applied and solved silently.

## Evaluation

Truffle now includes a deliberately modest 12-case starter benchmark and a
deterministic, offline-first scoring harness. It covers fresh specs, patches,
clarifications, adversarial wording, and infeasibility-prone requests. The
harness reports semantic exact match, parse-kind accuracy, constraint micro
precision/recall/F1, and clarification precision/recall/F1. Generated
constraint IDs and constraint ordering are normalized; substantive objective,
universe, and constraint fields are not.

Predictions are supplied as JSONL records shaped like
`{"case_id":"...","result":{...}}` and scored without constructing an LLM
client or making a network request:

```bash
python -m evaluation.run --predictions predictions.jsonl \
  --output evaluation-report.json
```

This starter corpus is infrastructure, not a publishable accuracy claim. No
parse-accuracy figure is reported yet; a larger reviewed corpus and a gated,
explicit live run are still required before publishing one.

## Roadmap

- [x] **Cardinality / max-names limits** (big-M MIP) — shipped. A "max N names"
  limit routes to a mixed-integer solver (HiGHS for MILP, SCIP for MIQP/MISOCP) and
  returns **conditional** shadow prices (see below).
- [x] **Infeasibility diagnosis** — normalized elastic relaxation, verified
  node-level IIS extraction, and ranked repairs that are re-solved before display.
- [x] **Walk-forward backtester** — delayed fills, drift, costs, deterministic
  tearsheets, equal-weight comparison, and no-lookahead regression tests.
- [x] **IID and block-bootstrap CVaR scenarios** — deterministic seeded APIs;
  block horizons are explicit and compounded.
- [x] **$100 local shadow experiment** — deterministic replay, exact cash and
  holdings accounting, conservative fill simulation, atomic risk gates, and
  parallel Truffle/equal-weight/market/cash exact and ideal arms.
- [x] **Non-submitting live shadow** — durable queued signals, strictly delayed
  exactly-once activation, SQLite recovery, explicit incidents/session closures,
  and a read-only Alpaca data boundary.
- [ ] **Broker-paper adapter** — remains a separate explicit workflow with
  reconciliation and deterministic client order IDs; no broker order submission
  is implemented today.
- [x] **Offline parse evaluation harness** — typed starter corpus, semantic
  normalization, deterministic reports, and injected-client tests; no accuracy
  claim until the corpus is expanded and a gated live run is performed.
- [x] **Deterministic Python facade** — validated specs and caller-supplied
  local data only; typed solver and backtest failures propagate unchanged.
- [ ] Multi-period optimization (stochastic programming over scenario trees)
- [ ] Factor-model risk (Fama–French exposure constraints) — generic factor-exposure constraints ship today; named Fama–French factors are future work.
- [ ] Black–Litterman view blending ("I think NVDA outperforms by 5%")
- [ ] Robust optimization (uncertainty sets on expected returns)

### Current limitations

- **Cardinality shadow prices are conditional.** A mixed-integer optimum has no
  native dual variables, so Truffle prices the constraints by fixing the chosen
  names (re-solving the continuous restriction) and harvesting *that* problem's
  duals. The reported shadow prices are therefore valid *given the selected name
  set*, not global sensitivities — the explanation states this explicitly.
  Cardinality currently composes with min-variance / mean-variance (MIQP) and
  min-CVaR (MILP), under a long-only book.
- **No broker execution.** The backtester and `$100` replay remain local. The
  optional Alpaca adapter performs read-only data GETs, and the live-shadow
  library simulates local fills; neither submits live or paper orders. There is
  no automatic scheduler, streaming daemon, or end-to-end broker command.
- **Execution models are deliberately bounded.** Historical backtests use the
  next observed close plus proportional cost. The local paper simulator adds
  bid/ask, adverse slippage, commissions, minimum fees, quantity steps, and
  conservative tick rounding, but not partial fills, exchange queues, market
  impact, or broker-specific behavior.
- **Max-Sharpe** uses the Charnes–Cooper transform and currently supports only `budget` + `long_only` + `box` constraints (and requires `long_only`). Combining it with group caps, turnover, transaction cost, tracking error, CVaR or factor constraints raises a clear error; use a min-variance / mean-variance / min-CVaR objective if you need those together. Transforming arbitrary constraints through the change of variables is future work.
- **Risk parity** is solved standalone (the convex log-barrier surrogate, normalized); it does not yet compose with additional hard constraints.

## Disclaimer

Truffle is a research and decision-support tool, not investment advice. Expected-return estimates are the weakest input in portfolio optimization; Truffle defaults toward risk-based objectives and shows you its assumptions. Nothing here is a recommendation to buy or sell any security.

## License

MIT
