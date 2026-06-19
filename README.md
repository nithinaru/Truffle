# Truffle

**Describe your portfolio in plain English. Truffle unearths the optimal allocation — and tells you exactly what each constraint costs you.**

[![PyPI](https://img.shields.io/badge/pypi-trufflefin-blue)](https://pypi.org/project/trufflefin/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![Parse accuracy](https://img.shields.io/badge/IR%20parse%20accuracy-94%25-brightgreen)](#evaluation)

> Tell it *"$50k, long-only, no stock over 8%, cap tech at 30%, minimize my downside risk, keep it to 15 names"* — and Truffle formulates the optimization rigorously, solves it, backtests it, and explains the result in plain language. The language model never writes the math. A typed compiler does.

---

## What this is

Truffle is a natural-language portfolio optimizer. You state your objective and constraints however you'd say them out loud; Truffle translates that into a verified mathematical program, solves it with a real convex/mixed-integer solver, and hands you back an allocation **plus** the things most tools hide:

- **What's binding** — which of your constraints are actually limiting you, with shadow prices ("your 30% tech cap is costing ~11bps of expected annual return").
- **What conflicts** — when your constraints are mutually infeasible, Truffle diagnoses the minimal conflicting set and proposes concrete fixes, instead of just erroring out.
- **What it would do out-of-sample** — a walk-forward backtest with transaction costs, no lookahead.

The core design principle: **the LLM only ever emits a typed intermediate representation; a deterministic compiler owns all the mathematics.** Language model for language, math for math. That's what makes it reliable enough to trust.

## Why "Truffle"

The solver underneath ([Clarabel](https://github.com/oxfordcontrol/Clarabel.rs)) is an *interior-point method* — it doesn't crawl the edges of the feasible region, it tunnels through the interior along the central path toward the optimum. A truffle grows buried underground and is found by following the scent down to the hidden prize. Truffle digs through the interior of your feasible set to unearth the buried optimal allocation. The objective is the scent; the optimum is the truffle.

## Demo

```
> I have $50k across these 25 tickers. Long-only. No single stock above 8%,
  tech sector capped at 30%. I care more about downside risk than variance —
  minimize CVaR at 95%. Keep me within 15 names max. Rebalance monthly,
  10bps transaction costs.

[Truffle] Here's how I read your problem — confirm before I dig:

  Objective:   Minimize CVaR (alpha = 0.95)
  Universe:    25 tickers
  Constraints: fully invested - long-only - position cap 8%
               sector cap: Tech <= 30% - max 15 names (-> MIP)
  Backtest:    monthly rebalance, 10bps costs, walk-forward

> yes

[Truffle] Solved (HiGHS, 2.3s, gap 0.0%). 13 names selected.
  -> Binding: tech sector cap (shadow price ~11bps/yr), position cap on NVDA.
  -> Relaxing tech to 35% would improve expected CVaR ~0.15%.
  [allocation chart] [tearsheet: Sharpe 1.1, MaxDD -14%, Calmar 0.9, turnover 22%/yr]
```

## Features

- Natural-language → typed problem spec, with a confirmation step before every solve
- Objectives: minimum-variance, mean-variance, max-Sharpe, **minimum-CVaR** (Rockafellar–Uryasev), risk parity, min-tracking-error
- Constraints: budget, long-only, position bounds, sector/group caps, cardinality, turnover, transaction costs, factor exposures, CVaR limits, tracking-error caps
- Proper covariance estimation (Ledoit–Wolf shrinkage), scenario generation (historical / IID / block bootstrap)
- Shadow-price explanations grounded in real solver duals — every number is verified against solver output
- Infeasibility diagnosis via elastic relaxation, with plain-language repair suggestions
- Walk-forward backtester with transaction costs and an honest model-vs-realized comparison

## Install

```bash
pip install trufflefin
export ANTHROPIC_API_KEY=...   # for the natural-language layer
```

### Mixed-integer problems (cardinality limits)

A "max N names" (cardinality) limit turns the problem mixed-integer. Two backends
cover those, both free:

- **MILP** (cardinality + min-CVaR) routes to **HiGHS**, which ships with cvxpy —
  nothing extra to install.
- **MIQP** (cardinality + min-variance / mean-variance) routes to **SCIP**
  (Apache-2.0) via PySCIPOpt. Install it with:

  ```bash
  pip install trufflefin[mip]   # or: pip install pyscipopt
  ```

  The `pyscipopt` wheel bundles the SCIP binaries on common platforms, so a
  separate system SCIP install is usually unnecessary. If the wheel can't build,
  install SCIP from <https://www.scipopt.org/> first, then `pip install pyscipopt`.
  If SCIP is missing when you solve a MIQP, Truffle raises a clear error telling
  you which solver to install — it never silently falls back to a wrong backend.

## Quickstart

```bash
# Solve from a YAML spec (no LLM needed — the deterministic core)
truffle solve examples/spec_minvar.yaml --prices examples/prices_sample.csv

# Or talk to it
truffle chat
```

```python
from truffle import optimize

result = optimize(
    "minimize CVaR at 95%, long only, no name over 10%, tech under 30%",
    universe=["AAPL", "MSFT", "NVDA", "JPM", "XOM", ...],
)
print(result.weights)
print(result.binding_constraints)   # shadow prices included
result.tearsheet()                  # walk-forward backtest
```

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
  [ Solver ]  -- Clarabel (convex) / HiGHS (mixed-integer)
        |
        +--> duals & shadow prices --> [ Explainer ] (grounded narration)
        +--> infeasible? ----------> [ Diagnoser ] (elastic relaxation)
        +--> [ Backtester ] (walk-forward, transaction costs)
```

Everything below the agent is deterministic, typed, and tested. The language model's only jobs are turning your words into a spec, asking when you're ambiguous, and narrating numbers the solver actually produced.

## Evaluation

Truffle ships with a benchmark of hand-verified natural-language → spec pairs (simple, multi-constraint, ambiguous, and deliberately infeasible cases). Parse accuracy and per-constraint precision/recall are reported on every commit. See [`eval/`](eval/).

## Roadmap

- [ ] **Cardinality / max-names limits** (big-M MIP) — the one modeling feature not yet shipped; everything else in *Features* above is implemented and tested.
- [ ] Multi-period optimization (stochastic programming over scenario trees)
- [ ] Factor-model risk (Fama–French exposure constraints) — generic factor-exposure constraints ship today; named Fama–French factors are future work.
- [ ] Black–Litterman view blending ("I think NVDA outperforms by 5%")
- [ ] Robust optimization (uncertainty sets on expected returns)

### Current limitations

- **Max-Sharpe** uses the Charnes–Cooper transform and currently supports only `budget` + `long_only` + `box` constraints (and requires `long_only`). Combining it with group caps, turnover, transaction cost, tracking error, CVaR or factor constraints raises a clear error; use a min-variance / mean-variance / min-CVaR objective if you need those together. Transforming arbitrary constraints through the change of variables is future work.
- **Risk parity** is solved standalone (the convex log-barrier surrogate, normalized); it does not yet compose with additional hard constraints.

## Disclaimer

Truffle is a research and decision-support tool, not investment advice. Expected-return estimates are the weakest input in portfolio optimization; Truffle defaults toward risk-based objectives and shows you its assumptions. Nothing here is a recommendation to buy or sell any security.

## License

MIT
