# Truffle Product Requirements

**Status:** Active  
**Version:** 1.2
**Last updated:** 2026-07-15
**Primary branch:** `main`
**Current checkpoint:** Sprint 2 is complete locally and awaits the owner's push
and remote Python 3.12 CI confirmation.

This document is the source of truth for Truffle's remaining product work. The
original [project blueprint](BLUEPRINT.md) remains useful design history; this
PRD reflects what is implemented now, the trust gaps found during the current
audit, and the staged path from research prototype to credible paper testing.

## 1. Product outcome

Truffle turns a plain-English portfolio request into a typed, inspectable
optimization problem, solves it deterministically, explains the result using
solver-owned facts, and evaluates the confirmed strategy without lookahead.

The product is successful when a user can:

1. Describe an objective and constraints in natural language.
2. Inspect and explicitly confirm the exact mathematical specification.
3. Receive a valid allocation, conflict diagnosis, and correctly typed risk and
   sensitivity metrics.
4. Run a reproducible historical evaluation of that confirmed specification.
5. Run a durable live-market shadow experiment with fake money and no order
   submission.
6. Compare Truffle with cash, equal-weight, and market baselines after
   conservative costs.

Broker-hosted paper orders are a later, separately gated workflow. Real-money
order submission is not a Truffle requirement.

## 2. Product principles

- **Language for intent; deterministic code for mathematics.** The language
  model may emit only typed intent. It never creates solver code or bypasses
  validation.
- **Confirm before action.** A portfolio specification and every operational
  workflow require explicit confirmation before execution.
- **Fail closed.** Invalid numbers, stale or incomplete data, unsupported
  solver states, journal inconsistencies, and reconciliation mismatches stop the
  workflow visibly.
- **No lookahead.** Every historical or forward decision records the exact
  information boundary available at signal time and a strictly later execution
  boundary.
- **Report meanings, not convenient numbers.** Objective terms, financial
  metrics, dual units, costs, and approximation status must be distinguishable
  in typed output.
- **Evidence before claims.** Synthetic examples test mechanics. Accuracy,
  performance, and readiness claims require frozen protocols and retained
  evidence.
- **Paper before broker; broker paper before any separate real-money project.**
  Live shadow must establish an operational record before broker write access is
  considered.

## 3. Users and primary workflows

### 3.1 Portfolio researcher

Builds constrained allocations, inspects binding constraints, diagnoses
infeasibility, and compares strategies across historical regimes.

### 3.2 Technical evaluator

Audits the IR, assumptions, solver selection, data timing, reports, and
reproducibility artifacts without relying on an LLM response.

### 3.3 Paper-test operator

Runs a small, long-only, regular-hours fake-money experiment; reviews incidents,
ledger state, baseline-relative results, and operational readiness.

## 4. Required capabilities

### 4.1 Typed portfolio construction

- Support the documented objective and constraint nodes through one validated
  `PortfolioSpec` contract.
- Reject non-finite values and numerically invalid arrays before compilation.
- Validate solver-domain requirements, including covariance PSD tolerance and
  transformed-objective budget semantics.
- Preserve deterministic solver routing and explicit missing-backend errors.
- Never silently ignore a requested amendment or workflow setting.

### 4.2 Trustworthy solve reports

- Separate the base financial objective from transaction-cost or other penalty
  terms.
- Report objective-appropriate metrics such as variance, volatility, expected
  return, CVaR, tracking error, or Sharpe ratio with explicit units.
- Preserve constraint row, side, sign, scaling, and units for sensitivity data.
- Mark mixed-integer sensitivities as conditional on the selected name set.
- Represent a time-limited mixed-integer solve only when a feasible incumbent
  was validated; report its actual optimality gap.

### 4.3 Natural-language workflow

- Preserve clarification context across turns.
- Echo the effective state that will actually be executed.
- Require confirmation after fresh specifications, amendments, and verified
  repairs.
- Route confirmed operational intent to a typed backtest, local replay, or live
  shadow workflow outside `PortfolioSpec`.
- Fall back to deterministic output on provider or narration failure.

### 4.4 Historical evaluation

- Estimate from trailing data only and execute strictly after the signal
  boundary.
- Treat scheduled failures and data outages explicitly; never shorten a sample
  silently.
- Retain specifications, data provenance or hashes, assumptions, trades,
  failures, curves, costs, and baseline-relative metrics.
- Distinguish execution-model limitations from observed strategy behavior.

### 4.5 Local paper replay

- Maintain exact cash, positions, marks, fees, orders, and fills from a `$100`
  seed account.
- Maintain isolated Truffle, equal-weight, market, and cash arms.
- Preserve a separate frictionless normalized ledger to expose rounding and
  execution drag.
- Use conservative bid/ask, slippage, fee, and quantity-step assumptions.

### 4.6 Non-submitting live shadow

- Queue a confirmed post-close signal durably before waiting for execution data.
- Activate it exactly once against a strictly later, fresh snapshot.
- Run the mandatory shadow-only risk gate and persist approval or incident-backed
  rejection atomically.
- Provide a one-shot operator command that wires calendar, data, strategy,
  journal, reporting, and session closure without any order-submission surface.
- Derive healthy, incident, and incomplete session closure from objective checks.
- Require 30 consecutive healthy official sessions for operational readiness.

### 4.7 Broker paper, gated

Broker paper begins only after the live-shadow operational gate and predeclared
performance/drawdown review pass.

- Hardcode paper-only hosts and allowlist the expected account.
- Use deterministic client order IDs, explicit confirmation, and a kill switch.
- Model submitted, accepted, partial, filled, canceled, rejected, expired, and
  late-fill states.
- Reconcile broker cash, positions, orders, and fills before and after each
  rebalance; block on any unexplained mismatch.
- Record dividends, splits, symbol changes, fees, and other account activity.
- Keep real-money endpoints unsupported.

## 5. Explicit non-goals

- Guaranteeing positive returns or claiming a durable trading edge.
- Real-money order submission.
- High-frequency, intraday, leveraged, short, options, crypto, or extended-hours
  trading in the first paper campaign.
- Treating broker-paper fills as realistic exchange execution.
- Letting an LLM select untyped operations, mutate a ledger, or send an order.
- Publishing accuracy or performance claims from the synthetic examples or the
  12-case starter parse corpus.

## 6. Success and release gates

### 6.1 Correctness gate

- Full supported test matrix and lint pass from a clean environment.
- Wheel build and isolated import pass.
- Invalid numerical inputs fail before solver invocation.
- No confirmed setting is silently ignored.
- Every narrated number maps to the correct typed entity, field, and unit.

### 6.2 Historical evidence gate

- Frozen strategies and thresholds are evaluated over multiple real-data
  universes and regimes.
- Cash, equal-weight, and market baselines are always present where applicable.
- Results include costs, turnover, drawdown, volatility/CVaR, and uncertainty;
  no conclusion depends only on ending balance.

### 6.3 Live-shadow operational gate

- No duplicate activations or forbidden operations.
- No unresolved ledger or journal integrity failure.
- All stale-data, restart, disconnect, and rejection tests fail safely.
- At least 30 consecutive official sessions close healthy.
- Performance and drawdown remain within thresholds frozen before the campaign.

### 6.4 Broker-paper gate

- All broker lifecycle and reconciliation tests pass with deterministic fakes.
- Paper-only endpoint and account protections are verified.
- No unresolved cash, position, fill, or open-order mismatch.
- Conservative local results remain available beside broker-reported results.

### 6.5 Public launch gate

- A reviewed 75-100+ prompt benchmark has reproducible model and prompt
  provenance.
- A safe demo exposes specification confirmation, allocation, conflicts,
  sensitivities, historical curves, drawdown, and costs.
- Package metadata, supported API, documentation, CI matrix, and release process
  are stable.

## 7. Current implementation status

### Shipped foundation

- Typed IR, deterministic compiler, six objective kinds, and ten constraint kinds.
- Convex and mixed-integer routing with explicit solver requirements.
- Confirmation-gated chat, grounded narration, and deterministic infeasibility
  diagnosis with verified repairs.
- No-lookahead walk-forward backtester and deterministic tearsheet.
- Exact `$100` four-arm replay, conservative local fills, atomic risk gates, and
  exact/ideal ledgers.
- Durable live-shadow primitives, append-only SQLite journal, incidents, session
  closures, and operational readiness evaluation.
- Read-only Alpaca IEX data adapter with official calendar and source provenance.
- Offline parse-scoring infrastructure and a 12-case starter corpus.

### Remaining blockers

- Clarification context and some patch semantics are incomplete.
- No downstream natural-language operation router exists.
- Live shadow is a library workflow, not yet an operator command or deployed run.
- No 30-session forward evidence or predeclared performance gate exists yet.
- Broker order lifecycle, reconciliation, and corporate actions are not built.
- The parser benchmark, empirical study, visual demo, and public release remain
  incomplete.

## 8. Delivery roadmap

Each sprint must end in a reviewable, independently testable state. No later
sprint may weaken an earlier safety gate.

### Sprint 0 - Baseline recovery and current PRD

- Establish this PRD as the active roadmap.
- Reproduce and fix the failing Python 3.12 CI contract.
- Make optional MIQP tests skip honestly when SCIP is absent and execute them in
  CI with the declared `mip` extra.
- Remove deprecated GitHub Actions runtime warnings.

**Exit:** lint and tests pass in the CI-equivalent Python 3.12 environment both
with the CI extras and with optional SCIP absent; the next push is ready to
confirm the GitHub-hosted run.

**Status:** Complete. The owner-pushed commit passed the GitHub-hosted CI run on
2026-07-15.

### Sprint 1 - Numerical trust boundary

- Reject NaN and infinity across the typed IR.
- Validate finiteness, dimensions, symmetry, and PSD tolerance for compiler
  arrays and all named vectors/matrices.
- Enforce supported budget semantics for Max-Sharpe and risk parity.
- Add direct regression and property-style boundary tests.

**Exit:** invalid numerical state cannot reach CVXPY, and transformed objectives
cannot return weights inconsistent with the confirmed budget.

**Status:** Complete. The owner-pushed commit `1d81a52` passed the GitHub-hosted
Python 3.12 CI run on 2026-07-15. The implementation:

- rejects NaN and infinity at typed-IR construction and revalidates the complete
  specification at compilation;
- coerces and validates all compiler arrays and supplied named vectors before
  CVXPY construction;
- uses scale-relative covariance symmetry and PSD tolerances, projecting only
  accepted roundoff to a numerically PSD matrix;
- preserves the documented implicit unit budget for Max-Sharpe and risk parity
  while rejecting every explicit non-unit budget; and
- covers the trust boundary with direct and parameterized regression tests.

Acceptance evidence: all 554 tests passed under Python 3.12 with the `mip` extra
and SCIP, repository-wide Ruff lint passed, and GitHub Actions run `29454520792`
completed successfully.

### Sprint 2 - Solver and report semantics

- Add typed objective decomposition and objective-appropriate portfolio metrics.
- Add unit-aware, row-aware sensitivity records.
- Correct time-limited MIP incumbent and gap handling.
- Update CLI, deterministic narration, LLM grounding, and documentation.

**Exit:** every displayed metric has a tested definition and unit; time-limited
results are either validated incumbents or explicit failures.

**Status:** Local acceptance complete on 2026-07-15; owner push and remote CI are
pending. The implementation now:

- publishes versioned `SolutionReport` schema `2.0`, with objective terms kept
  separate from objective-appropriate portfolio metrics for all six objectives;
- distinguishes annualized log-return moments, annualized variance/volatility,
  scenario-period VaR/CVaR, tracking error, Sharpe, risk-contribution deviation,
  L1 turnover, modeled cost, and transformed solver scores by explicit unit and
  definition;
- preserves every sensitivity row's constraint, ticker/label, side, bound,
  primal value, slack, raw dual, signed bound derivative, transform scale,
  objective unit, and conditionality, with explicit coverage reasons where a
  meaningful dual does not exist;
- suppresses non-identifiable multipliers from linearly dependent active rows,
  withholds derivatives from approximate solves, filters fixed-name rows with
  no selected-name support, and attaches conditional MIP sensitivities only
  when fix-and-resolve reproduces the reported portfolio;
- verifies backend-native MIP termination, every returned variable/domain,
  binary integrality, every model constraint, recovered weights, recomputed
  objective, and finite relative gap before reporting a time-limited incumbent;
- exposes typed CLI tables, `--time-limit-s`, deterministic solve-report JSON,
  field- and unit-aware LLM grounding, deterministic fallback wording,
  cross-field report invariants, and updated public report types/documentation;
  and
- fails closed on missing/malformed dual rows, ambiguous unit conversions,
  unreported small integers, invalid gap sentinels, and incomplete or infeasible
  MIP primals.

Local evidence: all 627 tests pass under Python 3.13 with Clarabel, HiGHS, and
SCIP installed; repository-wide Ruff lint and `git diff --check` pass. The
GitHub-hosted Python 3.12 run remains the sprint-closing gate.

### Sprint 3 - Conversational and workflow correctness

- Preserve clarification state across turns.
- Remove or implement reserved patch overrides; reject unknown removals.
- Normalize provider failures and deterministic fallbacks.
- Add typed operational intent and post-confirm routing for backtest, local
  replay, and live shadow.

**Exit:** the chat can complete clarified requests and route confirmed operations
without claiming an ignored or unstarted action.

### Sprint 4 - Live-shadow operator

- Add a one-shot, non-submitting operator and CLI.
- Wire official calendar, adjusted history, latest execution snapshot, strategy
  version, journal, risk configuration, incidents, and automatic session closure.
- Add restart, stale-data, missing-data, duplicate-run, and recovery tests using
  deterministic fakes.

**Exit:** an operator can run one complete official-session cycle repeatedly and
safely, with no broker write method present.

### Sprint 5 - Forward campaign and observability

- Freeze the first `$100` experiment manifest and thresholds.
- Add status/report commands, equity and drawdown curves, baseline comparisons,
  latency/staleness counters, unresolved-incident views, and alerts.
- Deploy the one-shot operator under an external scheduler.
- Accumulate the 30-session evidence record.

**Exit:** tooling is complete when deterministic operational tests pass; the
campaign gate remains open until 30 real official sessions and the predeclared
performance review are complete.

### Sprint 6 - Broker-paper lifecycle

- Add broker account, order, fill, activity, and reconciliation models.
- Add partial-fill, cancel/replace, rejection, timeout, late-fill, dividend, and
  corporate-action handling.
- Add a separate paper-only Alpaca execution adapter behind explicit confirmation
  and account protections.

**Entry:** Sprint 5's operational and performance gates are complete.  
**Exit:** a broker-paper campaign can run with deterministic IDs, exact
reconciliation, conservative parallel simulation, and no real-money surface.

### Sprint 7 - Evidence and launch

- Expand and review the parse benchmark; add explicit live evaluation with
  provenance and category-level reporting.
- Run and publish the frozen historical study without alpha guarantees.
- Build the safe visual demo and production documentation.
- Harden the package namespace, wheel CI, supported-version matrix, and release
  process.

**Exit:** Truffle is public, demoable, reproducible, and makes only evidence-backed
claims.

### Post-launch research

- Multi-period optimization over scenario trees.
- Named factor-model risk such as Fama-French.
- Black-Litterman view blending.
- Robust optimization under parameter uncertainty.
- Broader Max-Sharpe and risk-parity constraint composition.

## 9. Sprint handoff protocol

- Work directly on `main`, as requested by the repository owner.
- Complete only one sprint between owner checkpoints.
- At sprint end, report changed files, acceptance criteria, verification, and
  remaining risks.
- Do not commit or push; the repository owner reviews and pushes each sprint.
- A sprint closes only after the owner's push is green in remote CI. A failure
  reopens the same sprint rather than starting the next one.
- Default verification must remain offline and must not call paid or live
  services.
- Begin the next sprint only after the owner acknowledges the handoff.
