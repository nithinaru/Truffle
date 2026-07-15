"""Truffle CLI.

Usage:
    python cli.py solve examples/spec_minvar.yaml --prices examples/prices_sample.csv

Loads a YAML spec into the IR, validates it, estimates mu/sigma from a CSV
price panel, compiles and solves with Clarabel, and pretty-prints the
results (weights table, objective, problem class, solver stats, shadow
prices) using Rich.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from core.exceptions import InfeasibleError, SolverError, UnboundedError
from core.ir import PortfolioSpec
from core.report import ConflictReport, SolutionReport
from core.solve import solve_spec
from data.inputs import load_named_series, load_sectors

app = typer.Typer(add_completion=False, help="Truffle: typed portfolio optimization.")
console = Console()


@app.callback()
def _root() -> None:
    """Truffle: typed portfolio optimization.

    A no-op callback so Typer treats ``solve`` as an explicit subcommand
    rather than promoting it to the default entrypoint.
    """


def _load_spec(spec_path: Path) -> PortfolioSpec:
    with spec_path.open("r") as f:
        payload = yaml.safe_load(f)
    return PortfolioSpec.model_validate(payload)


def _load_prices(prices_path: Path, universe: list[str]) -> pd.DataFrame:
    df = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
    missing = [t for t in universe if t not in df.columns]
    if missing:
        raise SystemExit(
            f"Prices CSV is missing columns for universe tickers: {missing}.\n"
            f"  CSV columns: {list(df.columns)}"
        )
    # Reorder to match the spec's universe — order must be canonical
    # because compile_spec indexes mu/sigma by position.
    return df[universe]


def _render_weights(spec: PortfolioSpec, weights: list[float]) -> Table:
    t = Table(title="Optimal weights", show_lines=False)
    t.add_column("Ticker", style="bold cyan")
    t.add_column("Weight", justify="right")
    t.add_column("Weight %", justify="right")
    for ticker, w in zip(spec.universe, weights, strict=True):
        t.add_row(ticker, f"{w:.6f}", f"{100.0 * w:.2f}%")
    return t


def _render_binding(report: SolutionReport) -> Table:
    t = Table(title="Binding constraints (shadow prices, largest first)")
    t.add_column("Constraint", style="bold magenta")
    t.add_column("Shadow price", justify="right")
    for b in report.binding:
        t.add_row(b.human_name, f"{b.shadow_price:.6f}")
    return t


def _load_named(path: Path | None, *, label: str) -> dict[str, dict[str, float]] | None:
    return None if path is None else load_named_series(path, label=label)


@app.command()
def solve(
    spec_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML spec file."),
    prices: Path = typer.Option(
        ..., "--prices", exists=True, readable=True, help="CSV of historical prices."
    ),
    sectors: Path | None = typer.Option(
        None, "--sectors", exists=True, readable=True, help="ticker,sector CSV (for group caps)."
    ),
    benchmark: Path | None = typer.Option(
        None, "--benchmark", exists=True, readable=True,
        help="Wide ticker,<name> CSV of benchmark weights (tracking-error nodes).",
    ),
    factors: Path | None = typer.Option(
        None, "--factors", exists=True, readable=True,
        help="Wide ticker,<factor> CSV of factor loadings (factor-exposure nodes).",
    ),
    diagnose: bool = typer.Option(
        False,
        "--diagnose/--no-diagnose",
        help="On infeasibility, run deterministic conflict analysis and show verified repairs.",
    ),
) -> None:
    """Solve the portfolio problem described in ``spec_path`` against ``prices``."""
    try:
        spec = _load_spec(spec_path)
    except Exception as e:
        console.print(f"[red]Spec validation failed:[/red] {e}")
        raise typer.Exit(code=2) from None

    console.print(f"[bold]Spec loaded:[/bold] {spec_path}")
    console.print(
        f"  universe = {len(spec.universe)} tickers · "
        f"objective = [bold]{spec.objective.kind}[/bold] · "
        f"problem class = [bold]{spec.problem_class}[/bold]"
    )

    price_df = _load_prices(prices, spec.universe)
    sector_map = load_sectors(sectors) if sectors is not None else None
    try:
        _compiled, report = solve_spec(
            spec,
            price_df,
            sectors=sector_map,
            benchmarks=_load_named(benchmark, label="Benchmark"),
            factors=_load_named(factors, label="Factor"),
            diagnose=diagnose,
        )
    except InfeasibleError as e:
        console.print(f"[red]Problem is infeasible.[/red] {e}")
        if diagnose and isinstance(e.conflict_report, ConflictReport):
            _render_conflict(
                _deterministic_conflict_summary(e.conflict_report),
                e.conflict_report,
                interactive=False,
            )
        elif not diagnose:
            console.print(
                "[dim]Re-run with --diagnose to identify the conflicting constraints "
                "and show verified repairs.[/dim]"
            )
        raise typer.Exit(code=3) from None
    except UnboundedError as e:
        console.print(f"[red]Problem is unbounded.[/red] {e}")
        raise typer.Exit(code=4) from None
    except (SolverError, ValueError) as e:
        console.print(f"[red]Solve failed:[/red] {e}")
        raise typer.Exit(code=5) from None

    console.print(_render_weights(spec, [report.weights[t] for t in spec.universe]))
    var_line = f"  ·  VaR: {report.var:.6f}" if report.var is not None else ""
    console.print(
        f"\n[bold]Objective value:[/bold] {report.objective_value:.6f}{var_line}"
        f"  ·  [bold]Solver:[/bold] {report.solver}"
        f"  ·  [bold]Status:[/bold] {report.status}"
        f"  ·  [bold]Time:[/bold] {report.solve_time_ms:.1f} ms"
    )
    if report.binding:
        console.print(_render_binding(report))
    else:
        console.print("[dim]No constraints are binding at the optimum.[/dim]")
    if report.duals_conditional and report.selected_names is not None:
        gap = "n/a" if report.optimality_gap is None else f"{report.optimality_gap:.4f}"
        console.print(
            f"[yellow]Mixed-integer solve:[/yellow] {len(report.selected_names)} names "
            f"selected ({', '.join(report.selected_names)}); shadow prices above are "
            f"conditional on this selection. Optimality gap: {gap}."
        )


def _render_solved(report: SolutionReport, explanation: str) -> None:
    """Pretty-print a chat-mode solve: explanation, weights, binders."""
    console.print(f"\n[bold]Explanation[/bold]\n{explanation}\n")
    weights_table = Table(title="Optimal weights")
    weights_table.add_column("Ticker", style="bold cyan")
    weights_table.add_column("Weight", justify="right")
    weights_table.add_column("Weight %", justify="right")
    for ticker, w in report.weights.items():
        weights_table.add_row(ticker, f"{w:.6f}", f"{100.0 * w:.2f}%")
    console.print(weights_table)
    if report.binding:
        title = (
            "Binding constraints (conditional on selected names)"
            if report.duals_conditional
            else "Binding constraints"
        )
        binders = Table(title=title)
        binders.add_column("Constraint", style="bold magenta")
        binders.add_column("Shadow price", justify="right")
        for b in report.binding:
            binders.add_row(b.human_name, f"{b.shadow_price:.6f}")
        console.print(binders)
    if report.duals_conditional and report.selected_names is not None:
        gap = "n/a" if report.optimality_gap is None else f"{report.optimality_gap:.4f}"
        console.print(
            f"[yellow]Mixed-integer solve:[/yellow] {len(report.selected_names)} names "
            f"selected ({', '.join(report.selected_names)}); shadow prices are "
            f"conditional on this selection. Optimality gap: {gap}."
        )
    var_line = f"  ·  VaR: {report.var:.6f}" if report.var is not None else ""
    console.print(
        f"[dim]objective = {report.objective_value:.6f}{var_line}  ·  "
        f"solver = {report.solver}  ·  time = {report.solve_time_ms:.1f} ms[/dim]"
    )


def _render_backtest(sheet: object) -> None:
    """Render a compact deterministic tearsheet summary."""
    from backtest.tearsheet import Tearsheet  # noqa: PLC0415

    if not isinstance(sheet, Tearsheet):
        return
    summary = sheet.summary
    table = Table(title=f"Walk-forward results · {sheet.start_date} to {sheet.end_date}")
    table.add_column("Series", style="bold cyan")
    table.add_column("Total", justify="right")
    table.add_column("Annualized", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max drawdown", justify="right")
    rows = [
        ("Truffle (net)", summary.strategy),
        ("Truffle (gross)", summary.strategy_gross),
        ("Equal weight (net)", summary.equal_weight),
    ]
    if summary.market is not None:
        rows.append(("Market (gross)", summary.market))
    for label, metrics in rows:
        sharpe = "n/a" if metrics.sharpe is None else f"{metrics.sharpe:.3f}"
        table.add_row(
            label,
            f"{100.0 * metrics.total_return:.2f}%",
            f"{100.0 * metrics.annualized_return:.2f}%",
            f"{100.0 * metrics.annualized_volatility:.2f}%",
            sharpe,
            f"{100.0 * metrics.max_drawdown:.2f}%",
        )
    console.print(table)
    modeled = summary.holding_weighted_modeled_cvar
    realized = summary.holding_weighted_realized_cvar
    modeled_text = "n/a" if modeled is None else f"{100.0 * modeled:.3f}%"
    realized_text = "n/a" if realized is None else f"{100.0 * realized:.3f}%"
    console.print(
        f"[bold]Rebalances:[/bold] {len(sheet.rebalances)}  ·  "
        f"[bold]L1 turnover:[/bold] {summary.total_turnover:.3f}  ·  "
        f"[bold]Cost paid:[/bold] {100.0 * summary.total_cost_paid:.3f}% initial NAV  ·  "
        f"[bold]Cost drag:[/bold] {100.0 * summary.cost_drag:.3f}%"
    )
    console.print(
        f"[bold]Holding-weighted CVaR:[/bold] modeled {modeled_text}  ·  "
        f"realized {realized_text}  ·  alpha={sheet.config.cvar_alpha:g}"
    )


def _load_market_prices(path: Path, column: str | None) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    if column is None:
        if len(frame.columns) != 1:
            raise ValueError(
                "Market prices CSV has multiple value columns; pass --market-column."
            )
        column = str(frame.columns[0])
    if column not in frame.columns:
        raise ValueError(
            f"Market prices CSV has no column {column!r}; columns are {list(frame.columns)}."
        )
    return frame[column]


@app.command("backtest")
def backtest_command(
    spec_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML spec file."),
    prices: Path = typer.Option(
        ..., "--prices", exists=True, readable=True, help="CSV of adjusted historical closes."
    ),
    lookback: int = typer.Option(252, "--lookback", min=2, help="Trailing return observations."),
    rebalance: Literal["weekly", "monthly", "quarterly"] = typer.Option(
        "monthly", "--rebalance", help="Calendar rebalance frequency."
    ),
    cost_bps: float | None = typer.Option(
        None,
        "--cost-bps",
        min=0.0,
        help="Execution cost per unit of L1 turnover; defaults to spec cost nodes.",
    ),
    periods_per_year: int = typer.Option(252, "--periods-per-year", min=1),
    cvar_alpha: float = typer.Option(0.95, "--cvar-alpha", min=0.000001, max=0.999999),
    risk_free_rate: float = typer.Option(0.0, "--risk-free-rate", min=-0.999999),
    market_prices: Path | None = typer.Option(
        None,
        "--market-prices",
        exists=True,
        readable=True,
        help="Optional local CSV for a gross market/SPY baseline.",
    ),
    market_column: str | None = typer.Option(None, "--market-column"),
    sectors: Path | None = typer.Option(None, "--sectors", exists=True, readable=True),
    benchmark: Path | None = typer.Option(None, "--benchmark", exists=True, readable=True),
    factors: Path | None = typer.Option(None, "--factors", exists=True, readable=True),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Write the complete deterministic tearsheet JSON to this path.",
    ),
) -> None:
    """Run a delayed-fill, no-lookahead walk-forward backtest."""
    from backtest import BacktestConfig, BacktestError, run_backtest  # noqa: PLC0415

    try:
        spec = _load_spec(spec_path)
        panel = _load_prices(prices, spec.universe)
        config = BacktestConfig(
            lookback_returns=lookback,
            rebalance_frequency=rebalance,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
            execution_cost_bps=cost_bps,
            annual_risk_free_rate=risk_free_rate,
        )
        market = (
            None
            if market_prices is None
            else _load_market_prices(market_prices, market_column)
        )
        sheet = run_backtest(
            spec,
            panel,
            config=config,
            sectors=load_sectors(sectors) if sectors is not None else None,
            benchmarks=_load_named(benchmark, label="Benchmark"),
            factors=_load_named(factors, label="Factor"),
            market_prices=market,
        )
    except (BacktestError, SolverError, ValueError) as exc:
        console.print(f"[red]Backtest failed:[/red] {exc}")
        raise typer.Exit(code=6) from None

    _render_backtest(sheet)
    if json_out is not None:
        json_out.write_text(sheet.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]Wrote deterministic tearsheet JSON to {json_out}.[/dim]")


def _deterministic_conflict_summary(report: ConflictReport) -> str:
    """Describe a diagnosis using only trusted fields from ``report``."""
    names = ", ".join(member.human_name for member in report.conflict_set)
    if report.minimality_status == "verified_iis":
        opening = f"The verified irreducible conflict contains: {names}."
    else:
        opening = (
            "These constraints appear to conflict, but minimality was not "
            f"verified: {names}."
        )

    parts = [opening]
    parts.extend(evidence.text for evidence in report.evidence)
    structural = [
        member.human_name
        for member in report.conflict_set
        if member.relaxability != "relaxable"
    ]
    if structural:
        parts.append(
            "The conflict includes structural or non-negotiable constraints: "
            + ", ".join(structural)
            + "."
        )
    if report.repairs:
        choices = " ".join(
            f"{repair.rank}. {repair.description}" for repair in report.repairs
        )
        parts.append("Verified repair options: " + choices)
    else:
        parts.append("No verified single-change repair is available from this diagnosis.")
    return " ".join(parts)


def _render_conflict(
    explanation: str,
    report: object,
    *,
    interactive: bool = True,
) -> None:
    """Render a grounded conflict explanation and verified repair choices."""
    if not isinstance(report, ConflictReport):
        return
    console.print(f"\n[bold red]Constraints conflict[/bold red]\n{explanation}\n")
    if report.repairs:
        repairs = Table(title="Verified repairs")
        repairs.add_column("Choice", justify="right", style="bold cyan")
        repairs.add_column("Change")
        repairs.add_column("Type", style="dim")
        for repair in report.repairs:
            repairs.add_row(str(repair.rank), repair.description, repair.kind)
        console.print(repairs)
        if interactive:
            console.print(
                "[dim]Type a repair number to apply it; Truffle will re-echo before solving.[/dim]"
            )
        else:
            console.print(
                "[dim]Apply one of the verified changes to the YAML spec, then run solve again.[/dim]"
            )
    else:
        console.print("[yellow]No verified repair is available for automatic application.[/yellow]")


@app.command()
def chat(
    prices: Path = typer.Option(..., "--prices", exists=True, readable=True),
    sectors: Path | None = typer.Option(None, "--sectors", exists=True, readable=True),
    benchmark: Path | None = typer.Option(None, "--benchmark", exists=True, readable=True),
    factors: Path | None = typer.Option(None, "--factors", exists=True, readable=True),
) -> None:
    """Interactive natural-language session against ``--prices``."""
    # Local imports keep `solve` working without anthropic/loop dependencies
    # loaded for users who never enter chat mode.
    from agent.client import AnthropicClient  # noqa: PLC0415
    from agent.loop import ChatSession  # noqa: PLC0415

    try:
        client = AnthropicClient()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None

    price_df = pd.read_csv(prices, parse_dates=[0], index_col=0)
    sector_map = load_sectors(sectors) if sectors is not None else None
    session = ChatSession(
        client=client,
        prices=price_df,
        sectors=sector_map,
        benchmarks=_load_named(benchmark, label="Benchmark"),
        factors=_load_named(factors, label="Factor"),
    )

    console.print(
        "[bold]Truffle chat.[/bold] Tell me about the portfolio you want. "
        "Type 'start over' to reset, Ctrl-C to exit.\n"
    )
    while True:
        try:
            user_text = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            return
        if not user_text:
            continue

        result = session.handle_user_message(user_text)
        if result.kind == "clarification":
            console.print(f"[yellow]?[/yellow] {result.text}\n")
            continue
        if result.kind == "info":
            console.print(f"[dim]{result.text}[/dim]\n")
            continue
        if result.kind == "error":
            console.print(f"[red]{result.text}[/red]\n")
            continue
        if result.kind == "echo":
            console.print(result.text)
            try:
                decision = console.input(
                    "\n[bold]Proceed?[/bold] [y/n/edit]: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                return
            outcome = session.confirm_pending(decision)
            if outcome.kind == "solved" and outcome.report and outcome.explanation:
                _render_solved(outcome.report, outcome.explanation)
            elif (
                outcome.kind == "conflict"
                and outcome.conflict_report
                and outcome.explanation
            ):
                _render_conflict(outcome.explanation, outcome.conflict_report)
            elif outcome.kind == "error":
                console.print(f"[red]{outcome.text}[/red]\n")
            else:
                console.print(f"[dim]{outcome.text}[/dim]\n")
            continue
        if result.kind == "solved" and result.report and result.explanation:
            _render_solved(result.report, result.explanation)
            continue
        if result.kind == "conflict" and result.conflict_report and result.explanation:
            _render_conflict(result.explanation, result.conflict_report)
            continue


if __name__ == "__main__":
    app()
