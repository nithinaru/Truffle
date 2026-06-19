"""Truffle CLI.

Usage:
    python cli.py solve examples/spec_minvar.yaml --prices examples/prices_sample.csv

Loads a YAML spec into the IR, validates it, estimates mu/sigma from a CSV
price panel, compiles and solves with Clarabel, and pretty-prints the
results (weights table, objective, problem class, solver stats, shadow
prices) using Rich.
"""

from __future__ import annotations

import time
from pathlib import Path

import cvxpy as cp
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from core.compiler import compile_spec
from core.duals import harvest_duals
from core.exceptions import InfeasibleError, SolverError, TruffleError, UnboundedError
from core.ir import PortfolioSpec
from core.report import SolutionReport
from data.estimation import estimate_moments

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


def _render_duals(spec: PortfolioSpec, duals: dict[str, float]) -> Table:
    t = Table(title="Shadow prices (binding constraints first)")
    t.add_column("Constraint id", style="bold magenta")
    t.add_column("Kind")
    t.add_column("Shadow price", justify="right")
    t.add_column("Binding?", justify="center")
    kinds = {c.id: c.kind for c in spec.constraints}
    rows = sorted(duals.items(), key=lambda kv: -abs(kv[1]))
    for cid, sp in rows:
        binding = "yes" if abs(sp) > 1e-6 else "no"
        t.add_row(cid, kinds.get(cid, "?"), f"{sp:.6f}", binding)
    return t


@app.command()
def solve(
    spec_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML spec file."),
    prices: Path = typer.Option(
        ..., "--prices", exists=True, readable=True, help="CSV of historical prices."
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
    mu, sigma = estimate_moments(price_df)

    compiled = compile_spec(spec, mu, sigma)
    start = time.perf_counter()
    try:
        compiled.problem.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        raise SolverError(f"Clarabel failed: {e}") from e
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    status = compiled.problem.status
    if status in {"infeasible", "infeasible_inaccurate"}:
        console.print(f"[red]Problem is infeasible[/red] (status: {status}).")
        raise typer.Exit(code=3)
    if status in {"unbounded", "unbounded_inaccurate"}:
        console.print(f"[red]Problem is unbounded[/red] (status: {status}).")
        raise typer.Exit(code=4)

    weights = list(compiled.recovered_weights())
    console.print(_render_weights(spec, weights))

    console.print(
        f"\n[bold]Objective value:[/bold] {compiled.problem.value:.6f}"
        f"  ·  [bold]Solver:[/bold] Clarabel"
        f"  ·  [bold]Status:[/bold] {status}"
        f"  ·  [bold]Time:[/bold] {elapsed_ms:.1f} ms"
    )

    try:
        duals = harvest_duals(compiled)
    except (InfeasibleError, UnboundedError, TruffleError) as e:
        console.print(f"[yellow]Duals unavailable:[/yellow] {e}")
        return

    console.print(_render_duals(spec, duals))


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
        binders = Table(title="Binding constraints")
        binders.add_column("Constraint", style="bold magenta")
        binders.add_column("Shadow price", justify="right")
        for b in report.binding:
            binders.add_row(b.human_name, f"{b.shadow_price:.6f}")
        console.print(binders)
    var_line = f"  ·  VaR: {report.var:.6f}" if report.var is not None else ""
    console.print(
        f"[dim]objective = {report.objective_value:.6f}{var_line}  ·  "
        f"solver = {report.solver}  ·  time = {report.solve_time_ms:.1f} ms[/dim]"
    )


def _load_sectors(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    if not {"ticker", "sector"}.issubset(df.columns):
        raise SystemExit("--sectors CSV must have columns: ticker, sector.")
    return dict(zip(df["ticker"], df["sector"], strict=True))


@app.command()
def chat(
    prices: Path = typer.Option(..., "--prices", exists=True, readable=True),
    sectors: Path | None = typer.Option(None, "--sectors", exists=True, readable=True),
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
    sector_map = _load_sectors(sectors)
    session = ChatSession(client=client, prices=price_df, sectors=sector_map)

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
            elif outcome.kind == "error":
                console.print(f"[red]{outcome.text}[/red]\n")
            else:
                console.print(f"[dim]{outcome.text}[/dim]\n")
            continue
        if result.kind == "solved" and result.report and result.explanation:
            _render_solved(result.report, result.explanation)
            continue


if __name__ == "__main__":
    app()
