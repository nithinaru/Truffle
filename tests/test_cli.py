"""CLI coverage for opt-in, deterministic infeasibility diagnosis."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import cli
from core.exceptions import InfeasibleError
from core.patch import SpecPatch
from core.report import (
    ConflictEvidence,
    ConflictMember,
    ConflictReport,
    ElasticResult,
    GroundValue,
    Repair,
    RepairChange,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """\
universe: [AAA, BBB]
objective:
  kind: min_variance
constraints:
  - kind: budget
    id: budget
  - kind: long_only
    id: long_only
""",
        encoding="utf-8",
    )
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text(
        """\
date,AAA,BBB
2025-01-01,100,100
2025-01-02,101,99
2025-01-03,102,100
""",
        encoding="utf-8",
    )
    return spec_path, prices_path


def _conflict_report() -> ConflictReport:
    repair = Repair(
        repair_id="raise_cap",
        description="Raise the position cap from 40% to 50%.",
        patch=SpecPatch(remove_constraint_ids=["position_cap"]),
        changes=(
            RepairChange(
                constraint_id="position_cap",
                field="upper",
                direction="raise",
                old_value=0.4,
                solver_required_value=0.5,
                applied_value=0.5,
                required_change=0.1,
                normalized_change=0.25,
                unit="fraction",
            ),
        ),
        required_change=0.1,
        relative_change=0.25,
        kind="single_lever",
        rank=1,
        verified=True,
    )
    return ConflictReport(
        solver_status="infeasible",
        n_assets=2,
        minimality_status="verified_iis",
        conflict_scope="mixed",
        candidate_constraint_ids=("budget", "position_cap"),
        conflict_set=(
            ConflictMember(
                constraint_id="budget",
                constraint_kind="budget",
                human_name="Budget = 100%",
                relaxability="structural",
            ),
            ConflictMember(
                constraint_id="position_cap",
                constraint_kind="box",
                human_name="Position cap = 40%",
                relaxability="relaxable",
            ),
        ),
        elastic=ElasticResult(
            kind="soft_repair",
            status="optimal",
            solver="Clarabel",
            solve_time_ms=1.0,
            total_relative_slack=0.25,
        ),
        evidence=(
            ConflictEvidence(
                text="Budget requires 100%, while aggregate caps allow only 80%.",
                values=(
                    GroundValue(key="budget", value=1.0, unit="fraction", source="spec"),
                    GroundValue(key="aggregate_cap", value=0.8, unit="fraction", source="derived"),
                ),
            ),
        ),
        repairs=(repair,),
    )


@pytest.mark.parametrize("flag", [[], ["--no-diagnose"]])
def test_solve_diagnosis_is_off_by_default_and_explicitly_disableable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: list[str],
) -> None:
    spec_path, prices_path = _inputs(tmp_path)
    calls: list[bool] = []

    def fake_solve(*args: object, diagnose: bool, **kwargs: object) -> None:
        calls.append(diagnose)
        raise InfeasibleError("synthetic infeasibility")

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, color_system=None))
    monkeypatch.setattr(cli, "solve_spec", fake_solve)

    result = CliRunner().invoke(
        cli.app,
        ["solve", str(spec_path), "--prices", str(prices_path), *flag],
    )

    assert result.exit_code == 3
    assert calls == [False]
    output = stream.getvalue()
    assert "Re-run with --diagnose" in output
    assert "Constraints conflict" not in output


def test_solve_diagnose_renders_deterministic_report_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, prices_path = _inputs(tmp_path)
    report = _conflict_report()
    calls: list[bool] = []

    def fake_solve(*args: object, diagnose: bool, **kwargs: object) -> None:
        calls.append(diagnose)
        raise InfeasibleError("synthetic infeasibility", conflict_report=report)

    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, color_system=None))
    monkeypatch.setattr(cli, "solve_spec", fake_solve)

    result = CliRunner().invoke(
        cli.app,
        ["solve", str(spec_path), "--prices", str(prices_path), "--diagnose"],
    )

    assert result.exit_code == 3
    assert calls == [True]
    output = stream.getvalue()
    assert "The verified irreducible conflict contains" in output
    assert "Budget requires 100%, while aggregate caps allow only 80%." in output
    assert "Verified repairs" in output
    assert "Raise the position cap from 40% to 50%." in output
    assert "Apply one of the verified changes to the YAML spec" in output
    assert "Type a repair number" not in output


def test_solve_renders_typed_units_and_writes_versioned_report(tmp_path: Path) -> None:
    spec_path, prices_path = _inputs(tmp_path)
    json_path = tmp_path / "solve-report.json"

    result = CliRunner().invoke(
        cli.app,
        [
            "solve",
            str(spec_path),
            "--prices",
            str(prices_path),
            "--json-out",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Objective decomposition" in result.output
    assert "Portfolio metrics" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["field_units"]["objective_value"] == "objective_score"
    assert payload["field_units"]["weights"] == "portfolio_weight_fraction"
    assert payload["objective_decomposition"]["solver_unit"] == "objective_score"
    assert {metric["key"] for metric in payload["metrics"]} >= {
        "expected_return",
        "variance",
        "volatility",
    }
    variance = next(metric for metric in payload["metrics"] if metric["key"] == "variance")
    assert variance["unit"] == "fraction_squared_per_year"
    assert payload["sensitivities"]
    assert all("bound_unit" in item for item in payload["sensitivities"])


def test_backtest_command_writes_delayed_fill_tearsheet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """\
universe: [AAA]
objective:
  kind: min_variance
constraints:
  - kind: budget
    id: budget
  - kind: long_only
    id: long_only
""",
        encoding="utf-8",
    )
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text(
        """\
date,AAA
2024-01-29,100
2024-01-30,101
2024-01-31,100
2024-02-01,102
2024-02-02,103
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "tearsheet.json"
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, color_system=None))

    result = CliRunner().invoke(
        cli.app,
        [
            "backtest",
            str(spec_path),
            "--prices",
            str(prices_path),
            "--lookback",
            "2",
            "--rebalance",
            "monthly",
            "--json-out",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    output = stream.getvalue()
    assert "Walk-forward results" in output
    assert "Wrote deterministic tearsheet JSON" in output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "tearsheet"
    assert payload["rebalances"][0]["signal_date"] == "2024-01-31"
    assert payload["rebalances"][0]["fill_date"] == "2024-02-01"
    assert payload["rebalances"][0]["training_end"] == "2024-01-31"
