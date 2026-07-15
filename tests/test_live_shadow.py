"""Adversarial tests for durable, strictly delayed live-shadow execution."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

import paper.live_shadow as live_shadow
from paper.journal import (
    JournalCollisionError,
    JournalIntegrityError,
    SQLiteShadowJournal,
)
from paper.live_shadow import (
    ConfirmedShadowSignal,
    LiveShadowApprovedResult,
    LiveShadowFreshMarkError,
    LiveShadowQueuedResult,
    LiveShadowRejectedResult,
    LiveShadowSignalNotQueuedError,
    LiveShadowTimingError,
    queue_live_shadow_signal,
    run_live_shadow_step,
)
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    Quote,
    TargetAllocation,
)
from paper.planner import PaperPlanningError, PlanningConfig, plan_orders
from paper.risk import ShadowRiskLimits, gate_shadow_batch

SIGNAL_AT = datetime(2026, 7, 14, 20, tzinfo=UTC)
CONFIRMED_AT = SIGNAL_AT + timedelta(minutes=1)
EXECUTION_AT = SIGNAL_AT + timedelta(days=1)


def _snapshot(
    snapshot_id: str,
    at: datetime,
    **prices: str,
) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        as_of=at,
        quotes=tuple(
            Quote(ticker=ticker, bid=price, ask=price, last=price, as_of=at)
            for ticker, price in prices.items()
        ),
    )


def _signal(
    *,
    signal_id: str = "signal-1",
    confirmation_id: str = "confirmation-1",
    signal_snapshot: MarketSnapshot | None = None,
    confirmed_at: datetime = CONFIRMED_AT,
    weights: dict[str, str] | None = None,
) -> ConfirmedShadowSignal:
    snapshot = signal_snapshot or _snapshot("signal-close", SIGNAL_AT, AAA="10")
    return ConfirmedShadowSignal(
        signal_id=signal_id,
        confirmation_id=confirmation_id,
        strategy_version="minvar-policy@sha256:abc123",
        signal_snapshot=snapshot,
        confirmed_target=TargetAllocation(
            allocation_id=f"confirmed-{signal_id}",
            effective_at=snapshot.as_of,
            weights=weights or {"AAA": "0.5"},
        ),
        confirmed_at=confirmed_at,
    )


def _risk(
    *,
    symbols: tuple[str, ...] = ("AAA",),
    max_orders: int = 20,
    max_age: int = 300,
) -> ShadowRiskLimits:
    return ShadowRiskLimits(
        allowed_symbols=symbols,
        max_orders=max_orders,
        max_snapshot_age_seconds=max_age,
        max_order_notional="100",
        max_batch_notional="100",
        max_gross_exposure="100",
    )


def _run(
    journal: SQLiteShadowJournal,
    *,
    signal: ConfirmedShadowSignal | None = None,
    snapshot: MarketSnapshot | None = None,
    execute_not_before: datetime | None = None,
    evaluated_at: datetime | None = None,
    risk: ShadowRiskLimits | None = None,
    queue: bool = True,
):
    execution = snapshot or _snapshot("execution-close", EXECUTION_AT, AAA="10")
    confirmed_signal = signal or _signal()
    boundary = execute_not_before or CONFIRMED_AT
    if queue:
        queue_live_shadow_signal(
            journal,
            confirmed_signal,
            execute_not_before=boundary,
        )
    return run_live_shadow_step(
        journal,
        confirmed_signal,
        execution,
        execute_not_before=boundary,
        evaluated_at=evaluated_at or execution.as_of + timedelta(seconds=1),
        assumptions=ExecutionAssumptions(quantity_step="0.000001"),
        planning=PlanningConfig(),
        risk=risk or _risk(),
    )


def test_signal_requires_explicit_confirmation_at_completed_snapshot() -> None:
    with pytest.raises(ValidationError, match="strategy_version"):
        ConfirmedShadowSignal(
            signal_id="signal",
            confirmation_id="confirmation",
            strategy_version="",
            signal_snapshot=_snapshot("signal", SIGNAL_AT, AAA="10"),
            confirmed_target=TargetAllocation(
                allocation_id="target",
                effective_at=SIGNAL_AT,
                weights={"AAA": "1"},
            ),
            confirmed_at=CONFIRMED_AT,
        )

    with pytest.raises(ValidationError, match="cannot predate"):
        _signal(confirmed_at=SIGNAL_AT - timedelta(seconds=1))

    with pytest.raises(ValidationError, match="snapshot_status"):
        ConfirmedShadowSignal.model_validate(
            {
                **_signal().model_dump(),
                "snapshot_status": "still_open",
            }
        )


def test_execution_requires_a_durably_queued_signal(tmp_path: Path) -> None:
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        with pytest.raises(LiveShadowSignalNotQueuedError, match="durably queued"):
            _run(journal, queue=False)
        assert journal.read_records() == ()


@pytest.mark.parametrize(
    ("execution_at", "not_before", "match"),
    [
        (SIGNAL_AT, CONFIRMED_AT, "strictly later"),
        (
            EXECUTION_AT,
            EXECUTION_AT + timedelta(seconds=1),
            "earlier than execute_not_before",
        ),
        (
            CONFIRMED_AT - timedelta(seconds=1),
            SIGNAL_AT,
            "cannot predate explicit signal confirmation",
        ),
    ],
)
def test_same_close_or_early_execution_has_zero_journal_mutation(
    tmp_path: Path,
    execution_at: datetime,
    not_before: datetime,
    match: str,
) -> None:
    path = tmp_path / "shadow.sqlite"
    with SQLiteShadowJournal(path) as journal:
        signal = _signal()
        before = journal.read_records()
        with pytest.raises(LiveShadowTimingError, match=match):
            _run(
                journal,
                signal=signal,
                snapshot=_snapshot("too-early", execution_at, AAA="10"),
                execute_not_before=not_before,
                evaluated_at=max(CONFIRMED_AT, execution_at),
                queue=False,
            )
        assert journal.read_records() == before
        assert journal.load_ledger().state.sequence == 0


def test_evaluation_cannot_predate_confirmation_without_mutation(tmp_path: Path) -> None:
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        signal = _signal()
        before = journal.read_records()
        with pytest.raises(LiveShadowTimingError, match="risk evaluation cannot predate"):
            _run(journal, signal=signal, evaluated_at=SIGNAL_AT, queue=False)
        assert journal.read_records() == before


def test_approved_step_activates_weights_only_at_later_snapshot(tmp_path: Path) -> None:
    signal = _signal()
    execution = _snapshot("execution-close", EXECUTION_AT, AAA="10")
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        result = _run(journal, signal=signal, snapshot=execution)

        assert isinstance(result, LiveShadowApprovedResult)
        assert result.signal.signal_snapshot.as_of == SIGNAL_AT
        assert result.activated_target.effective_at == EXECUTION_AT
        assert result.activated_target.weights == signal.confirmed_target.weights
        assert result.decision.event.snapshot.snapshot_id == "execution-close"
        assert all(fill.executed_at == EXECUTION_AT for fill in result.decision.event.fills)
        assert result.ledger.sequence == 2  # fresh mark, then local simulated execution
        assert result.ledger.positions == {"AAA": Decimal("5.000000")}
        assert result.ledger.recorded_snapshot_ids == ("execution-close",)
        assert journal.verify_chain() == journal.read_records()

        kinds = tuple(record.kind for record in journal.read_records())
        assert "signal_confirmation" in kinds
        assert "confirmed_signal_queued" in kinds
        assert "market_snapshot" in kinds
        assert "confirmed_target" in kinds
        assert "planned_batch" in kinds
        assert "risk_decision" in kinds
        assert "mark" in kinds
        assert "shadow_execution" in kinds
        signal_records = journal.read_records(kind="signal_confirmation")
        confirmation = journal.decode_record(signal_records[0])
        assert confirmation.signal_id == signal.signal_id
        assert confirmation.strategy_version == signal.strategy_version
        assert confirmation.signal_snapshot == signal.signal_snapshot
        assert len(journal.read_records(kind="market_snapshot")) == 2
        assert len(journal.read_records(kind="confirmed_target")) == 2


def test_planning_sees_fresh_mark_and_gate_mode_is_always_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def inspect_plan(*args, **kwargs):
        ledger = args[2]
        observed["planning_sequence"] = ledger.sequence
        observed["planning_marks"] = dict(ledger.marks)
        return plan_orders(*args, **kwargs)

    def inspect_gate(*args, **kwargs):
        observed["mode"] = kwargs["mode"]
        return gate_shadow_batch(*args, **kwargs)

    monkeypatch.setattr(live_shadow, "plan_orders", inspect_plan)
    monkeypatch.setattr(live_shadow, "gate_shadow_batch", inspect_gate)
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        _run(journal)

    assert observed == {
        "planning_sequence": 1,
        "planning_marks": {"AAA": Decimal("10")},
        "mode": "shadow",
    }


def test_planning_failure_has_zero_journal_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_planning(*_args, **_kwargs):
        raise PaperPlanningError("deliberate planning failure")

    monkeypatch.setattr(live_shadow, "plan_orders", fail_planning)
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        signal = _signal()
        queue_live_shadow_signal(
            journal,
            signal,
            execute_not_before=CONFIRMED_AT,
        )
        before = journal.read_records()
        with pytest.raises(live_shadow.LiveShadowPlanningError, match="deliberate"):
            _run(journal, signal=signal, queue=False)
        assert journal.read_records() == before


def test_risk_rejection_journals_mark_and_incident_but_no_execution(tmp_path: Path) -> None:
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        result = _run(journal, risk=_risk(max_orders=0))

        assert isinstance(result, LiveShadowRejectedResult)
        assert {violation.code for violation in result.decision.violations} == {
            "too_many_orders"
        }
        assert result.ledger.sequence == 1
        assert result.ledger.positions == {}
        kinds = tuple(record.kind for record in journal.read_records())
        assert "risk_decision" in kinds
        assert "mark" in kinds
        assert "incident" in kinds
        assert "shadow_execution" not in kinds


def test_stale_evaluation_is_a_durable_rejection_not_an_execution(tmp_path: Path) -> None:
    with SQLiteShadowJournal(tmp_path / "shadow.sqlite") as journal:
        result = _run(
            journal,
            evaluated_at=EXECUTION_AT + timedelta(seconds=6),
            risk=_risk(max_age=5),
        )

        assert isinstance(result, LiveShadowRejectedResult)
        assert "stale_snapshot" in {
            violation.code for violation in result.decision.violations
        }
        assert result.ledger.sequence == 1
        assert result.ledger.executed_batch_ids == ()
        assert journal.read_records(kind="shadow_execution") == ()


def test_restart_and_byte_identical_retry_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    signal = _signal()
    with SQLiteShadowJournal(path) as journal:
        queued = queue_live_shadow_signal(
            journal,
            signal,
            execute_not_before=CONFIRMED_AT,
        )
        assert isinstance(queued, LiveShadowQueuedResult)
        queued_record_count = len(journal.read_records())
        queued_retry = queue_live_shadow_signal(
            journal,
            signal,
            execute_not_before=CONFIRMED_AT,
        )
        assert queued_retry.canonical_json() == queued.canonical_json()
        assert len(journal.read_records()) == queued_record_count

    with SQLiteShadowJournal(path) as restarted:
        assert restarted.read_confirmed_signal(signal.signal_id) == queued.queued_signal
        first = _run(restarted, signal=signal, queue=False)
        first_record_count = len(restarted.read_records())
        first_ledger_json = restarted.load_ledger().state.canonical_json()
        assert first_record_count > queued_record_count

    with SQLiteShadowJournal(path) as restarted_again:
        retry = _run(restarted_again, signal=signal, queue=False)
        assert retry.canonical_json() == first.canonical_json()
        assert len(restarted_again.read_records()) == first_record_count
        assert restarted_again.load_ledger().state.canonical_json() == first_ledger_json


def test_changed_queued_signal_content_collides_atomically(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    with SQLiteShadowJournal(path) as journal:
        queue_live_shadow_signal(
            journal,
            _signal(),
            execute_not_before=CONFIRMED_AT,
        )
        before = journal.read_records()

        with pytest.raises(JournalCollisionError):
            queue_live_shadow_signal(
                journal,
                _signal(weights={"AAA": "0.6"}),
                execute_not_before=CONFIRMED_AT,
            )
        assert journal.read_records() == before


def test_same_snapshot_identity_with_changed_prices_is_a_collision(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    with SQLiteShadowJournal(path) as journal:
        _run(journal)
        before = journal.read_records()

        with pytest.raises(JournalCollisionError):
            _run(
                journal,
                snapshot=_snapshot("execution-close", EXECUTION_AT, AAA="11"),
            )
        assert journal.read_records() == before


def test_one_queued_signal_cannot_execute_at_a_second_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    signal = _signal()
    with SQLiteShadowJournal(path) as journal:
        _run(journal, signal=signal)
        before_records = journal.read_records()
        before_ledger = journal.load_ledger().state.canonical_json()

        with pytest.raises(JournalCollisionError):
            _run(
                journal,
                signal=signal,
                snapshot=_snapshot(
                    "second-execution-close",
                    EXECUTION_AT + timedelta(days=1),
                    AAA="11",
                ),
                queue=False,
            )
        assert journal.read_records() == before_records
        assert journal.load_ledger().state.canonical_json() == before_ledger


def test_retry_result_stays_identical_after_a_later_signal_executes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite"
    first_signal = _signal()
    first_execution = _snapshot("execution-close", EXECUTION_AT, AAA="10")
    with SQLiteShadowJournal(path) as journal:
        first = _run(
            journal,
            signal=first_signal,
            snapshot=first_execution,
        )

        second_signal_at = EXECUTION_AT + timedelta(days=1)
        second_signal = _signal(
            signal_id="signal-2",
            confirmation_id="confirmation-2",
            signal_snapshot=_snapshot("signal-2-close", second_signal_at, AAA="11"),
            confirmed_at=second_signal_at + timedelta(minutes=1),
            weights={"AAA": "0.2"},
        )
        _run(
            journal,
            signal=second_signal,
            snapshot=_snapshot(
                "execution-2-close",
                second_signal_at + timedelta(days=1),
                AAA="12",
            ),
            execute_not_before=second_signal.confirmed_at,
        )
        count_after_second = len(journal.read_records())

        retry = _run(
            journal,
            signal=first_signal,
            snapshot=first_execution,
            queue=False,
        )
        assert retry.canonical_json() == first.canonical_json()
        assert len(journal.read_records()) == count_after_second


def test_execution_snapshot_must_freshly_mark_every_held_symbol(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    with SQLiteShadowJournal(path) as journal:
        _run(journal)
        before = journal.read_records()

        second_signal_at = EXECUTION_AT + timedelta(days=1)
        second_signal = _signal(
            signal_id="signal-2",
            confirmation_id="confirmation-2",
            signal_snapshot=_snapshot("signal-2-close", second_signal_at, BBB="20"),
            confirmed_at=second_signal_at + timedelta(minutes=1),
            weights={"BBB": "1"},
        )
        incomplete_execution = _snapshot(
            "execution-2-close",
            second_signal_at + timedelta(days=1),
            BBB="20",
        )
        queue_live_shadow_signal(
            journal,
            second_signal,
            execute_not_before=second_signal.confirmed_at,
        )
        before = journal.read_records()
        with pytest.raises(LiveShadowFreshMarkError, match="missing held positions"):
            _run(
                journal,
                signal=second_signal,
                snapshot=incomplete_execution,
                execute_not_before=second_signal.confirmed_at,
                risk=_risk(symbols=("AAA", "BBB")),
                queue=False,
            )
        assert journal.read_records() == before


def test_journal_tamper_error_propagates_without_new_records(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite"
    with SQLiteShadowJournal(path) as journal:
        _run(journal)

    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER journal_records_no_update")
        connection.execute(
            "UPDATE journal_records SET payload_json = '{}' WHERE sequence = 1"
        )
        connection.execute(
            "CREATE TRIGGER journal_records_no_update "
            "BEFORE UPDATE ON journal_records "
            "BEGIN SELECT RAISE(ABORT, 'journal records are append-only'); END"
        )

    with SQLiteShadowJournal(path) as tampered:
        with pytest.raises(JournalIntegrityError):
            _run(tampered)
