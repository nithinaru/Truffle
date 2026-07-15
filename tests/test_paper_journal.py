"""Durability and integrity tests for the local SQLite shadow journal."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

import paper.journal as journal_module
from paper.journal import (
    ConfirmedSignalQueued,
    JournalClosedError,
    JournalCollisionError,
    JournalIntegrityError,
    JournalSchemaError,
    OperationalIncident,
    OperationalSessionClosed,
    ShadowStepManifest,
    SignalConfirmation,
    SQLiteShadowJournal,
)
from paper.ledger import ShadowMarkRecorded
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    OrderBatch,
    OrderIntent,
    Quote,
    ShadowBatchExecuted,
    SimulationConfig,
    TargetAllocation,
)
from paper.risk import RiskViolation, ShadowRiskApproved, ShadowRiskRejected
from paper.simulator import simulate_shadow_batch

T0 = datetime(2026, 1, 2, 20, tzinfo=UTC)


def _snapshot(snapshot_id: str, at: datetime, price: str = "10.00") -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        as_of=at,
        quotes=(
            Quote(ticker="AAA", bid=price, ask=price, last=price, as_of=at),
        ),
    )


def _approved_step(
    *,
    event_sequence: int = 1,
    signal_id: str = "signal-1",
    provenance_id: str = "provenance-1",
    signal_snapshot_id: str = "signal-snapshot-1",
    signal_target_id: str = "signal-target-1",
    snapshot_id: str = "execution-snapshot-1",
    target_id: str = "activated-target-1",
    batch_id: str = "batch-1",
    order_id: str = "order-1",
    signal_at: datetime = T0,
    execution_at: datetime = T0 + timedelta(minutes=2),
    with_order: bool = True,
) -> tuple[
    SignalConfirmation,
    MarketSnapshot,
    TargetAllocation,
    OrderBatch,
    ShadowRiskApproved,
    tuple[ShadowBatchExecuted | ShadowMarkRecorded, ...],
]:
    signal_snapshot = _snapshot(signal_snapshot_id, signal_at)
    confirmed_target = TargetAllocation(
        allocation_id=signal_target_id,
        effective_at=signal_at,
        weights={"AAA": "0.5"},
    )
    snapshot = _snapshot(snapshot_id, execution_at)
    target = TargetAllocation(
        allocation_id=target_id,
        effective_at=execution_at,
        weights={"AAA": "0.5"},
    )
    provenance = SignalConfirmation(
        provenance_id=provenance_id,
        signal_id=signal_id,
        strategy_version="strategy-v1",
        signal_snapshot=signal_snapshot,
        confirmation_id=f"confirmation-{provenance_id}",
        confirmed_at=signal_at + timedelta(minutes=1),
        confirmed_target=confirmed_target,
        activated_target=target,
    )
    orders = (
        (
            OrderIntent(
                order_id=order_id,
                ticker="AAA",
                side="buy",
                quantity="5",
                reference_price="10.00",
            ),
        )
        if with_order
        else ()
    )
    batch = OrderBatch(
        batch_id=batch_id,
        allocation_id=target_id,
        effective_at=execution_at,
        orders=orders,
    )
    assumptions = ExecutionAssumptions()
    simulation = SimulationConfig()
    execution = simulate_shadow_batch(
        batch,
        snapshot,
        assumptions,
        sequence=event_sequence,
        config=simulation,
    )
    notional = sum((fill.notional for fill in execution.fills), Decimal("0"))
    decision = ShadowRiskApproved(
        evaluated_at=execution_at,
        snapshot_id=snapshot_id,
        batch_id=batch_id,
        projected_cash=Decimal("100") - notional,
        projected_gross_exposure=notional,
        batch_notional=notional,
        simulation=simulation,
        event=execution,
    )
    mark = ShadowMarkRecorded(sequence=event_sequence + 1, snapshot=snapshot)
    return provenance, snapshot, target, batch, decision, (execution, mark)


def _commit_approved(
    journal: SQLiteShadowJournal,
    step: tuple[
        SignalConfirmation,
        MarketSnapshot,
        TargetAllocation,
        OrderBatch,
        ShadowRiskApproved,
        tuple[ShadowBatchExecuted | ShadowMarkRecorded, ...],
    ],
):
    provenance, snapshot, target, batch, decision, events = step
    journal.record_confirmed_signal(
        ConfirmedSignalQueued(
            signal_id=provenance.signal_id,
            strategy_version=provenance.strategy_version,
            signal_snapshot=provenance.signal_snapshot,
            confirmation_id=provenance.confirmation_id,
            confirmed_at=provenance.confirmed_at,
            confirmed_target=provenance.confirmed_target,
            execute_not_before=target.effective_at,
        )
    )
    return journal.commit_shadow_step(
        signal_confirmation=provenance,
        snapshot=snapshot,
        target=target,
        batch=batch,
        decision=decision,
        ledger_events=events,
    )


def test_atomic_step_restart_recovery_and_idempotent_lookup(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    step = _approved_step()
    with SQLiteShadowJournal(path) as journal:
        first = _commit_approved(journal, step)
        retry = _commit_approved(journal, step)
        assert retry == first
        prior = journal.read_shadow_step("provenance-1")
        assert prior is not None
        assert prior[:-1] == first[:-1]
        assert isinstance(journal.decode_record(prior[-1]), ShadowStepManifest)
        expected_events = journal.replay_ledger_events()
        expected_state = journal.load_ledger().state

    with SQLiteShadowJournal(path) as restarted:
        assert restarted.replay_ledger_events() == expected_events
        recovered = restarted.load_ledger()
        assert recovered.events == expected_events
        assert recovered.state == expected_state
        assert recovered.state.cash == Decimal("50.00")
        assert recovered.state.positions == {"AAA": Decimal("5")}
        assert recovered.state.sequence == 2
        assert restarted.read_shadow_step("missing") is None


def test_confirmed_signal_is_durable_before_activation_and_collides_safely(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    provenance, _snapshot_value, target, *_rest = _approved_step()
    queued = ConfirmedSignalQueued(
        signal_id=provenance.signal_id,
        strategy_version=provenance.strategy_version,
        signal_snapshot=provenance.signal_snapshot,
        confirmation_id=provenance.confirmation_id,
        confirmed_at=provenance.confirmed_at,
        confirmed_target=provenance.confirmed_target,
        execute_not_before=target.effective_at,
    )
    with SQLiteShadowJournal(path) as journal:
        first = journal.record_confirmed_signal(queued)
        assert journal.record_confirmed_signal(queued) == first
    with SQLiteShadowJournal(path) as restarted:
        assert restarted.read_confirmed_signal("signal-1") == queued
        changed = queued.model_copy(update={"strategy_version": "strategy-v2"})
        with pytest.raises(JournalCollisionError, match="semantic key"):
            restarted.record_confirmed_signal(changed)
        assert restarted.read_confirmed_signal("signal-1") == queued


def test_activation_requires_a_preexisting_matching_queue(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    provenance, snapshot, target, batch, decision, events = _approved_step()
    with SQLiteShadowJournal(path) as journal:
        with pytest.raises(ValueError, match="durably queued"):
            journal.commit_shadow_step(
                signal_confirmation=provenance,
                snapshot=snapshot,
                target=target,
                batch=batch,
                decision=decision,
                ledger_events=events,
            )
        assert journal.read_records() == ()


def test_atomic_step_rolls_back_when_late_semantic_collision_occurs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    with SQLiteShadowJournal(path) as journal:
        _commit_approved(journal, _approved_step())
        colliding = _approved_step(
            event_sequence=3,
            signal_id="signal-2",
            provenance_id="provenance-2",
            signal_snapshot_id="signal-snapshot-2",
            signal_target_id="signal-target-2",
            snapshot_id="execution-snapshot-2",
            target_id="activated-target-1",
            batch_id="batch-2",
            signal_at=T0 + timedelta(days=1),
            execution_at=T0 + timedelta(days=1, minutes=2),
            with_order=False,
        )
        provenance, snapshot, target, batch, decision, events = colliding
        journal.record_confirmed_signal(
            ConfirmedSignalQueued(
                signal_id=provenance.signal_id,
                strategy_version=provenance.strategy_version,
                signal_snapshot=provenance.signal_snapshot,
                confirmation_id=provenance.confirmation_id,
                confirmed_at=provenance.confirmed_at,
                confirmed_target=provenance.confirmed_target,
                execute_not_before=target.effective_at,
            )
        )
        before = journal.read_records()
        with pytest.raises(JournalCollisionError, match="semantic key"):
            journal.commit_shadow_step(
                signal_confirmation=provenance,
                snapshot=snapshot,
                target=target,
                batch=batch,
                decision=decision,
                ledger_events=events,
            )
        assert journal.read_records() == before
        assert journal.read_shadow_step("provenance-2") is None


def test_identical_retry_is_noop_and_changed_key_payload_collides(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    original = _snapshot("same-id", T0)
    with SQLiteShadowJournal(path) as journal:
        first = journal.record_market_snapshot(original)
        assert journal.record_market_snapshot(original) == first
        assert len(journal.read_records()) == 1
        changed = _snapshot("same-id", T0 + timedelta(minutes=1), "11.00")
        with pytest.raises(JournalCollisionError, match="semantic key"):
            journal.record_market_snapshot(changed)


def test_explicit_record_id_collision_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "shadow.sqlite3"
    with SQLiteShadowJournal(path) as journal:
        first = journal.record_market_snapshot(_snapshot("one", T0))
        monkeypatch.setattr(journal_module, "_content_id", lambda *_args: first.record_id)
        with pytest.raises(JournalCollisionError, match="record ID"):
            journal.record_market_snapshot(
                _snapshot("two", T0 + timedelta(minutes=1))
            )


def test_hash_chain_detects_external_tampering(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    with SQLiteShadowJournal(path) as journal:
        journal.record_market_snapshot(_snapshot("one", T0))
        with sqlite3.connect(path) as attacker:
            attacker.execute("DROP TRIGGER journal_records_no_update")
            attacker.execute(
                "UPDATE journal_records SET payload_json = ? WHERE sequence = 1",
                ('{"tampered":true}',),
            )
        with pytest.raises(JournalIntegrityError):
            journal.verify_chain()


def test_append_only_trigger_blocks_normal_update(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    with SQLiteShadowJournal(path) as journal:
        journal.record_market_snapshot(_snapshot("one", T0))
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE journal_records SET semantic_key = 'changed' WHERE sequence = 1"
            )


def test_two_connections_converge_on_one_identical_record(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    snapshot = _snapshot("one", T0)
    barrier = Barrier(2)
    # Schema creation is a separate one-time migration concern.  Initialize it
    # before synchronizing the two independent writer transactions so neither
    # worker waits at the barrier while the other legitimately holds DDL locks.
    with SQLiteShadowJournal(path):
        pass

    def append() -> tuple[int, str]:
        with SQLiteShadowJournal(path, busy_timeout_ms=10_000) as journal:
            barrier.wait(timeout=5)
            record = journal.record_market_snapshot(snapshot)
            return record.sequence, record.record_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: append(), range(2)))
    assert results[0] == results[1]
    with SQLiteShadowJournal(path) as journal:
        assert len(journal.read_records()) == 1


def test_future_schema_and_nonjournal_database_are_rejected(tmp_path: Path) -> None:
    future = tmp_path / "future.sqlite3"
    SQLiteShadowJournal(future).close()
    with sqlite3.connect(future) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(JournalSchemaError, match="unsupported journal schema"):
        SQLiteShadowJournal(future)

    unrelated = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(JournalSchemaError, match="not a Truffle"):
        SQLiteShadowJournal(unrelated)


def test_incidents_and_explicit_market_session_health_are_persisted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    incident = OperationalIncident(
        incident_id="incident-1",
        occurred_at=T0,
        severity="error",
        code="stale_quote",
        message="quote exceeded the configured age",
        context={"symbol": "AAA"},
    )
    with SQLiteShadowJournal(path) as journal:
        journal.record_incident(incident)
        unhealthy = OperationalSessionClosed(
            session_id="session-1",
            market_session_date=date(2026, 1, 2),
            started_at=T0 - timedelta(hours=6),
            ended_at=T0,
            status="incident",
            incident_ids=("incident-1",),
        )
        journal.record_session_closed(unhealthy)
        healthy = OperationalSessionClosed(
            session_id="session-2",
            market_session_date=date(2026, 1, 5),
            started_at=T0 + timedelta(days=3, hours=-6),
            ended_at=T0 + timedelta(days=3),
            status="healthy",
        )
        journal.record_session_closed(healthy)
        decoded = [
            journal.decode_record(record)
            for record in journal.read_records("session_closed")
        ]
        assert decoded == [unhealthy, healthy]


def test_session_requires_existing_incidents_and_unique_exchange_date(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    missing = OperationalSessionClosed(
        session_id="session-missing",
        market_session_date=date(2026, 1, 2),
        started_at=T0 - timedelta(hours=6),
        ended_at=T0,
        status="incomplete",
        incident_ids=("not-recorded",),
    )
    with SQLiteShadowJournal(path) as journal:
        with pytest.raises(JournalIntegrityError, match="missing incidents"):
            journal.record_session_closed(missing)
        assert journal.read_records() == ()
        first = OperationalSessionClosed(
            session_id="session-1",
            market_session_date=date(2026, 1, 2),
            started_at=T0 - timedelta(hours=6),
            ended_at=T0,
            status="healthy",
        )
        journal.record_session_closed(first)
        duplicate_date = first.model_copy(
            update={"session_id": "session-2", "ended_at": T0 + timedelta(minutes=1)}
        )
        with pytest.raises(JournalIntegrityError, match="more than one closure"):
            journal.record_session_closed(duplicate_date)


def test_rejected_step_requires_mark_and_incident_and_is_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    snapshot = _snapshot("rejected-snapshot", T0)
    target = TargetAllocation(
        allocation_id="rejected-target", effective_at=T0, weights={}
    )
    batch = OrderBatch(
        batch_id="rejected-batch",
        allocation_id=target.allocation_id,
        effective_at=T0,
    )
    decision = ShadowRiskRejected(
        evaluated_at=T0,
        snapshot_id=snapshot.snapshot_id,
        batch_id=batch.batch_id,
        violations=(RiskViolation(code="stale", message="stale quote"),),
    )
    mark = ShadowMarkRecorded(sequence=1, snapshot=snapshot)
    incident = OperationalIncident(
        incident_id="rejection-incident",
        occurred_at=T0,
        severity="warning",
        code="risk_rejected",
        message="risk gate rejected the batch",
    )
    with SQLiteShadowJournal(path) as journal:
        with pytest.raises(ValueError, match="operational incident"):
            journal.commit_shadow_step(
                snapshot=snapshot,
                target=target,
                batch=batch,
                decision=decision,
                ledger_events=(mark,),
            )
        journal.commit_shadow_step(
            snapshot=snapshot,
            target=target,
            batch=batch,
            decision=decision,
            ledger_events=(mark,),
            incidents=(incident,),
        )
        assert journal.load_ledger().state.sequence == 1
        assert journal.read_records("incident")[0].semantic_key == "rejection-incident"


def test_approved_step_requires_both_execution_and_mark(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    provenance, snapshot, target, batch, decision, events = _approved_step()
    with SQLiteShadowJournal(path) as journal:
        journal.record_confirmed_signal(
            ConfirmedSignalQueued(
                signal_id=provenance.signal_id,
                strategy_version=provenance.strategy_version,
                signal_snapshot=provenance.signal_snapshot,
                confirmation_id=provenance.confirmation_id,
                confirmed_at=provenance.confirmed_at,
                confirmed_target=provenance.confirmed_target,
                execute_not_before=target.effective_at,
            )
        )
        before = journal.read_records()
        with pytest.raises(ValueError, match="one execution and one mark"):
            journal.commit_shadow_step(
                signal_confirmation=provenance,
                snapshot=snapshot,
                target=target,
                batch=batch,
                decision=decision,
                ledger_events=(events[0],),
            )
        assert journal.read_records() == before


def test_signal_source_rows_are_independently_collision_checked(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    with SQLiteShadowJournal(path) as journal:
        _commit_approved(journal, _approved_step())
        market_records = journal.read_records("market_snapshot")
        target_records = journal.read_records("confirmed_target")
        assert [record.semantic_key for record in market_records] == [
            "signal-snapshot-1",
            "execution-snapshot-1",
        ]
        assert [record.semantic_key for record in target_records] == [
            "signal-target-1",
            "activated-target-1",
        ]


def test_one_queued_signal_cannot_activate_on_two_execution_snapshots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    first_step = _approved_step()
    first_confirmation = first_step[0]
    with SQLiteShadowJournal(path) as journal:
        _commit_approved(journal, first_step)
        before = journal.read_records()
        second = _approved_step(
            event_sequence=3,
            signal_id=first_confirmation.signal_id,
            provenance_id="provenance-second-execution",
            signal_snapshot_id=first_confirmation.signal_snapshot.snapshot_id,
            signal_target_id=first_confirmation.confirmed_target.allocation_id,
            snapshot_id="different-execution-snapshot",
            target_id="different-activated-target",
            batch_id="different-batch",
            signal_at=first_confirmation.signal_snapshot.as_of,
            execution_at=T0 + timedelta(days=1),
            with_order=False,
        )
        provenance, snapshot, target, batch, decision, events = second
        provenance = provenance.model_copy(
            update={
                "strategy_version": first_confirmation.strategy_version,
                "signal_snapshot": first_confirmation.signal_snapshot,
                "confirmation_id": first_confirmation.confirmation_id,
                "confirmed_at": first_confirmation.confirmed_at,
                "confirmed_target": first_confirmation.confirmed_target,
            }
        )
        with pytest.raises(JournalCollisionError, match="semantic key"):
            journal.commit_shadow_step(
                signal_confirmation=provenance,
                snapshot=snapshot,
                target=target,
                batch=batch,
                decision=decision,
                ledger_events=events,
            )
        assert journal.read_records() == before


def test_naive_risk_record_time_is_rejected_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    rejected = ShadowRiskRejected(
        evaluated_at=None,
        snapshot_id="snapshot",
        batch_id="batch",
        violations=(RiskViolation(code="invalid_time", message="missing time"),),
    )
    with SQLiteShadowJournal(path) as journal:
        with pytest.raises(ValueError, match="timezone-aware"):
            journal.record_risk_decision(rejected, occurred_at=datetime(2026, 1, 2))
        assert journal.read_records() == ()


def test_obvious_secrets_are_rejected_and_never_written(tmp_path: Path) -> None:
    path = tmp_path / "shadow.sqlite3"
    secret = "VERY_PRIVATE_42"
    with pytest.raises(ValidationError, match="must not contain secrets"):
        OperationalIncident(
            incident_id="bad",
            occurred_at=T0,
            severity="error",
            code="configuration_error",
            message="invalid configuration",
            context={"api_key": secret},
        )
    with SQLiteShadowJournal(path):
        pass
    assert secret.encode() not in path.read_bytes()


def test_context_manager_closes_safely(tmp_path: Path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3")
    journal.close()
    journal.close()
    with pytest.raises(JournalClosedError):
        journal.read_records()
