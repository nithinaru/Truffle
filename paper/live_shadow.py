"""Durable, deterministic live-shadow steps with no order-submission surface.

This module is deliberately orchestration-only.  Every market snapshot, timestamp,
configuration value, and confirmed signal is supplied by the caller.  It has no
clock, scheduler, data provider, broker, credential, or network dependency.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import field_validator, model_validator

from paper.journal import (
    ConfirmedSignalQueued,
    JournalCollisionError,
    JournalRecord,
    OperationalIncident,
    SignalConfirmation,
    SQLiteShadowJournal,
)
from paper.ledger import (
    PaperLedgerError,
    ShadowLedgerSnapshot,
    ShadowMarkRecorded,
    reduce_shadow_ledger,
)
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    OrderBatch,
    PaperModel,
    ShadowBatchExecuted,
    TargetAllocation,
    UtcDatetime,
)
from paper.planner import PaperPlanningError, PlanningConfig, plan_orders
from paper.risk import (
    ShadowRiskApproved,
    ShadowRiskDecision,
    ShadowRiskLimits,
    ShadowRiskRejected,
    gate_shadow_batch,
)


class LiveShadowError(Exception):
    """Base class for a live-shadow step rejected before journal commit."""


class LiveShadowTimingError(LiveShadowError):
    """Raised when an execution snapshot violates delayed-activation timing."""


class LiveShadowPlanningError(LiveShadowError):
    """Raised when fresh-marked account state cannot produce an order plan."""


class LiveShadowFreshMarkError(LiveShadowError):
    """Raised when the execution snapshot cannot freshly mark every holding."""


class LiveShadowSignalNotQueuedError(LiveShadowError):
    """Raised when execution is attempted before durable signal queuing."""


def _nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("identifier must not be empty")
    return stripped


class ConfirmedShadowSignal(PaperModel):
    """One explicitly confirmed allocation captured at a completed snapshot."""

    signal_id: str
    confirmation_id: str
    strategy_version: str
    snapshot_status: Literal["completed"] = "completed"
    confirmation_status: Literal["confirmed"] = "confirmed"
    signal_snapshot: MarketSnapshot
    confirmed_target: TargetAllocation
    confirmed_at: UtcDatetime

    _validate_ids = field_validator(
        "signal_id",
        "confirmation_id",
        "strategy_version",
        mode="before",
    )(_nonempty)

    @model_validator(mode="after")
    def _captured_at_completed_snapshot(self) -> Self:
        if self.confirmed_target.effective_at != self.signal_snapshot.as_of:
            raise ValueError(
                "confirmed target must be effective at the completed signal snapshot"
            )
        if self.confirmed_at < self.signal_snapshot.as_of:
            raise ValueError("confirmation cannot predate the completed signal snapshot")
        return self


class LiveShadowApprovedResult(PaperModel):
    """Durable result for an approved, locally simulated shadow execution."""

    outcome: Literal["approved"] = "approved"
    step_id: str
    signal: ConfirmedShadowSignal
    execution_snapshot: MarketSnapshot
    activated_target: TargetAllocation
    planned_batch: OrderBatch
    decision: ShadowRiskApproved
    ledger: ShadowLedgerSnapshot
    journal_record_ids: tuple[str, ...]


class LiveShadowQueuedResult(PaperModel):
    """Durable result proving a confirmed signal can survive a restart."""

    outcome: Literal["queued"] = "queued"
    queued_signal: ConfirmedSignalQueued
    journal_record_ids: tuple[str, ...]


class LiveShadowRejectedResult(PaperModel):
    """Durable result for a risk rejection; the fresh mark remains journaled."""

    outcome: Literal["rejected"] = "rejected"
    step_id: str
    signal: ConfirmedShadowSignal
    execution_snapshot: MarketSnapshot
    activated_target: TargetAllocation
    planned_batch: OrderBatch
    decision: ShadowRiskRejected
    ledger: ShadowLedgerSnapshot
    journal_record_ids: tuple[str, ...]


type LiveShadowResult = LiveShadowApprovedResult | LiveShadowRejectedResult


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LiveShadowTimingError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _step_id(
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
) -> str:
    """Return a stable semantic key; content collisions stay visible to SQLite."""

    material = "\x1f".join(
        (
            signal.signal_id,
            signal.confirmation_id,
            signal.signal_snapshot.snapshot_id,
            execution_snapshot.snapshot_id,
        )
    ).encode()
    return f"live-shadow-{hashlib.sha256(material).hexdigest()}"


def _request_fingerprint(
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    *,
    execute_not_before: datetime,
    evaluated_at: datetime,
    assumptions: ExecutionAssumptions,
    planning: PlanningConfig,
    risk: ShadowRiskLimits,
) -> str:
    material = "\x1f".join(
        (
            signal.canonical_json(),
            execution_snapshot.canonical_json(),
            _utc(execute_not_before, label="execute_not_before").isoformat(),
            _utc(evaluated_at, label="evaluated_at").isoformat(),
            assumptions.canonical_json(),
            planning.canonical_json(),
            risk.canonical_json(),
        )
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _activated_target(
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    *,
    step_id: str,
    request_fingerprint: str,
) -> TargetAllocation:
    return TargetAllocation(
        allocation_id=f"{step_id}:target:{request_fingerprint}",
        effective_at=execution_snapshot.as_of,
        weights=dict(signal.confirmed_target.weights),
    )


def _validate_execution_time(
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    execute_not_before: datetime,
    evaluated_at: datetime,
) -> None:
    lower_bound = _utc(execute_not_before, label="execute_not_before")
    evaluation_time = _utc(evaluated_at, label="evaluated_at")
    if execution_snapshot.as_of <= signal.signal_snapshot.as_of:
        raise LiveShadowTimingError(
            "execution snapshot must be strictly later than the signal snapshot"
        )
    if execution_snapshot.as_of < lower_bound:
        raise LiveShadowTimingError(
            "execution snapshot is earlier than execute_not_before"
        )
    if execution_snapshot.as_of < signal.confirmed_at:
        raise LiveShadowTimingError(
            "execution snapshot cannot predate explicit signal confirmation"
        )
    if evaluation_time < signal.confirmed_at:
        raise LiveShadowTimingError(
            "risk evaluation cannot predate explicit signal confirmation"
        )


def _queued_signal(
    signal: ConfirmedShadowSignal,
    execute_not_before: datetime,
) -> ConfirmedSignalQueued:
    boundary = _utc(execute_not_before, label="execute_not_before")
    if boundary <= signal.signal_snapshot.as_of:
        raise LiveShadowTimingError(
            "execute_not_before must be strictly later than the signal snapshot"
        )
    if boundary < signal.confirmed_at:
        raise LiveShadowTimingError(
            "execute_not_before cannot predate explicit signal confirmation"
        )
    return ConfirmedSignalQueued(
        signal_id=signal.signal_id,
        strategy_version=signal.strategy_version,
        signal_snapshot=signal.signal_snapshot,
        confirmation_id=signal.confirmation_id,
        confirmed_at=signal.confirmed_at,
        confirmed_target=signal.confirmed_target,
        execute_not_before=boundary,
    )


def queue_live_shadow_signal(
    journal: SQLiteShadowJournal,
    signal: ConfirmedShadowSignal,
    *,
    execute_not_before: datetime,
) -> LiveShadowQueuedResult:
    """Atomically persist a confirmed signal before waiting for a later snapshot."""

    queued = _queued_signal(signal, execute_not_before)
    records = journal.record_confirmed_signal(queued)
    return LiveShadowQueuedResult(
        queued_signal=queued,
        journal_record_ids=tuple(record.record_id for record in records),
    )


def _require_queued_signal(
    journal: SQLiteShadowJournal,
    signal: ConfirmedShadowSignal,
    execute_not_before: datetime,
) -> ConfirmedSignalQueued:
    expected = _queued_signal(signal, execute_not_before)
    persisted = journal.read_confirmed_signal(signal.signal_id)
    if persisted is None:
        raise LiveShadowSignalNotQueuedError(
            "confirmed signal must be durably queued before live-shadow execution"
        )
    if persisted.canonical_json() != expected.canonical_json():
        raise JournalCollisionError(
            "queued signal ID is already bound to different canonical inputs"
        )
    return persisted


def _confirmation(
    signal: ConfirmedShadowSignal,
    activated_target: TargetAllocation,
    *,
    step_id: str,
) -> SignalConfirmation:
    return SignalConfirmation(
        provenance_id=step_id,
        signal_id=signal.signal_id,
        strategy_version=signal.strategy_version,
        signal_snapshot=signal.signal_snapshot,
        confirmation_id=signal.confirmation_id,
        confirmed_at=signal.confirmed_at,
        confirmed_target=signal.confirmed_target,
        activated_target=activated_target,
    )


def _result(
    *,
    step_id: str,
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    activated_target: TargetAllocation,
    batch: OrderBatch,
    decision: ShadowRiskDecision,
    ledger: ShadowLedgerSnapshot,
    records: tuple[JournalRecord, ...],
) -> LiveShadowResult:
    common = {
        "step_id": step_id,
        "signal": signal,
        "execution_snapshot": execution_snapshot,
        "activated_target": activated_target,
        "planned_batch": batch,
        "decision": decision,
        "ledger": ledger,
        "journal_record_ids": tuple(record.record_id for record in records),
    }
    if isinstance(decision, ShadowRiskApproved):
        return LiveShadowApprovedResult(**common)
    return LiveShadowRejectedResult(**common)


def _ledger_at_step(
    journal: SQLiteShadowJournal,
    records: tuple[JournalRecord, ...],
) -> ShadowLedgerSnapshot:
    step_events = tuple(
        model
        for record in records
        if record.kind in ("mark", "shadow_execution")
        and isinstance(
            model := journal.decode_record(record),
            (ShadowMarkRecorded, ShadowBatchExecuted),
        )
    )
    if not step_events:
        raise JournalCollisionError("existing live-shadow step has no ledger event")
    step_sequence = max(event.sequence for event in step_events)
    return reduce_shadow_ledger(
        [
            event
            for event in journal.replay_ledger_events()
            if event.sequence <= step_sequence
        ]
    )


def _existing_result(
    journal: SQLiteShadowJournal,
    records: tuple[JournalRecord, ...],
    *,
    step_id: str,
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    activated_target: TargetAllocation,
    confirmation: SignalConfirmation,
) -> LiveShadowResult:
    """Validate and recover a prior atomic step before deriving a new sequence."""

    decoded = tuple(
        (record, journal.decode_record(record))
        for record in records
    )

    def exactly_one(kind: str) -> PaperModel:
        matches = tuple(model for record, model in decoded if record.kind == kind)
        if len(matches) != 1:
            raise JournalCollisionError(
                f"existing live-shadow step has {len(matches)} {kind!r} records"
            )
        return matches[0]

    def by_semantic_key(kind: str, semantic_key: str) -> PaperModel:
        matches = tuple(
            model
            for record, model in decoded
            if record.kind == kind and record.semantic_key == semantic_key
        )
        if len(matches) != 1:
            raise JournalCollisionError(
                f"existing live-shadow step has {len(matches)} {kind!r} records "
                f"for semantic key {semantic_key!r}"
            )
        return matches[0]

    persisted_confirmation = by_semantic_key(
        "signal_confirmation",
        signal.signal_id,
    )
    persisted_signal_snapshot = by_semantic_key(
        "market_snapshot",
        signal.signal_snapshot.snapshot_id,
    )
    persisted_snapshot = by_semantic_key(
        "market_snapshot",
        execution_snapshot.snapshot_id,
    )
    persisted_confirmed_target = by_semantic_key(
        "confirmed_target",
        signal.confirmed_target.allocation_id,
    )
    persisted_target = by_semantic_key(
        "confirmed_target",
        activated_target.allocation_id,
    )
    batch = exactly_one("planned_batch")
    decision = exactly_one("risk_decision")
    if (
        persisted_confirmation.canonical_json() != confirmation.canonical_json()
        or persisted_signal_snapshot.canonical_json()
        != signal.signal_snapshot.canonical_json()
        or persisted_snapshot.canonical_json() != execution_snapshot.canonical_json()
        or persisted_confirmed_target.canonical_json()
        != signal.confirmed_target.canonical_json()
        or persisted_target.canonical_json() != activated_target.canonical_json()
    ):
        raise JournalCollisionError(
            "live-shadow step identity is already bound to different canonical inputs"
        )
    if not isinstance(batch, OrderBatch) or not isinstance(
        decision, (ShadowRiskApproved, ShadowRiskRejected)
    ):
        raise JournalCollisionError("existing live-shadow step has invalid typed records")
    return _result(
        step_id=step_id,
        signal=signal,
        execution_snapshot=execution_snapshot,
        activated_target=activated_target,
        batch=batch,
        decision=decision,
        ledger=_ledger_at_step(journal, records),
        records=records,
    )


def _rejection_incident(
    *,
    step_id: str,
    decision: ShadowRiskRejected,
    occurred_at: datetime,
) -> OperationalIncident:
    codes = ",".join(violation.code for violation in decision.violations)
    return OperationalIncident(
        incident_id=f"{step_id}:risk-rejected",
        occurred_at=occurred_at,
        severity="warning",
        code="shadow_risk_rejected",
        message="Live-shadow risk gate rejected the planned local batch.",
        context={
            "batch_id": decision.batch_id,
            "snapshot_id": decision.snapshot_id,
            "violation_codes": codes,
        },
    )


def run_live_shadow_step(
    journal: SQLiteShadowJournal,
    signal: ConfirmedShadowSignal,
    execution_snapshot: MarketSnapshot,
    *,
    execute_not_before: datetime,
    evaluated_at: datetime,
    assumptions: ExecutionAssumptions,
    planning: PlanningConfig,
    risk: ShadowRiskLimits,
) -> LiveShadowResult:
    """Activate one confirmed signal at a strictly later, injected snapshot.

    The execution snapshot is first staged as a fresh mark.  Planning and the
    mandatory ``mode='shadow'`` risk gate operate on that marked state.  The mark,
    complete provenance, and either the approved local execution or the rejection
    are then committed as one SQLite transaction.
    """

    _validate_execution_time(
        signal,
        execution_snapshot,
        execute_not_before,
        evaluated_at,
    )
    _require_queued_signal(journal, signal, execute_not_before)
    step_id = _step_id(signal, execution_snapshot)
    fingerprint = _request_fingerprint(
        signal,
        execution_snapshot,
        execute_not_before=execute_not_before,
        evaluated_at=evaluated_at,
        assumptions=assumptions,
        planning=planning,
        risk=risk,
    )
    target = _activated_target(
        signal,
        execution_snapshot,
        step_id=step_id,
        request_fingerprint=fingerprint,
    )
    confirmation = _confirmation(signal, target, step_id=step_id)
    existing = journal.read_shadow_step(step_id)
    if existing is not None:
        return _existing_result(
            journal,
            existing,
            step_id=step_id,
            signal=signal,
            execution_snapshot=execution_snapshot,
            activated_target=target,
            confirmation=confirmation,
        )

    prior_events = journal.replay_ledger_events()
    prior_state = reduce_shadow_ledger(prior_events)
    mark = ShadowMarkRecorded(
        sequence=prior_state.sequence + 1,
        snapshot=execution_snapshot,
    )
    try:
        marked_state = reduce_shadow_ledger([*prior_events, mark])
    except PaperLedgerError as exc:
        raise LiveShadowFreshMarkError(str(exc)) from exc
    try:
        batch = plan_orders(
            target,
            execution_snapshot,
            marked_state,
            assumptions,
            config=planning,
        )
    except PaperPlanningError as exc:
        raise LiveShadowPlanningError(str(exc)) from exc

    decision = gate_shadow_batch(
        batch,
        target,
        execution_snapshot,
        marked_state,
        assumptions,
        mode="shadow",
        evaluated_at=evaluated_at,
        limits=risk,
    )
    ledger_events = (
        (mark, decision.event)
        if isinstance(decision, ShadowRiskApproved)
        else (mark,)
    )
    incidents = (
        (_rejection_incident(
            step_id=step_id,
            decision=decision,
            occurred_at=_utc(evaluated_at, label="evaluated_at"),
        ),)
        if isinstance(decision, ShadowRiskRejected)
        else ()
    )
    records = journal.commit_shadow_step(
        snapshot=execution_snapshot,
        target=target,
        batch=batch,
        decision=decision,
        ledger_events=ledger_events,
        incidents=incidents,
        signal_confirmation=confirmation,
        decision_at=evaluated_at,
    )
    return _result(
        step_id=step_id,
        signal=signal,
        execution_snapshot=execution_snapshot,
        activated_target=target,
        batch=batch,
        decision=decision,
        ledger=_ledger_at_step(journal, records),
        records=records,
    )


__all__ = [
    "ConfirmedShadowSignal",
    "LiveShadowApprovedResult",
    "LiveShadowError",
    "LiveShadowFreshMarkError",
    "LiveShadowPlanningError",
    "LiveShadowQueuedResult",
    "LiveShadowRejectedResult",
    "LiveShadowResult",
    "LiveShadowSignalNotQueuedError",
    "LiveShadowTimingError",
    "queue_live_shadow_signal",
    "run_live_shadow_step",
]
