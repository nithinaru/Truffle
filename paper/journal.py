"""Durable, deterministic audit journal for local live-shadow operation.

The journal is deliberately a small SQLite boundary around the immutable paper
models.  It owns no clock, network client, broker credential, or executable
serialization format.  Every write is canonical JSON, content-addressed, and
linked to the preceding row with SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from paper.ledger import (
    ShadowLedger,
    ShadowLedgerEvent,
    ShadowMarkRecorded,
    reduce_shadow_ledger,
)
from paper.models import (
    MarketSnapshot,
    OrderBatch,
    PaperModel,
    ShadowBatchExecuted,
    TargetAllocation,
    UtcDatetime,
)
from paper.risk import ShadowRiskApproved, ShadowRiskDecision, ShadowRiskRejected

SCHEMA_VERSION = 1
_APPLICATION_ID = 0x5452464A  # ``TRFJ``
_GENESIS_HASH = "0" * 64

JournalKind = Literal[
    "confirmed_signal_queued",
    "signal_confirmation",
    "market_snapshot",
    "confirmed_target",
    "planned_batch",
    "risk_decision",
    "shadow_execution",
    "mark",
    "incident",
    "session_closed",
    "shadow_step",
]

_KINDS: tuple[JournalKind, ...] = (
    "confirmed_signal_queued",
    "signal_confirmation",
    "market_snapshot",
    "confirmed_target",
    "planned_batch",
    "risk_decision",
    "shadow_execution",
    "mark",
    "incident",
    "session_closed",
    "shadow_step",
)


class ShadowJournalError(Exception):
    """Base class for persistent shadow-journal failures."""


class JournalCollisionError(ShadowJournalError):
    """Raised when an ID or semantic key is reused with different bytes."""


class JournalIntegrityError(ShadowJournalError):
    """Raised when the append-only hash chain or a payload is invalid."""


class JournalSchemaError(ShadowJournalError):
    """Raised for an unknown, newer, or malformed journal schema."""


class JournalClosedError(ShadowJournalError):
    """Raised when an operation is attempted after closing the journal."""


def _nonempty(value: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError("value must not be empty")
    return result


class _FrozenStringDict(dict[str, str]):
    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("incident context cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> Self:
        return self

    def copy(self) -> Self:
        return self


_SECRET_KEY = re.compile(
    r"(?:api.?key|access.?token|refresh.?token|password|secret|authorization|"
    r"cookie|credential|private.?key)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:bearer\s+\S+|(?:api.?key|access.?token|refresh.?token|password|secret|"
    r"authorization|cookie|credential|private.?key)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


class SignalConfirmation(PaperModel):
    """Full immutable provenance from a completed signal to its activation."""

    provenance_id: str
    signal_id: str
    strategy_version: str
    signal_snapshot: MarketSnapshot
    confirmation_id: str
    confirmed_at: UtcDatetime
    confirmed_target: TargetAllocation
    activated_target: TargetAllocation

    _validate_ids = field_validator(
        "provenance_id",
        "signal_id",
        "strategy_version",
        "confirmation_id",
        mode="before",
    )(_nonempty)

    @model_validator(mode="after")
    def _consistent_provenance(self) -> Self:
        if self.confirmed_at < self.signal_snapshot.as_of:
            raise ValueError("confirmation cannot predate its signal snapshot")
        if self.confirmed_target.effective_at != self.signal_snapshot.as_of:
            raise ValueError("confirmed target must identify the signal snapshot instant")
        if self.confirmed_target.weights != self.activated_target.weights:
            raise ValueError("activation may change time and ID, but not confirmed weights")
        if self.activated_target.effective_at <= self.signal_snapshot.as_of:
            raise ValueError("activated target must be strictly later than the signal")
        if self.activated_target.effective_at < self.confirmed_at:
            raise ValueError("activated target cannot predate confirmation")
        return self

    @property
    def signal_snapshot_id(self) -> str:
        return self.signal_snapshot.snapshot_id

    @property
    def signal_as_of(self) -> datetime:
        return self.signal_snapshot.as_of


class ConfirmedSignalQueued(PaperModel):
    """A confirmed signal durably waiting for a later executable snapshot."""

    signal_id: str
    strategy_version: str
    signal_snapshot: MarketSnapshot
    confirmation_id: str
    confirmed_at: UtcDatetime
    confirmed_target: TargetAllocation
    execute_not_before: UtcDatetime

    _validate_ids = field_validator(
        "signal_id", "strategy_version", "confirmation_id", mode="before"
    )(_nonempty)

    @model_validator(mode="after")
    def _consistent_queue(self) -> Self:
        if self.confirmed_target.effective_at != self.signal_snapshot.as_of:
            raise ValueError("queued target must identify the signal snapshot instant")
        if self.confirmed_at < self.signal_snapshot.as_of:
            raise ValueError("confirmation cannot predate its signal snapshot")
        if self.execute_not_before <= self.signal_snapshot.as_of:
            raise ValueError("execution boundary must be strictly later than the signal")
        if self.execute_not_before < self.confirmed_at:
            raise ValueError("execution boundary cannot predate confirmation")
        return self


class OperationalIncident(PaperModel):
    """Explicit, secret-free operational anomaly persisted with its timestamp."""

    incident_id: str
    occurred_at: UtcDatetime
    severity: Literal["info", "warning", "error", "critical"]
    code: str
    message: str
    context: dict[str, str] = Field(default_factory=dict)

    _validate_strings = field_validator(
        "incident_id", "code", "message", mode="before"
    )(_nonempty)

    @field_validator("context", mode="after")
    @classmethod
    def _safe_context(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if _SECRET_KEY.search(key) or _SECRET_VALUE.search(item):
                raise ValueError("operational incidents must not contain secrets")
        return _FrozenStringDict(sorted(value.items()))

    @field_validator("message", mode="after")
    @classmethod
    def _safe_message(cls, value: str) -> str:
        if _SECRET_VALUE.search(value):
            raise ValueError("operational incidents must not contain secrets")
        return value


class OperationalSessionClosed(PaperModel):
    """Explicit proof that one live-market observation session ended."""

    session_id: str
    market_session_date: date
    started_at: UtcDatetime
    ended_at: UtcDatetime
    status: Literal["healthy", "incident", "incomplete"]
    incident_ids: tuple[str, ...] = ()

    _validate_id = field_validator("session_id", mode="before")(_nonempty)

    @field_validator("incident_ids", mode="after")
    @classmethod
    def _incident_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(sorted(_nonempty(value) for value in values))
        if len(canonical) != len(set(canonical)):
            raise ValueError("session incident IDs must be unique")
        return canonical

    @model_validator(mode="after")
    def _ordered_times(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("session end cannot predate its start")
        if self.status == "healthy" and self.incident_ids:
            raise ValueError("a healthy session cannot reference incidents")
        if self.status != "healthy" and not self.incident_ids:
            raise ValueError("incident and incomplete sessions must reference incidents")
        return self


class ShadowStepManifest(PaperModel):
    """Content-addressed index for restart-safe lookup of one atomic step."""

    step_id: str
    provenance_id: str | None = None
    strategy_version: str | None = None
    snapshot_id: str
    allocation_id: str
    batch_id: str
    record_ids: tuple[str, ...] = Field(min_length=5)

    _validate_required_ids = field_validator(
        "step_id", "snapshot_id", "allocation_id", "batch_id", mode="before"
    )(_nonempty)

    @field_validator("provenance_id", "strategy_version", mode="before")
    @classmethod
    def _validate_optional_ids(cls, value: str | None) -> str | None:
        return None if value is None else _nonempty(value)

    @field_validator("record_ids", mode="after")
    @classmethod
    def _unique_records(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("shadow-step manifest record IDs must be unique")
        return values


@dataclass(frozen=True, slots=True)
class JournalRecord:
    """One verified journal row in deterministic global sequence."""

    sequence: int
    record_id: str
    kind: JournalKind
    semantic_key: str
    occurred_at: datetime
    payload_json: str
    prev_hash: str
    record_hash: str

    def payload(self) -> object:
        """Decode the inert JSON payload (never code or pickle)."""

        return json.loads(self.payload_json)

    def canonical_json(self) -> str:
        """Return a stable representation of the complete persisted row."""

        return _canonical_json(
            {
                "kind": self.kind,
                "occurred_at": _utc_text(self.occurred_at),
                "payload_json": self.payload_json,
                "prev_hash": self.prev_hash,
                "record_hash": self.record_hash,
                "record_id": self.record_id,
                "semantic_key": self.semantic_key,
                "sequence": self.sequence,
            }
        )


@dataclass(frozen=True, slots=True)
class _PendingRecord:
    kind: JournalKind
    semantic_key: str
    occurred_at: datetime
    payload_json: str
    record_id: str


def _models_of_kind(
    records: tuple[JournalRecord, ...],
    decoded: dict[str, PaperModel],
    kind: JournalKind,
) -> tuple[PaperModel, ...]:
    return tuple(
        decoded[record.record_id] for record in records if record.kind == kind
    )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("journal timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JournalIntegrityError("journal contains an invalid timestamp") from exc
    if _utc_text(parsed) != value:
        raise JournalIntegrityError("journal timestamp is not canonical UTC")
    return parsed.astimezone(UTC)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_id(
    kind: JournalKind,
    semantic_key: str,
    occurred_at: datetime,
    payload_json: str,
) -> str:
    material = _canonical_json(
        {
            "kind": kind,
            "occurred_at": _utc_text(occurred_at),
            "payload_json": payload_json,
            "semantic_key": semantic_key,
        }
    ).encode()
    return f"journal_{hashlib.sha256(material).hexdigest()}"


def _chained_hash(
    sequence: int,
    record_id: str,
    kind: JournalKind,
    semantic_key: str,
    occurred_at_text: str,
    payload_json: str,
    prev_hash: str,
) -> str:
    envelope = _canonical_json(
        {
            "kind": kind,
            "occurred_at": occurred_at_text,
            "payload_json": payload_json,
            "prev_hash": prev_hash,
            "record_id": record_id,
            "semantic_key": semantic_key,
            "sequence": sequence,
        }
    ).encode()
    return hashlib.sha256(envelope).hexdigest()


def _pending(
    kind: JournalKind,
    semantic_key: str,
    occurred_at: datetime,
    model: PaperModel,
) -> _PendingRecord:
    key = _nonempty(semantic_key)
    payload = model.canonical_json()
    timestamp = _parse_utc(_utc_text(occurred_at))
    return _PendingRecord(
        kind=kind,
        semantic_key=key,
        occurred_at=timestamp,
        payload_json=payload,
        record_id=_content_id(kind, key, timestamp, payload),
    )


class SQLiteShadowJournal:
    """Append-only SQLite journal with atomic workflow commits and replay."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self._closed = False
        self._connection = sqlite3.connect(
            str(path),
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms:d}")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_schema()
        except Exception:
            self._connection.close()
            self._closed = True
            raise

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close safely; repeated closes are harmless."""

        if not self._closed:
            self._connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise JournalClosedError("shadow journal is closed")

    def _initialize_schema(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            }
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            application_id = int(
                self._connection.execute("PRAGMA application_id").fetchone()[0]
            )
            if not tables:
                if version != 0 or application_id not in (0, _APPLICATION_ID):
                    raise JournalSchemaError("empty database has incompatible schema metadata")
                self._create_schema_v1()
            else:
                if application_id != _APPLICATION_ID:
                    raise JournalSchemaError("database is not a Truffle shadow journal")
                if version != SCHEMA_VERSION:
                    raise JournalSchemaError(
                        f"unsupported journal schema {version}; expected {SCHEMA_VERSION}"
                    )
                self._validate_schema()
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def _create_schema_v1(self) -> None:
        allowed = ",".join(f"'{kind}'" for kind in _KINDS)
        self._connection.execute(
            "CREATE TABLE journal_schema (version INTEGER PRIMARY KEY) STRICT"
        )
        self._connection.execute(
            f"""CREATE TABLE journal_records (
                sequence INTEGER PRIMARY KEY,
                record_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK (kind IN ({allowed})),
                semantic_key TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE,
                UNIQUE (kind, semantic_key)
            ) STRICT"""
        )
        self._connection.execute(
            "CREATE TRIGGER journal_records_no_update BEFORE UPDATE ON journal_records "
            "BEGIN SELECT RAISE(ABORT, 'journal records are append-only'); END"
        )
        self._connection.execute(
            "CREATE TRIGGER journal_records_no_delete BEFORE DELETE ON journal_records "
            "BEGIN SELECT RAISE(ABORT, 'journal records are append-only'); END"
        )
        self._connection.execute(
            "INSERT INTO journal_schema(version) VALUES (?)", (SCHEMA_VERSION,)
        )
        self._connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _validate_schema(self) -> None:
        schema_rows = self._connection.execute(
            "SELECT version FROM journal_schema"
        ).fetchall()
        if [int(row[0]) for row in schema_rows] != [SCHEMA_VERSION]:
            raise JournalSchemaError("journal schema metadata is malformed")
        expected_columns = (
            "sequence",
            "record_id",
            "kind",
            "semantic_key",
            "occurred_at",
            "payload_json",
            "prev_hash",
            "record_hash",
        )
        actual_columns = tuple(
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(journal_records)")
        )
        if actual_columns != expected_columns:
            raise JournalSchemaError("journal record schema is malformed")
        unique_indexes: set[tuple[str, ...]] = set()
        for index in self._connection.execute("PRAGMA index_list(journal_records)"):
            if not bool(index[2]):
                continue
            index_name = str(index[1]).replace("'", "''")
            unique_indexes.add(
                tuple(
                    str(column[2])
                    for column in self._connection.execute(
                        f"PRAGMA index_info('{index_name}')"
                    )
                )
            )
        expected_unique_indexes = {
            ("record_id",),
            ("record_hash",),
            ("kind", "semantic_key"),
        }
        if not expected_unique_indexes.issubset(unique_indexes):
            raise JournalSchemaError("journal uniqueness constraints are malformed")
        triggers = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
        expected_triggers = {"journal_records_no_update", "journal_records_no_delete"}
        if not expected_triggers.issubset(triggers):
            raise JournalSchemaError("journal append-only triggers are missing")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JournalRecord:
        kind = str(row["kind"])
        if kind not in _KINDS:
            raise JournalIntegrityError(f"unknown journal record kind {kind!r}")
        return JournalRecord(
            sequence=int(row["sequence"]),
            record_id=str(row["record_id"]),
            kind=kind,  # type: ignore[arg-type]
            semantic_key=str(row["semantic_key"]),
            occurred_at=_parse_utc(str(row["occurred_at"])),
            payload_json=str(row["payload_json"]),
            prev_hash=str(row["prev_hash"]),
            record_hash=str(row["record_hash"]),
        )

    def _rows_unlocked(self) -> tuple[JournalRecord, ...]:
        return tuple(
            self._row_to_record(row)
            for row in self._connection.execute(
                "SELECT sequence, record_id, kind, semantic_key, occurred_at, "
                "payload_json, prev_hash, record_hash "
                "FROM journal_records ORDER BY sequence"
            )
        )

    def _verify_unlocked(self) -> tuple[JournalRecord, ...]:
        records = self._rows_unlocked()
        previous = _GENESIS_HASH
        for expected_sequence, record in enumerate(records, start=1):
            if record.sequence != expected_sequence:
                raise JournalIntegrityError("journal sequence is not contiguous")
            if record.prev_hash != previous:
                raise JournalIntegrityError("journal previous-hash link is broken")
            try:
                decoded = json.loads(record.payload_json)
            except json.JSONDecodeError as exc:
                raise JournalIntegrityError("journal payload is not valid JSON") from exc
            if _canonical_json(decoded) != record.payload_json:
                raise JournalIntegrityError("journal payload is not canonical JSON")
            expected_id = _content_id(
                record.kind,
                record.semantic_key,
                record.occurred_at,
                record.payload_json,
            )
            if record.record_id != expected_id:
                raise JournalIntegrityError("journal content-derived record ID is invalid")
            expected_hash = _chained_hash(
                record.sequence,
                record.record_id,
                record.kind,
                record.semantic_key,
                _utc_text(record.occurred_at),
                record.payload_json,
                record.prev_hash,
            )
            if record.record_hash != expected_hash:
                raise JournalIntegrityError("journal record hash is invalid")
            previous = record.record_hash
        for record in records:
            model = self.decode_record(record)
            expected_key = self._model_semantic_key(record.kind, model)
            if record.semantic_key != expected_key:
                raise JournalIntegrityError(
                    f"{record.kind} semantic key does not match its typed payload"
                )
        self._validate_relations(records)
        return records

    @staticmethod
    def _model_semantic_key(kind: JournalKind, model: PaperModel) -> str:
        if kind == "confirmed_signal_queued" and isinstance(
            model, ConfirmedSignalQueued
        ):
            return model.signal_id
        if kind == "signal_confirmation" and isinstance(model, SignalConfirmation):
            return model.signal_id
        if kind == "market_snapshot" and isinstance(model, MarketSnapshot):
            return model.snapshot_id
        if kind == "confirmed_target" and isinstance(model, TargetAllocation):
            return model.allocation_id
        if kind == "planned_batch" and isinstance(model, OrderBatch):
            return model.batch_id
        if kind == "risk_decision" and isinstance(
            model, (ShadowRiskApproved, ShadowRiskRejected)
        ):
            return model.batch_id
        if kind == "shadow_execution" and isinstance(model, ShadowBatchExecuted):
            return model.batch_id
        if kind == "mark" and isinstance(model, ShadowMarkRecorded):
            return model.snapshot_id
        if kind == "incident" and isinstance(model, OperationalIncident):
            return model.incident_id
        if kind == "session_closed" and isinstance(model, OperationalSessionClosed):
            return model.session_id
        if kind == "shadow_step" and isinstance(model, ShadowStepManifest):
            return model.step_id
        raise JournalIntegrityError(f"{kind} payload decoded to an unexpected type")

    def _validate_relations(self, records: tuple[JournalRecord, ...]) -> None:
        record_ids = {record.record_id for record in records}
        decoded = {
            record.record_id: self.decode_record(record)
            for record in records
        }
        market_snapshots = {
            record.semantic_key: decoded[record.record_id]
            for record in records
            if record.kind == "market_snapshot"
        }
        confirmed_targets = {
            record.semantic_key: decoded[record.record_id]
            for record in records
            if record.kind == "confirmed_target"
        }
        queued_signals = {
            record.semantic_key: decoded[record.record_id]
            for record in records
            if record.kind == "confirmed_signal_queued"
        }
        incidents = {
            record.semantic_key
            for record in records
            if record.kind == "incident"
        }
        by_record_id = {record.record_id: record for record in records}
        manifests = tuple(
            (record, decoded[record.record_id])
            for record in records
            if record.kind == "shadow_step"
        )
        session_dates: set[date] = set()
        for record in records:
            model = decoded[record.record_id]
            if record.kind == "confirmed_signal_queued":
                assert isinstance(model, ConfirmedSignalQueued)
                source_snapshot = market_snapshots.get(model.signal_snapshot.snapshot_id)
                source_target = confirmed_targets.get(model.confirmed_target.allocation_id)
                if (
                    not isinstance(source_snapshot, MarketSnapshot)
                    or source_snapshot.canonical_json()
                    != model.signal_snapshot.canonical_json()
                    or not isinstance(source_target, TargetAllocation)
                    or source_target.canonical_json()
                    != model.confirmed_target.canonical_json()
                ):
                    raise JournalIntegrityError(
                        "queued signal is missing its exact source snapshot or target"
                    )
            elif record.kind == "signal_confirmation":
                assert isinstance(model, SignalConfirmation)
                queued = queued_signals.get(model.signal_id)
                if not isinstance(queued, ConfirmedSignalQueued):
                    raise JournalIntegrityError(
                        "signal activation has no durable queued signal"
                    )
                matches = (
                    queued.strategy_version == model.strategy_version
                    and queued.signal_snapshot.canonical_json()
                    == model.signal_snapshot.canonical_json()
                    and queued.confirmation_id == model.confirmation_id
                    and queued.confirmed_at == model.confirmed_at
                    and queued.confirmed_target.canonical_json()
                    == model.confirmed_target.canonical_json()
                    and model.activated_target.effective_at
                    >= queued.execute_not_before
                )
                if not matches:
                    raise JournalIntegrityError(
                        "signal activation differs from its durable queued signal"
                    )
            elif record.kind == "session_closed":
                session = model
                assert isinstance(session, OperationalSessionClosed)
                missing = sorted(set(session.incident_ids) - incidents)
                if missing:
                    raise JournalIntegrityError(
                        f"session references missing incidents: {missing}"
                    )
                if session.market_session_date in session_dates:
                    raise JournalIntegrityError(
                        "market session date has more than one closure record"
                    )
                session_dates.add(session.market_session_date)
            elif record.kind == "shadow_step":
                manifest = model
                assert isinstance(manifest, ShadowStepManifest)
                missing_records = sorted(set(manifest.record_ids) - record_ids)
                if missing_records:
                    raise JournalIntegrityError(
                        "shadow-step manifest references missing records"
                    )

        for record in records:
            if record.kind not in ("shadow_execution", "signal_confirmation"):
                continue
            owners = tuple(
                manifest
                for _manifest_record, manifest in manifests
                if isinstance(manifest, ShadowStepManifest)
                and record.record_id in manifest.record_ids
            )
            if len(owners) != 1:
                raise JournalIntegrityError(
                    f"{record.kind} must belong to exactly one shadow-step manifest"
                )

        for _manifest_record, raw_manifest in manifests:
            assert isinstance(raw_manifest, ShadowStepManifest)
            constituent_records = tuple(
                by_record_id[record_id] for record_id in raw_manifest.record_ids
            )

            batches = _models_of_kind(constituent_records, decoded, "planned_batch")
            decisions = _models_of_kind(constituent_records, decoded, "risk_decision")
            marks = _models_of_kind(constituent_records, decoded, "mark")
            executions = _models_of_kind(
                constituent_records, decoded, "shadow_execution"
            )
            manifest_snapshot = market_snapshots.get(raw_manifest.snapshot_id)
            manifest_target = confirmed_targets.get(raw_manifest.allocation_id)
            if (
                len(batches) != 1
                or not isinstance(batches[0], OrderBatch)
                or batches[0].batch_id != raw_manifest.batch_id
                or batches[0].allocation_id != raw_manifest.allocation_id
                or len(decisions) != 1
                or not isinstance(
                    decisions[0], (ShadowRiskApproved, ShadowRiskRejected)
                )
                or decisions[0].batch_id != raw_manifest.batch_id
                or decisions[0].snapshot_id != raw_manifest.snapshot_id
                or len(marks) != 1
                or not isinstance(marks[0], ShadowMarkRecorded)
                or marks[0].snapshot_id != raw_manifest.snapshot_id
                or not isinstance(manifest_snapshot, MarketSnapshot)
                or not isinstance(manifest_target, TargetAllocation)
            ):
                raise JournalIntegrityError(
                    "shadow-step manifest has inconsistent workflow constituents"
                )
            decision = decisions[0]
            if isinstance(decision, ShadowRiskApproved):
                if (
                    len(executions) != 1
                    or not isinstance(executions[0], ShadowBatchExecuted)
                    or executions[0].canonical_json()
                    != decision.event.canonical_json()
                ):
                    raise JournalIntegrityError(
                        "approved shadow step lacks its exact approved execution"
                    )
            elif executions or not _models_of_kind(
                constituent_records, decoded, "incident"
            ):
                raise JournalIntegrityError(
                    "rejected shadow step must contain an incident and no execution"
                )
            if raw_manifest.provenance_id is not None:
                confirmations = _models_of_kind(
                    constituent_records, decoded, "signal_confirmation"
                )
                if (
                    len(confirmations) != 1
                    or not isinstance(confirmations[0], SignalConfirmation)
                    or confirmations[0].provenance_id
                    != raw_manifest.provenance_id
                    or confirmations[0].activated_target.canonical_json()
                    != manifest_target.canonical_json()
                ):
                    raise JournalIntegrityError(
                        "shadow-step manifest has inconsistent signal provenance"
                    )

    def verify_chain(self) -> tuple[JournalRecord, ...]:
        """Verify and return the complete ordered hash chain."""

        self._ensure_open()
        return self._verify_unlocked()

    def read_records(self, kind: JournalKind | None = None) -> tuple[JournalRecord, ...]:
        """Return verified records in deterministic global sequence."""

        self._ensure_open()
        records = self._verify_unlocked()
        if kind is None:
            return records
        if kind not in _KINDS:
            raise ValueError(f"unknown journal kind {kind!r}")
        return tuple(record for record in records if record.kind == kind)

    def decode_record(self, record: JournalRecord) -> PaperModel:
        """Decode a verified JSON row into its typed inert paper model."""

        decoders: dict[JournalKind, type[PaperModel]] = {
            "confirmed_signal_queued": ConfirmedSignalQueued,
            "signal_confirmation": SignalConfirmation,
            "market_snapshot": MarketSnapshot,
            "confirmed_target": TargetAllocation,
            "planned_batch": OrderBatch,
            "shadow_execution": ShadowBatchExecuted,
            "mark": ShadowMarkRecorded,
            "incident": OperationalIncident,
            "session_closed": OperationalSessionClosed,
            "shadow_step": ShadowStepManifest,
        }
        try:
            if record.kind == "risk_decision":
                raw = json.loads(record.payload_json)
                decision = raw.get("decision") if isinstance(raw, dict) else None
                if decision == "approved":
                    return ShadowRiskApproved.model_validate_json(record.payload_json)
                if decision == "rejected":
                    return ShadowRiskRejected.model_validate_json(record.payload_json)
                raise JournalIntegrityError("risk decision payload has an unknown decision")
            decoder = decoders[record.kind]
            return decoder.model_validate_json(record.payload_json)
        except JournalIntegrityError:
            raise
        except Exception as exc:
            raise JournalIntegrityError(
                f"invalid {record.kind} payload at sequence {record.sequence}"
            ) from exc

    def _insert_unlocked(self, pending: _PendingRecord) -> JournalRecord:
        by_id = self._connection.execute(
            "SELECT * FROM journal_records WHERE record_id = ?", (pending.record_id,)
        ).fetchone()
        by_key = self._connection.execute(
            "SELECT * FROM journal_records WHERE kind = ? AND semantic_key = ?",
            (pending.kind, pending.semantic_key),
        ).fetchone()
        existing_rows = [row for row in (by_id, by_key) if row is not None]
        for row in existing_rows:
            existing = self._row_to_record(row)
            identical = (
                existing.record_id == pending.record_id
                and existing.kind == pending.kind
                and existing.semantic_key == pending.semantic_key
                and existing.occurred_at == pending.occurred_at
                and existing.payload_json == pending.payload_json
            )
            if not identical:
                label = "record ID" if existing.record_id == pending.record_id else "semantic key"
                raise JournalCollisionError(
                    f"journal {label} is already bound to different canonical bytes"
                )
        if existing_rows:
            return self._row_to_record(existing_rows[0])

        tail = self._connection.execute(
            "SELECT sequence, record_hash FROM journal_records ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if tail is None else int(tail["sequence"]) + 1
        previous = _GENESIS_HASH if tail is None else str(tail["record_hash"])
        occurred_at = _utc_text(pending.occurred_at)
        record_hash = _chained_hash(
            sequence,
            pending.record_id,
            pending.kind,
            pending.semantic_key,
            occurred_at,
            pending.payload_json,
            previous,
        )
        try:
            self._connection.execute(
                "INSERT INTO journal_records "
                "(sequence, record_id, kind, semantic_key, occurred_at, payload_json, "
                "prev_hash, record_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sequence,
                    pending.record_id,
                    pending.kind,
                    pending.semantic_key,
                    occurred_at,
                    pending.payload_json,
                    previous,
                    record_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise JournalCollisionError("journal uniqueness collision") from exc
        row = self._connection.execute(
            "SELECT * FROM journal_records WHERE sequence = ?", (sequence,)
        ).fetchone()
        assert row is not None
        return self._row_to_record(row)

    def _ledger_events_unlocked(self) -> tuple[ShadowLedgerEvent, ...]:
        events: list[ShadowLedgerEvent] = []
        try:
            for record in self._rows_unlocked():
                if record.kind == "shadow_execution":
                    events.append(
                        ShadowBatchExecuted.model_validate_json(record.payload_json)
                    )
                elif record.kind == "mark":
                    events.append(
                        ShadowMarkRecorded.model_validate_json(record.payload_json)
                    )
        except Exception as exc:
            raise JournalIntegrityError("journal contains an invalid ledger event") from exc
        return tuple(events)

    def _append(self, pending: tuple[_PendingRecord, ...]) -> tuple[JournalRecord, ...]:
        self._ensure_open()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._verify_unlocked()
            records = tuple(self._insert_unlocked(item) for item in pending)
            if any(item.kind in ("shadow_execution", "mark") for item in pending):
                reduce_shadow_ledger(self._ledger_events_unlocked())
            self._verify_unlocked()
            self._connection.execute("COMMIT")
            return records
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def record_confirmed_signal(
        self,
        value: ConfirmedSignalQueued,
    ) -> tuple[JournalRecord, JournalRecord, JournalRecord]:
        """Atomically queue the source snapshot, target, and confirmation."""

        records = self._append(
            (
                _pending(
                    "market_snapshot",
                    value.signal_snapshot.snapshot_id,
                    value.signal_snapshot.as_of,
                    value.signal_snapshot,
                ),
                _pending(
                    "confirmed_target",
                    value.confirmed_target.allocation_id,
                    value.confirmed_target.effective_at,
                    value.confirmed_target,
                ),
                _pending(
                    "confirmed_signal_queued",
                    value.signal_id,
                    value.confirmed_at,
                    value,
                ),
            )
        )
        return records[0], records[1], records[2]

    def read_confirmed_signal(self, signal_id: str) -> ConfirmedSignalQueued | None:
        """Load a verified pending signal before planning an execution step."""

        key = _nonempty(signal_id)
        matches = tuple(
            record
            for record in self.read_records("confirmed_signal_queued")
            if record.semantic_key == key
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise JournalIntegrityError("confirmed signal has multiple queued records")
        model = self.decode_record(matches[0])
        if not isinstance(model, ConfirmedSignalQueued):
            raise JournalIntegrityError("queued signal decoded to the wrong type")
        return model

    def record_market_snapshot(self, value: MarketSnapshot) -> JournalRecord:
        return self._append(
            (_pending("market_snapshot", value.snapshot_id, value.as_of, value),)
        )[0]

    def record_confirmed_target(self, value: TargetAllocation) -> JournalRecord:
        return self._append(
            (_pending("confirmed_target", value.allocation_id, value.effective_at, value),)
        )[0]

    def record_planned_batch(self, value: OrderBatch) -> JournalRecord:
        return self._append(
            (_pending("planned_batch", value.batch_id, value.effective_at, value),)
        )[0]

    @staticmethod
    def _risk_time(
        value: ShadowRiskDecision,
        occurred_at: datetime | None,
    ) -> datetime:
        model_time = value.evaluated_at
        if occurred_at is not None:
            _utc_text(occurred_at)
        if occurred_at is not None and model_time is not None:
            if occurred_at.astimezone(UTC) != model_time:
                raise ValueError("risk record time must equal the decision evaluation time")
        result = model_time if occurred_at is None else occurred_at
        if result is None:
            raise ValueError("a rejected decision without evaluated_at needs occurred_at")
        return result

    def record_risk_decision(
        self,
        value: ShadowRiskDecision,
        *,
        occurred_at: datetime | None = None,
    ) -> JournalRecord:
        timestamp = self._risk_time(value, occurred_at)
        return self._append(
            (_pending("risk_decision", value.batch_id, timestamp, value),)
        )[0]

    def record_mark(self, value: ShadowMarkRecorded) -> JournalRecord:
        return self._append(
            (_pending("mark", value.snapshot_id, value.snapshot.as_of, value),)
        )[0]

    def record_incident(self, value: OperationalIncident) -> JournalRecord:
        return self._append(
            (_pending("incident", value.incident_id, value.occurred_at, value),)
        )[0]

    def record_session_closed(self, value: OperationalSessionClosed) -> JournalRecord:
        return self._append(
            (_pending("session_closed", value.session_id, value.ended_at, value),)
        )[0]

    def commit_shadow_step(
        self,
        *,
        snapshot: MarketSnapshot,
        target: TargetAllocation,
        batch: OrderBatch,
        decision: ShadowRiskDecision,
        ledger_events: tuple[ShadowLedgerEvent, ...],
        incidents: tuple[OperationalIncident, ...] = (),
        signal_confirmation: SignalConfirmation | None = None,
        decision_at: datetime | None = None,
    ) -> tuple[JournalRecord, ...]:
        """Atomically persist one complete, cross-validated shadow workflow.

        A byte-identical retry returns the original records.  Any reused
        semantic key with different bytes aborts the whole transaction.
        """

        if not 1 <= len(ledger_events) <= 2:
            raise ValueError("a complete shadow step needs one or two ledger events")
        if target.effective_at != snapshot.as_of or batch.effective_at != snapshot.as_of:
            raise ValueError("step snapshot, target, and batch timestamps must match")
        if batch.allocation_id != target.allocation_id:
            raise ValueError("step batch must identify the activated target")
        if decision.snapshot_id != snapshot.snapshot_id or decision.batch_id != batch.batch_id:
            raise ValueError("risk decision does not identify the step snapshot and batch")
        if signal_confirmation is not None:
            if signal_confirmation.activated_target.canonical_json() != target.canonical_json():
                raise ValueError("signal confirmation does not contain the activated target")
            queued = self.read_confirmed_signal(signal_confirmation.signal_id)
            if queued is None:
                raise ValueError("signal must be durably queued before activation")
            queue_matches = (
                queued.strategy_version == signal_confirmation.strategy_version
                and queued.signal_snapshot.canonical_json()
                == signal_confirmation.signal_snapshot.canonical_json()
                and queued.confirmation_id == signal_confirmation.confirmation_id
                and queued.confirmed_at == signal_confirmation.confirmed_at
                and queued.confirmed_target.canonical_json()
                == signal_confirmation.confirmed_target.canonical_json()
            )
            if not queue_matches:
                raise JournalCollisionError(
                    "signal activation differs from its durable queued provenance"
                )
            if target.effective_at < queued.execute_not_before:
                raise ValueError("signal activation is earlier than execute_not_before")

        executions = tuple(
            event for event in ledger_events if isinstance(event, ShadowBatchExecuted)
        )
        marks = tuple(event for event in ledger_events if isinstance(event, ShadowMarkRecorded))
        if len(executions) > 1 or len(marks) > 1:
            raise ValueError("a shadow step may contain at most one execution and one mark")
        for event in ledger_events:
            if event.snapshot.canonical_json() != snapshot.canonical_json():
                raise ValueError("every step ledger event must contain the exact step snapshot")
        if isinstance(decision, ShadowRiskApproved):
            if len(executions) != 1 or len(marks) != 1:
                raise ValueError(
                    "an approved step must persist exactly one execution and one mark"
                )
            if executions[0].canonical_json() != decision.event.canonical_json():
                raise ValueError("executed event differs from the risk-approved event")
        else:
            if executions or len(marks) != 1:
                raise ValueError(
                    "a rejected step must persist exactly one mark and no execution"
                )
            if not incidents:
                raise ValueError("a rejected step must persist an operational incident")

        ordered_events = tuple(sorted(ledger_events, key=lambda event: event.sequence))
        ordered_incidents = tuple(sorted(incidents, key=lambda item: item.incident_id))
        pending: list[_PendingRecord] = []
        if signal_confirmation is not None:
            pending.append(
                _pending(
                    "signal_confirmation",
                    signal_confirmation.signal_id,
                    signal_confirmation.confirmed_at,
                    signal_confirmation,
                )
            )
            pending.extend(
                (
                    _pending(
                        "market_snapshot",
                        signal_confirmation.signal_snapshot.snapshot_id,
                        signal_confirmation.signal_snapshot.as_of,
                        signal_confirmation.signal_snapshot,
                    ),
                    _pending(
                        "confirmed_target",
                        signal_confirmation.confirmed_target.allocation_id,
                        signal_confirmation.confirmed_target.effective_at,
                        signal_confirmation.confirmed_target,
                    ),
                )
            )
        pending.extend(
            (
                _pending("market_snapshot", snapshot.snapshot_id, snapshot.as_of, snapshot),
                _pending("confirmed_target", target.allocation_id, target.effective_at, target),
                _pending("planned_batch", batch.batch_id, batch.effective_at, batch),
                _pending(
                    "risk_decision",
                    decision.batch_id,
                    self._risk_time(decision, decision_at),
                    decision,
                ),
            )
        )
        for event in ordered_events:
            if isinstance(event, ShadowBatchExecuted):
                pending.append(
                    _pending(
                        "shadow_execution", event.batch_id, event.snapshot.as_of, event
                    )
                )
            else:
                pending.append(
                    _pending("mark", event.snapshot_id, event.snapshot.as_of, event)
                )
        pending.extend(
            _pending("incident", item.incident_id, item.occurred_at, item)
            for item in ordered_incidents
        )
        step_id = (
            signal_confirmation.provenance_id
            if signal_confirmation is not None
            else f"batch:{batch.batch_id}"
        )
        manifest = ShadowStepManifest(
            step_id=step_id,
            provenance_id=(
                signal_confirmation.provenance_id
                if signal_confirmation is not None
                else None
            ),
            strategy_version=(
                signal_confirmation.strategy_version
                if signal_confirmation is not None
                else None
            ),
            snapshot_id=snapshot.snapshot_id,
            allocation_id=target.allocation_id,
            batch_id=batch.batch_id,
            record_ids=tuple(item.record_id for item in pending),
        )
        pending.append(_pending("shadow_step", step_id, snapshot.as_of, manifest))
        return self._append(tuple(pending))

    def read_shadow_step(self, step_id: str) -> tuple[JournalRecord, ...] | None:
        """Return a prior atomic step by provenance ID (or ``batch:<id>``).

        This lookup lets a restarted runner recognize a committed retry before
        it computes a new ledger sequence or repeats any workflow work.
        """

        key = _nonempty(step_id)
        records = self.read_records()
        manifests = tuple(
            record
            for record in records
            if record.kind == "shadow_step" and record.semantic_key == key
        )
        if not manifests:
            return None
        if len(manifests) != 1:
            raise JournalIntegrityError("shadow step has multiple manifests")
        manifest_record = manifests[0]
        manifest = self.decode_record(manifest_record)
        if not isinstance(manifest, ShadowStepManifest):
            raise JournalIntegrityError("shadow step manifest decoded to the wrong type")
        by_id = {record.record_id: record for record in records}
        try:
            constituents = tuple(by_id[record_id] for record_id in manifest.record_ids)
        except KeyError as exc:
            raise JournalIntegrityError(
                "shadow step manifest references a missing record"
            ) from exc
        return (*constituents, manifest_record)

    def replay_ledger_events(self) -> tuple[ShadowLedgerEvent, ...]:
        """Return verified execution/mark events in exact append sequence."""

        self._ensure_open()
        self._verify_unlocked()
        events = self._ledger_events_unlocked()
        reduce_shadow_ledger(events)
        return events

    def load_ledger(self) -> ShadowLedger:
        """Reconstruct a fresh exact ``ShadowLedger`` from persisted events."""

        ledger = ShadowLedger()
        for event in self.replay_ledger_events():
            ledger.append(event)
        return ledger


__all__ = [
    "ConfirmedSignalQueued",
    "JournalClosedError",
    "JournalCollisionError",
    "JournalIntegrityError",
    "JournalKind",
    "JournalRecord",
    "JournalSchemaError",
    "OperationalIncident",
    "OperationalSessionClosed",
    "SCHEMA_VERSION",
    "SQLiteShadowJournal",
    "ShadowJournalError",
    "ShadowStepManifest",
    "SignalConfirmation",
]
