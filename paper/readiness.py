"""Operational readiness checks for the non-submitting live-shadow stage.

Readiness is evaluated against an injected official market-session calendar.
That makes an absent journal closure an explicit break in the streak instead of
silently treating "no incident row" as evidence that a session was healthy.
The result is intentionally operational-only and never authorizes real-money
trading.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Literal

from pydantic import Field

from paper.journal import OperationalSessionClosed, SQLiteShadowJournal
from paper.models import PaperModel


class OperationalReadinessError(ValueError):
    """Raised when session evidence or its official calendar is ambiguous."""


class OperationalReadinessPolicy(PaperModel):
    """Predeclared live-shadow machinery gate."""

    required_consecutive_healthy_sessions: int = Field(default=30, ge=1)


class OperationalReadinessReport(PaperModel):
    """Deterministic assessment of journaled live-market operations only."""

    scope: Literal["operational_only"] = "operational_only"
    status: Literal["ready", "not_ready"]
    real_money_authorized: Literal[False] = False
    required_consecutive_healthy_sessions: int
    consecutive_healthy_sessions: int
    official_sessions_evaluated: int
    closures_evaluated: int
    healthy_closures: int
    incident_closures: int
    incomplete_closures: int
    as_of_session_date: date | None
    streak_start_date: date | None
    missing_session_dates: tuple[date, ...]
    blockers: tuple[str, ...]


def _official_calendar(values: Iterable[date]) -> tuple[date, ...]:
    sessions = tuple(values)
    if any(not isinstance(value, date) for value in sessions):
        raise OperationalReadinessError(
            "official market sessions must be date instances"
        )
    if tuple(sorted(sessions)) != sessions:
        raise OperationalReadinessError(
            "official market sessions must be strictly increasing"
        )
    if len(sessions) != len(set(sessions)):
        raise OperationalReadinessError(
            "official market sessions must not contain duplicates"
        )
    return sessions


def _session_evidence(
    values: Iterable[OperationalSessionClosed],
) -> dict[date, OperationalSessionClosed]:
    sessions = tuple(values)
    if any(not isinstance(value, OperationalSessionClosed) for value in sessions):
        raise OperationalReadinessError(
            "session evidence must contain OperationalSessionClosed objects"
        )
    by_date: dict[date, OperationalSessionClosed] = {}
    session_ids: set[str] = set()
    for session in sessions:
        if session.market_session_date in by_date:
            raise OperationalReadinessError(
                "market session evidence contains duplicate dates"
            )
        if session.session_id in session_ids:
            raise OperationalReadinessError(
                "market session evidence contains duplicate session IDs"
            )
        by_date[session.market_session_date] = session
        session_ids.add(session.session_id)
    return by_date


def evaluate_operational_readiness(
    sessions: Iterable[OperationalSessionClosed],
    official_session_dates: Iterable[date],
    *,
    policy: OperationalReadinessPolicy | None = None,
) -> OperationalReadinessReport:
    """Count the trailing healthy streak across every official session.

    ``official_session_dates`` must contain every exchange session through the
    caller's chosen as-of date.  A missing closure, incident closure, or
    incomplete closure breaks the streak.  Dates outside the supplied calendar
    are rejected so a caller cannot accidentally assess mismatched evidence.
    """

    effective_policy = policy or OperationalReadinessPolicy()
    calendar = _official_calendar(official_session_dates)
    by_date = _session_evidence(sessions)
    unexpected = tuple(sorted(set(by_date) - set(calendar)))
    if unexpected:
        rendered = ", ".join(value.isoformat() for value in unexpected)
        raise OperationalReadinessError(
            f"session evidence is outside the official calendar: {rendered}"
        )

    missing = tuple(value for value in calendar if value not in by_date)
    ordered_evidence = tuple(
        by_date[value] for value in calendar if value in by_date
    )
    healthy = sum(session.status == "healthy" for session in ordered_evidence)
    incidents = sum(session.status == "incident" for session in ordered_evidence)
    incomplete = sum(session.status == "incomplete" for session in ordered_evidence)

    streak = 0
    streak_start: date | None = None
    for session_date in reversed(calendar):
        session = by_date.get(session_date)
        if session is None or session.status != "healthy":
            break
        streak += 1
        streak_start = session_date

    blockers: list[str] = []
    if not calendar:
        blockers.append("no_official_sessions")
    elif calendar[-1] not in by_date:
        blockers.append("latest_session_missing")
    elif by_date[calendar[-1]].status == "incident":
        blockers.append("latest_session_had_incident")
    elif by_date[calendar[-1]].status == "incomplete":
        blockers.append("latest_session_incomplete")
    if streak < effective_policy.required_consecutive_healthy_sessions:
        blockers.append("insufficient_consecutive_healthy_sessions")

    return OperationalReadinessReport(
        status="ready" if not blockers else "not_ready",
        required_consecutive_healthy_sessions=(
            effective_policy.required_consecutive_healthy_sessions
        ),
        consecutive_healthy_sessions=streak,
        official_sessions_evaluated=len(calendar),
        closures_evaluated=len(ordered_evidence),
        healthy_closures=healthy,
        incident_closures=incidents,
        incomplete_closures=incomplete,
        as_of_session_date=calendar[-1] if calendar else None,
        streak_start_date=streak_start,
        missing_session_dates=missing,
        blockers=tuple(blockers),
    )


def evaluate_journal_operational_readiness(
    journal: SQLiteShadowJournal,
    official_session_dates: Iterable[date],
    *,
    policy: OperationalReadinessPolicy | None = None,
) -> OperationalReadinessReport:
    """Verify the journal, decode session closures, and evaluate its streak."""

    sessions: list[OperationalSessionClosed] = []
    for record in journal.read_records(kind="session_closed"):
        decoded = journal.decode_record(record)
        if not isinstance(decoded, OperationalSessionClosed):
            raise OperationalReadinessError(
                "session-closed journal record decoded to the wrong model"
            )
        sessions.append(decoded)
    return evaluate_operational_readiness(
        sessions,
        official_session_dates,
        policy=policy,
    )


__all__ = [
    "OperationalReadinessError",
    "OperationalReadinessPolicy",
    "OperationalReadinessReport",
    "evaluate_journal_operational_readiness",
    "evaluate_operational_readiness",
]
