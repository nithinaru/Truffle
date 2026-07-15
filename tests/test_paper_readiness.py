"""Tests for the explicit 30-session live-shadow operational gate."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from paper.journal import (
    OperationalIncident,
    OperationalSessionClosed,
    SQLiteShadowJournal,
)
from paper.readiness import (
    OperationalReadinessError,
    OperationalReadinessPolicy,
    evaluate_journal_operational_readiness,
    evaluate_operational_readiness,
)

START = date(2025, 1, 2)


def _calendar(count: int) -> tuple[date, ...]:
    return tuple(START + timedelta(days=index) for index in range(count))


def _session(
    session_date: date,
    *,
    status: str = "healthy",
) -> OperationalSessionClosed:
    started = datetime.combine(session_date, datetime.min.time(), tzinfo=UTC)
    incident_ids = () if status == "healthy" else (f"incident-{session_date}",)
    return OperationalSessionClosed(
        session_id=f"session-{session_date}",
        market_session_date=session_date,
        started_at=started,
        ended_at=started + timedelta(hours=7),
        status=status,
        incident_ids=incident_ids,
    )


def test_default_gate_requires_thirty_consecutive_official_sessions() -> None:
    calendar = _calendar(30)
    report = evaluate_operational_readiness(
        (_session(value) for value in calendar),
        calendar,
    )

    assert report.status == "ready"
    assert report.consecutive_healthy_sessions == 30
    assert report.streak_start_date == calendar[0]
    assert report.blockers == ()
    assert report.scope == "operational_only"
    assert report.real_money_authorized is False


def test_twenty_nine_sessions_are_not_ready() -> None:
    calendar = _calendar(29)
    report = evaluate_operational_readiness(
        tuple(_session(value) for value in calendar),
        calendar,
    )

    assert report.status == "not_ready"
    assert report.consecutive_healthy_sessions == 29
    assert report.blockers == ("insufficient_consecutive_healthy_sessions",)


def test_incident_resets_streak_and_later_healthy_sessions_rebuild_it() -> None:
    calendar = _calendar(35)
    sessions = [
        _session(value, status="incident" if index == 4 else "healthy")
        for index, value in enumerate(calendar)
    ]
    report = evaluate_operational_readiness(sessions, calendar)

    assert report.status == "ready"
    assert report.consecutive_healthy_sessions == 30
    assert report.incident_closures == 1
    assert report.streak_start_date == calendar[5]


def test_missing_official_session_breaks_trailing_streak() -> None:
    calendar = _calendar(30)
    missing = calendar[-2]
    sessions = tuple(_session(value) for value in calendar if value != missing)
    report = evaluate_operational_readiness(sessions, calendar)

    assert report.status == "not_ready"
    assert report.consecutive_healthy_sessions == 1
    assert report.missing_session_dates == (missing,)
    assert "insufficient_consecutive_healthy_sessions" in report.blockers


@pytest.mark.parametrize(
    ("status", "blocker"),
    [
        ("incident", "latest_session_had_incident"),
        ("incomplete", "latest_session_incomplete"),
    ],
)
def test_latest_nonhealthy_session_is_an_explicit_block(
    status: str,
    blocker: str,
) -> None:
    calendar = _calendar(30)
    sessions = [
        _session(value, status=status if value == calendar[-1] else "healthy")
        for value in calendar
    ]
    report = evaluate_operational_readiness(sessions, calendar)

    assert report.status == "not_ready"
    assert report.consecutive_healthy_sessions == 0
    assert blocker in report.blockers


def test_duplicate_or_calendar_mismatched_evidence_is_rejected() -> None:
    calendar = _calendar(2)
    duplicate = _session(calendar[0])
    with pytest.raises(OperationalReadinessError, match="duplicate dates"):
        evaluate_operational_readiness((duplicate, duplicate), calendar)

    with pytest.raises(OperationalReadinessError, match="outside the official"):
        evaluate_operational_readiness((_session(START + timedelta(days=9)),), calendar)

    with pytest.raises(OperationalReadinessError, match="strictly increasing"):
        evaluate_operational_readiness((), tuple(reversed(calendar)))


def test_empty_calendar_is_never_operationally_ready() -> None:
    report = evaluate_operational_readiness((), ())

    assert report.status == "not_ready"
    assert report.as_of_session_date is None
    assert report.blockers == (
        "no_official_sessions",
        "insufficient_consecutive_healthy_sessions",
    )


def test_journal_evaluator_verifies_and_decodes_session_evidence(tmp_path: Path) -> None:
    calendar = _calendar(2)
    incident = OperationalIncident(
        incident_id=f"incident-{calendar[0]}",
        occurred_at=datetime.combine(calendar[0], datetime.min.time(), tzinfo=UTC),
        severity="warning",
        code="disconnect",
        message="Market-data connection ended unexpectedly.",
    )
    with SQLiteShadowJournal(tmp_path / "readiness.sqlite") as journal:
        journal.record_incident(incident)
        journal.record_session_closed(_session(calendar[0], status="incident"))
        journal.record_session_closed(_session(calendar[1]))
        report = evaluate_journal_operational_readiness(
            journal,
            calendar,
            policy=OperationalReadinessPolicy(
                required_consecutive_healthy_sessions=1
            ),
        )

    assert report.status == "ready"
    assert report.consecutive_healthy_sessions == 1
    assert report.incident_closures == 1

