"""Offline replay provider tests; no network or broker access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from paper.models import MarketSnapshot, Quote
from paper.provider import (
    LocalReplayProvider,
    ReplayDataError,
    SnapshotNotFoundError,
)

T0 = datetime(2026, 1, 2, 21, tzinfo=UTC)


def _snapshot(snapshot_id: str, as_of: datetime, **prices: str) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        as_of=as_of,
        quotes=tuple(
            Quote(ticker=ticker, bid=price, ask=price, last=price, as_of=as_of)
            for ticker, price in prices.items()
        ),
    )


def test_wide_replay_is_complete_ordered_and_has_no_future_leakage() -> None:
    prices = pd.DataFrame(
        {"bbb": ["20.00", "21.00", "22.00"], "aaa": ["10", "11", "12"]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    provider = LocalReplayProvider.from_wide_prices(prices)

    assert provider.symbols == ("AAA", "BBB")
    assert all(snapshot.as_of.tzinfo is UTC for snapshot in provider.snapshots)
    endpoint = datetime(2026, 1, 5, tzinfo=UTC)
    history = provider.history_through(endpoint, observations=2)
    assert list(history.columns) == ["AAA", "BBB"]
    assert list(history.index) == [
        pd.Timestamp("2026-01-02", tz=UTC),
        pd.Timestamp("2026-01-05", tz=UTC),
    ]
    assert history.loc[endpoint, "AAA"] == 11.0
    assert provider.snapshot_at(endpoint).quote("aaa").last == Decimal("11")
    assert provider.next_after(endpoint) == provider.snapshots[2]


def test_replay_requires_exact_lookup_and_complete_symbol_sets() -> None:
    first = _snapshot("s0", T0, AAA="10", BBB="20")
    incomplete = _snapshot("s1", T0 + timedelta(days=1), AAA="11")
    with pytest.raises(ReplayDataError, match="same complete symbol set"):
        LocalReplayProvider([first, incomplete])

    provider = LocalReplayProvider([first])
    with pytest.raises(SnapshotNotFoundError, match="no local replay snapshot"):
        provider.history_through(T0 + timedelta(seconds=1))


@pytest.mark.parametrize(
    "prices, message",
    [
        (
            pd.DataFrame(
                {"AAA": [10, 11]},
                index=pd.to_datetime(["2026-01-03", "2026-01-02"]),
            ),
            "strictly increasing",
        ),
        (
            pd.DataFrame(
                {"AAA": [10, 0]},
                index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
            ),
            "finite and positive",
        ),
    ],
)
def test_wide_replay_rejects_ambiguous_or_invalid_data(
    prices: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ReplayDataError, match=message):
        LocalReplayProvider.from_wide_prices(prices)


def test_csv_ingestion_preserves_decimal_text_and_interval_bounds(tmp_path: Path) -> None:
    path = tmp_path / "replay.csv"
    path.write_text(
        "as_of,AAA\n"
        "2026-01-02T21:00:00Z,10.0100\n"
        "2026-01-03T21:00:00Z,10.0200\n",
        encoding="utf-8",
    )
    provider = LocalReplayProvider.from_csv(path)

    assert provider.snapshots[0].quote("AAA").last == Decimal("10.0100")
    selected = tuple(
        provider.iter_snapshots(
            start=datetime(2026, 1, 3, 21, tzinfo=UTC),
            end=datetime(2026, 1, 3, 21, tzinfo=UTC),
        )
    )
    assert selected == (provider.snapshots[1],)
