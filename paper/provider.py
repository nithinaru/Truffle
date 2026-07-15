"""Offline market-data boundary for deterministic paper replays.

The first paper-testing stage deliberately has no network implementation.
``LocalReplayProvider`` accepts immutable :class:`MarketSnapshot` objects or a
caller-supplied wide price panel, then exposes exact snapshots and trailing
solver windows without ever reading beyond the requested instant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from paper.models import MarketSnapshot, Quote


class ReplayDataError(ValueError):
    """Raised when local replay data is incomplete, ambiguous, or unordered."""


class SnapshotNotFoundError(ReplayDataError):
    """Raised when an exact replay instant is unavailable."""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ReplayDataError("snapshot lookup timestamps must be timezone-aware")
    return value.astimezone(UTC)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Narrow read-only boundary shared by replay and future live adapters."""

    @property
    def symbols(self) -> tuple[str, ...]: ...

    def snapshot_at(self, as_of: datetime) -> MarketSnapshot: ...

    def history_through(
        self,
        as_of: datetime,
        *,
        observations: int | None = None,
    ) -> pd.DataFrame: ...


class LocalReplayProvider:
    """Strictly ordered, complete snapshots held entirely in local memory."""

    def __init__(self, snapshots: Iterable[MarketSnapshot]) -> None:
        ordered = tuple(snapshots)
        if not ordered:
            raise ReplayDataError("local replay requires at least one market snapshot")

        expected_symbols = tuple(quote.ticker for quote in ordered[0].quotes)
        seen_ids: set[str] = set()
        previous_time: datetime | None = None
        by_time: dict[datetime, MarketSnapshot] = {}
        for snapshot in ordered:
            symbols = tuple(quote.ticker for quote in snapshot.quotes)
            if symbols != expected_symbols:
                raise ReplayDataError(
                    "every replay snapshot must contain the same complete symbol set; "
                    f"expected {list(expected_symbols)}, got {list(symbols)} at "
                    f"{snapshot.as_of.isoformat()}"
                )
            if snapshot.snapshot_id in seen_ids:
                raise ReplayDataError(
                    f"duplicate replay snapshot ID: {snapshot.snapshot_id!r}"
                )
            if previous_time is not None and snapshot.as_of <= previous_time:
                raise ReplayDataError(
                    "replay snapshot timestamps must be unique and strictly increasing"
                )
            seen_ids.add(snapshot.snapshot_id)
            by_time[snapshot.as_of] = snapshot
            previous_time = snapshot.as_of

        self._snapshots = ordered
        self._symbols = expected_symbols
        self._by_time = by_time

    @classmethod
    def from_wide_prices(
        cls,
        prices: pd.DataFrame,
        *,
        snapshot_id_prefix: str = "local-replay",
    ) -> LocalReplayProvider:
        """Build zero-observed-spread quotes from a wide close-price panel.

        Naive timestamps are interpreted explicitly as UTC. Binary floating
        inputs are converted through their shortest decimal string at this
        ingestion boundary; all accounting models downstream receive Decimal.
        """

        if prices.empty or prices.shape[1] < 1:
            raise ReplayDataError("wide replay prices require at least one row and column")
        if not snapshot_id_prefix.strip():
            raise ReplayDataError("snapshot_id_prefix must not be empty")
        if prices.columns.has_duplicates:
            raise ReplayDataError("wide replay prices contain duplicate columns")

        symbols = [str(column).strip().upper() for column in prices.columns]
        if any(not symbol for symbol in symbols):
            raise ReplayDataError("wide replay tickers must not be empty")
        if len(set(symbols)) != len(symbols):
            raise ReplayDataError("wide replay tickers collide after canonicalization")

        try:
            index = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="raise"))
        except (TypeError, ValueError) as exc:
            raise ReplayDataError("wide replay index must contain valid timestamps") from exc
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ReplayDataError(
                "wide replay timestamps must be unique and strictly increasing"
            )
        if index.tz is None:
            index = index.tz_localize(UTC)
        else:
            index = index.tz_convert(UTC)

        snapshots: list[MarketSnapshot] = []
        for row_number, timestamp in enumerate(index):
            as_of = timestamp.to_pydatetime()
            quotes: list[Quote] = []
            for column_number, symbol in enumerate(symbols):
                raw = prices.iloc[row_number, column_number]
                try:
                    price = Decimal(str(raw))
                except (InvalidOperation, ValueError) as exc:
                    raise ReplayDataError(
                        f"invalid replay price for {symbol} at {as_of.isoformat()}: {raw!r}"
                    ) from exc
                if not price.is_finite() or price <= Decimal("0"):
                    raise ReplayDataError(
                        f"replay prices must be finite and positive; {symbol} at "
                        f"{as_of.isoformat()} is {raw!r}"
                    )
                quotes.append(
                    Quote(
                        ticker=symbol,
                        bid=price,
                        ask=price,
                        last=price,
                        as_of=as_of,
                    )
                )
            snapshots.append(
                MarketSnapshot(
                    snapshot_id=f"{snapshot_id_prefix}:{as_of.isoformat()}",
                    as_of=as_of,
                    quotes=tuple(quotes),
                )
            )
        return cls(snapshots)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        snapshot_id_prefix: str = "local-replay",
    ) -> LocalReplayProvider:
        """Load a local wide CSV whose first column contains timestamps."""

        frame = pd.read_csv(path, dtype=str)
        if frame.shape[1] < 2:
            raise ReplayDataError(
                "wide replay CSV needs a timestamp column and at least one ticker"
            )
        timestamp_column = frame.columns[0]
        values = frame.drop(columns=[timestamp_column])
        values.index = frame[timestamp_column]
        return cls.from_wide_prices(
            values,
            snapshot_id_prefix=snapshot_id_prefix,
        )

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def snapshots(self) -> tuple[MarketSnapshot, ...]:
        return self._snapshots

    def snapshot_at(self, as_of: datetime) -> MarketSnapshot:
        instant = _utc(as_of)
        try:
            return self._by_time[instant]
        except KeyError:
            raise SnapshotNotFoundError(
                f"no local replay snapshot exists at {instant.isoformat()}"
            ) from None

    def history_through(
        self,
        as_of: datetime,
        *,
        observations: int | None = None,
    ) -> pd.DataFrame:
        """Return closes through an exact snapshot, never future observations."""

        endpoint = self.snapshot_at(as_of)
        if observations is not None and (
            isinstance(observations, bool)
            or not isinstance(observations, Integral)
            or observations < 1
        ):
            raise ReplayDataError("observations must be a positive integer or None")
        endpoint_index = self._snapshots.index(endpoint) + 1
        count = None if observations is None else int(observations)
        start_index = 0 if count is None else max(0, endpoint_index - count)
        selected = self._snapshots[start_index:endpoint_index]
        values = [
            [float(snapshot.quote(symbol).last) for symbol in self._symbols]
            for snapshot in selected
        ]
        if not all(math.isfinite(value) for row in values for value in row):
            raise ReplayDataError("replay prices exceed the finite solver numeric range")
        return pd.DataFrame(
            values,
            index=pd.DatetimeIndex([snapshot.as_of for snapshot in selected]),
            columns=list(self._symbols),
            dtype=float,
        )

    def iter_snapshots(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[MarketSnapshot]:
        """Iterate a closed UTC interval in canonical replay order."""

        lower = None if start is None else _utc(start)
        upper = None if end is None else _utc(end)
        if lower is not None and upper is not None and lower > upper:
            raise ReplayDataError("replay interval start must not be after end")
        for snapshot in self._snapshots:
            if lower is not None and snapshot.as_of < lower:
                continue
            if upper is not None and snapshot.as_of > upper:
                continue
            yield snapshot

    def next_after(self, as_of: datetime) -> MarketSnapshot | None:
        """Return the first later local snapshot, or ``None`` at replay end."""

        instant = _utc(as_of)
        return next(
            (snapshot for snapshot in self._snapshots if snapshot.as_of > instant),
            None,
        )


__all__ = [
    "LocalReplayProvider",
    "MarketDataProvider",
    "ReplayDataError",
    "SnapshotNotFoundError",
]
