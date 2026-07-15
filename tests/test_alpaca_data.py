"""Offline-only tests for the capability-bounded Alpaca data adapter."""

from __future__ import annotations

import builtins
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import SecretStr

from paper.alpaca_data import (
    DATA_BASE_URL,
    PAPER_BASE_URL,
    AlpacaAssetEligibilityError,
    AlpacaAuthenticationError,
    AlpacaConfigurationError,
    AlpacaCredentialError,
    AlpacaCredentials,
    AlpacaDataConfig,
    AlpacaDataProvider,
    AlpacaDependencyError,
    AlpacaEndpointError,
    AlpacaHistoryError,
    AlpacaHttpError,
    AlpacaHttpResponse,
    AlpacaRateLimitError,
    AlpacaResponseTooLargeError,
    AlpacaSchemaError,
    AlpacaSessionError,
    AlpacaSnapshotError,
    AlpacaStaleDataError,
    AlpacaTransportFailureError,
    HttpxAlpacaTransport,
    MarketSession,
    SourceTimestamp,
)
from paper.journal import SQLiteShadowJournal
from paper.models import MarketSnapshot

SESSION = MarketSession(
    session_date=date(2026, 1, 2),
    open_at=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
    close_at=datetime(2026, 1, 2, 21, 0, tzinfo=UTC),
)
CAPTURED = datetime(2026, 1, 2, 15, 0, 10, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Call:
    endpoint: str
    symbol: str | None
    params: Mapping[str, str]
    headers: Mapping[str, str]
    timeout_seconds: float
    max_response_bytes: int


class FakeTransport:
    """One ordered fake; queue entries are responses or transport failures."""

    def __init__(self, *results: AlpacaHttpResponse | Exception) -> None:
        self.results = list(results)
        self.calls: list[Call] = []
        self.close_count = 0

    def get(
        self,
        endpoint: str,
        *,
        symbol: str | None,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> AlpacaHttpResponse:
        self.calls.append(
            Call(
                endpoint=endpoint,
                symbol=symbol,
                params=dict(params),
                headers=dict(headers),
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        )
        if not self.results:
            raise AssertionError("fake transport has no queued result")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.close_count += 1


def _response(
    payload: object = None,
    *,
    raw: bytes | None = None,
    status: int = 200,
    request_id: str | None = "request-1",
    headers: Mapping[str, str] | None = None,
) -> AlpacaHttpResponse:
    response_headers = dict(headers or {})
    if request_id is not None:
        response_headers["X-Request-ID"] = request_id
    return AlpacaHttpResponse(
        status_code=status,
        headers=response_headers,
        content=json.dumps(payload, separators=(",", ":")).encode() if raw is None else raw,
    )


def _wire_symbol(
    *,
    quote_time: str = "2026-01-02T15:00:09.123456789Z",
    trade_time: str = "2026-01-02T15:00:08.987654321Z",
    bid: object = "10.0100",
    ask: object = "10.0200",
    last: object = "10.0150",
) -> dict[str, object]:
    return {
        "latestQuote": {
            "t": quote_time,
            "bp": bid,
            "ap": ask,
            "bx": "V",
            "ax": "V",
            "c": ["R"],
        },
        "latestTrade": {
            "t": trade_time,
            "p": last,
            "x": "V",
            "c": ["@", "I"],
            "i": 12345,
        },
    }


def _snapshot_payload(**overrides: object) -> dict[str, object]:
    aaa = _wire_symbol()
    aaa.update(overrides)
    return {"AAA": aaa}


def _credentials() -> AlpacaCredentials:
    return AlpacaCredentials(key_id="key-id-value", secret_key="secret-value")


def _provider(
    transport: FakeTransport,
    *,
    symbols: tuple[str, ...] = ("AAA",),
    sleeper: Callable[[float], None] | None = None,
    jitter: Callable[[float], float] | None = None,
    **config: Any,
) -> AlpacaDataProvider:
    return AlpacaDataProvider(
        AlpacaDataConfig(symbols=symbols, **config),
        _credentials(),
        transport=transport,
        sleeper=sleeper,
        jitter=jitter,
    )


def _session(day: int, *, close_hour: int = 21) -> MarketSession:
    return MarketSession(
        session_date=date(2026, 1, day),
        open_at=datetime(2026, 1, day, 14, 30, tzinfo=UTC),
        close_at=datetime(2026, 1, day, close_hour, tzinfo=UTC),
    )


def _bar(day: str, close: str) -> dict[str, object]:
    return {"t": f"2026-01-{day}T05:00:00Z", "c": close}


def test_credentials_are_explicit_redacted_and_load_only_from_injected_mapping() -> None:
    credentials = AlpacaCredentials.from_env(
        {
            "TRUFFLE_ALPACA_KEY_ID": "paper-key",
            "TRUFFLE_ALPACA_SECRET_KEY": "paper-secret",
        }
    )

    assert isinstance(credentials.key_id, SecretStr)
    assert credentials.key_id.get_secret_value() == "paper-key"
    assert "paper-key" not in repr(credentials)
    assert "paper-secret" not in repr(credentials)
    with pytest.raises(AlpacaCredentialError, match="missing required credential"):
        AlpacaCredentials.from_env({})
    with pytest.raises(AlpacaCredentialError, match="non-empty"):
        AlpacaCredentials(key_id=" ", secret_key="secret")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"symbols": ()}, "1-30"),
        ({"symbols": tuple(f"S{i}" for i in range(31))}, "1-30"),
        ({"symbols": ("aaa", "AAA")}, "collide"),
        ({"symbols": ("../orders",)}, "US-equity identifiers"),
        ({"symbols": ("AAA",), "feed": "sip"}, "only feed='iex'"),
        ({"symbols": ("AAA",), "request_timeout_seconds": 0}, "positive"),
        ({"symbols": ("AAA",), "max_pages": 0}, "between"),
    ],
)
def test_configuration_is_canonical_and_bounded(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AlpacaConfigurationError, match=message):
        AlpacaDataConfig(**kwargs)  # type: ignore[arg-type]

    assert AlpacaDataConfig(symbols=("brk.b", "aapl")).symbols == ("AAPL", "BRK.B")


def test_capture_preserves_decimal_nanoseconds_provenance_and_safe_request() -> None:
    # Numeric JSON (rather than strings) proves parse_float=Decimal preserves text precision.
    raw = (
        b'{"AAA":{"latestQuote":{"t":"2026-01-02T15:00:09.123456789Z",'
        b'"bp":10.0100,"ap":10.0200,"bx":"V","ax":"V","c":["R"]},'
        b'"latestTrade":{"t":"2026-01-02T15:00:08.987654321Z",'
        b'"p":10.0150,"x":"V","c":["@","I"],"i":12345}}}'
    )
    transport = FakeTransport(_response(raw=raw, request_id="snapshot-request"))
    result = _provider(transport).capture_execution_snapshot(
        captured_at=CAPTURED,
        session=SESSION,
    )

    source = result.symbols[0]
    assert source.bid == Decimal("10.0100")
    assert source.ask == Decimal("10.0200")
    assert source.last == Decimal("10.0150")
    assert source.quote_timestamp.text == "2026-01-02T15:00:09.123456789Z"
    assert source.quote_timestamp.epoch_ns % 1_000_000_000 == 123_456_789
    assert source.trade_timestamp.epoch_ns % 1_000_000_000 == 987_654_321
    assert source.bid_exchange == source.ask_exchange == source.trade_exchange == "V"
    assert source.quote_conditions == ("R",)
    assert source.trade_conditions == ("@", "I")
    assert result.captured_at == CAPTURED
    assert result.request_id == "snapshot-request"
    assert result.capture_id.startswith("alpaca:iex:")
    assert result.snapshot.snapshot_id == result.capture_id
    assert result.snapshot.as_of == datetime(2026, 1, 2, 15, 0, 8, 987654, tzinfo=UTC)
    assert result.snapshot.quote("AAA").bid == Decimal("10.0100")

    call = transport.calls[0]
    assert call.endpoint == "snapshots"
    assert call.symbol is None
    assert call.params == {"symbols": "AAA", "feed": "iex", "currency": "USD"}
    assert call.timeout_seconds == 10.0
    assert call.max_response_bytes == 2_000_000
    assert call.headers["APCA-API-KEY-ID"] == "key-id-value"
    assert call.headers["APCA-API-SECRET-KEY"] == "secret-value"


def test_capture_id_is_stable_for_identical_evidence() -> None:
    response = _response(_snapshot_payload(), request_id="same-request")
    first = _provider(FakeTransport(response)).capture_execution_snapshot(
        captured_at=CAPTURED,
        session=SESSION,
    )
    second = _provider(FakeTransport(response)).capture_execution_snapshot(
        captured_at=CAPTURED,
        session=SESSION,
    )
    changed = _provider(
        FakeTransport(_response(_snapshot_payload(), request_id="other-request"))
    ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)

    assert first.capture_id == second.capture_id
    assert changed.capture_id != first.capture_id


def test_normalized_snapshot_durably_retains_full_source_provenance(
    tmp_path: Path,
) -> None:
    observed = _provider(
        FakeTransport(_response(_snapshot_payload(), request_id="alpaca-request-42"))
    ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)

    with SQLiteShadowJournal(tmp_path / "observations.sqlite") as journal:
        record = journal.record_market_snapshot(observed.snapshot)

    with SQLiteShadowJournal(tmp_path / "observations.sqlite") as restarted:
        decoded = restarted.decode_record(restarted.read_records()[0])

    assert record.kind == "market_snapshot"
    assert isinstance(decoded, MarketSnapshot)
    assert decoded.source_provenance_json is not None
    provenance = json.loads(decoded.source_provenance_json)
    assert provenance["provider"] == "alpaca"
    assert provenance["request_id"] == "alpaca-request-42"
    assert provenance["symbols"][0]["quote_timestamp"]["epoch_ns"] > 0
    assert provenance["session"]["session_date"] == "2026-01-02"


def test_effective_snapshot_time_is_oldest_source_floored_conservatively() -> None:
    captured = datetime(2026, 1, 2, 15, 0, 9, 123456, tzinfo=UTC)
    response = _response(
        {
            "AAA": _wire_symbol(
                quote_time="2026-01-02T15:00:09.123456001Z",
                trade_time="2026-01-02T15:00:09.123456001Z",
            )
        }
    )
    result = _provider(FakeTransport(response)).capture_execution_snapshot(
        captured_at=captured,
        session=SESSION,
    )

    assert result.snapshot.as_of == datetime(2026, 1, 2, 15, 0, 9, 123456, tzinfo=UTC)

    older_trade = _provider(
        FakeTransport(
            _response(
                {
                    "AAA": _wire_symbol(
                        quote_time="2026-01-02T15:00:09.999999999Z",
                        trade_time="2026-01-02T15:00:01.000000001Z",
                    )
                }
            )
        )
    ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)
    assert older_trade.snapshot.as_of == datetime(2026, 1, 2, 15, 0, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, r"missing=\['AAA'\]"),
        ({"AAA": _wire_symbol(), "BBB": _wire_symbol()}, r"unexpected=\['BBB'\]"),
        ({"AAA": _wire_symbol(bid="11", ask="10")}, "prices"),
        ({"AAA": _wire_symbol(bid="0")}, "prices"),
        ({"AAA": {"latestQuote": _wire_symbol()["latestQuote"]}}, "latestTrade"),
    ],
)
def test_capture_fails_closed_for_incomplete_or_invalid_market_data(
    payload: object,
    message: str,
) -> None:
    with pytest.raises((AlpacaSnapshotError, AlpacaSchemaError), match=message):
        _provider(FakeTransport(_response(payload))).capture_execution_snapshot(
            captured_at=CAPTURED,
            session=SESSION,
        )


def test_capture_requires_regular_hours_and_honors_early_close() -> None:
    early = MarketSession(
        session_date=SESSION.session_date,
        open_at=SESSION.open_at,
        close_at=datetime(2026, 1, 2, 18, tzinfo=UTC),
    )
    with pytest.raises(AlpacaSessionError, match="inside"):
        _provider(FakeTransport()).capture_execution_snapshot(
            captured_at=early.close_at,
            session=early,
        )
    assert not FakeTransport().calls


@pytest.mark.parametrize(
    "wire, config, exception, message",
    [
        (
            _wire_symbol(quote_time="2026-01-02T14:59:59Z"),
            {},
            AlpacaStaleDataError,
            "quote exceeds",
        ),
        (
            _wire_symbol(trade_time="2026-01-02T14:58:00Z"),
            {},
            AlpacaStaleDataError,
            "trade exceeds",
        ),
        (
            _wire_symbol(quote_time="2026-01-02T15:00:13Z"),
            {},
            AlpacaStaleDataError,
            "max_future_skew",
        ),
        (
            _wire_symbol(quote_time="2026-01-02T14:29:59.999999999Z"),
            {"max_quote_age_seconds": 3600},
            AlpacaSessionError,
            "outside",
        ),
    ],
)
def test_capture_checks_each_source_for_freshness_future_skew_and_session(
    wire: dict[str, object],
    config: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _provider(FakeTransport(_response({"AAA": wire})), **config).capture_execution_snapshot(
            captured_at=CAPTURED,
            session=SESSION,
        )


def test_asset_metadata_requires_exact_fractional_us_equity_eligibility() -> None:
    transport = FakeTransport(
        _response(
            {
                "symbol": "AAA",
                "class": "us_equity",
                "status": "active",
                "tradable": True,
                "fractionable": True,
            },
            request_id="asset-aaa",
        ),
        _response(
            {
                "symbol": "BRK.B",
                "class": "us_equity",
                "status": "active",
                "tradable": True,
                "fractionable": True,
            },
            request_id="asset-brk",
        ),
    )

    assets = _provider(transport, symbols=("brk.b", "aaa")).asset_metadata()

    assert tuple(asset.symbol for asset in assets) == ("AAA", "BRK.B")
    assert tuple(call.endpoint for call in transport.calls) == ("asset", "asset")
    assert tuple(call.symbol for call in transport.calls) == ("AAA", "BRK.B")
    assert assets[0].request_id == "asset-aaa"


@pytest.mark.parametrize(
    "field, value",
    [
        ("class", "crypto"),
        ("status", "inactive"),
        ("tradable", False),
        ("fractionable", False),
    ],
)
def test_asset_metadata_rejects_every_ineligible_dimension(field: str, value: object) -> None:
    payload = {
        "symbol": "AAA",
        "class": "us_equity",
        "status": "active",
        "tradable": True,
        "fractionable": True,
    }
    payload[field] = value

    with pytest.raises(AlpacaAssetEligibilityError, match="active, tradable, fractionable"):
        _provider(FakeTransport(_response(payload))).asset_metadata()


def test_calendar_normalizes_new_york_times_and_preserves_early_close() -> None:
    transport = FakeTransport(
        _response([{"date": "2026-01-02", "open": "09:30", "close": "13:00"}])
    )

    sessions = _provider(transport).fetch_market_sessions(
        date(2026, 1, 2),
        date(2026, 1, 2),
    )

    assert sessions == (
        MarketSession(
            session_date=date(2026, 1, 2),
            open_at=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
            close_at=datetime(2026, 1, 2, 18, 0, tzinfo=UTC),
        ),
    )
    assert transport.calls[0].endpoint == "calendar"
    assert transport.calls[0].params == {
        "start": "2026-01-02",
        "end": "2026-01-02",
        "date_type": "TRADING",
    }


def test_calendar_rejects_duplicate_or_out_of_range_rows() -> None:
    duplicate = {"date": "2026-01-02", "open": "09:30", "close": "16:00"}
    with pytest.raises(AlpacaSchemaError, match="duplicate"):
        _provider(FakeTransport(_response([duplicate, duplicate]))).fetch_market_sessions(
            date(2026, 1, 1),
            date(2026, 1, 3),
        )
    with pytest.raises(AlpacaSchemaError, match="outside"):
        _provider(FakeTransport(_response([duplicate]))).fetch_market_sessions(
            date(2026, 1, 3),
            date(2026, 1, 3),
        )


def test_market_session_instants_must_match_the_new_york_session_date() -> None:
    with pytest.raises(ValueError, match="session_date"):
        MarketSession(
            session_date=date(2026, 1, 3),
            open_at=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
            close_at=datetime(2026, 1, 2, 21, 0, tzinfo=UTC),
        )


def test_history_paginates_symbol_first_and_relabels_to_official_closes() -> None:
    sessions = (_session(2), _session(5))
    transport = FakeTransport(
        _response(
            {
                "bars": {"AAA": [_bar("02", "10.0100"), _bar("05", "10.0200")]},
                "next_page_token": "page-two",
            },
            request_id="bars-one",
        ),
        _response(
            {
                "bars": {"BBB": [_bar("02", "20.0100"), _bar("05", "20.0200")]},
                "next_page_token": None,
            },
            request_id="bars-two",
        ),
    )
    as_of = datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC)

    history = _provider(transport, symbols=("BBB", "AAA")).history_through(
        as_of,
        observations=2,
        sessions=sessions,
    )

    assert list(history.columns) == ["AAA", "BBB"]
    assert list(history.index) == [
        pd.Timestamp("2026-01-02T21:00:00Z"),
        pd.Timestamp("2026-01-05T21:00:00Z"),
    ]
    assert history.loc[pd.Timestamp("2026-01-02T21:00:00Z"), "AAA"] == 10.01
    assert history.loc[pd.Timestamp("2026-01-05T21:00:00Z"), "BBB"] == 20.02
    first, second = transport.calls
    assert first.endpoint == second.endpoint == "bars"
    assert first.params["symbols"] == "AAA,BBB"
    assert first.params["adjustment"] == "all"
    assert first.params["feed"] == "iex"
    assert first.params["currency"] == "USD"
    assert first.params["timeframe"] == "1Day"
    assert "page_token" not in first.params
    assert second.params["page_token"] == "page-two"


def test_history_can_fetch_calendar_before_bars_without_using_a_clock() -> None:
    calendar = [
        {"date": "2026-01-02", "open": "09:30", "close": "16:00"},
        {"date": "2026-01-05", "open": "09:30", "close": "16:00"},
    ]
    transport = FakeTransport(
        _response(calendar, request_id="calendar"),
        _response(
            {
                "bars": {"AAA": [_bar("02", "10"), _bar("05", "11")]},
                "next_page_token": None,
            },
            request_id="bars",
        ),
    )

    history = _provider(transport).history_through(
        datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC),
        observations=2,
    )

    assert len(history) == 2
    assert tuple(call.endpoint for call in transport.calls) == ("calendar", "bars")


def test_history_rejects_incomplete_current_duplicate_and_unexpected_panels() -> None:
    sessions = (_session(2), _session(5))
    with pytest.raises(AlpacaHistoryError, match="completed sessions"):
        _provider(FakeTransport()).history_through(
            sessions[-1].close_at,
            observations=2,
            sessions=sessions,
        )

    missing = _response(
        {"bars": {"AAA": [_bar("02", "10")]}, "next_page_token": None}
    )
    with pytest.raises(AlpacaHistoryError, match="complete aligned panel"):
        _provider(FakeTransport(missing)).history_through(
            datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC),
            observations=2,
            sessions=sessions,
        )

    duplicated = _response(
        {
            "bars": {"AAA": [_bar("02", "10"), _bar("02", "10")]},
            "next_page_token": None,
        }
    )
    with pytest.raises(AlpacaHistoryError, match="repeats"):
        _provider(FakeTransport(duplicated)).history_through(
            datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC),
            observations=2,
            sessions=sessions,
        )

    unexpected = _response(
        {"bars": {"BBB": [_bar("02", "10")]}, "next_page_token": None}
    )
    with pytest.raises(AlpacaHistoryError, match="unexpected symbol"):
        _provider(FakeTransport(unexpected)).history_through(
            datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC),
            observations=2,
            sessions=sessions,
        )

    duplicate_symbol = _response(
        {
            "bars": {"AAA": [], "aaa": []},
            "next_page_token": None,
        }
    )
    with pytest.raises(AlpacaHistoryError, match="duplicate canonical symbol"):
        _provider(FakeTransport(duplicate_symbol)).history_through(
            datetime(2026, 1, 5, 21, 0, 1, tzinfo=UTC),
            observations=2,
            sessions=sessions,
        )


def test_history_rejects_repeated_tokens_and_page_limit() -> None:
    sessions = (_session(2),)
    repeated = FakeTransport(
        _response({"bars": {}, "next_page_token": "same"}),
        _response({"bars": {}, "next_page_token": "same"}),
    )
    with pytest.raises(AlpacaHistoryError, match="repeated"):
        _provider(repeated).history_through(
            datetime(2026, 1, 2, 21, 0, 1, tzinfo=UTC),
            observations=1,
            sessions=sessions,
        )

    limited = FakeTransport(_response({"bars": {}, "next_page_token": "more"}))
    with pytest.raises(AlpacaHistoryError, match="max_pages"):
        _provider(limited, max_pages=1).history_through(
            datetime(2026, 1, 2, 21, 0, 1, tzinfo=UTC),
            observations=1,
            sessions=sessions,
        )


@pytest.mark.parametrize("observations", [0, -1, True, 1.5])
def test_history_requires_a_positive_integer_observation_count(observations: object) -> None:
    with pytest.raises(AlpacaHistoryError, match="positive integer"):
        _provider(FakeTransport()).history_through(
            datetime(2026, 1, 3, tzinfo=UTC),
            observations=observations,  # type: ignore[arg-type]
            sessions=(),
        )


def test_auth_and_other_client_errors_are_not_retried_or_leaked() -> None:
    for status, exception in ((401, AlpacaAuthenticationError), (403, AlpacaAuthenticationError), (400, AlpacaHttpError)):
        transport = FakeTransport(_response({}, status=status))
        with pytest.raises(exception) as raised:
            _provider(transport, max_retries=4).capture_execution_snapshot(
                captured_at=CAPTURED,
                session=SESSION,
            )
        assert len(transport.calls) == 1
        assert "key-id-value" not in str(raised.value)
        assert "secret-value" not in str(raised.value)


def test_transport_and_transient_http_failures_retry_with_bounded_delays() -> None:
    transport = FakeTransport(
        AlpacaTransportFailureError("socket closed"),
        _response({}, status=503),
        _response(_snapshot_payload()),
    )
    delays: list[float] = []
    provider = _provider(
        transport,
        sleeper=delays.append,
        jitter=lambda _maximum: 0.0,
        max_retries=2,
        retry_base_seconds=0.25,
        retry_max_seconds=1.0,
    )

    result = provider.capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)

    assert result.snapshot.quote("AAA").last == Decimal("10.0150")
    assert len(transport.calls) == 3
    assert delays == [0.25, 0.5]

    secret_failure = FakeTransport(AlpacaTransportFailureError("secret-value"))
    with pytest.raises(AlpacaTransportFailureError) as raised:
        _provider(secret_failure, max_retries=0).capture_execution_snapshot(
            captured_at=CAPTURED,
            session=SESSION,
        )
    assert "secret-value" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_rate_limit_is_bounded_and_honors_bounded_retry_after() -> None:
    transport = FakeTransport(
        _response({}, status=429, headers={"Retry-After": "60"}),
        _response({}, status=429),
    )
    delays: list[float] = []
    with pytest.raises(AlpacaRateLimitError, match="rate-limited"):
        _provider(
            transport,
            sleeper=delays.append,
            jitter=lambda _maximum: 0,
            max_retries=1,
            retry_max_seconds=2,
        ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)

    assert len(transport.calls) == 2
    assert delays == [2.0]


def test_malformed_oversized_or_unattributed_responses_fail_before_use() -> None:
    with pytest.raises(AlpacaSchemaError, match="valid finite UTF-8 JSON"):
        _provider(FakeTransport(_response(raw=b"{not-json"))).capture_execution_snapshot(
            captured_at=CAPTURED,
            session=SESSION,
        )
    with pytest.raises(AlpacaSchemaError, match="X-Request-ID"):
        _provider(
            FakeTransport(_response(_snapshot_payload(), request_id=None))
        ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)
    with pytest.raises(AlpacaResponseTooLargeError, match="max_response_bytes"):
        _provider(
            FakeTransport(_response(raw=b"{}")),
            max_response_bytes=1,
        ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)
    with pytest.raises(AlpacaSchemaError, match="finite UTF-8 JSON"):
        _provider(
            FakeTransport(_response(raw=b'{"AAA":NaN}'))
        ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)
    duplicate_key = (
        b'{"AAA":{"latestQuote":{}},"AAA":{"latestTrade":{}}}'
    )
    with pytest.raises(AlpacaSchemaError, match="valid finite UTF-8 JSON"):
        _provider(
            FakeTransport(_response(raw=duplicate_key))
        ).capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)


def test_source_timestamp_model_rejects_text_epoch_mismatch() -> None:
    with pytest.raises(ValueError, match="identify one instant"):
        SourceTimestamp(text="2026-01-02T15:00:00.000000001Z", epoch_ns=1)


def test_provider_context_manager_closes_transport_once_and_refuses_reuse() -> None:
    transport = FakeTransport(_response(_snapshot_payload()))
    provider = _provider(transport)

    with provider as entered:
        assert entered.capture_execution_snapshot(
            captured_at=CAPTURED,
            session=SESSION,
        ).request_id == "request-1"

    assert transport.close_count == 1
    provider.close()
    assert transport.close_count == 1
    with pytest.raises(AlpacaTransportFailureError, match="closed"):
        provider.capture_execution_snapshot(captured_at=CAPTURED, session=SESSION)


def test_history_uses_new_york_date_for_fetched_calendar_end() -> None:
    transport = FakeTransport(
        _response(
            [{"date": "2026-01-05", "open": "09:30", "close": "16:00"}],
            request_id="calendar",
        ),
        _response(
            {"bars": {"AAA": [_bar("05", "11")]}, "next_page_token": None},
            request_id="bars",
        ),
    )
    # Jan 6 in UTC, but still Jan 5 in New York.
    _provider(transport).history_through(
        datetime(2026, 1, 6, 0, 30, tzinfo=UTC),
        observations=1,
    )

    assert transport.calls[0].params["end"] == "2026-01-05"


def test_concrete_transport_has_only_fixed_get_endpoints_and_lazy_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert HttpxAlpacaTransport._url("snapshots", None) == (
        f"{DATA_BASE_URL}/v2/stocks/snapshots"
    )
    assert HttpxAlpacaTransport._url("bars", None) == f"{DATA_BASE_URL}/v2/stocks/bars"
    assert HttpxAlpacaTransport._url("calendar", None) == f"{PAPER_BASE_URL}/v2/calendar"
    assert HttpxAlpacaTransport._url("asset", "BRK.B") == (
        f"{PAPER_BASE_URL}/v2/assets/BRK.B"
    )
    with pytest.raises(AlpacaEndpointError, match="allow-list"):
        HttpxAlpacaTransport._url("orders", None)
    with pytest.raises(AlpacaEndpointError, match="allow-list"):
        HttpxAlpacaTransport._url("asset", "../orders")
    assert not hasattr(HttpxAlpacaTransport, "post")
    assert not hasattr(AlpacaDataProvider, "snapshot_at")
    assert not any(
        hasattr(AlpacaDataProvider, method)
        for method in ("submit_order", "place_order", "cancel_order", "post")
    )

    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "httpx":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(AlpacaDependencyError, match="alpaca-data extra"):
        HttpxAlpacaTransport()


def test_concrete_transport_bounds_content_length_and_streamed_body_before_allocation() -> None:
    class RequestError(Exception):
        pass

    class Response:
        def __init__(self, chunks: tuple[bytes, ...], headers: Mapping[str, str]) -> None:
            self.status_code = 200
            self.headers = headers
            self._chunks = chunks
            self.iterated = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def iter_bytes(self):  # type: ignore[no-untyped-def]
            self.iterated = True
            yield from self._chunks

    class Client:
        def __init__(self, *responses: Response) -> None:
            self.responses = list(responses)
            self.calls: list[tuple[object, ...]] = []

        def stream(self, *args: object, **kwargs: object) -> Response:
            self.calls.append((*args, kwargs))
            return self.responses.pop(0)

    declared = Response((b"not-read",), {"Content-Length": "4"})
    chunked = Response((b"ab", b"cd"), {})
    complete = Response((b"{}",), {"X-Request-ID": "safe"})
    client = Client(declared, chunked, complete)
    transport = object.__new__(HttpxAlpacaTransport)
    transport._httpx = type("Httpx", (), {"RequestError": RequestError})  # type: ignore[attr-defined]
    transport._client = client  # type: ignore[attr-defined]

    with pytest.raises(AlpacaResponseTooLargeError):
        transport.get(
            "snapshots",
            symbol=None,
            params={},
            headers={},
            timeout_seconds=1,
            max_response_bytes=3,
        )
    assert not declared.iterated
    with pytest.raises(AlpacaResponseTooLargeError):
        transport.get(
            "bars",
            symbol=None,
            params={},
            headers={},
            timeout_seconds=1,
            max_response_bytes=3,
        )
    assert chunked.iterated
    response = transport.get(
        "calendar",
        symbol=None,
        params={},
        headers={},
        timeout_seconds=1,
        max_response_bytes=3,
    )
    assert response.content == b"{}"
    assert client.calls[0][0:2] == ("GET", f"{DATA_BASE_URL}/v2/stocks/snapshots")
