"""Strict, read-only Alpaca market-data boundary for live shadow experiments.

This module deliberately does not implement the exact-time ``snapshot_at``
contract used by local replay data.  Alpaca's latest endpoints return an
observation assembled from independently timestamped quote and trade events;
``ObservedMarketSnapshot`` preserves that provenance before exposing the
normalized :class:`paper.models.MarketSnapshot` used by local paper execution.

There is no order, account, position, POST, PATCH, or DELETE capability here.
The optional concrete transport maps four named read operations to fixed HTTPS
hosts and paths.  Tests can inject the small transport protocol and never need
network access or credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Literal, Protocol, Self, runtime_checkable
from urllib.parse import quote as url_quote
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field, SecretStr, field_validator, model_validator

from paper.models import MarketSnapshot, PaperModel, Quote, StrictDecimal, UtcDatetime

type AlpacaEndpoint = Literal["snapshots", "bars", "asset", "calendar"]
type AlpacaFeed = Literal["iex"]

DATA_BASE_URL = "https://data.alpaca.markets"
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
KEY_ID_ENV = "TRUFFLE_ALPACA_KEY_ID"
SECRET_KEY_ENV = "TRUFFLE_ALPACA_SECRET_KEY"

_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,31}$")
_SOURCE_TIME_PATTERN = re.compile(
    r"^(?P<day>\d{4}-\d{2}-\d{2})T"
    r"(?P<clock>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,9})?"
    r"(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NEW_YORK = ZoneInfo("America/New_York")
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AlpacaDataError(RuntimeError):
    """Base class for a read-only Alpaca data failure."""


class AlpacaConfigurationError(AlpacaDataError, ValueError):
    """Raised when the bounded adapter configuration is invalid."""


class AlpacaCredentialError(AlpacaDataError, ValueError):
    """Raised when explicit data credentials are absent or blank."""


class AlpacaDependencyError(AlpacaDataError, ImportError):
    """Raised when the optional concrete HTTP dependency is unavailable."""


class AlpacaEndpointError(AlpacaDataError, ValueError):
    """Raised when a transport request is outside the fixed read allow-list."""


class AlpacaTransportFailureError(AlpacaDataError):
    """Raised by a transport for a timeout or connection-level failure."""


class AlpacaAuthenticationError(AlpacaDataError):
    """Raised for HTTP 401 or 403 without retrying."""


class AlpacaRateLimitError(AlpacaDataError):
    """Raised when bounded retries cannot clear an HTTP 429."""


class AlpacaHttpError(AlpacaDataError):
    """Raised for a non-successful response that is not retried further."""


class AlpacaResponseTooLargeError(AlpacaDataError):
    """Raised before parsing a response that exceeds the configured bound."""


class AlpacaSchemaError(AlpacaDataError):
    """Raised when an authenticated response has malformed or unsafe data."""


class AlpacaSessionError(AlpacaDataError):
    """Raised when data cannot be tied to an official regular-hours session."""


class AlpacaSnapshotError(AlpacaDataError):
    """Raised when a latest snapshot is incomplete or internally invalid."""


class AlpacaStaleDataError(AlpacaSnapshotError):
    """Raised when a quote or trade is stale or materially future-dated."""


class AlpacaAssetEligibilityError(AlpacaDataError):
    """Raised when an asset is not an active, tradable, fractionable US equity."""


class AlpacaHistoryError(AlpacaDataError):
    """Raised when adjusted daily history is incomplete or temporally unsafe."""


def _secret_text(value: str | SecretStr, *, field: str) -> SecretStr:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    if not isinstance(raw, str) or not raw.strip():
        raise AlpacaCredentialError(f"{field} must be a non-empty string")
    return SecretStr(raw.strip())


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Explicit credentials whose representation never exposes either value."""

    key_id: SecretStr
    secret_key: SecretStr

    def __init__(self, *, key_id: str | SecretStr, secret_key: str | SecretStr) -> None:
        object.__setattr__(self, "key_id", _secret_text(key_id, field="key_id"))
        object.__setattr__(
            self,
            "secret_key",
            _secret_text(secret_key, field="secret_key"),
        )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str],
        *,
        key_id_name: str = KEY_ID_ENV,
        secret_key_name: str = SECRET_KEY_ENV,
    ) -> Self:
        """Load from an injected mapping; this method never reads process state."""

        try:
            key_id = environ[key_id_name]
        except KeyError:
            raise AlpacaCredentialError(
                f"missing required credential variable {key_id_name!r}"
            ) from None
        try:
            secret_key = environ[secret_key_name]
        except KeyError:
            raise AlpacaCredentialError(
                f"missing required credential variable {secret_key_name!r}"
            ) from None
        return cls(key_id=key_id, secret_key=secret_key)


def _canonical_symbol(value: object) -> str:
    if not isinstance(value, str):
        raise AlpacaConfigurationError("symbols must be strings")
    symbol = value.strip().upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise AlpacaConfigurationError(
            "symbols must be 1-32 character US-equity identifiers containing "
            "only letters, digits, dots, or hyphens"
        )
    return symbol


def _positive_finite_number(value: object, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AlpacaConfigurationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise AlpacaConfigurationError(f"{field} must be a finite {comparator} number")
    return number


def _bounded_integer(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AlpacaConfigurationError(f"{field} must be an integer")
    integer = int(value)
    if not minimum <= integer <= maximum:
        raise AlpacaConfigurationError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return integer


@dataclass(frozen=True, slots=True)
class AlpacaDataConfig:
    """All operational limits for the first IEX-only data adapter."""

    symbols: tuple[str, ...]
    feed: AlpacaFeed = "iex"
    max_quote_age_seconds: float = 10.0
    max_trade_age_seconds: float = 60.0
    max_future_skew_seconds: float = 2.0
    request_timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_base_seconds: float = 0.25
    retry_max_seconds: float = 5.0
    retry_jitter_seconds: float = 0.10
    max_response_bytes: int = 2_000_000
    max_pages: int = 100

    def __post_init__(self) -> None:
        if self.feed != "iex":
            raise AlpacaConfigurationError("the initial Alpaca adapter supports only feed='iex'")
        if isinstance(self.symbols, (str, bytes)):
            raise AlpacaConfigurationError("symbols must be a collection, not one string")
        raw_symbols = tuple(self.symbols)
        if not 1 <= len(raw_symbols) <= 30:
            raise AlpacaConfigurationError("Alpaca IEX configuration requires 1-30 symbols")
        symbols = tuple(sorted(_canonical_symbol(symbol) for symbol in raw_symbols))
        if len(symbols) != len(set(symbols)):
            raise AlpacaConfigurationError("symbols collide after canonicalization")
        object.__setattr__(self, "symbols", symbols)

        for field in (
            "max_quote_age_seconds",
            "max_trade_age_seconds",
            "request_timeout_seconds",
            "retry_base_seconds",
            "retry_max_seconds",
        ):
            object.__setattr__(
                self,
                field,
                _positive_finite_number(getattr(self, field), field=field),
            )
        for field in ("max_future_skew_seconds", "retry_jitter_seconds"):
            object.__setattr__(
                self,
                field,
                _positive_finite_number(
                    getattr(self, field),
                    field=field,
                    allow_zero=True,
                ),
            )
        object.__setattr__(
            self,
            "max_retries",
            _bounded_integer(self.max_retries, field="max_retries", minimum=0, maximum=10),
        )
        object.__setattr__(
            self,
            "max_response_bytes",
            _bounded_integer(
                self.max_response_bytes,
                field="max_response_bytes",
                minimum=1,
                maximum=50_000_000,
            ),
        )
        object.__setattr__(
            self,
            "max_pages",
            _bounded_integer(self.max_pages, field="max_pages", minimum=1, maximum=10_000),
        )
        if self.retry_base_seconds > self.retry_max_seconds:
            raise AlpacaConfigurationError(
                "retry_base_seconds must not exceed retry_max_seconds"
            )


@dataclass(frozen=True, slots=True)
class AlpacaHttpResponse:
    """Transport-neutral response used by deterministic fakes."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes


@runtime_checkable
class AlpacaHttpTransport(Protocol):
    """Capability transport: four named GETs, never a caller-supplied URL."""

    def get(
        self,
        endpoint: AlpacaEndpoint,
        *,
        symbol: str | None,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> AlpacaHttpResponse: ...


class HttpxAlpacaTransport:
    """Optional HTTPS transport with a closed host/path/method allow-list."""

    def __init__(self) -> None:
        try:
            import httpx
        except ImportError:
            raise AlpacaDependencyError(
                "live Alpaca REST access requires the optional HTTP client; "
                "install Truffle's alpaca-data extra or `pip install httpx>=0.28,<1`"
            ) from None
        self._httpx = httpx
        self._client = httpx.Client(
            verify=True,
            follow_redirects=False,
            trust_env=False,
        )

    @staticmethod
    def _url(endpoint: object, symbol: str | None) -> str:
        if endpoint == "snapshots" and symbol is None:
            return f"{DATA_BASE_URL}/v2/stocks/snapshots"
        if endpoint == "bars" and symbol is None:
            return f"{DATA_BASE_URL}/v2/stocks/bars"
        if endpoint == "calendar" and symbol is None:
            return f"{PAPER_BASE_URL}/v2/calendar"
        if endpoint == "asset" and symbol is not None:
            try:
                canonical = _canonical_symbol(symbol)
            except AlpacaConfigurationError as exc:
                raise AlpacaEndpointError(
                    "asset symbol is outside the fixed Alpaca GET allow-list"
                ) from exc
            encoded = url_quote(canonical, safe="")
            return f"{PAPER_BASE_URL}/v2/assets/{encoded}"
        raise AlpacaEndpointError("request is outside the fixed Alpaca GET allow-list")

    def get(
        self,
        endpoint: AlpacaEndpoint,
        *,
        symbol: str | None,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> AlpacaHttpResponse:
        url = self._url(endpoint, symbol)
        try:
            with self._client.stream(
                "GET",
                url,
                params=dict(params),
                headers=dict(headers),
                timeout=timeout_seconds,
            ) as response:
                raw_length = response.headers.get("Content-Length")
                if raw_length is not None:
                    try:
                        content_length = int(raw_length)
                    except ValueError:
                        raise AlpacaSchemaError(
                            "Alpaca Content-Length header must be an integer"
                        ) from None
                    if content_length < 0:
                        raise AlpacaSchemaError(
                            "Alpaca Content-Length header must not be negative"
                        )
                    if content_length > max_response_bytes:
                        raise AlpacaResponseTooLargeError(
                            f"Alpaca {endpoint} response exceeds max_response_bytes"
                        )
                content = bytearray()
                for chunk in response.iter_bytes():
                    if len(content) + len(chunk) > max_response_bytes:
                        raise AlpacaResponseTooLargeError(
                            f"Alpaca {endpoint} response exceeds max_response_bytes"
                        )
                    content.extend(chunk)
                status_code = response.status_code
                response_headers = dict(response.headers)
        except self._httpx.RequestError:
            raise AlpacaTransportFailureError(
                f"Alpaca {endpoint} GET failed before a response was received"
            ) from None
        return AlpacaHttpResponse(
            status_code=status_code,
            headers=response_headers,
            content=bytes(content),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.close()


class SourceTimestamp(PaperModel):
    """Canonical UTC source timestamp retaining Alpaca's nanosecond precision."""

    text: str
    epoch_ns: int

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        canonical, epoch_ns = _parse_source_timestamp(self.text, label="source")
        if self.text != canonical or self.epoch_ns != epoch_ns:
            raise ValueError("source timestamp text and epoch_ns must identify one instant")
        return self


class MarketSession(PaperModel):
    """One official regular-hours session, including early closes."""

    session_date: date
    open_at: UtcDatetime
    close_at: UtcDatetime

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.open_at >= self.close_at:
            raise ValueError("market session open must precede close")
        if (
            self.open_at.astimezone(_NEW_YORK).date() != self.session_date
            or self.close_at.astimezone(_NEW_YORK).date() != self.session_date
        ):
            raise ValueError("market session instants must fall on session_date in New York")
        return self


class ObservedSymbolMarketData(PaperModel):
    """Raw quote/trade evidence retained alongside a normalized paper quote."""

    ticker: str
    bid: StrictDecimal
    ask: StrictDecimal
    last: StrictDecimal
    quote_timestamp: SourceTimestamp
    trade_timestamp: SourceTimestamp
    bid_exchange: str
    ask_exchange: str
    trade_exchange: str
    quote_conditions: tuple[str, ...] = ()
    trade_conditions: tuple[str, ...] = ()
    trade_id: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, value: object) -> str:
        return _canonical_symbol(value)

    @field_validator("bid_exchange", "ask_exchange", "trade_exchange", mode="before")
    @classmethod
    def _exchange(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("exchange codes must be non-empty strings")
        return value.strip()

    @model_validator(mode="after")
    def _valid_market(self) -> Self:
        if self.bid <= Decimal("0") or self.ask <= Decimal("0") or self.last <= Decimal("0"):
            raise ValueError("bid, ask, and last prices must be positive")
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")
        return self


class ObservedMarketSnapshot(PaperModel):
    """Complete Alpaca capture with source-time evidence and normalized snapshot."""

    capture_id: str
    feed: Literal["iex"] = "iex"
    captured_at: UtcDatetime
    request_id: str
    session: MarketSession
    symbols: tuple[ObservedSymbolMarketData, ...] = Field(min_length=1, max_length=30)
    snapshot: MarketSnapshot

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.snapshot.snapshot_id != self.capture_id:
            raise ValueError("normalized snapshot ID must equal capture ID")
        evidence = tuple(item.ticker for item in self.symbols)
        normalized = tuple(quote.ticker for quote in self.snapshot.quotes)
        if evidence != normalized:
            raise ValueError("normalized snapshot symbols must match source evidence")
        expected_provenance = _canonical_source_provenance(
            captured_at=self.captured_at,
            request_id=self.request_id,
            session=self.session,
            symbols=self.symbols,
        )
        if self.snapshot.source_provenance_json != expected_provenance:
            raise ValueError(
                "normalized snapshot must retain the complete source provenance"
            )
        return self


class AlpacaAssetMetadata(PaperModel):
    """Eligibility fields needed by an exact-$100 fractional experiment."""

    symbol: str
    asset_class: str
    status: str
    tradable: bool
    fractionable: bool
    request_id: str

    @field_validator("symbol", mode="before")
    @classmethod
    def _symbol(cls, value: object) -> str:
        return _canonical_symbol(value)


def _utc(value: datetime, *, label: str, error_type: type[AlpacaDataError]) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise error_type(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _datetime_epoch_ns(value: datetime) -> int:
    utc = value.astimezone(UTC)
    delta = utc - _EPOCH
    total_microseconds = (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    )
    return total_microseconds * 1_000


def _datetime_from_epoch_ns_floor(epoch_ns: int) -> datetime:
    seconds, nanoseconds = divmod(epoch_ns, 1_000_000_000)
    microseconds = nanoseconds // 1_000
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=microseconds)


def _parse_source_timestamp(value: object, *, label: str) -> tuple[str, int]:
    if not isinstance(value, str):
        raise AlpacaSchemaError(f"{label} timestamp must be an RFC-3339 string")
    match = _SOURCE_TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        raise AlpacaSchemaError(
            f"{label} timestamp must be RFC-3339 with no more than 9 fractional digits"
        )
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    try:
        whole = datetime.fromisoformat(f"{match.group('day')}T{match.group('clock')}{zone}")
    except ValueError:
        raise AlpacaSchemaError(f"{label} timestamp is not a valid instant") from None
    whole_utc = whole.astimezone(UTC)
    delta = whole_utc - _EPOCH
    seconds = delta.days * 86_400 + delta.seconds
    fractional = (match.group("fraction") or ".")[1:]
    nanoseconds = int(fractional.ljust(9, "0")) if fractional else 0
    epoch_ns = seconds * 1_000_000_000 + nanoseconds
    canonical_seconds = datetime.fromtimestamp(seconds, tz=UTC)
    text = f"{canonical_seconds:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"
    return text, epoch_ns


def _source_timestamp(value: object, *, label: str) -> SourceTimestamp:
    text, epoch_ns = _parse_source_timestamp(value, label=label)
    try:
        return SourceTimestamp(text=text, epoch_ns=epoch_ns)
    except ValueError as exc:
        raise AlpacaSchemaError(f"{label} timestamp is internally inconsistent") from exc


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise AlpacaSchemaError(f"{label} must not pass through a binary float")
    if not isinstance(value, (Decimal, Integral, str)):
        raise AlpacaSchemaError(f"{label} must be a decimal JSON number or string")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise AlpacaSchemaError(f"{label} is not a valid decimal") from None
    if not result.is_finite():
        raise AlpacaSchemaError(f"{label} must be finite")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AlpacaSchemaError(f"{label} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, label: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        raise AlpacaSchemaError(f"{label} is missing required field {key!r}") from None


def _conditions(value: object, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AlpacaSchemaError(f"{label} conditions must be a JSON string array")
    return tuple(value)


def _request_id(headers: Mapping[str, str]) -> str:
    for key, value in headers.items():
        if key.lower() == "x-request-id" and isinstance(value, str) and value.strip():
            return value.strip()
    raise AlpacaSchemaError("Alpaca response is missing the required X-Request-ID header")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_json(content: bytes) -> Any:
    try:
        text = content.decode("utf-8")
        return json.loads(
            text,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise AlpacaSchemaError("Alpaca response is not valid finite UTF-8 JSON") from None


def _canonical_source_provenance(
    *,
    captured_at: datetime,
    request_id: str,
    session: MarketSession,
    symbols: tuple[ObservedSymbolMarketData, ...],
) -> str:
    payload = {
        "captured_at": captured_at.isoformat(timespec="microseconds"),
        "feed": "iex",
        "provider": "alpaca",
        "request_id": request_id,
        "session": session.model_dump(mode="json"),
        "symbols": [item.model_dump(mode="json") for item in symbols],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _canonical_capture_id(source_provenance_json: str) -> str:
    return f"alpaca:iex:{hashlib.sha256(source_provenance_json.encode()).hexdigest()}"


def _session_instant(
    session_date: date,
    raw: object,
    *,
    label: str,
) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise AlpacaSchemaError(f"calendar {label} must be a time string")
    value = raw.strip()
    if "T" in value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise AlpacaSchemaError(f"calendar {label} is invalid") from None
        return _utc(parsed, label=f"calendar {label}", error_type=AlpacaSchemaError)
    try:
        parsed_time = time.fromisoformat(value)
    except ValueError:
        raise AlpacaSchemaError(f"calendar {label} is invalid") from None
    if parsed_time.tzinfo is not None:
        combined = datetime.combine(session_date, parsed_time)
    else:
        combined = datetime.combine(session_date, parsed_time, tzinfo=_NEW_YORK)
    return combined.astimezone(UTC)


class AlpacaDataProvider:
    """Read-only IEX data provider for shadow capture and adjusted analytics."""

    def __init__(
        self,
        config: AlpacaDataConfig,
        credentials: AlpacaCredentials,
        *,
        transport: AlpacaHttpTransport | None = None,
        sleeper: Callable[[float], None] | None = None,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        self._config = config
        self._credentials = credentials
        self._transport = transport if transport is not None else HttpxAlpacaTransport()
        self._sleeper = sleeper if sleeper is not None else __import__("time").sleep
        self._jitter = (
            jitter if jitter is not None else lambda maximum: random.SystemRandom().uniform(0, maximum)
        )
        self._closed = False

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._config.symbols

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "APCA-API-KEY-ID": self._credentials.key_id.get_secret_value(),
            "APCA-API-SECRET-KEY": self._credentials.secret_key.get_secret_value(),
        }

    def _retry_delay(self, attempt: int, response: AlpacaHttpResponse | None) -> float:
        exponential = min(
            self._config.retry_base_seconds * (2**attempt),
            self._config.retry_max_seconds,
        )
        retry_after = 0.0
        if response is not None:
            for key, raw in response.headers.items():
                if key.lower() == "retry-after":
                    try:
                        retry_after = max(0.0, float(raw))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                    break
        jitter = self._jitter(self._config.retry_jitter_seconds)
        if not isinstance(jitter, Real) or isinstance(jitter, bool):
            raise AlpacaConfigurationError("injected retry jitter must return a number")
        jitter_number = float(jitter)
        if not math.isfinite(jitter_number) or not 0 <= jitter_number <= self._config.retry_jitter_seconds:
            raise AlpacaConfigurationError(
                "injected retry jitter must be finite and inside the configured bound"
            )
        return min(max(exponential, retry_after) + jitter_number, self._config.retry_max_seconds)

    def _request_json(
        self,
        endpoint: AlpacaEndpoint,
        *,
        symbol: str | None = None,
        params: Mapping[str, str] | None = None,
    ) -> tuple[Any, str]:
        if self._closed:
            raise AlpacaTransportFailureError("Alpaca data provider is closed")
        response: AlpacaHttpResponse | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._transport.get(
                    endpoint,
                    symbol=symbol,
                    params={} if params is None else dict(params),
                    headers=self._headers(),
                    timeout_seconds=self._config.request_timeout_seconds,
                    max_response_bytes=self._config.max_response_bytes,
                )
            except AlpacaTransportFailureError:
                if attempt == self._config.max_retries:
                    raise AlpacaTransportFailureError(
                        f"Alpaca {endpoint} GET failed after {attempt + 1} attempts"
                    ) from None
                self._sleeper(self._retry_delay(attempt, None))
                continue

            if not isinstance(response.content, bytes):
                raise AlpacaSchemaError("transport response content must be bytes")
            if len(response.content) > self._config.max_response_bytes:
                raise AlpacaResponseTooLargeError(
                    f"Alpaca {endpoint} response exceeds max_response_bytes"
                )
            if response.status_code in (401, 403):
                raise AlpacaAuthenticationError(
                    f"Alpaca {endpoint} authentication or entitlement failed "
                    f"with HTTP {response.status_code}"
                )
            if response.status_code in _TRANSIENT_STATUS_CODES:
                if attempt < self._config.max_retries:
                    self._sleeper(self._retry_delay(attempt, response))
                    continue
                if response.status_code == 429:
                    raise AlpacaRateLimitError(
                        f"Alpaca {endpoint} remained rate-limited after {attempt + 1} attempts"
                    )
                raise AlpacaHttpError(
                    f"Alpaca {endpoint} remained unavailable with HTTP "
                    f"{response.status_code} after {attempt + 1} attempts"
                )
            if response.status_code != 200:
                raise AlpacaHttpError(
                    f"Alpaca {endpoint} GET returned non-retryable HTTP "
                    f"{response.status_code}"
                )
            request_id = _request_id(response.headers)
            return _parse_json(response.content), request_id

        raise AssertionError("bounded retry loop exited unexpectedly")

    def close(self) -> None:
        """Close an owned or injected transport once; no network action is implied."""

        if self._closed:
            return
        self._closed = True
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Self:
        if self._closed:
            raise AlpacaTransportFailureError("Alpaca data provider is closed")
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def asset_metadata(self) -> tuple[AlpacaAssetMetadata, ...]:
        """Fetch and require paper-eligible fractional US equities."""

        assets: list[AlpacaAssetMetadata] = []
        for symbol in self.symbols:
            payload, request_id = self._request_json("asset", symbol=symbol)
            raw = _mapping(payload, label=f"asset {symbol}")
            wire_symbol = _required(raw, "symbol", label=f"asset {symbol}")
            asset_class = _required(raw, "class", label=f"asset {symbol}")
            status = _required(raw, "status", label=f"asset {symbol}")
            tradable = _required(raw, "tradable", label=f"asset {symbol}")
            fractionable = _required(raw, "fractionable", label=f"asset {symbol}")
            if not isinstance(asset_class, str) or not isinstance(status, str):
                raise AlpacaSchemaError("asset class and status must be strings")
            if not isinstance(tradable, bool) or not isinstance(fractionable, bool):
                raise AlpacaSchemaError("asset tradable and fractionable must be booleans")
            try:
                metadata = AlpacaAssetMetadata(
                    symbol=wire_symbol,
                    asset_class=asset_class,
                    status=status,
                    tradable=tradable,
                    fractionable=fractionable,
                    request_id=request_id,
                )
            except ValueError as exc:
                raise AlpacaSchemaError(f"asset response for {symbol} is invalid") from exc
            if metadata.symbol != symbol:
                raise AlpacaSchemaError(
                    f"asset endpoint returned {metadata.symbol} while {symbol} was requested"
                )
            if (
                metadata.asset_class != "us_equity"
                or metadata.status != "active"
                or not metadata.tradable
                or not metadata.fractionable
            ):
                raise AlpacaAssetEligibilityError(
                    f"{symbol} must be an active, tradable, fractionable US equity"
                )
            assets.append(metadata)
        return tuple(assets)

    def fetch_market_sessions(self, start: date, end: date) -> tuple[MarketSession, ...]:
        """Fetch the official regular-hours calendar, including early closes."""

        if not isinstance(start, date) or not isinstance(end, date) or start > end:
            raise AlpacaSessionError("calendar start and end must be ordered dates")
        payload, _request_id_value = self._request_json(
            "calendar",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "date_type": "TRADING",
            },
        )
        if not isinstance(payload, list):
            raise AlpacaSchemaError("calendar response must be a JSON array")
        sessions: list[MarketSession] = []
        for index, item in enumerate(payload):
            raw = _mapping(item, label=f"calendar row {index}")
            raw_date = _required(raw, "date", label=f"calendar row {index}")
            if not isinstance(raw_date, str):
                raise AlpacaSchemaError("calendar date must be an ISO date string")
            try:
                session_date = date.fromisoformat(raw_date)
            except ValueError:
                raise AlpacaSchemaError("calendar date must be an ISO date string") from None
            try:
                session = MarketSession(
                    session_date=session_date,
                    open_at=_session_instant(
                        session_date,
                        _required(raw, "open", label=f"calendar row {index}"),
                        label="open",
                    ),
                    close_at=_session_instant(
                        session_date,
                        _required(raw, "close", label=f"calendar row {index}"),
                        label="close",
                    ),
                )
            except ValueError as exc:
                raise AlpacaSchemaError(f"calendar row {index} is invalid") from exc
            sessions.append(session)
        ordered = tuple(sorted(sessions, key=lambda session: session.open_at))
        dates = [session.session_date for session in ordered]
        if len(dates) != len(set(dates)):
            raise AlpacaSchemaError("calendar response contains duplicate session dates")
        if any(not start <= session.session_date <= end for session in ordered):
            raise AlpacaSchemaError("calendar response contains a date outside the request")
        return ordered

    def _observed_symbol(
        self,
        symbol: str,
        raw_snapshot: object,
    ) -> ObservedSymbolMarketData:
        snapshot = _mapping(raw_snapshot, label=f"snapshot {symbol}")
        quote = _mapping(
            _required(snapshot, "latestQuote", label=f"snapshot {symbol}"),
            label=f"latest quote {symbol}",
        )
        trade = _mapping(
            _required(snapshot, "latestTrade", label=f"snapshot {symbol}"),
            label=f"latest trade {symbol}",
        )
        trade_id_value = trade.get("i")
        if trade_id_value is not None and isinstance(trade_id_value, (dict, list, bool)):
            raise AlpacaSchemaError(f"latest trade {symbol} ID has an invalid type")
        try:
            return ObservedSymbolMarketData(
                ticker=symbol,
                bid=_decimal(_required(quote, "bp", label=f"latest quote {symbol}"), label="bid"),
                ask=_decimal(_required(quote, "ap", label=f"latest quote {symbol}"), label="ask"),
                last=_decimal(
                    _required(trade, "p", label=f"latest trade {symbol}"),
                    label="last trade",
                ),
                quote_timestamp=_source_timestamp(
                    _required(quote, "t", label=f"latest quote {symbol}"),
                    label=f"{symbol} quote",
                ),
                trade_timestamp=_source_timestamp(
                    _required(trade, "t", label=f"latest trade {symbol}"),
                    label=f"{symbol} trade",
                ),
                bid_exchange=_required(quote, "bx", label=f"latest quote {symbol}"),
                ask_exchange=_required(quote, "ax", label=f"latest quote {symbol}"),
                trade_exchange=_required(trade, "x", label=f"latest trade {symbol}"),
                quote_conditions=_conditions(
                    quote.get("c"),
                    label=f"latest quote {symbol}",
                ),
                trade_conditions=_conditions(
                    trade.get("c"),
                    label=f"latest trade {symbol}",
                ),
                trade_id=None if trade_id_value is None else str(trade_id_value),
            )
        except ValueError as exc:
            raise AlpacaSnapshotError(f"snapshot prices for {symbol} are invalid") from exc

    def capture_execution_snapshot(
        self,
        *,
        captured_at: datetime,
        session: MarketSession,
    ) -> ObservedMarketSnapshot:
        """Capture one complete, fresh, regular-hours raw quote/trade snapshot."""

        captured = _utc(
            captured_at,
            label="captured_at",
            error_type=AlpacaSessionError,
        )
        if not session.open_at <= captured < session.close_at:
            raise AlpacaSessionError(
                "execution capture must occur inside the supplied regular-hours session"
            )
        payload, request_id = self._request_json(
            "snapshots",
            params={
                "symbols": ",".join(self.symbols),
                "feed": self._config.feed,
                "currency": "USD",
            },
        )
        root = _mapping(payload, label="snapshot response")
        raw_snapshots: Mapping[str, Any]
        if set(root) == {"snapshots"}:
            raw_snapshots = _mapping(root["snapshots"], label="snapshot response snapshots")
        else:
            raw_snapshots = root

        returned_symbols: dict[str, Any] = {}
        for raw_symbol, value in raw_snapshots.items():
            try:
                symbol = _canonical_symbol(raw_symbol)
            except AlpacaConfigurationError as exc:
                raise AlpacaSchemaError("snapshot response contains an invalid symbol") from exc
            if symbol in returned_symbols:
                raise AlpacaSnapshotError(
                    f"snapshot response contains duplicate canonical symbol {symbol}"
                )
            returned_symbols[symbol] = value
        expected = set(self.symbols)
        returned = set(returned_symbols)
        if returned != expected:
            missing = sorted(expected - returned)
            unexpected = sorted(returned - expected)
            raise AlpacaSnapshotError(
                f"snapshot symbol set is incomplete or unexpected; "
                f"missing={missing}, unexpected={unexpected}"
            )

        observed = tuple(
            self._observed_symbol(symbol, returned_symbols[symbol])
            for symbol in self.symbols
        )
        captured_ns = _datetime_epoch_ns(captured)
        open_ns = _datetime_epoch_ns(session.open_at)
        close_ns = _datetime_epoch_ns(session.close_at)
        future_skew_ns = int(self._config.max_future_skew_seconds * 1_000_000_000)
        quote_age_ns = int(self._config.max_quote_age_seconds * 1_000_000_000)
        trade_age_ns = int(self._config.max_trade_age_seconds * 1_000_000_000)
        source_times: list[int] = []
        for item in observed:
            for kind, source, max_age in (
                ("quote", item.quote_timestamp, quote_age_ns),
                ("trade", item.trade_timestamp, trade_age_ns),
            ):
                source_times.append(source.epoch_ns)
                if not open_ns <= source.epoch_ns <= close_ns:
                    raise AlpacaSessionError(
                        f"{item.ticker} {kind} is outside the supplied market session"
                    )
                if source.epoch_ns > captured_ns + future_skew_ns:
                    raise AlpacaStaleDataError(
                        f"{item.ticker} {kind} exceeds max_future_skew_seconds"
                    )
                if captured_ns - source.epoch_ns > max_age:
                    raise AlpacaStaleDataError(
                        f"{item.ticker} {kind} exceeds its configured freshness limit"
                    )

        # The normalized paper model has one timestamp.  Use the oldest
        # required source so the downstream age gate cannot mistake an old
        # trade for a fresh response receipt.  Flooring sub-microsecond data is
        # conservative as well: the normalized timestamp is never newer than
        # the actual source event.
        effective_ns = min(source_times)
        effective_at = _datetime_from_epoch_ns_floor(effective_ns)
        source_provenance_json = _canonical_source_provenance(
            captured_at=captured,
            request_id=request_id,
            session=session,
            symbols=observed,
        )
        capture_id = _canonical_capture_id(source_provenance_json)
        normalized = MarketSnapshot(
            snapshot_id=capture_id,
            as_of=effective_at,
            quotes=tuple(
                Quote(
                    ticker=item.ticker,
                    bid=item.bid,
                    ask=item.ask,
                    last=item.last,
                    as_of=effective_at,
                )
                for item in observed
            ),
            source_provenance_json=source_provenance_json,
        )
        return ObservedMarketSnapshot(
            capture_id=capture_id,
            captured_at=captured,
            request_id=request_id,
            session=session,
            symbols=observed,
            snapshot=normalized,
        )

    @staticmethod
    def _materialize_sessions(
        sessions: Iterable[MarketSession],
    ) -> tuple[MarketSession, ...]:
        supplied = tuple(sessions)
        if any(not isinstance(session, MarketSession) for session in supplied):
            raise AlpacaSessionError("sessions must contain MarketSession objects")
        materialized = tuple(sorted(supplied, key=lambda session: session.open_at))
        dates = [session.session_date for session in materialized]
        if len(dates) != len(set(dates)):
            raise AlpacaSessionError("sessions contain duplicate dates")
        return materialized

    def history_through(
        self,
        as_of: datetime,
        *,
        observations: int,
        sessions: Iterable[MarketSession] | None = None,
    ) -> pd.DataFrame:
        """Return complete adjusted daily closes through completed sessions only.

        JSON prices remain ``Decimal`` until this final solver-facing DataFrame
        boundary, where validated values are deliberately converted to float.
        """

        if isinstance(observations, bool) or not isinstance(observations, Integral):
            raise AlpacaHistoryError("observations must be a positive integer")
        count = int(observations)
        if count < 1:
            raise AlpacaHistoryError("observations must be a positive integer")
        endpoint = _utc(as_of, label="as_of", error_type=AlpacaHistoryError)
        if sessions is None:
            local_end = endpoint.astimezone(_NEW_YORK).date()
            calendar_start = local_end - timedelta(days=max(30, count * 3))
            available_sessions = self.fetch_market_sessions(calendar_start, local_end)
        else:
            available_sessions = self._materialize_sessions(sessions)
        completed = tuple(session for session in available_sessions if session.close_at < endpoint)
        if len(completed) < count:
            raise AlpacaHistoryError(
                f"only {len(completed)} completed sessions are available; {count} required"
            )
        selected = completed[-count:]
        selected_by_date = {session.session_date: session for session in selected}

        common_params = {
            "symbols": ",".join(self.symbols),
            "timeframe": "1Day",
            "start": selected[0].session_date.isoformat(),
            "end": selected[-1].close_at.isoformat(),
            "adjustment": "all",
            "feed": self._config.feed,
            "currency": "USD",
            "sort": "asc",
            "limit": "10000",
            "asof": selected[-1].session_date.isoformat(),
        }
        by_symbol: dict[str, dict[date, Decimal]] = {symbol: {} for symbol in self.symbols}
        seen_tokens: set[str] = set()
        page_token: str | None = None
        finished = False
        for _page_number in range(self._config.max_pages):
            params = dict(common_params)
            if page_token is not None:
                params["page_token"] = page_token
            payload, _request_id_value = self._request_json("bars", params=params)
            root = _mapping(payload, label="bars response")
            bars = _mapping(_required(root, "bars", label="bars response"), label="bars")
            page_symbols: set[str] = set()
            for raw_symbol, raw_rows in bars.items():
                try:
                    symbol = _canonical_symbol(raw_symbol)
                except AlpacaConfigurationError as exc:
                    raise AlpacaHistoryError("bars response contains an invalid symbol") from exc
                if symbol in page_symbols:
                    raise AlpacaHistoryError(
                        f"bars page contains duplicate canonical symbol {symbol}"
                    )
                page_symbols.add(symbol)
                if symbol not in by_symbol:
                    raise AlpacaHistoryError(f"bars response contains unexpected symbol {symbol}")
                if not isinstance(raw_rows, list):
                    raise AlpacaSchemaError(f"bars for {symbol} must be a JSON array")
                for row_number, raw_row in enumerate(raw_rows):
                    row = _mapping(raw_row, label=f"bar {symbol}[{row_number}]")
                    source = _source_timestamp(
                        _required(row, "t", label=f"bar {symbol}[{row_number}]"),
                        label=f"{symbol} daily bar",
                    )
                    stamp_utc = _datetime_from_epoch_ns_floor(source.epoch_ns)
                    session_date = stamp_utc.astimezone(_NEW_YORK).date()
                    session = selected_by_date.get(session_date)
                    if session is None:
                        raise AlpacaHistoryError(
                            f"{symbol} bar maps to unexpected session {session_date.isoformat()}"
                        )
                    if session.close_at >= endpoint:
                        raise AlpacaHistoryError(
                            f"{symbol} bar belongs to a current or future incomplete session"
                        )
                    close = _decimal(
                        _required(row, "c", label=f"bar {symbol}[{row_number}]"),
                        label=f"{symbol} close",
                    )
                    if close <= Decimal("0"):
                        raise AlpacaHistoryError(f"{symbol} close must be positive")
                    if session_date in by_symbol[symbol]:
                        raise AlpacaHistoryError(
                            f"bars response repeats {symbol} session {session_date.isoformat()}"
                        )
                    by_symbol[symbol][session_date] = close

            raw_next = root.get("next_page_token")
            if raw_next is None:
                finished = True
                break
            if not isinstance(raw_next, str) or not raw_next:
                raise AlpacaHistoryError("next_page_token must be a non-empty string or null")
            if raw_next in seen_tokens:
                raise AlpacaHistoryError("bars pagination repeated a page token")
            seen_tokens.add(raw_next)
            page_token = raw_next
        if not finished:
            raise AlpacaHistoryError("bars pagination exceeded max_pages")

        session_dates = [session.session_date for session in selected]
        missing: list[str] = []
        for symbol in self.symbols:
            for session_date in session_dates:
                if session_date not in by_symbol[symbol]:
                    missing.append(f"{symbol}:{session_date.isoformat()}")
        if missing:
            raise AlpacaHistoryError(f"adjusted history is not a complete aligned panel: {missing}")

        values = [
            [float(by_symbol[symbol][session.session_date]) for symbol in self.symbols]
            for session in selected
        ]
        if not all(math.isfinite(value) and value > 0 for row in values for value in row):
            raise AlpacaHistoryError("history exceeds the finite solver numeric range")
        return pd.DataFrame(
            values,
            index=pd.DatetimeIndex([session.close_at for session in selected]),
            columns=list(self.symbols),
            dtype=float,
        )


__all__ = [
    "AlpacaAssetEligibilityError",
    "AlpacaAssetMetadata",
    "AlpacaAuthenticationError",
    "AlpacaConfigurationError",
    "AlpacaCredentialError",
    "AlpacaCredentials",
    "AlpacaDataConfig",
    "AlpacaDataError",
    "AlpacaDataProvider",
    "AlpacaDependencyError",
    "AlpacaEndpointError",
    "AlpacaHistoryError",
    "AlpacaHttpError",
    "AlpacaHttpResponse",
    "AlpacaHttpTransport",
    "AlpacaRateLimitError",
    "AlpacaResponseTooLargeError",
    "AlpacaSchemaError",
    "AlpacaSessionError",
    "AlpacaSnapshotError",
    "AlpacaStaleDataError",
    "AlpacaTransportFailureError",
    "HttpxAlpacaTransport",
    "MarketSession",
    "ObservedMarketSnapshot",
    "ObservedSymbolMarketData",
    "SourceTimestamp",
]
