"""Immutable, deterministic value objects for local paper execution.

The paper layer deliberately accepts decimal strings (or ``Decimal`` values)
instead of binary floats for anything that enters accounting.  This keeps a
replayed event log byte-for-byte stable and prevents a value such as ``0.1``
from silently becoming a long binary approximation before it reaches the
ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)
from numbers import Integral, Real
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

PAPER_DECIMAL_PRECISION = 96
PAPER_MAX_SIGNIFICANT_DIGITS = 36
PAPER_MAX_INTEGER_DIGITS = 18
PAPER_MAX_FRACTIONAL_DIGITS = 18
_PAPER_DECIMAL_CONTEXT = Context(
    prec=PAPER_DECIMAL_PRECISION,
    rounding=ROUND_HALF_EVEN,
)


def decimal_context():
    """Return an isolated, deterministic context for every paper calculation."""

    return localcontext(_PAPER_DECIMAL_CONTEXT)


def _bounded_decimal(value: Decimal) -> Decimal:
    """Bound accounting inputs so fixed-context arithmetic remains exact."""

    if not value.is_finite():
        raise ValueError("decimal values must be finite")
    _sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("decimal values must be finite")
    significant_digits = len(digits)
    integer_digits = max(significant_digits + exponent, 0)
    fractional_digits = max(-exponent, 0)
    if significant_digits > PAPER_MAX_SIGNIFICANT_DIGITS:
        raise ValueError(
            f"decimal values may have at most {PAPER_MAX_SIGNIFICANT_DIGITS} "
            "significant digits"
        )
    if integer_digits > PAPER_MAX_INTEGER_DIGITS:
        raise ValueError(
            f"decimal values may have at most {PAPER_MAX_INTEGER_DIGITS} integer digits"
        )
    if fractional_digits > PAPER_MAX_FRACTIONAL_DIGITS:
        raise ValueError(
            f"decimal values may have at most {PAPER_MAX_FRACTIONAL_DIGITS} "
            "fractional digits"
        )
    return value


def _bounded_derived_decimal(value: Decimal) -> Decimal:
    """Validate a deterministic result produced inside the fixed context."""

    if not value.is_finite():
        raise ValueError("derived decimal values must be finite")
    _sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("derived decimal values must be finite")
    integer_digits = max(len(digits) + exponent, 0)
    fractional_digits = max(-exponent, 0)
    if len(digits) > PAPER_DECIMAL_PRECISION:
        raise ValueError(
            f"derived values may have at most {PAPER_DECIMAL_PRECISION} significant digits"
        )
    if integer_digits > PAPER_MAX_INTEGER_DIGITS * 2:
        raise ValueError("derived decimal value exceeds the supported accounting range")
    max_fractional_digits = PAPER_DECIMAL_PRECISION + PAPER_MAX_INTEGER_DIGITS
    if fractional_digits > max_fractional_digits:
        raise ValueError(
            f"derived values may have at most {max_fractional_digits} fractional digits"
        )
    return value


class FrozenDecimalDict(dict[str, Decimal]):
    """JSON-friendly decimal map that cannot be mutated after validation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen decimal mappings cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> FrozenDecimalDict:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenDecimalDict:
        return self

    def __hash__(self) -> int:
        return hash(tuple(self.items()))

    def copy(self) -> FrozenDecimalDict:
        return self


def freeze_decimal_map(value: dict[str, Decimal]) -> dict[str, Decimal]:
    """Return a sorted, deeply immutable copy while retaining dict JSON support."""

    return FrozenDecimalDict(sorted(value.items()))


def _reject_binary_float(value: object) -> object:
    """Reject binary floats before Pydantic can coerce them to ``Decimal``."""

    if isinstance(value, bool):
        raise ValueError("booleans are not decimal values")
    if isinstance(value, Real) and not isinstance(value, Integral):
        raise ValueError(
            "binary floats are not accepted; pass Decimal, an integer, or a decimal string"
        )
    return value


def _as_utc(value: datetime) -> datetime:
    """Require an unambiguous instant and canonicalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


StrictDecimal = Annotated[
    Decimal,
    BeforeValidator(_reject_binary_float),
    Field(allow_inf_nan=False),
    AfterValidator(_bounded_decimal),
]
DerivedDecimal = Annotated[
    Decimal,
    BeforeValidator(_reject_binary_float),
    Field(allow_inf_nan=False),
    AfterValidator(_bounded_derived_decimal),
]
NonNegativeDecimal = Annotated[StrictDecimal, Field(ge=Decimal("0"))]
PositiveDecimal = Annotated[StrictDecimal, Field(gt=Decimal("0"))]
WeightDecimal = Annotated[
    StrictDecimal,
    Field(ge=Decimal("0"), le=Decimal("1")),
]
NonNegativeDerivedDecimal = Annotated[DerivedDecimal, Field(ge=Decimal("0"))]
PositiveDerivedDecimal = Annotated[DerivedDecimal, Field(gt=Decimal("0"))]
DerivedWeightDecimal = Annotated[
    DerivedDecimal,
    Field(ge=Decimal("0"), le=Decimal("1")),
]
UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class PaperModel(BaseModel):
    """Frozen base model with a canonical JSON representation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    def canonical_json(self) -> str:
        """Return stable JSON suitable for hashing and event comparison."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


def _canonical_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("ticker must not be empty")
    return ticker


def _nonempty_id(value: str) -> str:
    identifier = value.strip()
    if not identifier:
        raise ValueError("identifier must not be empty")
    return identifier


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


class Quote(PaperModel):
    """One executable and markable quote at a precise instant."""

    ticker: str
    bid: PositiveDecimal
    ask: PositiveDecimal
    last: PositiveDecimal
    as_of: UtcDatetime

    _normalize_ticker = field_validator("ticker", mode="before")(_canonical_ticker)

    @model_validator(mode="after")
    def _ordered_market(self) -> Self:
        if self.bid > self.ask:
            raise ValueError("bid must be less than or equal to ask")
        return self


class MarketSnapshot(PaperModel):
    """A complete, duplicate-free set of quotes observed at one instant."""

    snapshot_id: str
    as_of: UtcDatetime
    quotes: tuple[Quote, ...] = Field(min_length=1)
    source_provenance_json: str | None = Field(default=None, max_length=200_000)

    _validate_id = field_validator("snapshot_id", mode="before")(_nonempty_id)

    @field_validator("quotes", mode="after")
    @classmethod
    def _canonical_quote_order(cls, quotes: tuple[Quote, ...]) -> tuple[Quote, ...]:
        return tuple(sorted(quotes, key=lambda quote: quote.ticker))

    @field_validator("source_provenance_json", mode="after")
    @classmethod
    def _canonical_source_provenance(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            decoded = json.loads(
                value,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ValueError("source provenance must be finite JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("source provenance must be a JSON object")
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if value != canonical:
            raise ValueError("source provenance JSON must be canonical")
        return value

    @model_validator(mode="after")
    def _consistent_quotes(self) -> Self:
        tickers = [quote.ticker for quote in self.quotes]
        if len(tickers) != len(set(tickers)):
            raise ValueError("market snapshot contains duplicate tickers")
        if any(quote.as_of != self.as_of for quote in self.quotes):
            raise ValueError("every quote must have the snapshot timestamp")
        return self

    def quote(self, ticker: str) -> Quote:
        """Return a quote by canonical ticker, raising ``KeyError`` if absent."""

        canonical = _canonical_ticker(ticker)
        for quote in self.quotes:
            if quote.ticker == canonical:
                return quote
        raise KeyError(canonical)

    def marks(self) -> dict[str, Decimal]:
        """Return canonical last-price marks."""

        return {quote.ticker: quote.last for quote in self.quotes}


class TargetAllocation(PaperModel):
    """Long-only target weights, with any remainder held as cash."""

    allocation_id: str
    effective_at: UtcDatetime
    weights: dict[str, WeightDecimal]

    _validate_id = field_validator("allocation_id", mode="before")(_nonempty_id)

    @field_validator("weights", mode="after")
    @classmethod
    def _canonical_weights(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        canonical: dict[str, Decimal] = {}
        for raw_ticker, weight in value.items():
            ticker = _canonical_ticker(raw_ticker)
            if ticker in canonical:
                raise ValueError(f"duplicate ticker after canonicalization: {ticker}")
            canonical[ticker] = weight
        with decimal_context():
            if sum(canonical.values(), Decimal("0")) > Decimal("1"):
                raise ValueError("target weights must sum to at most 1")
        return freeze_decimal_map(canonical)

    @property
    def cash_weight(self) -> Decimal:
        with decimal_context():
            return Decimal("1") - sum(self.weights.values(), Decimal("0"))


class OrderIntent(PaperModel):
    """One positive-quantity buy or sell requested by a deterministic planner."""

    order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: PositiveDecimal
    reference_price: PositiveDecimal

    _validate_id = field_validator("order_id", mode="before")(_nonempty_id)
    _normalize_ticker = field_validator("ticker", mode="before")(_canonical_ticker)


class OrderBatch(PaperModel):
    """An ordered, atomic collection of order intents for one target."""

    batch_id: str
    allocation_id: str
    effective_at: UtcDatetime
    orders: tuple[OrderIntent, ...] = ()

    _validate_ids = field_validator("batch_id", "allocation_id", mode="before")(_nonempty_id)

    @model_validator(mode="after")
    def _unique_orders(self) -> Self:
        order_ids = [order.order_id for order in self.orders]
        if len(order_ids) != len(set(order_ids)):
            raise ValueError("order batch contains duplicate order IDs")
        tickers = [order.ticker for order in self.orders]
        if len(tickers) != len(set(tickers)):
            raise ValueError("order batch must contain at most one net order per ticker")
        return self


class Fill(PaperModel):
    """A complete local fill for one order intent."""

    fill_id: str
    order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: PositiveDecimal
    price: PositiveDecimal
    fee: NonNegativeDecimal = Decimal("0")
    executed_at: UtcDatetime

    _validate_ids = field_validator("fill_id", "order_id", mode="before")(_nonempty_id)
    _normalize_ticker = field_validator("ticker", mode="before")(_canonical_ticker)

    @property
    def notional(self) -> Decimal:
        with decimal_context():
            return self.quantity * self.price


class ExecutionAssumptions(PaperModel):
    """Explicit local assumptions used to turn intents into simulated fills."""

    half_spread_bps: NonNegativeDecimal = Decimal("0")
    slippage_bps: NonNegativeDecimal = Decimal("0")
    commission_bps: NonNegativeDecimal = Decimal("0")
    minimum_fee: NonNegativeDecimal = Decimal("0")
    quantity_step: PositiveDecimal = Decimal("0.000001")


class SimulationConfig(PaperModel):
    """Deterministic local exchange ticks persisted with every shadow event."""

    price_tick: PositiveDecimal = Decimal("0.01")
    fee_tick: PositiveDecimal = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class FillEconomics:
    """Pure expected economics shared by validation, planning, and simulation."""

    price: Decimal
    fee: Decimal
    notional: Decimal
    cash_change: Decimal


_BASIS_POINTS = Decimal("10000")


def _round_to_step(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    with decimal_context():
        units = (value / step).to_integral_value(rounding=rounding)
        return units * step


def expected_fill_economics(
    order: OrderIntent,
    quote: Quote,
    assumptions: ExecutionAssumptions,
    simulation: SimulationConfig,
) -> FillEconomics:
    """Recompute one conservative fill from fully persisted trusted inputs."""

    if assumptions.slippage_bps >= _BASIS_POINTS:
        raise ValueError("slippage_bps must be less than 10000")
    reference_price = quote.ask if order.side == "buy" else quote.bid
    if order.reference_price != reference_price:
        raise ValueError(
            f"{order.side} order {order.order_id!r} reference price must equal "
            f"the snapshot {'ask' if order.side == 'buy' else 'bid'}"
        )
    with decimal_context():
        if order.quantity % assumptions.quantity_step != Decimal("0"):
            raise ValueError(
                f"order {order.order_id!r} quantity is not a multiple of quantity_step"
            )
        slippage = assumptions.slippage_bps / _BASIS_POINTS
        if order.side == "buy":
            raw_price = quote.ask * (Decimal("1") + slippage)
            price = _round_to_step(raw_price, simulation.price_tick, ROUND_CEILING)
        else:
            raw_price = quote.bid * (Decimal("1") - slippage)
            if raw_price <= Decimal("0"):
                raise ValueError("sell assumptions must imply a positive fill price")
            price = _round_to_step(raw_price, simulation.price_tick, ROUND_FLOOR)
            if price <= Decimal("0"):
                raise ValueError(
                    "price_tick is too coarse to represent a positive conservative sell price"
                )

        notional = order.quantity * price
        proportional_fee = notional * assumptions.commission_bps / _BASIS_POINTS
        raw_fee = max(assumptions.minimum_fee, proportional_fee)
        fee = (
            Decimal("0")
            if raw_fee == Decimal("0")
            else _round_to_step(raw_fee, simulation.fee_tick, ROUND_CEILING)
        )
        cash_change = -(notional + fee) if order.side == "buy" else notional - fee
    _bounded_derived_decimal(notional)
    _bounded_derived_decimal(cash_change)
    _bounded_decimal(price)
    _bounded_decimal(fee)
    return FillEconomics(
        price=price,
        fee=fee,
        notional=notional,
        cash_change=cash_change,
    )


class ShadowBatchExecuted(PaperModel):
    """One all-or-nothing paper execution event.

    The event requires exactly one complete fill per order.  A later simulator
    may decide that an order cannot fill, but it must do so before constructing
    this event; the exact ledger never records a half-applied batch.
    """

    sequence: int = Field(ge=1)
    batch: OrderBatch
    snapshot: MarketSnapshot
    fills: tuple[Fill, ...]
    assumptions: ExecutionAssumptions
    simulation: SimulationConfig

    @model_validator(mode="after")
    def _fills_match_orders(self) -> Self:
        if self.snapshot.as_of != self.batch.effective_at:
            raise ValueError("batch and snapshot must have the same effective timestamp")
        if len(self.fills) != len(self.batch.orders):
            raise ValueError("executed batch requires exactly one fill per order")

        fill_ids = [fill.fill_id for fill in self.fills]
        if len(fill_ids) != len(set(fill_ids)):
            raise ValueError("executed batch contains duplicate fill IDs")

        orders = {order.order_id: order for order in self.batch.orders}
        if len(orders) != len(self.batch.orders):
            raise ValueError("executed batch contains duplicate order IDs")
        expected_order_ids = tuple(order.order_id for order in self.batch.orders)
        fill_order_ids = tuple(fill.order_id for fill in self.fills)
        if fill_order_ids != expected_order_ids:
            raise ValueError(
                "executed batch fills must match every order exactly once and in batch order"
            )
        for fill in self.fills:
            order = orders.get(fill.order_id)
            if order is None:
                raise ValueError(f"fill {fill.fill_id!r} does not belong to the batch")
            if (fill.ticker, fill.side, fill.quantity) != (
                order.ticker,
                order.side,
                order.quantity,
            ):
                raise ValueError(f"fill {fill.fill_id!r} does not match its order intent")
            if fill.executed_at != self.batch.effective_at:
                raise ValueError("fills must execute at the batch effective timestamp")
            try:
                quote = self.snapshot.quote(order.ticker)
            except KeyError:
                raise ValueError(
                    f"snapshot {self.snapshot.snapshot_id!r} has no quote for {order.ticker}"
                ) from None
            expected = expected_fill_economics(
                order,
                quote,
                self.assumptions,
                self.simulation,
            )
            if fill.price != expected.price or fill.fee != expected.fee:
                raise ValueError(
                    f"fill {fill.fill_id!r} price or fee does not match persisted "
                    "snapshot and execution assumptions"
                )
        return self

    @property
    def batch_id(self) -> str:
        return self.batch.batch_id

    @property
    def order_ids(self) -> tuple[str, ...]:
        return tuple(order.order_id for order in self.batch.orders)


__all__ = [
    "DerivedDecimal",
    "DerivedWeightDecimal",
    "ExecutionAssumptions",
    "Fill",
    "FillEconomics",
    "FrozenDecimalDict",
    "MarketSnapshot",
    "NonNegativeDerivedDecimal",
    "OrderBatch",
    "OrderIntent",
    "PaperModel",
    "Quote",
    "PositiveDerivedDecimal",
    "ShadowBatchExecuted",
    "SimulationConfig",
    "TargetAllocation",
    "decimal_context",
    "expected_fill_economics",
    "freeze_decimal_map",
]
