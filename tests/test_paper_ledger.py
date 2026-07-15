"""Exact, fully local accounting tests for the paper-testing foundation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from paper import (
    ExecutionAssumptions,
    Fill,
    IdealLedger,
    IdealLedgerError,
    LedgerAccountingError,
    LedgerIdentifierCollisionError,
    LedgerSequenceError,
    MarketSnapshot,
    OrderBatch,
    OrderIntent,
    Quote,
    ShadowBatchExecuted,
    ShadowLedger,
    SimulationConfig,
    TargetAllocation,
    reduce_shadow_ledger,
)

T0 = datetime(2026, 1, 2, 21, tzinfo=UTC)


def _quote(ticker: str, price: str, as_of: datetime) -> Quote:
    return Quote(ticker=ticker, bid=price, ask=price, last=price, as_of=as_of)


def _market(snapshot_id: str, as_of: datetime, **prices: str) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=snapshot_id,
        as_of=as_of,
        quotes=tuple(_quote(ticker, price, as_of) for ticker, price in prices.items()),
    )


def _execution(
    *,
    sequence: int,
    batch_id: str,
    order_id: str,
    side: str,
    quantity: str,
    price: str,
    fee: str,
    at: datetime,
) -> ShadowBatchExecuted:
    order = OrderIntent(
        order_id=order_id,
        ticker="AAA",
        side=side,
        quantity=quantity,
        reference_price=price,
    )
    return ShadowBatchExecuted(
        sequence=sequence,
        batch=OrderBatch(
            batch_id=batch_id,
            allocation_id=f"allocation-{batch_id}",
            effective_at=at,
            orders=(order,),
        ),
        snapshot=_market(f"snapshot-{batch_id}", at, AAA=price),
        fills=(
            Fill(
                fill_id=f"fill-{order_id}",
                order_id=order_id,
                ticker="AAA",
                side=side,
                quantity=quantity,
                price=price,
                fee=fee,
                executed_at=at,
            ),
        ),
        assumptions=ExecutionAssumptions(minimum_fee=fee),
        simulation=SimulationConfig(),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Quote(ticker="AAA", bid=9.0, ask="10", last="10", as_of=T0),
        lambda: TargetAllocation(
            allocation_id="a", effective_at=T0, weights={"AAA": 0.5}
        ),
        lambda: OrderIntent(
            order_id="o", ticker="AAA", side="buy", quantity=1.0, reference_price="10"
        ),
        lambda: Fill(
            fill_id="f",
            order_id="o",
            ticker="AAA",
            side="buy",
            quantity="1",
            price="10",
            fee=0.0,
            executed_at=T0,
        ),
        lambda: ExecutionAssumptions(slippage_bps=1.0),
    ],
)
def test_accounting_models_reject_binary_float_inputs(factory) -> None:
    with pytest.raises(ValidationError, match="binary floats are not accepted"):
        factory()


def test_models_canonicalize_tickers_time_and_json() -> None:
    offset = timezone(timedelta(hours=-8))
    local_time = datetime(2026, 1, 2, 13, tzinfo=offset)
    snapshot = MarketSnapshot(
        snapshot_id="close",
        as_of=T0,
        quotes=(
            _quote("bbb", "20.00", local_time),
            _quote("aaa", "10.00", local_time),
        ),
    )

    assert [quote.ticker for quote in snapshot.quotes] == ["AAA", "BBB"]
    assert snapshot.as_of == T0
    assert snapshot.canonical_json() == snapshot.canonical_json()
    assert '"last":"10.00"' in snapshot.canonical_json()
    assert snapshot.marks() == {"AAA": Decimal("10.00"), "BBB": Decimal("20.00")}


def test_shadow_ledger_exact_buy_mark_sell_and_replay() -> None:
    ledger = ShadowLedger()
    assert ledger.state.cash == Decimal("100.00")
    assert ledger.state.equity == Decimal("100.00")
    assert ledger.state.gross_exposure == Decimal("0")

    buy = _execution(
        sequence=1,
        batch_id="batch-buy",
        order_id="order-buy",
        side="buy",
        quantity="2",
        price="10",
        fee="0.25",
        at=T0,
    )
    bought = ledger.execute(buy)
    assert bought.cash == Decimal("79.75")
    assert bought.positions == {"AAA": Decimal("2")}
    assert bought.cumulative_fees == Decimal("0.25")
    assert bought.equity == Decimal("99.75")
    assert bought.gross_exposure == Decimal("20")

    marked = ledger.mark(_market("mark-1", T0 + timedelta(days=1), AAA="12"))
    assert marked.sequence == 2
    assert marked.equity == Decimal("103.75")
    assert marked.gross_exposure == Decimal("24")

    sell = _execution(
        sequence=3,
        batch_id="batch-sell",
        order_id="order-sell",
        side="sell",
        quantity="0.5",
        price="12",
        fee="0.10",
        at=T0 + timedelta(days=1),
    )
    sold = ledger.execute(sell)
    assert sold.cash == Decimal("85.65")
    assert sold.positions == {"AAA": Decimal("1.5")}
    assert sold.cumulative_fees == Decimal("0.35")
    assert sold.equity == Decimal("103.65")
    assert sold.gross_exposure == Decimal("18.0")
    assert sold.executed_batch_ids == ("batch-buy", "batch-sell")
    assert sold.executed_order_ids == ("order-buy", "order-sell")
    assert reduce_shadow_ledger(ledger.events) == sold


def test_shadow_batch_failure_is_atomic() -> None:
    ledger = ShadowLedger()
    unaffordable = _execution(
        sequence=1,
        batch_id="too-large",
        order_id="buy-too-large",
        side="buy",
        quantity="11",
        price="10",
        fee="0",
        at=T0,
    )
    with pytest.raises(LedgerAccountingError, match="negative cash"):
        ledger.execute(unaffordable)
    assert ledger.events == ()
    assert ledger.state.cash == Decimal("100.00")

    short_sale = _execution(
        sequence=1,
        batch_id="short",
        order_id="sell-unowned",
        side="sell",
        quantity="1",
        price="10",
        fee="0",
        at=T0,
    )
    with pytest.raises(LedgerAccountingError, match="short positions"):
        ledger.execute(short_sale)
    assert ledger.events == ()


def test_shadow_retries_are_idempotent_but_id_reuse_is_an_error() -> None:
    ledger = ShadowLedger()
    event = _execution(
        sequence=1,
        batch_id="batch-1",
        order_id="order-1",
        side="buy",
        quantity="1",
        price="10",
        fee="0",
        at=T0,
    )
    first = ledger.execute(event)
    retry = ledger.execute(event)
    assert retry == first
    assert ledger.events == (event,)

    collision = _execution(
        sequence=2,
        batch_id="batch-1",
        order_id="order-new",
        side="buy",
        quantity="2",
        price="10",
        fee="0",
        at=T0 + timedelta(days=1),
    )
    with pytest.raises(LedgerIdentifierCollisionError, match="batch ID"):
        ledger.execute(collision)
    assert ledger.events == (event,)

    skipped = _execution(
        sequence=3,
        batch_id="batch-3",
        order_id="order-3",
        side="buy",
        quantity="1",
        price="10",
        fee="0",
        at=T0 + timedelta(days=1),
    )
    with pytest.raises(LedgerSequenceError, match="expected event sequence 2"):
        ledger.execute(skipped)


def test_ideal_ledger_is_unit_normalized_frictionless_and_isolated() -> None:
    shadow = ShadowLedger()
    ideal = IdealLedger()
    assert ideal.state.nav == Decimal("1")
    assert ideal.state.cash == Decimal("1")

    initial_market = _market("ideal-0", T0, AAA="10", BBB="20")
    initial_target = TargetAllocation(
        allocation_id="target-0",
        effective_at=T0,
        weights={"BBB": "0.4", "AAA": "0.6"},
    )
    applied = ideal.apply(initial_target, initial_market)
    assert applied.nav == Decimal("1.000")
    assert applied.cash == Decimal("0.000")
    assert applied.weights == {"AAA": Decimal("0.6"), "BBB": Decimal("0.4")}
    assert applied.quantities == {"AAA": Decimal("0.06"), "BBB": Decimal("0.02")}

    next_time = T0 + timedelta(days=1)
    marked = ideal.mark(_market("ideal-1", next_time, AAA="11", BBB="18"))
    assert marked.nav == Decimal("1.02")

    second_target = TargetAllocation(
        allocation_id="target-1",
        effective_at=next_time,
        weights={"AAA": "1"},
    )
    rebalanced = ideal.apply(
        second_target,
        _market("ideal-1-rebalance", next_time, AAA="11", BBB="18"),
    )
    assert rebalanced.nav == Decimal("1.02")
    assert rebalanced.weights == {"AAA": Decimal("1")}
    assert shadow.state.cash == Decimal("100.00")
    assert shadow.events == ()


def test_ideal_allocation_requires_same_effective_timestamp() -> None:
    ideal = IdealLedger()
    target = TargetAllocation(
        allocation_id="late-target",
        effective_at=T0 + timedelta(seconds=1),
        weights={"AAA": "1"},
    )
    with pytest.raises(IdealLedgerError, match="same effective timestamp"):
        ideal.apply(target, _market("market", T0, AAA="10"))
    assert ideal.state.nav == Decimal("1")
    assert len(ideal.history) == 1


def test_executed_batch_requires_complete_fills_and_verified_economics() -> None:
    orders = (
        OrderIntent(
            order_id="order-a",
            ticker="AAA",
            side="buy",
            quantity="1",
            reference_price="10",
        ),
        OrderIntent(
            order_id="order-b",
            ticker="BBB",
            side="buy",
            quantity="1",
            reference_price="20",
        ),
    )
    batch = OrderBatch(
        batch_id="batch-complete",
        allocation_id="allocation-complete",
        effective_at=T0,
        orders=orders,
    )
    snapshot = _market("snapshot-complete", T0, AAA="10", BBB="20")
    duplicate_order_fills = (
        Fill(
            fill_id="fill-a-1",
            order_id="order-a",
            ticker="AAA",
            side="buy",
            quantity="1",
            price="10",
            executed_at=T0,
        ),
        Fill(
            fill_id="fill-a-2",
            order_id="order-a",
            ticker="AAA",
            side="buy",
            quantity="1",
            price="10",
            executed_at=T0,
        ),
    )
    with pytest.raises(ValidationError, match="every order exactly once"):
        ShadowBatchExecuted(
            sequence=1,
            batch=batch,
            snapshot=snapshot,
            fills=duplicate_order_fills,
            assumptions=ExecutionAssumptions(),
            simulation=SimulationConfig(),
        )

    one_order = OrderBatch(
        batch_id="batch-economics",
        allocation_id="allocation-economics",
        effective_at=T0,
        orders=(orders[0],),
    )
    with pytest.raises(ValidationError, match="does not match persisted"):
        ShadowBatchExecuted(
            sequence=1,
            batch=one_order,
            snapshot=snapshot,
            fills=(
                Fill(
                    fill_id="fabricated-fill",
                    order_id="order-a",
                    ticker="AAA",
                    side="buy",
                    quantity="1",
                    price="11",
                    fee="1",
                    executed_at=T0,
                ),
            ),
            assumptions=ExecutionAssumptions(),
            simulation=SimulationConfig(),
        )


def test_accounting_maps_are_deeply_immutable() -> None:
    target = TargetAllocation(
        allocation_id="immutable-target",
        effective_at=T0,
        weights={"AAA": "0.5"},
    )
    with pytest.raises(TypeError, match="cannot be mutated"):
        target.weights["AAA"] = Decimal("1")

    ledger = ShadowLedger()
    bought = ledger.execute(
        _execution(
            sequence=1,
            batch_id="immutable-buy",
            order_id="immutable-order",
            side="buy",
            quantity="1",
            price="10",
            fee="0",
            at=T0,
        )
    )
    with pytest.raises(TypeError, match="cannot be mutated"):
        bought.positions["AAA"] = Decimal("2")
    assert ledger.state.positions == {"AAA": Decimal("1")}

    ideal = IdealLedger()
    ideal.apply(target, _market("immutable-ideal", T0, AAA="10"))
    with pytest.raises(TypeError, match="cannot be mutated"):
        ideal.state.weights["AAA"] = Decimal("0")


def test_shadow_replay_ignores_ambient_decimal_precision() -> None:
    assumptions = ExecutionAssumptions(quantity_step="0.000000000000000001")
    simulation = SimulationConfig(price_tick="0.000000000000000001")
    order = OrderIntent(
        order_id="precision-order",
        ticker="AAA",
        side="buy",
        quantity="3.123456789012345678",
        reference_price="2.123456789012345678",
    )
    batch = OrderBatch(
        batch_id="precision-batch",
        allocation_id="precision-allocation",
        effective_at=T0,
        orders=(order,),
    )
    snapshot = _market(
        "precision-snapshot",
        T0,
        AAA="2.123456789012345678",
    )
    # Zero slippage means the exact persisted fill equals the quoted price.
    event = ShadowBatchExecuted(
        sequence=1,
        batch=batch,
        snapshot=snapshot,
        fills=(
            Fill(
                fill_id="precision-fill",
                order_id=order.order_id,
                ticker="AAA",
                side="buy",
                quantity=order.quantity,
                price=order.reference_price,
                executed_at=T0,
            ),
        ),
        assumptions=assumptions,
        simulation=simulation,
    )

    with localcontext() as context:
        context.prec = 6
        low_precision = reduce_shadow_ledger([event])
    with localcontext() as context:
        context.prec = 60
        high_precision = reduce_shadow_ledger([event])

    assert low_precision.canonical_json() == high_precision.canonical_json()
    assert low_precision.cash == Decimal("93.367474476185032773472031700234720316")


def test_execution_snapshot_marks_last_and_exposes_spread_and_slippage_drag() -> None:
    order = OrderIntent(
        order_id="drag-order",
        ticker="AAA",
        side="buy",
        quantity="1",
        reference_price="10.10",
    )
    event = ShadowBatchExecuted(
        sequence=1,
        batch=OrderBatch(
            batch_id="drag-batch",
            allocation_id="drag-allocation",
            effective_at=T0,
            orders=(order,),
        ),
        snapshot=MarketSnapshot(
            snapshot_id="drag-snapshot",
            as_of=T0,
            quotes=(
                Quote(
                    ticker="AAA",
                    bid="9.90",
                    ask="10.10",
                    last="10.00",
                    as_of=T0,
                ),
            ),
        ),
        fills=(
            Fill(
                fill_id="drag-fill",
                order_id=order.order_id,
                ticker="AAA",
                side="buy",
                quantity="1",
                price="10.21",
                executed_at=T0,
            ),
        ),
        assumptions=ExecutionAssumptions(slippage_bps="100"),
        simulation=SimulationConfig(),
    )
    state = ShadowLedger().execute(event)

    assert state.cash == Decimal("89.79")
    assert state.marks["AAA"] == Decimal("10.00")
    assert state.equity == Decimal("99.79")


def test_shadow_mark_requires_every_held_asset_and_is_atomic() -> None:
    ledger = ShadowLedger()
    ledger.execute(
        _execution(
            sequence=1,
            batch_id="full-mark-buy",
            order_id="full-mark-order",
            side="buy",
            quantity="1",
            price="10",
            fee="0",
            at=T0,
        )
    )
    before = ledger.state
    with pytest.raises(LedgerAccountingError, match="missing held positions"):
        ledger.mark(_market("partial-mark", T0 + timedelta(days=1), BBB="20"))
    assert ledger.state == before


def test_ideal_marks_are_idempotent_and_snapshot_collisions_fail() -> None:
    ideal = IdealLedger()
    initial = _market("ideal-id", T0, AAA="10")
    first = ideal.mark(initial)
    assert ideal.mark(initial) == first
    assert len(ideal.history) == 2

    collision = _market("ideal-id", T0, AAA="11")
    with pytest.raises(IdealLedgerError, match="different content"):
        ideal.mark(collision)
    assert ideal.state == first


def test_numeric_scale_change_is_not_a_byte_identical_event_retry() -> None:
    ledger = ShadowLedger()
    first = _execution(
        sequence=1,
        batch_id="scale-batch",
        order_id="scale-order",
        side="buy",
        quantity="1",
        price="10",
        fee="0",
        at=T0,
    )
    ledger.execute(first)
    changed_scale = _execution(
        sequence=1,
        batch_id="scale-batch",
        order_id="scale-order",
        side="buy",
        quantity="1.0",
        price="10.00",
        fee="0.00",
        at=T0,
    )
    with pytest.raises(LedgerIdentifierCollisionError, match="batch ID"):
        ledger.execute(changed_scale)
