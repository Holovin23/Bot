from typing import Any

import pytest
import requests
from pybit.exceptions import FailedRequestError, InvalidRequestError

from bot.core.config import InstrumentConfig, Secrets
from bot.core.models import OrderSide, OrderStatus, Side
from bot.execution.broker import (
    BrokerError,
    DuplicateOrderError,
    OrderRejectedError,
    TransientBrokerError,
    close_request,
)
from bot.execution.bybit import BybitBroker, format_step, parse_order, parse_position

BTC = {"BTCUSDT": InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001)}

FILLED = {
    "orderId": "abc",
    "orderLinkId": "b-BTCUSDT-1h-20261012T1000-L-X1",
    "symbol": "BTCUSDT",
    "side": "Sell",
    "orderType": "Market",
    "qty": "0.102",
    "price": "0",
    "orderStatus": "Filled",
    "cumExecQty": "0.102",
    "avgPrice": "65450.5",
    "reduceOnly": True,
    "createdTime": "1760266800000",
    "updatedTime": "1760266800100",
}


def invalid(code: int) -> InvalidRequestError:
    return InvalidRequestError("req", "msg", code, "00:00:00", None)


class FakeHTTP:
    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        def call(**params: Any) -> Any:
            self.calls.append((name, params))
            value = self.responses.get(name, {"result": {"list": []}})
            if isinstance(value, Exception):
                raise value
            return value

        return call


def test_format_step() -> None:
    assert format_step(0.10299, 0.001) == "0.102"
    assert format_step(65000.06, 0.1, down=False) == "65000.1"
    assert format_step(3, 0.01) == "3"


def test_close_order_params() -> None:
    http = FakeHTTP(get_open_orders={"result": {"list": [FILLED]}})
    broker = BybitBroker(http, BTC)  # type: ignore[arg-type]
    order = broker.place_order(
        close_request("BTCUSDT", Side.LONG, 0.10299, "b-BTCUSDT-1h-20261012T1000-L")
    )
    name, params = http.calls[0]
    assert name == "place_order"
    assert params["qty"] == "0.102" and params["reduceOnly"] is True
    assert params["side"] == "Sell" and params["positionIdx"] == 0
    assert params["orderLinkId"] == "b-BTCUSDT-1h-20261012T1000-L-X1"
    assert "stopLoss" not in params
    assert order.status is OrderStatus.FILLED and order.avg_fill_price == 65450.5


def test_errors_are_classified() -> None:
    cases: list[tuple[Exception, type[BrokerError]]] = [
        (invalid(110072), DuplicateOrderError),
        (invalid(10006), TransientBrokerError),
        (invalid(110007), OrderRejectedError),
        (
            FailedRequestError("r", "HTTP status code is not 200.", 502, "t", None),
            TransientBrokerError,
        ),
        (requests.exceptions.ReadTimeout(), TransientBrokerError),
    ]
    for error, kind in cases:
        broker = BybitBroker(FakeHTTP(get_positions=error), BTC)  # type: ignore[arg-type]
        with pytest.raises(kind):
            broker.get_positions()


def test_not_modified_is_ok() -> None:
    broker = BybitBroker(FakeHTTP(set_leverage=invalid(110043)), BTC)  # type: ignore[arg-type]
    broker.set_leverage("BTCUSDT", 3)


def test_parse_position_and_empty() -> None:
    raw = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "size": "0.1",
        "avgPrice": "65000",
        "leverage": "3",
        "stopLoss": "",
        "takeProfit": "0",
        "liqPrice": "43500",
        "unrealisedPnl": "12.5",
        "createdTime": "1760266800000",
    }
    pos = parse_position(raw)
    assert pos is not None and pos.side is Side.LONG and pos.stop_loss is None
    assert pos.liq_price == 43500
    assert parse_position({**raw, "size": "0", "side": ""}) is None
    assert parse_order(FILLED).side is OrderSide.SELL


def test_demo_only() -> None:
    live = Secrets(bybit_env="live", allow_live=True, bybit_api_key="k", bybit_api_secret="s")
    with pytest.raises(BrokerError, match="BYBIT_ENV=demo"):
        BybitBroker.demo(live, BTC)
    with pytest.raises(BrokerError, match="BYBIT_API_KEY"):
        BybitBroker.demo(Secrets(_env_file=None), BTC)  # type: ignore[call-arg]


def test_live_needs_flags_and_confirmation() -> None:
    demo = Secrets(_env_file=None, bybit_api_key="k", bybit_api_secret="s")  # type: ignore[call-arg]
    with pytest.raises(BrokerError, match="выключен"):
        BybitBroker.live(demo, BTC, confirmed=True)
    live = Secrets(bybit_env="live", allow_live=True, bybit_api_key="k", bybit_api_secret="s")
    with pytest.raises(BrokerError, match="не подтверждён"):
        BybitBroker.live(live, BTC, confirmed=False)
    assert BybitBroker.live(live, BTC, confirmed=True).name == "live"
