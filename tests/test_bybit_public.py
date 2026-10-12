from bot.data.bybit_public import parse_instrument

# Сокращённый ответ /v5/market/instruments-info для BTCUSDT (значения иллюстративные).
RAW = {
    "symbol": "BTCUSDT",
    "priceFilter": {"minPrice": "0.10", "maxPrice": "1999999.80", "tickSize": "0.10"},
    "lotSizeFilter": {
        "maxOrderQty": "1190.000",
        "minOrderQty": "0.001",
        "qtyStep": "0.001",
        "minNotionalValue": "5",
    },
    "leverageFilter": {"minLeverage": "1", "maxLeverage": "100.00", "leverageStep": "0.01"},
}


def test_parse_instrument() -> None:
    info = parse_instrument(RAW)
    assert info.symbol == "BTCUSDT"
    assert info.tick_size == 0.1
    assert info.qty_step == 0.001
    assert info.min_order_qty == 0.001
    assert info.min_notional == 5
    assert info.max_leverage == 100
