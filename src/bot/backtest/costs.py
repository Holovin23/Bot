"""Издержки сделки: комиссии, проскальзывание, funding. Чистые функции, без сети.

Значения — из config/default.yaml (раздел costs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from bot.core.config import CostsConfig
from bot.core.models import Side


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.00055
    maker_fee: float = 0.0002
    slippage: dict[str, float] = field(default_factory=dict)
    stop_slippage_multiplier: float = 2.0
    default_slippage: float = 0.0005  # для символа, которого нет в конфиге

    @classmethod
    def from_config(cls, costs: CostsConfig) -> CostModel:
        return cls(
            taker_fee=costs.taker_fee,
            maker_fee=costs.maker_fee,
            slippage=dict(costs.slippage),
            stop_slippage_multiplier=costs.stop_slippage_multiplier,
        )

    def slippage_rate(self, symbol: str, stop: bool = False) -> float:
        rate = self.slippage.get(symbol, self.default_slippage)
        return rate * self.stop_slippage_multiplier if stop else rate

    def fee(self, notional: float, maker: bool = False) -> float:
        """Комиссия в USDT: |объём × цена| × ставка."""
        return abs(notional) * (self.maker_fee if maker else self.taker_fee)


def apply_slippage(price: float, buy: bool, rate: float) -> float:
    """Покупка исполняется дороже, продажа — дешевле."""
    return price * (1 + rate) if buy else price * (1 - rate)


def entry_fill(price: float, side: Side, rate: float) -> float:
    """Вход: LONG покупает, SHORT продаёт."""
    return apply_slippage(price, buy=side is Side.LONG, rate=rate)


def exit_fill(price: float, side: Side, rate: float) -> float:
    """Выход: LONG продаёт, SHORT покупает."""
    return apply_slippage(price, buy=side is Side.SHORT, rate=rate)


def funding_payment(side: Side, qty: float, price: float, rate: float) -> float:
    """Денежный поток funding для позиции (>0 — получили, <0 — заплатили).

    При положительной ставке лонги платят шортам, при отрицательной — наоборот.
    """
    sign = 1.0 if side is Side.LONG else -1.0
    return -sign * qty * price * rate


def funding_events(
    funding: pd.DataFrame, entry_time: pd.Timestamp, exit_time: pd.Timestamp
) -> pd.DataFrame:
    """Выплаты funding, на момент которых позиция была открыта: entry < time <= exit."""
    mask = (funding["time"] > entry_time) & (funding["time"] <= exit_time)
    result: pd.DataFrame = funding.loc[mask]
    return result
