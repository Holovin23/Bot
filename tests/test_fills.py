import pytest

from bot.backtest.fills import ExitReason, Hit, check_exit
from bot.core.models import Side

# LONG: стоп 95, тейк 110. SHORT: стоп 105, тейк 90.
LONG = (Side.LONG, 95.0, 110.0)
SHORT = (Side.SHORT, 105.0, 90.0)


@pytest.mark.parametrize(
    ("position", "candle", "expected"),
    [
        # ничего не задето
        (LONG, (100, 105, 96), None),
        (SHORT, (100, 104, 91), None),
        # стоп внутри свечи — по цене стопа
        (LONG, (100, 101, 94), Hit(ExitReason.STOP, 95, gap=False)),
        (SHORT, (100, 106, 99), Hit(ExitReason.STOP, 105, gap=False)),
        # тейк внутри свечи — по цене тейка
        (LONG, (100, 111, 99), Hit(ExitReason.TAKE, 110, gap=False)),
        (SHORT, (100, 101, 89), Hit(ExitReason.TAKE, 90, gap=False)),
        # оба в одной свече — стоп (пессимистично)
        (LONG, (100, 111, 94), Hit(ExitReason.STOP, 95, gap=False)),
        (SHORT, (100, 106, 89), Hit(ExitReason.STOP, 105, gap=False)),
        # гэп за стоп — по open, хуже стопа
        (LONG, (93, 96, 92), Hit(ExitReason.STOP, 93, gap=True)),
        (SHORT, (107, 108, 104), Hit(ExitReason.STOP, 107, gap=True)),
        # гэп за тейк — по open, лучше тейка
        (LONG, (112, 113, 111), Hit(ExitReason.TAKE, 112, gap=True)),
        (SHORT, (88, 89, 87), Hit(ExitReason.TAKE, 88, gap=True)),
        # касание ровно уровня считается срабатыванием
        (LONG, (100, 110, 99), Hit(ExitReason.TAKE, 110, gap=False)),
        (LONG, (100, 101, 95), Hit(ExitReason.STOP, 95, gap=False)),
    ],
)
def test_check_exit(
    position: tuple[Side, float, float],
    candle: tuple[float, float, float],
    expected: Hit | None,
) -> None:
    side, stop, take = position
    open_, high, low = candle
    assert check_exit(side, stop, take, open_, high, low) == expected


def test_without_take() -> None:
    assert check_exit(Side.LONG, 95, None, 100, 1000, 96) is None
    assert check_exit(Side.LONG, 95, None, 100, 1000, 90) == Hit(ExitReason.STOP, 95, gap=False)
