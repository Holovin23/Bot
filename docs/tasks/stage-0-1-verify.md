# Как проверить этапы 0 и 1 на своём компьютере

Все команды выполняются в PowerShell из папки проекта `D:\Github\Bot`
(в VS Code: меню Terminal → New Terminal). Каждая задача из [stage-0-1.md](stage-0-1.md)
сделана отдельным коммитом, проверять можно по порядку.

## Один раз: подготовка

1. Установить Git: https://git-scm.com/download/win
2. Установить uv (в PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Закрыть и снова открыть терминал. Python 3.12 отдельно ставить не обязательно:
   `uv sync` сам скачает нужную версию.
3. Создать файл с ключами: скопировать `.env.example` в `.env`
   ```powershell
   Copy-Item .env.example .env
   ```
   Открыть `.env` в VS Code и вписать `BYBIT_API_KEY` и `BYBIT_API_SECRET` демо-счёта
   (Bybit → Demo Trading → API, права «чтение + торговля», **без вывода**).
   Ключи никуда не пересылать и в чат не вставлять.

## Этап 0

| Шаг | Команда | Что должно получиться |
|-----|---------|-----------------------|
| 0.1 Каркас | `uv sync` | установились зависимости, без ошибок |
| | `uv run pytest` | все тесты зелёные (`passed`) |
| | `uv run ruff check` | `All checks passed!` |
| | `uv run mypy src` | `Success: no issues found` |
| | `git status` | файла `.env` в списке нет |
| 0.2 Конфиг и логи | `uv run python -m bot` | строка `bot_started bybit_env=demo keys_loaded=True`; самих ключей в выводе и в `logs\bot.log` нет |
| 0.3 Модели | `uv run pytest tests/test_models.py` | тесты зелёные |
| 0.4 Подключение | `uv run python scripts/check_connection.py` | серверное время, параметры BTCUSDT/ETHUSDT, **баланс демо-счёта**. Цифры шага цены/лота вписать в таблицу в `docs/09-start-checklist.md` |

Если разница времени больше ±1000 мс: Параметры Windows → Время и язык → «Синхронизировать сейчас».

## Этап 1

| Шаг | Команда | Что должно получиться |
|-----|---------|-----------------------|
| 1.1 Хранилище | `uv run pytest tests/test_storage.py` | тесты зелёные |
| 1.2 Загрузка (проба, месяц) | `uv run python -m bot.data.downloader --symbol BTCUSDT --from 2026-09-01` | прогресс в процентах, файлы в `data\klines\BTCUSDT\1m\` |
| | открыть 3 случайные минуты на графике Bybit (BTCUSDT Perpetual, 1m) и сравнить open/high/low/close | совпадают |
| | запустить загрузку ещё раз, прервать `Ctrl+C`, запустить снова | продолжает с места остановки, «новых свечей» при повторе — 0 |
| 1.2 Полная история | `uv run python -m bot.data.downloader --from 2021-01-01` | обе монеты, около 3 млн свечей на символ; можно прерывать и продолжать |
| 1.3 Агрегация | `uv run pytest tests/test_resample.py` | тесты зелёные; 1h-свечи сверяются на графике в шаге 1.6 |
| 1.4 Качество | `uv run python -m bot.data.quality` | отчёт в консоли и в `data\reports\quality_BTCUSDT.md`; пропуски сверить с известными остановками биржи |
| 1.5 Живой поток | `uv run python -m bot.data.live_feed --hours 24` | раз в минуту строка с закрытой свечой. Для проверки отключить интернет на 5 минут: после восстановления пропуск дозагрузится |
| | после 24 ч: `uv run python -m bot.data.quality --from <дата начала> ` | пропусков за период нет |
| 1.6 График | `uv run python -m bot.app.api`, открыть http://127.0.0.1:8000 | свечи, EMA20/50/200 и полосы Боллинджера; сравнить 3 часовые свечи и вид индикаторов с графиком Bybit |

Остановить любую команду: `Ctrl+C`.

## Что проверить глазами

- `.env` не попал в git: `git status` его не показывает.
- В `logs\bot.log` нет ключей.
- Свечи и часовые свечи совпадают с графиком Bybit (время на графике бота — локальное, у Bybit можно выбрать часовой пояс внизу графика).
