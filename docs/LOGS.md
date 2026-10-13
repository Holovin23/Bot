# Журналы для разбора сделок (JSON Lines)

Зачем: по журналу можно восстановить каждое решение бота и каждую сделку бэктеста — почему был сигнал, что решил риск-менеджер, какой ордер ушёл, что ответила биржа, где стоял стоп, чем закончилось. Файлы удобно приложить к вопросу ИИ (Claude): «проверь по журналу, правильно ли бот открывал и закрывал сделки».

Формат — JSON Lines: **одна строка — одно событие**, каждая строка — отдельный JSON-объект. Время — UTC (ISO 8601). Ключей API в журналах нет.

## Где лежат

| Файл | Что внутри |
|------|-----------|
| `logs\journal\paper-<id>.jsonl` | paper-бот интерфейса `<id>` (вкладка «Paper-боты») |
| `logs\journal\paper.jsonl`, `demo.jsonl`, `live.jsonl` | бот из терминала (`bot.runner.live`) в этом режиме |
| `logs\journal\backtests.jsonl` | по строке на каждый бэктест: стратегия, параметры, период, итог, папка |
| `data\backtests\<папка>\run.json` | настройки бэктеста: стратегия и версия, параметры, период, капитал, риск, издержки, правила симуляции, все метрики |
| `data\backtests\<папка>\trades.jsonl` | каждая сделка бэктеста целиком (ниже) |
| `data\backtests\<папка>\skipped.jsonl` | сигналы, которые не стали сделками, и почему |
| `logs\jobs\*.log` | вывод команд и фоновых задач интерфейса (загрузка, бэктест) |

В интерфейсе: вкладка **«Журналы»** (просмотр и скачивание), ссылки на файлы бэктеста — во вкладке **«Отчёт»**.

## Журнал бота: события

Общие поля каждой строки: `time`, `account` (например, `paper-breakout-btcusdt-1h-1`), `kind`.

| `kind` | Когда | Главные поля |
|--------|-------|--------------|
| `bar` | закрылась свеча таймфрейма стратегии | `strategy`, `symbol`, `timeframe`, `bar_open`, `bar` (OHLCV), `features` (все индикаторы стратегии на этой свече), `atr`, `open_trade` |
| `signal` | стратегия дала сигнал | `signal`: `signal_id`, `side`, `entry_price` (close свечи), `stop_loss`, `take_profit`, `time_stop_bars`, `reason` (словами), `features` |
| `risk_decision` | проверки 1–13 риск-менеджера | `approved`, `reasons` (почему отклонён, с номером проверки из docs/05), `notes`, `qty`, `risk_amount`, `equity`, `open_risks`, `atr` |
| `order_sent` | ордер уходит на биржу | `purpose` (`entry`, `close:time`, `close:kill`), `request` (символ, сторона, объём, стоп, тейк, `reduce_only`, `order_link_id`) |
| `order_result` | ответ биржи | `order`: статус, исполненный объём, средняя цена |
| `order_rejected` / `order_unknown` / `order_failed` | отказ биржи / нет ответа (решит сверка) / ошибка закрытия | `order_link_id`, `error` |
| `trade_opened` | вход исполнен | `fill_price`, `signal_price`, `qty`, `stop`, `take`, `risk_amount` |
| `stop_moved` | трейлинг-стоп подтянут | `signal_id`, `old`, `new` |
| `time_exit` | выход по времени | `signal_id`, `bars_held` |
| `trade_closed` | сделка закрыта | `source` (`bot` — закрыл бот; `reconcile` — стоп или тейк сработал на бирже), `trade`: вход, выход, причина, `net_pnl`, `bars_held` |
| `event` | всё, что видно в «Событиях» (запуск, funding, сверка, аварийная остановка, сбои связи) | `event`, `level`, `message` |

Пример (строка сокращена):

```json
{"time":"2026-10-13T14:00:01+00:00","account":"paper-breakout-btcusdt-1h-1","kind":"risk_decision","signal_id":"b-BTCUSDT-1h-20261013T1300-L","approved":false,"reasons":["9: стоп 0.3 ATR ближе минимума 0.5 ATR"],"qty":0.0,"equity":10000.0}
```

## Бэктест: `trades.jsonl`

Одна строка — одна сделка, в порядке входа:

`number`, `signal_id`, `side`, `reason`, `features` (индикаторы на сигнальной свече), `signal_time`, `signal_price`, `entry_time`, `entry_price_raw` (open свечи входа), `entry_price` (с проскальзыванием), `qty`, `initial_stop`, `take`, `time_stop_bars`, `risk_amount`, `stop_moves` (каждый перенос стопа: время и уровень), `final_stop`, `exit_time`, `exit_reason` (`stop`, `take`, `time`, `end_of_data`), `exit_price_raw`, `exit_price`, `gap_exit`, `bars_held`, `entry_fee`, `exit_fee`, `slippage`, `funding`, `funding_events`, `gross_pnl`, `net_pnl`, `r_multiple`.

Проверка любой строки: `net_pnl = gross_pnl − entry_fee − exit_fee + funding`, `r_multiple = net_pnl / risk_amount`, `gross_pnl = (exit_price − entry_price) × qty` для LONG (для SHORT — наоборот).

`skipped.jsonl`: `signal_id`, `why_skipped` (например, «риск: 9: стоп слишком близко», «цена входа уже за стопом или тейком»), `signal` (сам сигнал целиком).

## Как дать журнал ИИ на проверку

1. Вкладка «Журналы» → «скачать» нужный файл (или возьмите файлы из папки бэктеста).
2. Приложите файл к сообщению и опишите, что проверить. Примеры вопросов:
   - «Это журнал paper-бота. Найди сделки, где вход был не по правилам стратегии из docs/09 (раздел «Правила стратегий v1»).»
   - «Почему бот за неделю не открыл ни одной сделки? Посмотри `risk_decision` и `bar`.»
   - «Сверь trades.jsonl бэктеста: правильно ли посчитаны комиссии, funding и R для каждой сделки (настройки — в run.json).»
3. Большой журнал можно сократить: нужен период — возьмите только строки с этими датами (каждая строка самостоятельна).
