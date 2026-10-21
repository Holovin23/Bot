# Журнал открытий hold-out

Каждая строка — один взгляд на hold-out. Если после взгляда стратегия менялась, hold-out для неё «сгорел».

| Дата (UTC) | Стратегия | Символ | Таймфрейм | Параметры |
|---|---|---|---|---|
| 2026-10-13 16:36 | trend_following | BTCUSDT | 1h | ema_fast=20, ema_mid=50, ema_slow=200, atr_length=14, stop_atr=2, trail_atr_length=22, trail_atr=3, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 16:37 | mean_reversion | BTCUSDT | 1h | bb_length=20, bb_std=2, rsi_length=14, rsi_low=30, rsi_high=70, adx_length=14, adx_max=20, atr_length=14, stop_atr=1.5, time_stop_bars=24, min_rr=1, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 16:39 | breakout | BTCUSDT | 1h | bb_length=20, bb_std=2, squeeze_lookback=100, squeeze_percentile=0.1, channel_length=20, volume_length=20, volume_mult=1.5, take_r=2, time_stop_bars=48, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 16:40 | composite | BTCUSDT | 1h | mode=regime, members=mean_reversion,trend_following,breakout (бэктест из интерфейса) |
| 2026-10-13 16:41 | random_entry | BTCUSDT | 1h | probability=0.05, stop_atr=1.5, tp_atr=2, time_stop_bars=24, atr_length=14, seed=1 (бэктест из интерфейса) |
| 2026-10-13 16:44 | mean_reversion | ETHUSDT | 1h | bb_length=20, bb_std=2, rsi_length=14, rsi_low=30, rsi_high=70, adx_length=14, adx_max=20, atr_length=14, stop_atr=1.5, time_stop_bars=24, min_rr=1, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 16:45 | breakout | ETHUSDT | 1h | bb_length=20, bb_std=2, squeeze_lookback=100, squeeze_percentile=0.1, channel_length=20, volume_length=20, volume_mult=1.5, take_r=2, time_stop_bars=48, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 16:45 | composite | ETHUSDT | 5m | mode=regime, members=mean_reversion,trend_following,breakout (бэктест из интерфейса) |
| 2026-10-13 17:02 | trend_following | BTCUSDT | 1h | ema_fast=20, ema_mid=50, ema_slow=200, atr_length=14, stop_atr=2, trail_atr_length=22, trail_atr=3, allow_short=True (бэктест из интерфейса) |
| 2026-10-13 17:02 | mean_reversion | ETHUSDT | 1h | bb_length=20, bb_std=2, rsi_length=14, rsi_low=30, rsi_high=70, adx_length=14, adx_max=20, atr_length=14, stop_atr=1.5, time_stop_bars=24, min_rr=1, allow_short=True (бэктест из интерфейса) |
