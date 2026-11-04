# How to use technical indicators

The technical entries in `qm.zoo` are ordinary models: deterministic,
checkpoint-ready, and strictly backward-looking. Each indicator line is
its own factory, so any line can be another model's `source`.

## OHLCV mapping

OHLCV entries default to the feature names `"open"`, `"high"`, `"low"`,
`"close"`, `"volume"` — pass your own inputs as
`{"high": high_frame, "low": low_frame, ...}` or override per input
(`stochastic_percent_k(high=my_model, ...)`). Per-bar integrity is
enforced: `low <= high`, non-negative volume, and a NaN in any
open/high/low/close cell blanks that whole price tuple. Bar frequency,
sessions, and corporate-action adjustment are your input pipeline's job.

## Which variant is implemented

The formulas follow the plain rolling-sum/rolling-mean textbook variants,
and the names say so — TA-Lib compatibility is not a goal:

- `rolling_relative_strength_index` — RSI with simple `window`-mean gains
  and losses (not Wilder smoothing; a Wilder variant would be a separate
  `wilder_*` name). All-gain windows are 100, all-loss 0, flat 50.
- `rolling_positive_directional_indicator`, `..._negative_...`,
  `rolling_directional_index`, `rolling_average_directional_index` — the
  rolling-sum DMI family: `DI = 100 * sum(DM) / sum(true_range)`,
  `DX = 100 * |+DI - -DI| / (+DI + -DI)`, `ADX = rolling_mean(DX)`.
  Tied up/down moves count as zero on both sides.
- `stochastic_percent_k` / `stochastic_percent_d` /
  `slow_stochastic_percent_d` — `%K` is close's position in the window's
  high-low range on 0–100; `%D` is its `smooth`-mean (slow `%K` equals
  `%D`, so it has no separate factory); slow `%D` smooths twice.
- `psychological_line` — up-step rate on 0–100; flat steps count in the
  denominator only.
- `rank_correlation_index` — Spearman rho of price against time order,
  on -100–100 (ties averaged; a constant window is NaN).
- `moving_average_deviation_percent` — percent scale (5% is `5.0`); use
  `divide`/`subtract` composition if you need the fraction.
- `macd_line` / `macd_signal` / `macd_histogram` — recursive EWMs with
  `alpha = 2 / (span + 1)`, seeded at the first finite price. Warm-up is
  explicit: the line is NaN until `slow_span` finite prices, the signal
  and histogram until `signal_span` further finite line values. Not
  TA-Lib compatible.
- `bollinger_*` — sample std (`ddof=1`) by default; `percent_b` and
  `bandwidth` are fractions and are never clamped.
- `moving_average_envelope_*` — `width` is a fraction (5% is `0.05`).
- Moving averages: `rolling_mean` (SMA), `recursive_ewm_mean`
  (`adjust=False`, first-finite seed), `linear_weighted_moving_average`
  (weights `1..window`, newest heaviest). A volume moving average is just
  `rolling_mean("volume", window)` — no dedicated entry.

## Ichimoku and causal alignment

`ichimoku_conversion_line`, `ichimoku_base_line`, and the two leading
spans are implemented; the lagging span is not, because plotting today's
close at a past timestamp (or reading a future close at today's) breaks
the available-at contract. The leading spans are aligned causally: the
output at chart time `t` is the span value computed `displacement` steps
earlier — exactly what a chart displays at `t` today, never a value
written into the future.

```python
span_a = qm.zoo.ichimoku_leading_span_a()      # 9/26 windows, shifted 26
span_b = qm.zoo.ichimoku_leading_span_b()      # 52 window, shifted 26
```

## Warm-up

Strict windows apply everywhere: an indicator over `window` steps is NaN
until the window is full, differencing indicators (RSI, psychological
line, DMI) need one extra bar, smoothed lines add their smoothing windows
(`ADX` first appears after `2 * window` bars), and displaced spans add
`displacement` steps.

## No signals

No entry emits buy/sell decisions — `rsi > 70` is one line in your own
model, and which threshold (and which bar) constitutes a signal is a
strategy decision the zoo deliberately does not make. Golden/dead-cross
helpers are omitted for the same reason: comparing two moving averages is
a one-liner with `subtract`.
