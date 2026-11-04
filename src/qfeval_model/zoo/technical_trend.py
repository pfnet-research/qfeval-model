"""Trend-following technical indicators: bands, envelopes, Ichimoku, DMI.

Multi-line indicators are split into one factory per line (upper band,
conversion line, +DI, ...) so each line is an ordinary model that can feed
another model's ``source``. All lines follow the strict rolling contract
(NaN until the window is full, NaN in a window poisons that cell). The
Ichimoku leading spans output, at each chart timestamp, the span value
computed ``displacement`` steps earlier — the causal reading of "plotted
ahead" — and the non-causal lagging span is deliberately absent.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from qfeval_model.core import DEFAULT_FEATURE, Context, Model, model
from qfeval_model.zoo._ohlcv import read_ohlcv, register_ohlcv, window_high_low
from qfeval_model.zoo._source import (
    finite_or_nan,
    lagged_value,
    recursive_ewm_update,
    register_source,
    source_value,
    state_array,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    validate_ddof,
    validate_integer,
    validate_window,
    window_moments,
    window_multi_sum,
    window_sum,
)


def _validate_deviations(deviations: float) -> None:
    if not np.isfinite(deviations) or deviations <= 0.0:
        raise ValueError(
            f"deviations must be a positive finite number, got {deviations!r}"
        )


def _validate_envelope_width(width: float) -> None:
    if not 0.0 <= width < 1.0:
        raise ValueError(
            f"width must be a fraction in [0, 1) — 5% is 0.05 — got {width!r}"
        )


def _envelope_model(
    ctx: Context, source: str | Model, window: int, width: float, sign: float
) -> None:
    """Shared scaffold for the two envelope lines (``sign`` is +1.0 or -1.0)."""
    validate_window(window)
    _validate_envelope_width(width)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, count = window_sum(ctx.state, x, window)
        with np.errstate(over="ignore", invalid="ignore"):
            line = total / window * (1.0 + sign * width)
        step.output = valid_or_nan(line, count == window)


def _rolling_mean_std(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    ddof: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Return strict rolling mean, std, and validity mask."""
    mean, m2, count = window_moments(state, x, window)
    return mean, np.sqrt(m2 / (window - ddof)), count == window


@model
def bollinger_upper_band(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 20,
    deviations: float = 2.0,
    ddof: int = 1,
) -> None:
    """Bollinger upper band ``rolling_mean + deviations * rolling_std``.

    The middle band is plain ``rolling_mean(source, window)``; the default
    ``ddof=1`` matches the sample standard deviation used by the textbook
    formula.
    """
    validate_ddof(window, ddof)
    _validate_deviations(deviations)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        mean, std, valid = _rolling_mean_std(ctx.state, x, window, ddof)
        with np.errstate(over="ignore", invalid="ignore"):
            upper = mean + deviations * std
        step.output = valid_or_nan(upper, valid)


@model
def bollinger_lower_band(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 20,
    deviations: float = 2.0,
    ddof: int = 1,
) -> None:
    """Bollinger lower band ``rolling_mean - deviations * rolling_std``."""
    validate_ddof(window, ddof)
    _validate_deviations(deviations)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        mean, std, valid = _rolling_mean_std(ctx.state, x, window, ddof)
        with np.errstate(over="ignore", invalid="ignore"):
            lower = mean - deviations * std
        step.output = valid_or_nan(lower, valid)


@model
def bollinger_bandwidth(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 20,
    deviations: float = 2.0,
    ddof: int = 1,
) -> None:
    """Bollinger bandwidth ``(upper - lower) / middle`` (fraction, not percent).

    A zero middle band yields NaN.
    """
    validate_ddof(window, ddof)
    _validate_deviations(deviations)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        middle, std, valid = _rolling_mean_std(ctx.state, x, window, ddof)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            width = 2.0 * deviations * std / middle
        step.output = valid_or_nan(width, valid & (middle != 0.0))


@model
def bollinger_percent_b(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 20,
    deviations: float = 2.0,
    ddof: int = 1,
) -> None:
    """Bollinger %B ``(source - lower) / (upper - lower)``.

    Despite the conventional name, the output is a fraction (0.5 at the
    middle band, outside ``[0, 1]`` beyond the bands — never clamped); a
    zero band width yields NaN.
    """
    validate_ddof(window, ddof)
    _validate_deviations(deviations)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        mean, std, valid = _rolling_mean_std(ctx.state, x, window, ddof)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lower = mean - deviations * std
            percent_b = (x - lower) / (2.0 * deviations * std)
        step.output = valid_or_nan(percent_b, valid & (std > 0.0))


@model
def moving_average_envelope_upper(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 25,
    width: float = 0.05,
) -> None:
    """Envelope upper line ``rolling_mean * (1 + width)``.

    ``width`` is a fraction (5% is ``0.05``), never a percent value.
    """
    _envelope_model(ctx, source, window, width, 1.0)


@model
def moving_average_envelope_lower(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    window: int = 25,
    width: float = 0.05,
) -> None:
    """Envelope lower line ``rolling_mean * (1 - width)``.

    ``width`` is a fraction (5% is ``0.05``), never a percent value.
    """
    _envelope_model(ctx, source, window, width, -1.0)


def _midpoint(
    state: dict[str, Any],
    values: dict[str, npt.NDArray[Any]],
    window: int,
    key: str,
) -> npt.NDArray[Any]:
    """``(highest high + lowest low) / 2`` over the window (NaN in warm-up)."""
    highest, lowest, full = window_high_low(state, values, window, key)
    return valid_or_nan(highest / 2.0 + lowest / 2.0, full)


@model
def ichimoku_conversion_line(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    window: int = 9,
) -> None:
    """Ichimoku conversion line: the 9-step high-low midpoint by default.

    ``(rolling_max(high, window) + rolling_min(low, window)) / 2``.
    """
    validate_window(window)
    sources = {"high": high, "low": low}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = _midpoint(ctx.state, values, window, "mid")


@model
def ichimoku_base_line(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    window: int = 26,
) -> None:
    """Ichimoku base line: the 26-step high-low midpoint by default.

    ``(rolling_max(high, window) + rolling_min(low, window)) / 2``.
    """
    validate_window(window)
    sources = {"high": high, "low": low}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = _midpoint(ctx.state, values, window, "mid")


@model
def ichimoku_leading_span_a(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    conversion_window: int = 9,
    base_window: int = 26,
    displacement: int = 26,
) -> None:
    """Ichimoku leading span A, aligned causally to the chart timestamp.

    The output at time ``t`` is ``(conversion + base) / 2`` as computed at
    ``t - displacement`` — the value a chart displays at ``t`` today, never
    a value written into the future.
    """
    validate_window(conversion_window)
    validate_window(base_window)
    validate_integer(displacement, "displacement")
    sources = {"high": high, "low": low}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        conversion = _midpoint(ctx.state, values, conversion_window, "conversion")
        base = _midpoint(ctx.state, values, base_window, "base")
        span = conversion / 2.0 + base / 2.0
        step.output = lagged_value(ctx.state, "displaced", span, displacement)


@model
def ichimoku_leading_span_b(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    window: int = 52,
    displacement: int = 26,
) -> None:
    """Ichimoku leading span B, aligned causally to the chart timestamp.

    The output at time ``t`` is the ``window``-step high-low midpoint as
    computed at ``t - displacement``.
    """
    validate_window(window)
    validate_integer(displacement, "displacement")
    sources = {"high": high, "low": low}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        midpoint = _midpoint(ctx.state, values, window, "mid")
        step.output = lagged_value(ctx.state, "displaced", midpoint, displacement)


def _directional_moves(
    state: dict[str, Any], values: dict[str, npt.NDArray[Any]]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Per-step (+DM, -DM, true range); NaN on the first step of a cell."""
    high, low, close = values["high"], values["low"], values["close"]
    prev_high = state_array(state, "prev_high", high, np.nan)
    prev_low = state_array(state, "prev_low", low, np.nan)
    prev_close = state_array(state, "prev_close", close, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        up = high - prev_high
        down = prev_low - low
    valid = ~np.isnan(up) & ~np.isnan(down)
    plus_dm = valid_or_nan(np.where((up > 0.0) & (up > down), up, 0.0), valid)
    minus_dm = valid_or_nan(np.where((down > 0.0) & (down > up), down, 0.0), valid)
    with np.errstate(over="ignore", invalid="ignore"):
        true_range = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
    true_range = finite_or_nan(true_range)
    np.copyto(prev_high, high)
    np.copyto(prev_low, low)
    np.copyto(prev_close, close)
    return plus_dm, minus_dm, true_range


def _directional_indicators(
    state: dict[str, Any],
    values: dict[str, npt.NDArray[Any]],
    window: int,
    side: Literal["positive", "negative", "both"] = "both",
) -> list[npt.NDArray[Any]]:
    """The requested DIs on 0-100; NaN in warm-up and on a zero true-range sum."""
    plus_dm, minus_dm, true_range = _directional_moves(state, values)
    dms = {"positive": [plus_dm], "negative": [minus_dm], "both": [plus_dm, minus_dm]}
    fields = np.stack([*dms[side], true_range])
    totals, count = window_multi_sum(state, fields, window, key="directional")
    *dm_sums, tr_sum = totals
    valid = (count == window) & (tr_sum != 0.0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return [valid_or_nan(100.0 * dm_sum / tr_sum, valid) for dm_sum in dm_sums]


def _directional_index(
    state: dict[str, Any],
    values: dict[str, npt.NDArray[Any]],
    window: int,
) -> npt.NDArray[Any]:
    """DX on 0-100; NaN when ``+DI + -DI`` is zero."""
    plus_di, minus_di = _directional_indicators(state, values, window)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        total = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / total
    return valid_or_nan(dx, total != 0.0)


@model
def rolling_positive_directional_indicator(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
) -> None:
    """+DI on 0-100, the rolling-sum DMI variant (not Wilder smoothing).

    ``100 * rolling_sum(+DM, window) / rolling_sum(true_range, window)``
    where ``+DM = max(high - prev_high, 0)`` only when it exceeds the down
    move (ties are zero). The first output needs ``window + 1`` bars; a
    zero true-range sum yields NaN.
    """
    validate_window(window)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        (step.output,) = _directional_indicators(
            ctx.state, values, window, side="positive"
        )


@model
def rolling_negative_directional_indicator(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
) -> None:
    """-DI on 0-100, the rolling-sum DMI variant (not Wilder smoothing).

    ``100 * rolling_sum(-DM, window) / rolling_sum(true_range, window)``
    where ``-DM = max(prev_low - low, 0)`` only when it exceeds the up move
    (ties are zero).
    """
    validate_window(window)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        (step.output,) = _directional_indicators(
            ctx.state, values, window, side="negative"
        )


@model
def rolling_directional_index(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
) -> None:
    """DX on 0-100: ``100 * |+DI - -DI| / (+DI + -DI)`` (zero sum is NaN)."""
    validate_window(window)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = _directional_index(ctx.state, values, window)


@model
def rolling_average_directional_index(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
) -> None:
    """ADX on 0-100: the ``window``-step rolling mean of DX.

    The first output needs ``2 * window`` bars (one window to form DX, one
    to average it).
    """
    validate_window(window)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        dx = _directional_index(ctx.state, values, window)
        total, count = window_sum(ctx.state, dx, window, key="dx")
        step.output = valid_or_nan(total / window, count == window)


def _validate_macd_spans(fast_span: int, slow_span: int) -> None:
    if not 1 <= fast_span < slow_span:
        raise ValueError(
            f"spans must satisfy 1 <= fast_span < slow_span, "
            f"got {fast_span} and {slow_span}"
        )
    validate_integer(fast_span, "fast_span", 1)
    validate_integer(slow_span, "slow_span", 1)


def _span_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _macd_line_value(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    fast_span: int,
    slow_span: int,
) -> npt.NDArray[Any]:
    """The MACD line, NaN until ``slow_span`` finite prices have been seen.

    The EWM states update from the very first finite price; only the output
    is masked during warm-up (this is the contract fixed by the plan's MACD
    RFC — see the ``macd_line`` docstring).
    """
    count = state_array(state, "price_count", x, 0.0)
    np.add(count, ~np.isnan(x), out=count)
    fast = recursive_ewm_update(state, "fast", x, _span_alpha(fast_span))
    slow = recursive_ewm_update(state, "slow", x, _span_alpha(slow_span))
    with np.errstate(over="ignore", invalid="ignore"):
        line = fast - slow
    return valid_or_nan(line, count >= slow_span)


def _macd_signal_values(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    fast_span: int,
    slow_span: int,
    signal_span: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """Update MACD line and signal state and return their readiness mask."""
    line = _macd_line_value(state, x, fast_span, slow_span)
    signal = recursive_ewm_update(state, "signal", line, _span_alpha(signal_span))
    line_count = state_array(state, "line_count", x, 0.0)
    np.add(line_count, ~np.isnan(line), out=line_count)
    return line, signal, line_count >= signal_span


@model
def macd_line(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    fast_span: int = 12,
    slow_span: int = 26,
) -> None:
    """MACD line: fast recursive EWM minus slow recursive EWM.

    ``alpha(span) = 2 / (span + 1)``; both EWMs seed at the first finite
    price and carry through missing inputs, exactly like
    ``recursive_ewm_mean``. Warm-up contract: the states update from the
    start, but the line is NaN until ``slow_span`` finite prices have been
    observed. After warm-up a missing price carries the previous line value
    instead of poisoning it. Not TA-Lib compatible.
    """
    _validate_macd_spans(fast_span, slow_span)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        step.output = _macd_line_value(ctx.state, x, fast_span, slow_span)


@model
def macd_signal(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> None:
    """MACD signal: the ``signal_span`` recursive EWM of the MACD line.

    The signal EWM consumes the (warm-up-masked) line values and is itself
    NaN until ``signal_span`` finite line values have accumulated on top of
    the line's own ``slow_span`` warm-up.
    """
    _validate_macd_spans(fast_span, slow_span)
    validate_integer(signal_span, "signal_span", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, signal, ready = _macd_signal_values(
            ctx.state, x, fast_span, slow_span, signal_span
        )
        step.output = np.where(ready, signal, np.nan)


@model
def macd_histogram(
    ctx: Context,
    source: str | Model = DEFAULT_FEATURE,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> None:
    """MACD histogram: the MACD line minus its signal line.

    NaN until the signal line's warm-up completes (``slow_span`` finite
    prices plus ``signal_span`` finite line values).
    """
    _validate_macd_spans(fast_span, slow_span)
    validate_integer(signal_span, "signal_span", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        line, signal, ready = _macd_signal_values(
            ctx.state, x, fast_span, slow_span, signal_span
        )
        with np.errstate(over="ignore", invalid="ignore"):
            histogram = line - signal
        step.output = valid_or_nan(histogram, ready)
