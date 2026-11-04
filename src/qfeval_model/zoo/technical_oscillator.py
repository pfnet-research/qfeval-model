"""Oscillator technical indicators (0-100 or -100 to 100 market scales).

Unlike the rest of the zoo (which favors ``[0, 1]`` fractions), these
indicators keep their conventional market scales — RSI, stochastics, and
the psychological line output 0-100, RCI outputs -100 to 100 — as their names
and docstrings state. No entry emits buy/sell decisions; thresholding is
the caller's model. All entries follow the strict rolling contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import DEFAULT_FEATURE, Context, Model, model
from qfeval_model.zoo._ohlcv import read_ohlcv, register_ohlcv, window_high_low
from qfeval_model.zoo._rank import average_ranks_axis0
from qfeval_model.zoo._source import (
    finite_or_nan,
    lagged_value,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    full_block,
    validate_window,
    window_boolean_count,
    window_multi_sum,
    window_sum,
)


def _delta(state: dict[str, Any], x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """One-step difference against the previous observation (NaN at start)."""
    prev = lagged_value(state, "prev", x, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        return finite_or_nan(x - prev)


def _rolling_mean(
    state: dict[str, Any], x: npt.NDArray[Any], window: int, key: str
) -> npt.NDArray[Any]:
    """Strict ``window``-mean of ``x``: NaN until the window is fully finite."""
    total, count = window_sum(state, x, window, key=key)
    total /= window
    return valid_or_nan(total, count == window)


@model
def rolling_relative_strength_index(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, window: int = 14
) -> None:
    """RSI on 0-100 with plain rolling means of gains and losses.

    ``RSI = 100 * mean(gain) / (mean(gain) + mean(loss))`` over the last
    ``window`` one-step deltas — the simple n-period average variant, not
    Wilder smoothing. All-gain windows are 100, all-loss windows are 0, and
    a flat window is the neutral 50. The first output needs ``window + 1``
    prices.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        delta = _delta(ctx.state, x)
        fields = np.stack((np.maximum(delta, 0.0), np.maximum(-delta, 0.0)))
        totals, count = window_multi_sum(ctx.state, fields, window, key="changes")
        gain_sum, loss_sum = totals
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            total = gain_sum + loss_sum
            rsi = 100.0 * gain_sum / total
        rsi[total == 0.0] = 50.0
        step.output = valid_or_nan(rsi, count == window)


def _percent_k(
    state: dict[str, Any],
    values: dict[str, npt.NDArray[Any]],
    window: int,
) -> npt.NDArray[Any]:
    """The raw %K line; NaN during warm-up and on a zero high-low range."""
    highest, lowest, full = window_high_low(state, values, window, "k")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        percent_k = 100.0 * (values["close"] - lowest) / (highest - lowest)
    return valid_or_nan(percent_k, full)


@model
def stochastic_percent_k(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
) -> None:
    """Stochastic %K on 0-100: close's position in the window's high-low range.

    ``100 * (close - lowest_low) / (highest_high - lowest_low)``; a zero
    range yields NaN. A close outside the range (inconsistent inputs) is
    emitted as-is beyond 0-100 rather than clamped.
    """
    validate_window(window)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = _percent_k(ctx.state, values, window)


@model
def stochastic_percent_d(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
    smooth: int = 3,
) -> None:
    """Stochastic %D on 0-100: the ``smooth``-step rolling mean of %K.

    Also known as slow %K; a zero high-low range anywhere in the smoothing
    window yields NaN.
    """
    validate_window(window)
    validate_window(smooth)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        percent_k = _percent_k(ctx.state, values, window)
        step.output = _rolling_mean(ctx.state, percent_k, smooth, "k_buffer")


@model
def slow_stochastic_percent_d(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> None:
    """Slow stochastic %D on 0-100: %K smoothed by ``smooth_k`` then ``smooth_d``.

    The intermediate ``smooth_k``-mean of %K is the slow %K, which equals
    ``stochastic_percent_d`` and therefore has no separate factory.
    """
    validate_window(window)
    validate_window(smooth_k)
    validate_window(smooth_d)
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        percent_k = _percent_k(ctx.state, values, window)
        slow_k = _rolling_mean(ctx.state, percent_k, smooth_k, "k_buffer")
        step.output = _rolling_mean(ctx.state, slow_k, smooth_d, "d_buffer")


@model
def psychological_line(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, window: int = 12
) -> None:
    """Psychological line on 0-100: the up-step rate over the window.

    ``100 * count(delta > 0) / window`` over the last ``window`` one-step
    deltas; a flat step counts in the denominator but not as an up step.
    The first output needs ``window + 1`` prices.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        delta = _delta(ctx.state, x)
        total, count, _ = window_boolean_count(
            ctx.state, delta > 0.0, window, valid=~np.isnan(delta)
        )
        rate = 100.0 * total / window
        step.output = valid_or_nan(rate, count == window)


@model
def rank_correlation_index(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, window: int = 9
) -> None:
    """RCI on -100 to 100: Spearman rank correlation of price against time.

    Time ranks run oldest=1 to newest=``window``; price ranks are ascending
    with ties averaged. A strictly rising window is +100, strictly falling
    is -100, and a constant window (zero rank variance) is NaN.
    """
    validate_window(window, minimum=2)
    register_source(ctx, source)
    centered_time = np.arange(1.0, window + 1.0)
    centered_time -= centered_time.mean()
    time_m2 = float(np.sum(centered_time**2))
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        ranks, counts = average_ranks_axis0(block)
        centered = ranks.reshape(window, -1) - (window + 1.0) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            comoment = np.sum(centered * centered_time[:, np.newaxis], axis=0)
            rank_m2 = np.sum(centered**2, axis=0)
            rho = 100.0 * comoment / np.sqrt(rank_m2 * time_m2)
        step.output = valid_or_nan(rho, counts == window).reshape(x.shape)


@model
def moving_average_deviation_percent(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, window: int = 25
) -> None:
    """Deviation from the moving average in percent (5% is ``5.0``).

    ``100 * (source / rolling_mean(source, window) - 1)``; a zero moving
    average yields NaN. Compose ``divide``/``subtract`` instead if you need
    the fraction scale.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, count = window_sum(ctx.state, x, window)
        mean = total / window
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            deviation = 100.0 * (x / mean - 1.0)
        step.output = valid_or_nan(deviation, (count == window) & (mean != 0.0))
