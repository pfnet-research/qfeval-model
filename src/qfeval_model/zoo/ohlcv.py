"""OHLCV primitives: true range, candle geometry, and price-volume stats.

Every input accepts a feature name or a child model per source; a NaN in
any open/high/low/close cell makes the whole price tuple NaN for that cell,
``low > high`` and negative volume raise ValueError, and candle geometry
that would be negative (``high < max(open, close)`` or
``low > min(open, close)``) also raises rather than silently clipping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._ohlcv import read_ohlcv, register_ohlcv
from qfeval_model.zoo._source import finite_or_nan, lagged_value, valid_or_nan
from qfeval_model.zoo._window import validate_window, window_multi_sum, window_sum


def _require_non_negative(values: npt.NDArray[Any], message: str) -> None:
    if (values < 0.0).any():
        raise ValueError(message)


@model
def true_range(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
) -> None:
    """True range: ``max(high - low, |high - prev_close|, |low - prev_close|)``.

    The first step (and any step after a missing close) is NaN because the
    previous close is unknown.
    """
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        prev_close = lagged_value(ctx.state, "prev_close", values["close"], 1)
        with np.errstate(over="ignore", invalid="ignore"):
            true_range = np.maximum(
                values["high"] - values["low"],
                np.maximum(
                    np.abs(values["high"] - prev_close),
                    np.abs(values["low"] - prev_close),
                ),
            )
        step.output = finite_or_nan(true_range)


@model
def typical_price(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
) -> None:
    """Typical price ``(high + low + close) / 3``."""
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = values["high"] / 3.0 + values["low"] / 3.0 + values["close"] / 3.0


@model
def weighted_close(
    ctx: Context,
    high: str | Model = "high",
    low: str | Model = "low",
    close: str | Model = "close",
) -> None:
    """Weighted close ``(high + low + 2 * close) / 4``."""
    sources = {"high": high, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        step.output = values["high"] / 4.0 + values["low"] / 4.0 + values["close"] / 2.0


@model
def rolling_volume_weighted_mean(
    ctx: Context,
    price: str | Model = "close",
    volume: str | Model = "volume",
    window: int = 20,
) -> None:
    """Volume-weighted mean price ``sum(price * volume) / sum(volume)``.

    Strict window; a NaN price or volume in the window poisons the cell,
    and a zero window volume yields NaN.
    """
    validate_window(window)
    sources = {"price": price, "volume": volume}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            traded_value = values["price"] * values["volume"]
        totals, count = window_multi_sum(
            ctx.state,
            np.stack((traded_value, values["volume"])),
            window,
            key="price_volume",
        )
        total_traded, total_volume = totals
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = total_traded / total_volume
        step.output = valid_or_nan(mean, count == window)


@model
def amihud_illiquidity(
    ctx: Context,
    source: str | Model,
    price: str | Model = "close",
    volume: str | Model = "volume",
    window: int = 20,
) -> None:
    """Amihud illiquidity: rolling ``mean(abs(return) / (price * volume))``.

    ``source`` is the per-step return series (compose ``simple_return()``
    explicitly — no return type is assumed). Strict window; a zero traded
    value ``price * volume`` yields NaN.
    """
    validate_window(window)
    sources = {"source": source, "price": price, "volume": volume}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = np.abs(values["source"]) / (values["price"] * values["volume"])
        total, count = window_sum(ctx.state, ratio, window)
        total /= window
        step.output = valid_or_nan(total, count == window)


@model
def candle_signed_body(
    ctx: Context, open: str | Model = "open", close: str | Model = "close"
) -> None:
    """Signed candle body ``close - open`` (negative on a down candle)."""
    sources = {"open": open, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            body = values["close"] - values["open"]
        step.output = finite_or_nan(body)


@model
def candle_real_body(
    ctx: Context, open: str | Model = "open", close: str | Model = "close"
) -> None:
    """Candle real body ``abs(close - open)``."""
    sources = {"open": open, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            body = np.abs(values["close"] - values["open"])
        step.output = finite_or_nan(body)


@model
def candle_upper_shadow(
    ctx: Context,
    open: str | Model = "open",
    high: str | Model = "high",
    close: str | Model = "close",
) -> None:
    """Upper shadow ``high - max(open, close)`` (ValueError if negative)."""
    sources = {"open": open, "high": high, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            shadow = values["high"] - np.maximum(values["open"], values["close"])
        shadow = finite_or_nan(shadow)
        _require_non_negative(shadow, "invalid OHLC bar: high < max(open, close)")
        step.output = shadow


@model
def candle_lower_shadow(
    ctx: Context,
    open: str | Model = "open",
    low: str | Model = "low",
    close: str | Model = "close",
) -> None:
    """Lower shadow ``min(open, close) - low`` (ValueError if negative)."""
    sources = {"open": open, "low": low, "close": close}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            shadow = np.minimum(values["open"], values["close"]) - values["low"]
        shadow = finite_or_nan(shadow)
        _require_non_negative(shadow, "invalid OHLC bar: low > min(open, close)")
        step.output = shadow


@model
def candle_range(
    ctx: Context, high: str | Model = "high", low: str | Model = "low"
) -> None:
    """Candle range ``high - low`` (ValueError on ``low > high``)."""
    sources = {"high": high, "low": low}
    register_ohlcv(ctx, sources)
    for step in ctx:
        values = read_ohlcv(step, ctx.state, sources)
        with np.errstate(over="ignore", invalid="ignore"):
            candle_range = values["high"] - values["low"]
        step.output = finite_or_nan(candle_range)
