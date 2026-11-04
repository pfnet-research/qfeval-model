"""Backward-looking risk statistics: drawdowns, historical VaR, and ES.

Sign conventions: return/P&L sources are positive for profit and negative
for loss; the loss sample used by VaR / Expected Shortfall is ``L = -x``.
Drawdown sources are positive wealth or price levels — returns are never
implicitly compounded into wealth here (compose
``rolling_compounded_return`` explicitly instead).
"""

from __future__ import annotations

import math

import numpy as np

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import (
    register_source,
    source_value,
    state_array,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    full_block,
    validate_window,
    window_extrema,
    window_max_drawdown,
    window_product,
    window_sum,
)


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be in (0, 1) — e.g. 0.95, not a percent value "
            f"like 95 — got {confidence!r}"
        )


@model
def rolling_compounded_return(ctx: Context, source: str | Model, window: int) -> None:
    """Compounded return ``product(1 + r) - 1`` over ``window`` return steps."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        product, count = window_product(ctx.state, 1.0 + x, window)
        step.output = valid_or_nan(product - 1.0, count == window)


@model
def rolling_worst_loss(ctx: Context, source: str | Model, window: int) -> None:
    """The worst loss ``-min(x)`` of a return/P&L source over ``window`` steps.

    Positive when the window contains a loss; negative when even the worst
    step was a profit.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        minimum, _, count = window_extrema(ctx.state, x, window, direction="min")
        step.output = valid_or_nan(-minimum, count == window)


@model
def drawdown(ctx: Context, source: str | Model) -> None:
    """Drawdown ``1 - x / running_max`` of a positive wealth/price source.

    The running maximum is over all finite observations so far; NaN inputs
    stay NaN without resetting the peak, and a non-positive running peak
    yields NaN.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        peak = state_array(ctx.state, "peak", x, np.nan)
        np.fmax(peak, x, out=peak)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 1.0 - x / peak
        step.output = valid_or_nan(dd, peak > 0.0)


@model
def rolling_drawdown(ctx: Context, source: str | Model, window: int) -> None:
    """Drawdown of the current value from its trailing ``window`` peak.

    ``1 - x / rolling_max``; strict window, non-positive peaks yield NaN.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, peak, count = window_extrema(ctx.state, x, window, direction="max")
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 1.0 - x / peak
        step.output = valid_or_nan(dd, (count == window) & (peak > 0.0))


@model
def rolling_max_drawdown(ctx: Context, source: str | Model, window: int) -> None:
    """The largest peak-to-subsequent-trough drawdown within the window.

    ``max_j (1 - x_j / max_{i <= j} x_i)`` over the window. The source
    contract is positive wealth/price (NaN is allowed for missing data); a
    finite non-positive cell raises ``ValueError``. The rolling window is
    strict, so a NaN in it yields NaN.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        if (x <= 0.0).any():
            raise ValueError("rolling_max_drawdown source must be positive or NaN")
        ratio, count = window_max_drawdown(ctx.state, x, window)
        step.output = valid_or_nan(1.0 - ratio, count == window)


@model
def drawdown_duration(ctx: Context, source: str | Model) -> None:
    """Steps since the running maximum was last set or matched (0 at a peak).

    NaN inputs extend the duration (the peak was not revisited); the output
    is NaN until the first finite observation.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        peak = state_array(ctx.state, "peak", x, np.nan)
        np.fmax(peak, x, out=peak)
        duration = state_array(ctx.state, "duration", x, np.nan)
        np.add(duration, 1.0, out=duration)
        np.copyto(duration, 0.0, where=x >= peak)
        step.output = duration


@model
def rolling_downside_deviation(
    ctx: Context, source: str | Model, window: int, threshold: float = 0.0
) -> None:
    """Square root of the order-2 lower partial moment below ``threshold``.

    ``sqrt(mean(max(threshold - x, 0) ** 2))`` over the window.
    """
    validate_window(window)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        with np.errstate(over="ignore", invalid="ignore"):
            squared = np.subtract(threshold, x)
            np.maximum(squared, 0.0, out=squared)
            np.square(squared, out=squared)
        total, count = window_sum(ctx.state, squared, window)
        deviation = np.sqrt(total / window)
        step.output = valid_or_nan(deviation, count == window)


@model
def rolling_historical_value_at_risk(
    ctx: Context, source: str | Model, window: int, confidence: float
) -> None:
    """Historical VaR of a return/P&L source at ``confidence`` in ``(0, 1)``.

    With the loss sample ``L = -x`` over the window,
    ``VaR_c = inf {l | F_n(l) >= c}`` — the empirical ``inverted_cdf``
    quantile, with no interpolation options. Positive means a loss; a
    profit-only window yields a negative VaR (never clamped to zero). No
    horizon scaling or annualization is applied.
    """
    validate_window(window)
    _validate_confidence(confidence)
    register_source(ctx, source)
    order = window - math.ceil(confidence * window)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        selected = -np.partition(block, order, axis=0)[order]
        step.output = valid_or_nan(selected, ~np.isnan(block).any(axis=0))


@model
def rolling_historical_expected_shortfall(
    ctx: Context, source: str | Model, window: int, confidence: float
) -> None:
    """Historical Expected Shortfall (CVaR) at ``confidence`` in ``(0, 1)``.

    ``ES_c = integral[c, 1](VaR_u du) / (1 - c)`` on the empirical loss
    distribution ``L = -x``: the mean of the largest ``window * (1 -
    confidence)`` losses, where the boundary sample enters with fractional
    weight (this is why ``ES >= VaR`` always holds, with equality on a
    constant loss). Positive means a loss; nothing is clamped, scaled, or
    annualized.
    """
    validate_window(window)
    _validate_confidence(confidence)
    register_source(ctx, source)
    # Tail-mass weights per descending-sorted loss: the top ``full`` samples
    # count 1, the boundary sample counts ``ceil(c*n) - c*n``.
    full = window - math.ceil(confidence * window)
    fraction = math.ceil(confidence * window) - confidence * window
    tail_mass = window * (1.0 - confidence)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        # Largest losses are the smallest returns.  Select only the boundary
        # order statistic instead of fully sorting every cell's window.
        scale = np.abs(block).max(axis=0)
        normalized = np.divide(
            block,
            scale,
            out=np.zeros(block.shape, dtype=np.float64),
            where=scale > 0.0,
        )
        selected = np.partition(normalized, full, axis=0)
        total = selected[:full].sum(axis=0)
        total += fraction * selected[full]
        step.output = -total / tail_mass * scale
