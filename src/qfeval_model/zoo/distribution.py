"""Rolling distribution statistics: quantiles, ranks, and threshold rates.

All entries follow the strict rolling contract (NaN until the window is
full, NaN in a window poisons that cell). Quantile methods are restricted
to ``"linear"`` and ``"inverted_cdf"`` — arbitrary strings are not passed
through to NumPy.
"""

from __future__ import annotations

import numpy as np

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import (
    finite_or_nan,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    full_block,
    validate_integer,
    validate_quantile_method,
    validate_window,
    window_boolean_count,
    window_extrema,
    window_sum,
)


@model
def rolling_quantile(
    ctx: Context,
    source: str | Model,
    window: int,
    q: float,
    *,
    method: str,
) -> None:
    """Rolling sample quantile ``q`` over ``window`` steps.

    ``method`` is required: ``"linear"`` interpolates between order
    statistics, ``"inverted_cdf"`` returns the smallest sample at or above
    the requested probability mass.
    """
    validate_window(window)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    chosen = validate_quantile_method(method)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        with np.errstate(over="ignore", invalid="ignore"):
            quantile = np.quantile(block, q, axis=0, method=chosen)
        step.output = finite_or_nan(quantile)


@model
def rolling_rank(ctx: Context, source: str | Model, window: int) -> None:
    """Percentile rank of the current value within its ``window`` in ``[0, 1]``.

    Ties are averaged (the current value counts as a window member, so the
    output of a one-step window is 0.5); a NaN in the window yields NaN.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        if window == 1:
            step.output = np.where(np.isnan(x), np.nan, 0.5)
            continue
        below = (block < x).sum(axis=0)
        equal = (block == x).sum(axis=0)
        rank = (below + (equal - 1) / 2.0) / (window - 1)
        step.output = valid_or_nan(rank, ~np.isnan(block).any(axis=0))


def _threshold_rate(
    ctx: Context, source: str | Model, window: int, threshold: float, compare: np.ufunc
) -> None:
    """Fraction of the window strictly on one side of ``threshold``."""
    validate_window(window)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, count, _ = window_boolean_count(
            ctx.state, compare(x, threshold), window, valid=~np.isnan(x)
        )
        step.output = valid_or_nan(total / window, count == window)


@model
def rolling_below_threshold_rate(
    ctx: Context, source: str | Model, window: int, threshold: float
) -> None:
    """Fraction of window values strictly below the finite ``threshold``."""
    _threshold_rate(ctx, source, window, threshold, np.less)


@model
def rolling_above_threshold_rate(
    ctx: Context, source: str | Model, window: int, threshold: float
) -> None:
    """Fraction of window values strictly above the finite ``threshold``."""
    _threshold_rate(ctx, source, window, threshold, np.greater)


@model
def rolling_lower_partial_moment(
    ctx: Context,
    source: str | Model,
    window: int,
    threshold: float = 0.0,
    order: int = 1,
) -> None:
    """Rolling ``mean(max(threshold - x, 0) ** order)`` over ``window`` steps.

    ``order=2`` is the semivariance below ``threshold`` (its square root is
    ``rolling_downside_deviation``).
    """
    validate_window(window)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    validate_integer(order, "order", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        with np.errstate(over="ignore", invalid="ignore"):
            partial = np.subtract(threshold, x)
            np.maximum(partial, 0.0, out=partial)
            if order == 2:
                np.square(partial, out=partial)
            elif order != 1:
                np.power(partial, order, out=partial)
        total, count = window_sum(ctx.state, partial, window)
        total /= window
        step.output = valid_or_nan(total, count == window)


@model
def rolling_range_position(ctx: Context, source: str | Model, window: int) -> None:
    """Position of the current value in its window range, ``0`` low to ``1`` high.

    ``(x - rolling_min) / (rolling_max - rolling_min)``; a zero window range
    yields NaN.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        minimum, maximum, count = window_extrema(ctx.state, x, window)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            position = (x - minimum) / (maximum - minimum)
        step.output = valid_or_nan(position, count == window)
