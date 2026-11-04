"""Rolling-window statistics over trailing step windows.

Common contract (see the package docstring for the shared source rules):
``window`` counts observation steps, the window trails and includes the
current step, warm-up is strict (``min_periods=window``: NaN until the
window is full, like pandas), and a NaN anywhere in a cell's window makes
that cell's output NaN. ``rolling_count`` and the ``rolling_valid_*`` family
are the deliberate missing-data exceptions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import (
    recursive_ewm_update,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    MomentAggregate,
    SumAggregate,
    full_block,
    validate_ddof,
    validate_window,
    window_boolean_count,
    window_extrema,
    window_indexed_extrema,
    window_linear_stats,
    window_moments,
    window_product,
    window_sum,
    window_valid_moments,
    window_valid_sum,
)


def _windowed_sum(
    ctx: Context,
    source: str | Model,
    window: int,
    kernel: Callable[..., SumAggregate],
    denominator: float,
) -> None:
    """Windowed sum scaled by ``denominator``, NaN until ``count == window``."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, count = kernel(ctx.state, x, window)
        total /= denominator
        step.output = valid_or_nan(total, count == window)


def _windowed_deviation(
    ctx: Context,
    source: str | Model,
    window: int,
    ddof: int,
    kernel: Callable[..., MomentAggregate],
    root: bool,
) -> None:
    """Windowed ``M2 / (window - ddof)`` (its root for std), NaN until full."""
    validate_ddof(window, ddof)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, m2, count = kernel(ctx.state, x, window)
        result = m2 / (window - ddof)
        if root:
            np.sqrt(result, out=result)
        step.output = valid_or_nan(result, count == window)


@model
def rolling_mean(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling arithmetic mean over ``window`` steps (NaN until full, like pandas)."""
    _windowed_sum(ctx, source, window, window_sum, window)


@model
def rolling_sum(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling sum over ``window`` steps (strict warm-up, NaN propagates)."""
    _windowed_sum(ctx, source, window, window_sum, 1)


@model
def rolling_valid_sum(ctx: Context, source: str | Model, window: int) -> None:
    """Sum each cell's latest ``window`` finite observations.

    NaN does not consume an observation or change that cell's state. Output
    is NaN until the cell has seen ``window`` finite values; afterwards a NaN
    input repeats the preceding sum. This differs from :func:`rolling_sum`,
    whose ``window`` counts timestamp steps and propagates NaN.
    """
    _windowed_sum(ctx, source, window, window_valid_sum, 1)


@model
def rolling_valid_mean(ctx: Context, source: str | Model, window: int) -> None:
    """Mean of each cell's latest ``window`` finite observations.

    Exactly ``rolling_valid_sum / window`` through the shared kernel: NaN
    does not consume an observation or change that cell's state. Output is
    NaN until the cell has seen ``window`` finite values; afterwards a NaN
    input repeats the preceding mean.
    """
    _windowed_sum(ctx, source, window, window_valid_sum, window)


@model
def rolling_valid_var(
    ctx: Context, source: str | Model, window: int, ddof: int = 1
) -> None:
    """Variance of each cell's latest ``window`` finite observations.

    The finite-observation counterpart of :func:`rolling_var` with explicit
    ``ddof``: NaN does not consume an observation or change that cell's
    state. Output is NaN until the cell has seen ``window`` finite values;
    afterwards a NaN input repeats the preceding variance.
    """
    _windowed_deviation(ctx, source, window, ddof, window_valid_moments, root=False)


@model
def rolling_valid_std(
    ctx: Context, source: str | Model, window: int, ddof: int = 1
) -> None:
    """Standard deviation of each cell's latest ``window`` finite observations.

    The finite-observation counterpart of :func:`rolling_std` with explicit
    ``ddof``: NaN does not consume an observation or change that cell's
    state. Output is NaN until the cell has seen ``window`` finite values;
    afterwards a NaN input repeats the preceding standard deviation.
    """
    _windowed_deviation(ctx, source, window, ddof, window_valid_moments, root=True)


@model
def rolling_var(ctx: Context, source: str | Model, window: int, ddof: int = 1) -> None:
    """Rolling variance over ``window`` steps with explicit ``ddof``."""
    _windowed_deviation(ctx, source, window, ddof, window_moments, root=False)


@model
def rolling_std(ctx: Context, source: str | Model, window: int, ddof: int = 1) -> None:
    """Rolling standard deviation over ``window`` steps with explicit ``ddof``."""
    _windowed_deviation(ctx, source, window, ddof, window_moments, root=True)


@model
def rolling_zscore(
    ctx: Context, source: str | Model, window: int, ddof: int = 1
) -> None:
    """Rolling z-score ``(x - rolling_mean) / rolling_std`` over ``window`` steps.

    The standard deviation uses the explicit ``ddof``; a zero standard
    deviation yields NaN.
    """
    validate_ddof(window, ddof)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        mean, m2, count = window_moments(ctx.state, x, window)
        std = np.sqrt(m2 / (window - ddof))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scored = (x - mean) / std
        step.output = valid_or_nan(scored, count == window)


@model
def rolling_min(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling minimum over ``window`` steps (strict warm-up, NaN propagates)."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        minimum, _, count = window_extrema(ctx.state, x, window, direction="min")
        step.output = np.where(count == window, minimum, np.nan)


@model
def rolling_max(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling maximum over ``window`` steps (strict warm-up, NaN propagates)."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, maximum, count = window_extrema(ctx.state, x, window, direction="max")
        step.output = np.where(count == window, maximum, np.nan)


@model
def rolling_range(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling ``max - min`` over ``window`` steps (strict warm-up, NaN propagates)."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        minimum, maximum, count = window_extrema(ctx.state, x, window)
        with np.errstate(over="ignore", invalid="ignore"):
            span = maximum - minimum
        step.output = valid_or_nan(span, count == window)


@model
def rolling_count(ctx: Context, source: str | Model, window: int) -> None:
    """Number of finite values in the ``window`` (the NaN-propagation exception).

    Strict warm-up still applies: the output is NaN until ``window`` steps
    have been observed; after that a NaN in the window lowers the count
    instead of poisoning it.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        count, _, size = window_boolean_count(ctx.state, ~np.isnan(x), window)
        step.output = np.where(size == window, count.astype(np.float64), np.nan)


@model
def rolling_median(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling median over ``window`` steps (an even count averages the middle two)."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        step.output = np.median(full_block(ctx.state, x, window), axis=0)


@model
def rolling_product(ctx: Context, source: str | Model, window: int) -> None:
    """Rolling product over ``window`` steps (strict warm-up, NaN propagates)."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        product, count = window_product(ctx.state, x, window)
        step.output = np.where(count == window, product, np.nan)


def _windowed_arg_extremum(
    ctx: Context, source: str | Model, window: int, use_min: bool
) -> None:
    """Steps since the window extremum (0 = current step), NaN until full."""
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, minimum_index, _, maximum_index, count, current_index = (
            window_indexed_extrema(ctx.state, x, window)
        )
        index = minimum_index if use_min else maximum_index
        result = (current_index - index).astype(np.float64)
        step.output = valid_or_nan(result, count == window)


@model
def rolling_argmin(ctx: Context, source: str | Model, window: int) -> None:
    """How many steps ago the window minimum occurred (0 = current step).

    Ties resolve to the most recent occurrence; a NaN in the window makes
    the output NaN.
    """
    _windowed_arg_extremum(ctx, source, window, use_min=True)


@model
def rolling_argmax(ctx: Context, source: str | Model, window: int) -> None:
    """How many steps ago the window maximum occurred (0 = current step).

    Ties resolve to the most recent occurrence; a NaN in the window makes
    the output NaN.
    """
    _windowed_arg_extremum(ctx, source, window, use_min=False)


@model
def recursive_ewm_mean(ctx: Context, source: str | Model, alpha: float) -> None:
    """Recursive exponential mean ``s_t = alpha * x_t + (1 - alpha) * s_{t-1}``.

    The recursive (``adjust=False``) variant: the state is seeded at the
    first finite value, and missing inputs carry the previous average
    forward without decaying it (``ignore_na=True``).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        step.output = recursive_ewm_update(ctx.state, "mean", x, alpha)


@model
def linear_weighted_moving_average(
    ctx: Context, source: str | Model, window: int
) -> None:
    """Linearly weighted moving average: weights ``1..window``, newest heaviest.

    ``sum(w_i * x_i) / sum(w_i)`` with ``w = 1`` on the oldest and
    ``w = window`` on the newest step (strict warm-up, NaN propagates).
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, centered, _, count = window_linear_stats(ctx.state, x, window)
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_mean = total / window + centered / (window * (window + 1.0) / 2.0)
        step.output = valid_or_nan(weighted_mean, count == window)
