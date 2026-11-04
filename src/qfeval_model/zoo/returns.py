"""Backward-looking returns (forward returns live in ``qm.metrics``)."""

from __future__ import annotations

import numpy as np

from qfeval_model.core import DEFAULT_FEATURE, Context, Model, model
from qfeval_model.zoo._source import (
    finite_or_nan,
    lagged_value,
    register_source,
    source_value,
)
from qfeval_model.zoo._window import validate_integer


@model
def simple_return(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, periods: int = 1
) -> None:
    """Simple return ``x_t / x_{t-periods} - 1`` (first ``periods`` steps NaN).

    A zero denominator yields NaN (never infinity, never a NumPy warning).
    """
    validate_integer(periods, "periods", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        base = lagged_value(ctx.state, "buffer", x, periods)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step.output = finite_or_nan(x / base - 1.0)


@model
def log_return(
    ctx: Context, source: str | Model = DEFAULT_FEATURE, periods: int = 1
) -> None:
    """Log return ``log(x_t / x_{t-periods})`` (additive across steps).

    A non-positive ratio (zero denominator included) yields NaN (never
    infinity, never a NumPy warning).
    """
    validate_integer(periods, "periods", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        base = lagged_value(ctx.state, "buffer", x, periods)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step.output = finite_or_nan(np.log(x / base))
