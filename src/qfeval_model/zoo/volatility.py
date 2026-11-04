"""Annualized historical volatility of an explicit return series."""

from __future__ import annotations

import math

import numpy as np

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import register_source, source_value, valid_or_nan
from qfeval_model.zoo._window import validate_ddof, window_moments


@model
def annualized_historical_volatility(
    ctx: Context,
    source: str | Model,
    window: int,
    periods_per_year: float,
    ddof: int = 1,
) -> None:
    """Rolling return volatility scaled to one year.

    ``rolling_std(source, window, ddof) * sqrt(periods_per_year)``. The
    source is an already-computed return series — compose ``log_return()``
    or ``simple_return()`` explicitly; nothing converts prices here.
    ``periods_per_year`` is required (e.g. 252 for daily bars) because bar
    frequency is never assumed.
    """
    validate_ddof(window, ddof)
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError(
            f"periods_per_year must be a positive finite number, "
            f"got {periods_per_year!r}"
        )
    register_source(ctx, source)
    scale = math.sqrt(periods_per_year)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, m2, count = window_moments(ctx.state, x, window)
        with np.errstate(over="ignore", invalid="ignore"):
            volatility = np.sqrt(m2 / (window - ddof)) * scale
        step.output = valid_or_nan(volatility, count == window)
