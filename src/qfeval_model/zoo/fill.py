"""Time alignment: lagging, forward filling, staleness, and history buffers."""

from __future__ import annotations

import numpy as np

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import (
    lagged_value,
    register_source,
    source_value,
    state_array,
)
from qfeval_model.zoo._window import full_block, validate_integer, validate_window


@model
def lag(ctx: Context, source: str | Model, periods: int = 1) -> None:
    """The value ``periods`` steps ago (``pandas.shift(periods)``).

    The first ``periods`` steps are NaN and missing values propagate as-is.
    """
    validate_integer(periods, "periods", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        step.output = lagged_value(ctx.state, "buffer", x, periods)


@model
def ffill(ctx: Context, source: str | Model) -> None:
    """Last observation carried forward including today (the as-of value)."""
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        carry = state_array(ctx.state, "carry", x, np.nan)
        np.copyto(carry, x, where=~np.isnan(x))
        step.output = carry


@model
def stale_age(ctx: Context, source: str | Model) -> None:
    """Steps since the last finite value (0 on a finite step).

    NaN until the first finite value has been observed. One of the explicit
    missing-data models that are exempt from NaN propagation.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        age = state_array(ctx.state, "age", x, np.nan)
        age += 1.0
        np.copyto(age, 0.0, where=~np.isnan(x))
        step.output = age


@model
def with_history(ctx: Context, source: str | Model, window: int) -> None:
    """The current value, plus a rolling buffer of the last ``window`` values.

    The main output is the current value; the buffer is exposed as the
    ``(window, S, *trailing)`` side output ``"history"`` (NaN-padded until
    full). The buffer lives in state and therefore survives
    checkpoint/restore.
    """
    validate_window(window)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        block = full_block(ctx.state, x, window)
        step.output = x
        step.side_outputs["history"] = block
