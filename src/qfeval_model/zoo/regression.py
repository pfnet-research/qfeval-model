"""Rolling ordinary-least-squares trend against the step index.

The regressor is the in-window step index ``t = 0 .. window - 1`` (0 =
oldest step, ``window - 1`` = current step); the slope is therefore in
source units per step. All entries follow the strict rolling contract
(NaN until the window is full, NaN in a window poisons that cell).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step, model
from qfeval_model.zoo._source import (
    lagged_value,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    validate_integer,
    validate_window,
    window_bivariate_moments,
    window_linear_stats,
)


def _time_m2(window: int) -> float:
    """Centered sum of squares of ``0 .. window - 1``."""
    return window * (window**2 - 1.0) / 12.0


_Array = npt.NDArray[Any]


def _trend_steps(
    ctx: Context, source: str | Model, window: int
) -> Iterator[tuple[Step, _Array, _Array, _Array, _Array]]:
    """Yield ``(step, x, window total, OLS slope, count)`` for every step."""
    validate_window(window, minimum=2)
    register_source(ctx, source)
    time_m2 = _time_m2(window)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        total, centered, _, count = window_linear_stats(ctx.state, x, window)
        yield step, x, total, centered / time_m2, count


@model
def rolling_linear_trend_slope(ctx: Context, source: str | Model, window: int) -> None:
    """OLS slope of the source against the step index (units per step)."""
    for step, _, _, slope, count in _trend_steps(ctx, source, window):
        step.output = valid_or_nan(slope, count == window)


@model
def rolling_linear_trend_intercept(
    ctx: Context, source: str | Model, window: int
) -> None:
    """OLS intercept at the oldest window step (``t = 0``)."""
    for step, _, total, slope, count in _trend_steps(ctx, source, window):
        intercept = total / window - slope * ((window - 1) / 2.0)
        step.output = valid_or_nan(intercept, count == window)


@model
def rolling_linear_trend_r2(ctx: Context, source: str | Model, window: int) -> None:
    """R² of the OLS trend fit (NaN for a zero-variance window)."""
    validate_window(window, minimum=2)
    register_source(ctx, source)
    time_m2 = _time_m2(window)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        _, time_comoment, source_m2, count = window_linear_stats(
            ctx.state, x, window, with_m2=True
        )
        source_m2 = cast(npt.NDArray[Any], source_m2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            correlation = time_comoment / np.sqrt(time_m2) / np.sqrt(source_m2)
            r2 = correlation**2
        np.clip(r2, 0.0, 1.0, out=r2)
        step.output = valid_or_nan(r2, (count == window) & (source_m2 > 0.0))


@model
def rolling_linear_trend_residual(
    ctx: Context, source: str | Model, window: int
) -> None:
    """Current value minus the OLS trend's fitted current value."""
    for step, x, total, slope, count in _trend_steps(ctx, source, window):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = x - (total / window + slope * ((window - 1) / 2.0))
        step.output = valid_or_nan(residual, count == window)


@model
def rolling_autocorr(
    ctx: Context, source: str | Model, window: int, lag: int = 1
) -> None:
    """Pearson correlation of the source with its ``lag``-step-old self.

    Computed over the last ``window`` pairs ``(x_t, x_{t-lag})``, so the
    first output needs ``window + lag`` observations; a zero variance on
    either side yields NaN.
    """
    validate_window(window, minimum=2)
    validate_integer(lag, "lag", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        lagged = lagged_value(ctx.state, "lag_values", x, lag)
        _, _, m2x, m2y, comoment, count = window_bivariate_moments(
            ctx.state, x, lagged, window, key="pairs"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = comoment / np.sqrt(m2x) / np.sqrt(m2y)
        np.clip(corr, -1.0, 1.0, out=corr)
        step.output = valid_or_nan(corr, (count == window) & (m2x > 0.0) & (m2y > 0.0))
