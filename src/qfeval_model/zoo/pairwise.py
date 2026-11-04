"""Pairwise rolling statistics between two aligned sources.

Both sources must have exactly the same shape; statistics are computed per
cell over the trailing window with the strict rolling contract (NaN until
the window is full, and a NaN in either source's window makes that cell's
output NaN).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step, model
from qfeval_model.zoo._source import (
    finite_or_nan,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import (
    validate_ddof,
    validate_window,
    window_bivariate_moments,
    window_moments,
)

_Array = npt.NDArray[Any]


def _pair_steps(
    ctx: Context, source: str | Model, other: str | Model, name: str
) -> Iterator[tuple[Step, _Array, _Array]]:
    """Register both sources and yield ``(step, x, y)`` for every step."""
    register_source(ctx, source, "source")
    register_source(ctx, other, name)
    for step in ctx:
        x = source_value(step, source, ctx.state, name="source")
        yield step, x, source_value(step, other, ctx.state, name=name)


@model
def rolling_cov(
    ctx: Context,
    source: str | Model,
    other: str | Model,
    window: int,
    ddof: int = 1,
) -> None:
    """Rolling covariance of corresponding cells over ``window`` steps."""
    validate_ddof(window, ddof)
    for step, x, y in _pair_steps(ctx, source, other, "other"):
        *_, comoment, count = window_bivariate_moments(ctx.state, x, y, window)
        covariance = comoment / (window - ddof)
        step.output = valid_or_nan(covariance, count == window)


@model
def rolling_corr(
    ctx: Context, source: str | Model, other: str | Model, window: int
) -> None:
    """Rolling Pearson correlation of corresponding cells over ``window`` steps.

    A zero standard deviation on either side yields NaN.
    """
    validate_window(window, minimum=2)
    for step, x, y in _pair_steps(ctx, source, other, "other"):
        _, _, m2x, m2y, comoment, count = window_bivariate_moments(
            ctx.state, x, y, window
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = comoment / np.sqrt(m2x) / np.sqrt(m2y)
        np.clip(corr, -1.0, 1.0, out=corr)
        step.output = valid_or_nan(corr, (count == window) & (m2x > 0.0) & (m2y > 0.0))


@model
def rolling_beta(
    ctx: Context,
    source: str | Model,
    benchmark: str | Model,
    window: int,
    ddof: int = 1,
) -> None:
    """Rolling OLS beta: ``cov(source, benchmark) / var(benchmark)``.

    A zero benchmark variance yields NaN.
    """
    validate_ddof(window, ddof)
    for step, x, y in _pair_steps(ctx, source, benchmark, "benchmark"):
        _, _, _, benchmark_m2, comoment, count = window_bivariate_moments(
            ctx.state, x, y, window
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = comoment / benchmark_m2
        step.output = valid_or_nan(beta, (count == window) & (benchmark_m2 > 0.0))


@model
def rolling_tracking_error(
    ctx: Context,
    source: str | Model,
    benchmark: str | Model,
    window: int,
    ddof: int = 1,
) -> None:
    """Rolling std of the active return ``source - benchmark`` (not annualized)."""
    validate_ddof(window, ddof)
    for step, x, y in _pair_steps(ctx, source, benchmark, "benchmark"):
        with np.errstate(over="ignore", invalid="ignore"):
            active = x - y
        _, m2, count = window_moments(ctx.state, finite_or_nan(active), window)
        error = np.sqrt(m2 / (window - ddof))
        step.output = valid_or_nan(error, count == window)
