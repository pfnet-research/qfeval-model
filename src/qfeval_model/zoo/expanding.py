"""Expanding statistics over all finite observations so far.

Unlike the strict rolling family, expanding statistics skip NaN inputs
instead of propagating them: each cell aggregates every finite value
observed so far and is NaN only until its first finite observation (or
while below the degrees of freedom required by ``ddof``). Aggregates live
in ``ctx.state`` as owned per-cell arrays (checkpoint state schema:
``count`` / ``sum`` / ``mean`` / ``m2`` / ``min`` / ``max`` as each model
needs, all float64 of the locked source shape).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._source import (
    finite_or_nan,
    register_source,
    source_value,
    state_array,
)
from qfeval_model.zoo._window import validate_integer


@model
def expanding_sum(ctx: Context, source: str | Model) -> None:
    """Sum of all finite observations so far (NaN until the first one)."""
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        finite = ~np.isnan(x)
        total = state_array(ctx.state, "sum", x, 0.0)
        count = state_array(ctx.state, "count", x, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add(total, x, out=total, where=finite)
        np.add(count, finite, out=count)
        step.output = np.where(count > 0, finite_or_nan(total), np.nan)


@model
def expanding_mean(ctx: Context, source: str | Model) -> None:
    """Mean of all finite observations so far (NaN until the first one)."""
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        finite = ~np.isnan(x)
        count = state_array(ctx.state, "count", x, 0.0)
        mean = state_array(ctx.state, "mean", x, 0.0)
        np.add(count, finite, out=count)
        with np.errstate(invalid="ignore", over="ignore"):
            change = np.divide(x, count, out=np.zeros_like(mean), where=finite)
            change -= np.divide(mean, count, out=np.zeros_like(mean), where=finite)
            np.add(mean, change, out=mean, where=finite)
        step.output = np.where(count > 0, finite_or_nan(mean), np.nan)


def _welford_update(
    state: dict[str, Any], x: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Advance the shared Welford aggregates by one observation step."""
    finite = ~np.isnan(x)
    count = state_array(state, "count", x, 0.0)
    mean = state_array(state, "mean", x, 0.0)
    m2 = state_array(state, "m2", x, 0.0)
    np.add(count, finite, out=count)
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.subtract(x, mean, where=finite, out=np.zeros_like(mean))
        adjustment = np.divide(delta, count, where=finite, out=np.zeros_like(mean))
        np.add(mean, adjustment, out=mean, where=finite)
        remainder = np.subtract(x, mean, where=finite, out=np.zeros_like(mean))
        np.multiply(delta, remainder, out=delta, where=finite)
        np.add(m2, delta, out=m2, where=finite)
    m2[~np.isfinite(m2)] = np.nan
    return count, mean, m2


def _welford_var(
    count: npt.NDArray[Any], m2: npt.NDArray[Any], ddof: int
) -> npt.NDArray[Any]:
    """Variance from Welford aggregates (NaN at or below ``ddof`` observations)."""
    variance = np.full_like(m2, np.nan)
    np.divide(m2, count - ddof, out=variance, where=count > ddof)
    return variance


@model
def expanding_var(ctx: Context, source: str | Model, ddof: int = 1) -> None:
    """Variance of all finite observations so far (Welford, explicit ``ddof``)."""
    validate_integer(ddof, "ddof")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        count, _, m2 = _welford_update(ctx.state, x)
        step.output = _welford_var(count, m2, ddof)


@model
def expanding_std(ctx: Context, source: str | Model, ddof: int = 1) -> None:
    """Standard deviation of all finite observations so far (Welford)."""
    validate_integer(ddof, "ddof")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        count, _, m2 = _welford_update(ctx.state, x)
        step.output = np.sqrt(_welford_var(count, m2, ddof))


@model
def expanding_zscore(ctx: Context, source: str | Model, ddof: int = 1) -> None:
    """Z-score of the current value against the expanding mean and std.

    The current value is included in the statistics. NaN while the input is
    NaN, below ``ddof + 1`` observations, or where the std is zero.
    """
    validate_integer(ddof, "ddof")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        count, mean, m2 = _welford_update(ctx.state, x)
        std = np.sqrt(_welford_var(count, m2, ddof))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scored = (x - mean) / std
        step.output = finite_or_nan(scored)


def _expanding_extreme(
    ctx: Context, source: str | Model, key: str, op: np.ufunc
) -> None:
    """Advance a NaN-seeded expanding extreme state array in place."""
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        current = state_array(ctx.state, key, x, np.nan)
        op(current, x, out=current)
        step.output = current


@model
def expanding_min(ctx: Context, source: str | Model) -> None:
    """Minimum of all finite observations so far (NaN until the first one)."""
    _expanding_extreme(ctx, source, "min", np.fmin)


@model
def expanding_max(ctx: Context, source: str | Model) -> None:
    """Maximum of all finite observations so far (NaN until the first one)."""
    _expanding_extreme(ctx, source, "max", np.fmax)
