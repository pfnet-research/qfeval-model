"""Elementwise transforms and stateless algebra over sources.

Shared contract for this module:

- Binary entries take a ``source`` left operand (feature name or child
  model) and accept a finite Python scalar as the right operand
  (``multiply("close", 2.0)``); a scalar left operand is not accepted —
  reorder or compose instead.
- Two panel operands must have exactly the same shape; NumPy's implicit
  broadcasting across sources is rejected.
- NaN inputs propagate elementwise (``fill_constant``, ``finite_indicator``,
  ``coalesce``, and ``where`` are the explicit missing-data exceptions).
- Domain errors, zero division, and overflow never emit warnings or
  infinities: any non-finite result cell becomes NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step, model
from qfeval_model.zoo._source import (
    finite_or_nan,
    lagged_value,
    register_source,
    source_value,
)
from qfeval_model.zoo._window import validate_integer

Operand = str | Model | float


def _register_operand(ctx: Context, operand: Operand, name: str) -> None:
    """Register a model operand as a child; validate a scalar operand."""
    if isinstance(operand, str | Model):
        register_source(ctx, operand, name)
    elif not np.isfinite(operand):
        raise ValueError(f"scalar operands must be finite, got {operand!r}")


def _operand_value(
    step: Step, operand: Operand, state: dict[str, Any], name: str
) -> npt.NDArray[Any] | float:
    """Read a panel operand through the shared source boundary, or pass a scalar."""
    if isinstance(operand, str | Model):
        return source_value(step, operand, state, name=name)
    return float(operand)


@model
def diff(ctx: Context, source: str | Model, periods: int = 1) -> None:
    """Difference ``x_t - x_{t-periods}`` (first ``periods`` steps NaN)."""
    validate_integer(periods, "periods", 1)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, ctx.state)
        base = lagged_value(ctx.state, "buffer", x, periods)
        with np.errstate(over="ignore", invalid="ignore"):
            step.output = finite_or_nan(x - base)


@model
def fill_constant(ctx: Context, source: str | Model, value: float = 0.0) -> None:
    """Replace NaN cells with the finite constant ``value`` (others unchanged)."""
    if not np.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        step.output = np.where(np.isnan(x), value, x)


@model
def finite_indicator(ctx: Context, source: str | Model) -> None:
    """1.0 where the source is finite, 0.0 elsewhere (NaN and infinity).

    The one zoo model that reads its source without the finiteness check,
    so it can flag infinite inputs instead of rejecting them.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source, allow_nonfinite=True)
        step.output = np.isfinite(x).astype(np.float64)


@model
def coalesce(ctx: Context, sources: Sequence[str | Model]) -> None:
    """The first finite source per cell (NaN where no source is finite)."""
    if len(sources) < 1:
        raise ValueError("coalesce needs at least one source")
    for index, source in enumerate(sources):
        register_source(ctx, source, f"s{index}")
    for step in ctx:
        out = source_value(step, sources[0], ctx.state, name="s0").copy()
        for index, source in enumerate(sources[1:], start=1):
            value = source_value(step, source, ctx.state, name=f"s{index}")
            np.copyto(out, value, where=np.isnan(out))
        step.output = out


@model
def clip(
    ctx: Context,
    source: str | Model,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Clip to fixed finite bounds (NaN cells stay NaN)."""
    for name, bound in (("lower", lower), ("upper", upper)):
        if bound is not None and not np.isfinite(bound):
            raise ValueError(f"{name} must be finite, got {bound!r}")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"lower must be <= upper, got {lower} > {upper}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        step.output = np.clip(x, lower, upper)


def _binary(ctx: Context, source: str | Model, other: Operand, op: np.ufunc) -> None:
    """Elementwise ``op(source, other)`` with non-finite results as NaN."""
    register_source(ctx, source)
    _register_operand(ctx, other, "other")
    for step in ctx:
        x = source_value(step, source, ctx.state)
        y = _operand_value(step, other, ctx.state, "other")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step.output = finite_or_nan(op(x, y))


@model
def add(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise ``source + other`` (non-finite results are NaN)."""
    _binary(ctx, source, other, np.add)


@model
def subtract(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise ``source - other`` (non-finite results are NaN)."""
    _binary(ctx, source, other, np.subtract)


@model
def multiply(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise ``source * other`` (non-finite results are NaN)."""
    _binary(ctx, source, other, np.multiply)


@model
def divide(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise ``source / other`` (zero denominators are NaN, never inf)."""
    _binary(ctx, source, other, np.divide)


@model
def minimum(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise minimum of ``source`` and ``other`` (NaN propagates)."""
    _binary(ctx, source, other, np.minimum)


@model
def maximum(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise maximum of ``source`` and ``other`` (NaN propagates)."""
    _binary(ctx, source, other, np.maximum)


@model
def power(ctx: Context, source: str | Model, other: Operand) -> None:
    """Elementwise ``source ** other`` (domain errors and overflow are NaN)."""
    _binary(ctx, source, other, np.power)


def _unary(ctx: Context, source: str | Model, op: np.ufunc) -> None:
    """Elementwise ``op(source)`` with non-finite results as NaN."""
    register_source(ctx, source)
    for step in ctx:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step.output = finite_or_nan(op(source_value(step, source)))


@model
def negate(ctx: Context, source: str | Model) -> None:
    """Elementwise ``-source``."""
    _unary(ctx, source, np.negative)


@model
def absolute(ctx: Context, source: str | Model) -> None:
    """Elementwise ``abs(source)``."""
    _unary(ctx, source, np.absolute)


@model
def sign(ctx: Context, source: str | Model) -> None:
    """Elementwise sign: -1.0, 0.0, or 1.0 (NaN stays NaN)."""
    _unary(ctx, source, np.sign)


@model
def log(ctx: Context, source: str | Model) -> None:
    """Elementwise natural log (non-positive inputs are NaN, no warnings)."""
    _unary(ctx, source, np.log)


@model
def exp(ctx: Context, source: str | Model) -> None:
    """Elementwise ``e ** source`` (overflow is NaN, never inf)."""
    _unary(ctx, source, np.exp)


@model
def where(ctx: Context, condition: str | Model, x: Operand, y: Operand) -> None:
    """``x`` where ``condition`` is nonzero, ``y`` where zero, NaN where NaN."""
    register_source(ctx, condition, "condition")
    _register_operand(ctx, x, "x")
    _register_operand(ctx, y, "y")
    for step in ctx:
        cond = source_value(step, condition, ctx.state, name="condition")
        xv = _operand_value(step, x, ctx.state, "x")
        yv = _operand_value(step, y, ctx.state, "y")
        result = np.where(cond != 0.0, xv, yv)
        result[np.isnan(cond)] = np.nan
        step.output = result
