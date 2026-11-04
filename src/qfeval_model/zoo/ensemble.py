"""Combining child models into one output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step, model
from qfeval_model.zoo._source import finite_or_nan, valid_or_nan
from qfeval_model.zoo._window import validate_integer


def _register_children(ctx: Context, children: Sequence[Model]) -> None:
    if len(children) < 1:
        raise ValueError("an ensemble needs at least one child model")
    for index, child in enumerate(children):
        ctx.children[f"m{index}"] = child


def _child_output(
    step: Step, index: int, shape: tuple[int, ...] | None = None
) -> npt.NDArray[Any]:
    output = np.asarray(step.children[f"m{index}"].array, dtype=np.float64)
    if shape is not None and output.shape != shape:
        raise ValueError(
            f"ensemble child shape mismatch: expected {shape}, got {output.shape}"
        )
    return output


def _scale(step: Step, count: int, extremum: np.ufunc = np.maximum) -> npt.NDArray[Any]:
    scale = np.abs(_child_output(step, 0))
    scratch = np.empty(scale.shape, dtype=np.float64)
    for index in range(1, count):
        np.abs(_child_output(step, index, scale.shape), out=scratch)
        extremum(scale, scratch, out=scale)
    return scale


def _weighted_sum(ctx: Context, count: int, weights: npt.NDArray[Any]) -> None:
    """Emit the scale-normalized weighted sum of the child outputs per step."""
    for step in ctx:
        scale = _scale(step, count)
        combined = np.zeros(scale.shape, dtype=np.float64)
        scratch = np.empty(scale.shape, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for index in range(count):
                output = _child_output(step, index, scale.shape)
                scratch.fill(0.0)
                np.divide(output, scale, out=scratch, where=scale > 0.0)
                np.multiply(scratch, weights[index], out=scratch)
                np.add(combined, scratch, out=combined)
            np.multiply(combined, scale, out=combined)
        step.output = finite_or_nan(combined)


@model
def mean_ensemble(ctx: Context, children: Sequence[Model]) -> None:
    """The elementwise mean of the child models' outputs (NaN propagates)."""
    _register_children(ctx, children)
    count = len(children)
    _weighted_sum(ctx, count, np.full(count, 1.0 / count))


@model
def nanmean_ensemble(
    ctx: Context, children: Sequence[Model], min_children: int = 1
) -> None:
    """The elementwise mean of the finite child outputs per cell.

    Cells with fewer than ``min_children`` finite child values are NaN.
    """
    validate_integer(min_children, "min_children", 1)
    _register_children(ctx, children)
    child_count = len(children)
    for step in ctx:
        scale = _scale(step, child_count, np.fmax)
        counts = np.zeros(scale.shape, dtype=np.int64)
        totals = np.zeros(scale.shape, dtype=np.float64)
        scratch = np.empty(scale.shape, dtype=np.float64)
        for index in range(child_count):
            output = _child_output(step, index, scale.shape)
            valid = ~np.isnan(output)
            np.add(counts, valid, out=counts)
            scratch.fill(0.0)
            np.divide(output, scale, out=scratch, where=valid & (scale > 0.0))
            np.add(totals, scratch, out=totals)
        with np.errstate(invalid="ignore"):
            mean = totals / counts * scale
        step.output = valid_or_nan(mean, counts >= min_children)


@model
def linear_combination(
    ctx: Context, children: Sequence[Model], weights: Sequence[float]
) -> None:
    """The weighted sum of the child models' outputs (no implicit normalization).

    ``weights`` are finite scalars, one per child; NaN child cells propagate.
    """
    _register_children(ctx, children)
    if len(weights) != len(children):
        raise ValueError(
            f"weights and children must have the same length, "
            f"got {len(weights)} and {len(children)}"
        )
    if not all(np.isfinite(w) for w in weights):
        raise ValueError(f"weights must be finite, got {list(weights)!r}")
    _weighted_sum(ctx, len(children), np.asarray(weights, dtype=np.float64))
