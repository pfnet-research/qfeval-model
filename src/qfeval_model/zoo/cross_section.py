"""Cross-sectional transforms over the symbol axis at one timestamp."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, model
from qfeval_model.zoo._rank import average_ranks_axis0
from qfeval_model.zoo._source import (
    finite_or_nan,
    register_source,
    source_value,
    valid_or_nan,
)
from qfeval_model.zoo._window import validate_integer, validate_quantile_method


def _normalized(x: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], ...]:
    """Valid mask, count, max-abs scale, ``x / scale``, and normalized mean.

    Dividing by the per-column max-abs first keeps every later sum
    overflow-free for arbitrarily large finite inputs.
    """
    valid = ~np.isnan(x)
    counts = valid.sum(axis=0, keepdims=True)
    scale = np.where(valid, np.abs(x), 0.0).max(axis=0, keepdims=True)
    normalized = np.divide(
        x,
        scale,
        out=np.zeros(x.shape, dtype=np.float64),
        where=valid & (scale > 0.0),
    )
    mean = np.divide(
        normalized.sum(axis=0, keepdims=True),
        counts,
        out=np.zeros_like(scale),
        where=counts > 0,
    )
    return valid, counts, scale, normalized, mean


@model
def cross_sectional_rank(ctx: Context, source: str | Model) -> None:
    """Cross-sectional rank in ``[0, 1]`` at each timestamp (ties averaged).

    Follows the quant convention (WorldQuant's ``rank(x)``): the rank is over
    the symbol axis at one timestamp, computed independently along every
    trailing axis of the source. NaN inputs stay NaN; a single valid symbol
    maps to 0.5.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        ranks, flat_counts = average_ranks_axis0(x)
        counts = flat_counts.reshape((1, *x.shape[1:]))
        normalized = np.divide(
            ranks - 1.0,
            counts - 1.0,
            out=np.full_like(ranks, 0.5),
            where=counts > 1,
        )
        normalized[np.isnan(x)] = np.nan
        step.output = normalized


@model
def cross_sectional_demean(ctx: Context, source: str | Model) -> None:
    """Subtract the symbol-axis mean at each timestamp (market-neutral).

    The mean is taken over the symbol axis, independently along every
    trailing axis of the source; NaN cells are excluded from the mean and
    stay NaN in the output.
    """
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        _, _, scale, _, mean = _normalized(x)
        np.clip(mean, -1.0, 1.0, out=mean)
        with np.errstate(over="ignore", invalid="ignore"):
            demeaned = x - mean * scale
        step.output = finite_or_nan(demeaned)


@model
def cross_sectional_zscore(ctx: Context, source: str | Model, ddof: int = 1) -> None:
    """Symbol-axis z-score at each timestamp (finite cells only).

    NaN cells stay NaN; a zero standard deviation or fewer than ``ddof + 1``
    valid symbols yields NaN for the whole cross-section.
    """
    validate_integer(ddof, "ddof")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        valid, counts, _, normalized, mean = _normalized(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            squared = np.where(valid, (normalized - mean) ** 2, 0.0).sum(
                axis=0, keepdims=True
            )
            scored = (normalized - mean) / np.sqrt(squared / (counts - ddof))
        step.output = valid_or_nan(scored, valid & (counts > ddof))


@model
def cross_sectional_scale(
    ctx: Context, source: str | Model, gross: float = 1.0
) -> None:
    """Scale each cross-section so its finite ``sum(abs(x))`` equals ``gross``.

    NaN cells stay NaN; an all-zero (or all-NaN) cross-section yields NaN.
    """
    if not np.isfinite(gross) or gross <= 0.0:
        raise ValueError(f"gross must be a positive finite number, got {gross!r}")
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        valid, _, _, normalized, _ = _normalized(x)
        with np.errstate(invalid="ignore", divide="ignore"):
            denominator = np.abs(normalized).sum(axis=0, keepdims=True)
            scaled = normalized * (gross / denominator)
        step.output = valid_or_nan(scaled, valid)


@model
def cross_sectional_winsorize(
    ctx: Context,
    source: str | Model,
    lower: float = 0.05,
    upper: float = 0.95,
    method: str = "linear",
) -> None:
    """Clip each cross-section to its ``[lower, upper]`` sample quantiles.

    Quantiles are computed over the finite cells of the symbol axis with the
    explicit ``method`` (``"linear"`` or ``"inverted_cdf"``); NaN cells stay
    NaN.
    """
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(
            f"quantiles must satisfy 0 <= lower <= upper <= 1, got {lower} and {upper}"
        )
    chosen = validate_quantile_method(method)
    register_source(ctx, source)
    for step in ctx:
        x = source_value(step, source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            bounds = np.nanquantile(x, [lower, upper], axis=0, method=chosen)
        step.output = finite_or_nan(np.clip(x, bounds[0], bounds[1]))
