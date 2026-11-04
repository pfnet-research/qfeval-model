"""Shared ``source`` resolution: an input feature name or a child model."""

from __future__ import annotations

from collections import deque
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step


def register_source(ctx: Context, source: str | Model, name: str = "source") -> None:
    """Register ``source`` as the child ``name`` when it is a model."""
    if not isinstance(source, str):
        ctx.children[name] = source


def source_value(
    step: Step,
    source: str | Model,
    state: dict[str, Any] | None = None,
    name: str = "source",
    allow_nonfinite: bool = False,
) -> npt.NDArray[Any]:
    """Return the step's real per-timestamp value of ``source`` as float64.

    The shape is ``(S, *trailing)``: a feature name reads ``step.inputs``,
    a child model reads the child registered under ``name``. Complex and
    infinite values are rejected at this shared boundary (``finite_indicator``
    is the one caller that passes ``allow_nonfinite=True``). Stateful
    transforms pass their ``ctx.state``: they hold prior-shape arrays there,
    and a differently shaped source would silently broadcast against them —
    so the first read locks the source shape for the model's lifetime, and
    every source of a multi-source model must match it.
    """
    if isinstance(source, str):
        value = step.inputs[source]
    else:
        value = step.children[name].array
    if np.iscomplexobj(value):
        raise ValueError(
            "zoo sources must be real numeric; complex values are not supported"
        )
    array = np.asarray(value, dtype=np.float64)
    if not allow_nonfinite and np.isinf(array).any():
        raise ValueError(
            "zoo sources must be finite or NaN; infinite values are not supported"
        )
    if state is not None:
        shape = state.setdefault("shape", array.shape)
        if array.shape != shape:
            raise ValueError(
                f"source shape changed: expected {shape}, got {array.shape}; "
                f"every source of a stateful model must keep one shape "
                f"across steps and across sources"
            )
    return array


def nan_like(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return a writable NaN array with the shape of ``x``."""
    return np.full(x.shape, np.nan, dtype=np.float64)


def lagged_value(
    state: dict[str, Any],
    key: str,
    x: npt.NDArray[Any],
    periods: int,
) -> npt.NDArray[Any]:
    """Store ``x`` and return its ``periods``-step lag, NaN until available."""
    if periods == 0:
        return x
    buffer: deque[npt.NDArray[Any]] = state.setdefault(key, deque(maxlen=periods))
    lagged = buffer[0] if len(buffer) == periods else nan_like(x)
    # A row view would retain the full input array across predict calls.
    buffer.append(x.copy())
    return lagged


def state_array(
    state: dict[str, Any],
    key: str,
    x: npt.NDArray[Any],
    fill: float,
) -> npt.NDArray[Any]:
    """Return an owned float64 state array, allocating it only once.

    ``dict.get(key, np.full(...))`` evaluates the default allocation even when
    ``key`` is already present.  Stateful zoo models use this helper at their
    initialization boundary and then update the returned buffer in place.
    Source-shape locking guarantees that an existing array has the right shape.
    """
    value = state.get(key)
    if value is None:
        value = np.full(x.shape, fill, dtype=np.float64)
        state[key] = value
    return cast(npt.NDArray[Any], value)


def recursive_ewm_update(
    state: dict[str, Any],
    key: str,
    x: npt.NDArray[Any],
    alpha: float,
) -> npt.NDArray[Any]:
    """Update a recursive EWM seeded by the first finite value, carrying NaN."""
    current = state_array(state, key, x, np.nan)
    finite = ~np.isnan(x)
    initialized = ~np.isnan(current)
    update = finite & initialized
    with np.errstate(over="ignore", invalid="ignore"):
        np.multiply(current, 1.0 - alpha, out=current, where=update)
        np.add(current, x * alpha, out=current, where=update)
    np.copyto(current, x, where=finite & ~initialized)
    return current


def finite_or_nan(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Sanitize a writable arithmetic result in place and return it."""
    x[~np.isfinite(x)] = np.nan
    return x


def valid_or_nan(x: npt.NDArray[Any], valid: npt.NDArray[np.bool_]) -> npt.NDArray[Any]:
    """Sanitize an owned result and mask cells outside its readiness contract."""
    x[~valid | ~np.isfinite(x)] = np.nan
    return x
