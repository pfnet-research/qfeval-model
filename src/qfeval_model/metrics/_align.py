"""Shared alignment and correlation helpers for the metrics package."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from qfeval_model.core import (
    Features,
    RawData,
    as_wide,
)


def wide_price(inputs: RawData, price: str) -> pd.DataFrame:
    """Pivot any accepted inputs shape into a wide price table (UTC-naive index).

    The single evaluation-side gate on the price feature: it must be a flat
    ``(timestamp, symbol)`` table of real numbers. Trailing axes and
    non-real dtypes are rejected here rather than surfacing as silent NaN
    metrics downstream (when in doubt, refuse).
    """
    frames = as_wide(inputs)
    if price not in frames:
        raise ValueError(f"inputs must have a {price!r} input")
    frame = frames[price]
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(
            f"price feature {price!r} must be a flat (timestamp, symbol) "
            "table, but it carries anonymous trailing axes; select one "
            "horizon per evaluation"
        )
    bad = sorted({str(dtype) for dtype in frame.dtypes if dtype.kind not in "iuf"})
    if bad:
        raise ValueError(
            f"price feature {price!r} must hold real numbers, got dtype(s) {bad}"
        )
    # A long frame is dense over the union of every input's timestamps, so a
    # sparse price feature can acquire rows that contain no price at all.  Such
    # rows are not price observations and must not consume a forward-return
    # period; dropping them also keeps equivalent wide and long inputs equal.
    frame = frame.dropna(how="all")
    # pandas nullable numerics can otherwise make a mixed-dtype frame's
    # ``to_numpy()`` object-typed (with ``pd.NA`` values).  Downstream metrics
    # use NumPy arithmetic / ``isnan``, so normalize every accepted real
    # panel to their common numeric representation here.
    values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"price feature {price!r} must not contain infinity")
    if (values == 0.0).any():
        raise ValueError(f"price feature {price!r} must not contain zero")
    return pd.DataFrame(
        values,
        index=frame.index,
        columns=frame.columns,
    )


def forward_wide(wide: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Forward return ``price_{t+periods} / price_t - 1`` on a wide price table.

    One period counts one row on the price-observation axis (rows containing
    no price for any symbol are removed by :func:`wide_price`).  ``aligned``
    and ``metrics.forward_returns`` both use this definition so the
    leak-sensitive formula and its ``periods >= 1`` constraint exist exactly
    once.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    return wide.shift(-periods) / wide - 1.0


def require_symbol_output(result: Features) -> None:
    """Reject results whose output carries anonymous trailing axes.

    The financial meaning of the metrics is tied to the plain ``(T, S)``
    shape; rather than silently averaging extra axes away, the caller must
    select one position explicitly (when in doubt, refuse).
    """
    if result.array.ndim != 2:
        raise ValueError(
            "metrics require a plain (T, S) output, got output shape "
            f"{result.array.shape} with anonymous trailing axes; "
            "select one position before evaluating with "
            "result.select_trailing(...)"
        )


def aligned(
    result: Features,
    inputs: RawData,
    price: str,
    periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (output, forward-return) wide frames on the output axes."""
    require_symbol_output(result)
    forward = forward_wide(wide_price(inputs, price), periods)
    pred = result.to_wide()
    forward = forward.reindex(index=pred.index, columns=pred.columns)
    return pred, forward


def average_rank(values: Any) -> Any:
    """Return 0-based average-tie ranks of a 1-D array with no NaN."""
    n = values.shape[0]
    order = np.argsort(values, kind="stable")
    positions = np.empty(n, dtype=np.float64)
    positions[order] = np.arange(n, dtype=np.float64)
    _, inverse = np.unique(values, return_inverse=True)
    sums = np.bincount(inverse, weights=positions)
    counts = np.bincount(inverse)
    return sums[inverse] / counts[inverse]


def corr(x: Any, y: Any, *, method: Literal["spearman", "pearson"]) -> float:
    """Pairwise-complete correlation of two 1-D arrays (NaN if undefined)."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    if int(valid.sum()) < 2:
        return float("nan")
    a, b = x[valid], y[valid]
    if method == "spearman":
        a, b = average_rank(a), average_rank(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)
