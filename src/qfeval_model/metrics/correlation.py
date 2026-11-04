"""Output-vs-forward-return correlation metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from qfeval_model.core import DEFAULT_FEATURE, Features, RawData
from qfeval_model.metrics._align import aligned, corr


def information_coefficient(
    result: Features,
    inputs: RawData,
    *,
    price: str = DEFAULT_FEATURE,
    periods: int = 1,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.Series:
    """Per-timestamp rank correlation between outputs and forward returns.

    Args:
        result: The output to evaluate.
        inputs: Inputs in any accepted shape (a wide mapping, a flat wide
            DataFrame, or a long DataFrame) carrying the ``price`` input
            (must extend ``periods`` price-observation rows beyond a timestamp
            for it to score).
        price: Price feature name.
        periods: Forward horizon in price-observation rows. Rows with no price
            for any symbol do not consume a period.
        method: Correlation method (``"spearman"`` or ``"pearson"``).

    Returns:
        A Series of per-timestamp correlations indexed by timestamp
        (NaN where undefined, e.g. the last ``periods`` timestamps).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    pred, forward = aligned(result, inputs, price, periods)
    p = pred.to_numpy(dtype=np.float64)
    f = forward.to_numpy(dtype=np.float64)
    values = [corr(p[row], f[row], method=method) for row in range(p.shape[0])]
    return pd.Series(
        values, index=pred.index, name="information_coefficient", dtype=np.float64
    )


def hit_rate(
    result: Features,
    inputs: RawData,
    *,
    price: str = DEFAULT_FEATURE,
    periods: int = 1,
) -> float:
    """Fraction of cells where the output's sign matches the return's sign.

    Cells where either side is NaN or the output is exactly zero are
    excluded. A forward return of exactly zero is *not* excluded: no nonzero
    output matches its sign, so it counts as a miss (directional accuracy —
    a flat market scores against a directional bet).
    """
    pred, forward = aligned(result, inputs, price, periods)
    p = pred.to_numpy()
    f = forward.to_numpy()
    valid = ~np.isnan(p) & ~np.isnan(f) & (p != 0.0)
    if not valid.any():
        return float("nan")
    return float((np.sign(p[valid]) == np.sign(f[valid])).mean())
