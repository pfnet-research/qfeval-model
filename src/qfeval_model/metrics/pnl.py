"""Portfolio-style PnL metrics."""

from __future__ import annotations

import pandas as pd

from qfeval_model.core import DEFAULT_FEATURE, Features, RawData
from qfeval_model.metrics._align import aligned


def long_short_pnl(
    result: Features,
    inputs: RawData,
    *,
    price: str = DEFAULT_FEATURE,
    periods: int = 1,
) -> pd.Series:
    """Per-timestamp PnL of a rank-weighted, market-neutral long-short book.

    Weights are the cross-sectional ranks of the output, demeaned and
    scaled so absolute weights sum to 1 (long the highest-ranked symbols,
    short the lowest). PnL at ``t`` is ``sum(w_t * forward_return_t)``.
    A timestamp whose book holds a symbol with a missing forward return is
    NaN: dropping that leg would make the evaluated book neither gross-1
    nor market-neutral, so NaN is the honest answer.
    """
    pred, forward = aligned(result, inputs, price, periods)
    ranks = pred.rank(axis=1)
    weights = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = weights.abs().sum(axis=1)
    weights = weights.div(gross.where(gross > 0.0), axis=0)
    pnl = (weights * forward).sum(axis=1, min_count=1)
    held = weights.notna() & weights.ne(0.0)
    return pnl.mask((held & forward.isna()).any(axis=1)).rename("long_short_pnl")
