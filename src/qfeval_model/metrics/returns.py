"""Forward returns — the evaluation-side counterpart of the zoo's returns."""

from __future__ import annotations

from typing import cast

import pandas as pd

from qfeval_model.core import DEFAULT_FEATURE, RawData
from qfeval_model.metrics._align import forward_wide, wide_price


def forward_returns(
    inputs: RawData,
    *,
    price: str = DEFAULT_FEATURE,
    periods: int = 1,
) -> pd.DataFrame:
    """Return per-``(timestamp, symbol)`` forward returns as a long DataFrame.

    The forward return at ``t`` is ``price_{t+periods} / price_t - 1`` — the
    return realized *after* the prediction made at ``t``. Rows whose future
    price does not exist yet are NaN.

    Args:
        inputs: Inputs in any accepted shape (a wide mapping, a flat wide
            DataFrame, or a long DataFrame) carrying the ``price`` input.
        price: Price feature name.
        periods: Forward horizon in price-observation rows. Rows with no price
            for any symbol do not consume a period.

    Returns:
        A long-format DataFrame with columns ``timestamp``, ``symbol``,
        ``forward_return``.

    Raises:
        ValueError: If ``periods < 1``.
    """
    forward = forward_wide(wide_price(inputs, price), periods)
    stacked = cast(pd.Series, forward.stack(future_stack=True))
    return stacked.rename("forward_return").reset_index()
