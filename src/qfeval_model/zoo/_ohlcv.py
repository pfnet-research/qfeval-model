"""Private OHLCV source handling: registration, integrity, high-low windows.

Bar frequency, trading sessions, and corporate-action adjustment are the
input side's responsibility — the read helpers only enforce per-bar integrity:
``low <= high``, non-negative volume, exact shape agreement across sources
(via the shared shape lock), and the convention that a missing cell in any
of open/high/low/close makes the whole price tuple NaN for that cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.core import Context, Model, Step
from qfeval_model.zoo._source import register_source, source_value
from qfeval_model.zoo._window import window_extrema

PRICE_FIELDS = ("open", "high", "low", "close")


def window_high_low(
    state: dict[str, Any],
    values: dict[str, npt.NDArray[Any]],
    window: int,
    key: str,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.bool_]]:
    """Strict rolling highest-high / lowest-low and their joint readiness."""
    _, highest, high_count = window_extrema(
        state, values["high"], window, key=f"{key}_high", direction="max"
    )
    lowest, _, low_count = window_extrema(
        state, values["low"], window, key=f"{key}_low", direction="min"
    )
    return highest, lowest, (high_count == window) & (low_count == window)


def register_ohlcv(ctx: Context, sources: dict[str, str | Model]) -> None:
    """Register every named OHLCV source (models become children)."""
    for name, source in sources.items():
        register_source(ctx, source, name)


def read_ohlcv(
    step: Step,
    state: dict[str, Any],
    sources: dict[str, str | Model],
) -> dict[str, npt.NDArray[Any]]:
    """Read all sources for one step, validate the bar, join price NaNs.

    Raises ValueError on ``low > high`` or negative volume (NaN cells are
    exempt); a NaN in any open/high/low/close cell blanks that cell in all
    of them, so candle arithmetic never mixes a real price with a missing
    one.
    """
    values = {
        name: source_value(step, source, state, name=name)
        for name, source in sources.items()
    }
    prices = [name for name in PRICE_FIELDS if name in values]
    if len(prices) > 1:
        missing = np.isnan(values[prices[0]])
        for name in prices[1:]:
            missing |= np.isnan(values[name])
        # Dense bars are the common path.  Keep the source row views when no
        # joint missing cell needs to be materialized.
        if missing.any():
            for name in prices:
                values[name] = np.where(missing, np.nan, values[name])
    if "high" in values and "low" in values and (values["low"] > values["high"]).any():
        raise ValueError("invalid OHLC bar: low > high")
    if "volume" in values and (values["volume"] < 0.0).any():
        raise ValueError("invalid OHLCV bar: volume < 0")
    return values
