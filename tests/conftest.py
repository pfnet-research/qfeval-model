"""Shared test helpers: plain functions and one hypothesis strategy.

No fixtures on purpose — tests call these helpers directly so each test
reads top-to-bottom. Test data is written as literal 2-D lists (via
``wide``) so expected values stay hand-checkable. ``long_of`` is an
independent wide-to-long conversion used as the reference side of
"wide ≡ long" equivalence tests; it must not use the library's own
``as_frame``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from hypothesis import strategies as st


def days(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """Daily DatetimeIndex of ``n`` rows; every test's time axis comes from here."""
    return pd.date_range(start, periods=n, freq="D")


def wide(
    values: Sequence[Sequence[float]],
    symbols: Sequence[str] = ("A", "B"),
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """2-D literal list -> wide frame (index=days(T), columns=symbols)."""
    return pd.DataFrame(
        [list(row) for row in values],
        index=days(len(values), start=start),
        columns=list(symbols),
        dtype=np.float64,
    )


def trailing_wide(
    rows: Sequence[Sequence[float]],
    symbols: Sequence[str] = ("A", "B"),
    k: int = 2,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Rows of S*K floats -> wide frame with MultiIndex (symbol, 0..k-1) columns."""
    return pd.DataFrame(
        [list(row) for row in rows],
        index=days(len(rows), start=start),
        columns=pd.MultiIndex.from_product([list(symbols), range(k)]),
        dtype=np.float64,
    )


def long_of(w: pd.DataFrame | dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Naive independent wide->long conversion (reference side of wide ≡ long)."""
    frames = {"close": w} if isinstance(w, pd.DataFrame) else w
    rows: dict[tuple[pd.Timestamp, str], dict[str, object]] = {}
    for name, frame in frames.items():
        for raw_ts, row in frame.iterrows():
            ts = pd.Timestamp(str(raw_ts))
            for symbol, value in row.items():
                key = (ts, str(symbol))
                rows.setdefault(key, {"timestamp": ts, "symbol": str(symbol)})[name] = (
                    value
                )
    return pd.DataFrame(list(rows.values()))


@st.composite
def _panel(draw: st.DrawFn) -> dict[str, pd.DataFrame]:
    t = draw(st.integers(1, 8))
    s = draw(st.integers(1, 4))
    n_features = draw(st.integers(1, 2))
    symbols = [f"S{i}" for i in range(s)]
    values = st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ) | st.just(float("nan"))
    result: dict[str, pd.DataFrame] = {}
    for i in range(n_features):
        cells = draw(
            st.lists(st.lists(values, min_size=s, max_size=s), min_size=t, max_size=t)
        )
        result[f"f{i}"] = wide(cells, symbols=symbols)
    return result


def panels() -> st.SearchStrategy[dict[str, pd.DataFrame]]:
    """Random wide-dict panels for test_properties.py."""
    return _panel()
