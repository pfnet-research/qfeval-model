"""Exercise the zoo hotspots identified by ``REVIEW_ZOO.md``.

Run with::

    uv run python benchmarks/bench_zoo_hotspots.py

The grid covers dense and 20%-missing panels, narrow through wide symbol
axes, and short through long windows. Reported peak memory includes the
public input/output path; persistent state counts unique NumPy buffers only.
"""

from __future__ import annotations

import time
import tracemalloc
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from qfeval_model import Model, zoo


def _state_mib(value: Any, seen: set[int] | None = None) -> float:
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        if id(value) in seen:
            return 0.0
        seen.add(id(value))
        return value.nbytes / 2**20
    if isinstance(value, dict):
        return sum(_state_mib(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, deque)):
        return sum(_state_mib(item, seen) for item in value)
    return 0.0


def _frame(values: np.ndarray) -> pd.DataFrame:
    length, symbols = values.shape
    return pd.DataFrame(
        values,
        index=pd.date_range("2020-01-01", periods=length, tz="UTC"),
        columns=[f"S{index:04d}" for index in range(symbols)],
    )


def _measure(factory: Callable[[], Model], inputs: Any) -> tuple[float, ...]:
    model = factory()
    tracemalloc.start()
    started = time.perf_counter()
    prediction = model.predict(inputs)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if np.isinf(prediction.array).any():
        raise AssertionError("zoo hotspot emitted infinity")
    return elapsed, _state_mib(model.state), peak / 2**20


def main() -> None:
    rng = np.random.default_rng(20_260_819)
    print("case                                 time       state      peak")
    for missing_rate in (0.0, 0.2):
        for symbols in (1, 64, 512):
            for window in (5, 63, 252):
                length = max(384, 2 * window + 8)
                raw = rng.normal(size=(length, symbols))
                if missing_rate:
                    raw[rng.random(raw.shape) < missing_rate] = np.nan
                inputs = {
                    "returns": _frame(raw),
                    "wealth": _frame(np.exp(raw / 10.0)),
                }
                cases: tuple[tuple[str, Callable[[], Model], pd.DataFrame], ...] = (
                    (
                        "rci",
                        lambda w=window: zoo.rank_correlation_index("close", w),
                        inputs["returns"],
                    ),
                    (
                        "max-drawdown",
                        lambda w=window: zoo.rolling_max_drawdown("close", w),
                        inputs["wealth"],
                    ),
                    (
                        "expected-shortfall",
                        lambda w=window: zoo.rolling_historical_expected_shortfall(
                            "close", w, 0.975
                        ),
                        inputs["returns"],
                    ),
                    (
                        "count",
                        lambda w=window: zoo.rolling_count("close", w),
                        inputs["returns"],
                    ),
                    (
                        "trend-r2",
                        lambda w=window: zoo.rolling_linear_trend_r2("close", w),
                        inputs["returns"],
                    ),
                )
                density = "dense" if missing_rate == 0.0 else "nan20"
                for name, factory, frame in cases:
                    elapsed, state, peak = _measure(factory, frame)
                    label = f"{name} {density} S={symbols} W={window}"
            print(
                f"{label:36s} {elapsed * 1e3:7.1f} ms {state:7.2f} MiB {peak:7.2f} MiB"
            )

    for missing_rate in (0.0, 0.2):
        values = rng.normal(size=(384, 512))
        if missing_rate:
            values[rng.random(values.shape) < missing_rate] = np.nan
        close = _frame(100.0 + values)
        representative: dict[str, pd.DataFrame] = {
            "close": close,
            "high": _frame(101.0 + values),
            "low": _frame(99.0 + values),
            "fallback": _frame(100.0 + rng.normal(size=values.shape)),
        }
        density = "dense" if missing_rate == 0.0 else "nan20"
        cases = (
            ("ewm-state", lambda: zoo.recursive_ewm_mean("close", 0.1), close),
            ("ohlcv", lambda: zoo.typical_price(), representative),
            (
                "coalesce",
                lambda: zoo.coalesce(["close", "fallback"]),
                representative,
            ),
            (
                "ensemble",
                lambda: zoo.mean_ensemble(
                    [zoo.lag("close"), zoo.rolling_mean("close", 5)]
                ),
                close,
            ),
        )
        for name, factory, inputs in cases:
            elapsed, state, peak = _measure(factory, inputs)
            label = f"{name} {density} S=512"
            print(
                f"{label:36s} {elapsed * 1e3:7.1f} ms {state:7.2f} MiB {peak:7.2f} MiB"
            )

    for missing_rate in (0.0, 0.2):
        for symbols in (1, 64, 512):
            values = rng.normal(size=(384, symbols))
            if missing_rate:
                values[rng.random(values.shape) < missing_rate] = np.nan
            density = "dense" if missing_rate == 0.0 else "nan20"
            elapsed, state, peak = _measure(
                lambda: zoo.cross_sectional_rank("close"), _frame(values)
            )
            label = f"cross-rank {density} S={symbols}"
            print(
                f"{label:36s} {elapsed * 1e3:7.1f} ms {state:7.2f} MiB {peak:7.2f} MiB"
            )


if __name__ == "__main__":
    main()
