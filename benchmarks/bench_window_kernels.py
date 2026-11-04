"""Benchmark the fixed-capacity kernels shared by rolling zoo models.

Run with::

    uv run python benchmarks/bench_window_kernels.py

The benchmark reports kernel-only time and persistent NumPy state.  Numerical
and streaming equivalence are covered by the zoo tests; this script is the
small, repeatable performance canary for their common implementation layer.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.zoo._window import (
    full_block,
    window_bivariate_moments,
    window_extrema,
    window_linear_stats,
    window_moments,
    window_product,
)

Array = npt.NDArray[np.float64]
Runner = Callable[[dict[str, Any]], None]


def _state_mib(state: dict[str, Any]) -> float:
    seen: set[int] = set()

    def size(value: Any) -> int:
        if isinstance(value, np.ndarray):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            return value.nbytes
        if isinstance(value, dict):
            return sum(size(item) for item in value.values())
        if isinstance(value, (list, tuple, deque)):
            return sum(size(item) for item in value)
        return 0

    return size(state) / 2**20


def _unary(
    kernel: Callable[..., object],
    values: Array,
    window: int,
    **kwargs: object,
) -> Runner:
    def run(state: dict[str, Any]) -> None:
        for row in values:
            kernel(state, row, window, **kwargs)

    return run


def _bivariate(x: Array, y: Array, window: int) -> Runner:
    def run(state: dict[str, Any]) -> None:
        for row_x, row_y in zip(x, y, strict=True):
            window_bivariate_moments(state, row_x, row_y, window)

    return run


def _measure(run: Runner, repeats: int = 5) -> tuple[float, float]:
    samples = []
    final_state: dict[str, Any] = {}
    for _ in range(repeats):
        state: dict[str, Any] = {}
        started = time.perf_counter()
        run(state)
        samples.append(time.perf_counter() - started)
        final_state = state
    return statistics.median(samples), _state_mib(final_state)


def main() -> None:
    length, symbols, window = 1_536, 128, 63
    rng = np.random.default_rng(482)
    x = 1e6 + rng.normal(size=(length, symbols))
    y = -2e6 + rng.normal(size=(length, symbols))
    x[211::397, ::17] = np.nan
    y[307::431, 3::19] = np.nan
    product_input = 1.0 + x * 1e-7

    cases: list[tuple[str, Runner]] = [
        ("moments", _unary(window_moments, x, window)),
        ("bivariate", _bivariate(x, y, window)),
        ("product", _unary(window_product, product_input, window)),
        ("extrema-values", _unary(window_extrema, x, window)),
        ("linear-stats", _unary(window_linear_stats, x, window)),
        ("full-block", _unary(full_block, x, window)),
    ]
    print(f"T={length}, S={symbols}, window={window}")
    for name, run in cases:
        elapsed, state_mib = _measure(run)
        print(
            f"  {name:16s} {elapsed * 1e3:8.1f} ms  "
            f"{length / elapsed:9.0f} rows/s  state={state_mib:5.2f} MiB"
        )


if __name__ == "__main__":
    main()
