"""Compare stable streaming kernels for ``rolling_valid_sum``.

This is an implementation-selection benchmark, not a public API benchmark.
It compares three fixed-capacity designs which never subtract an expired
observation from a long-lived total:

1. resumming the complete ring on every update (O(window));
2. updating a pairwise segment tree (O(log(window)));
3. merging a previous-chunk suffix with a current-chunk prefix (amortized O(1)).

Run with::

    uv run python benchmarks/bench_rolling_valid_sum_candidates.py
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qfeval_model.zoo._window import window_valid_sum

Array = npt.NDArray[np.float64]
Kernel = Callable[[dict[str, Any], Array, int], tuple[Array, npt.NDArray[np.int64]]]


def _ring_resum(
    state: dict[str, Any], x: Array, window: int
) -> tuple[Array, npt.NDArray[np.int64]]:
    tracker = state.get("buffer")
    if tracker is None:
        tracker = {
            "values": np.zeros((x.size, window), dtype=np.float64),
            "positions": np.zeros(x.size, dtype=np.int64),
            "counts": np.zeros(x.size, dtype=np.int64),
        }
        state["buffer"] = tracker
    values = tracker["values"]
    positions = tracker["positions"]
    counts = tracker["counts"]
    finite = ~np.isnan(x)
    cells = np.flatnonzero(finite)
    values[cells, positions[cells]] = x[cells]
    positions[cells] = (positions[cells] + 1) % window
    counts[cells] = np.minimum(counts[cells] + 1, window)
    with np.errstate(over="ignore", invalid="ignore"):
        total = values.sum(axis=1)
    return np.where(np.isfinite(total), total, np.nan), counts


def _segment_tree(
    state: dict[str, Any], x: Array, window: int
) -> tuple[Array, npt.NDArray[np.int64]]:
    tracker = state.get("buffer")
    if tracker is None:
        base = 1 << (window - 1).bit_length()
        tracker = {
            "tree": np.zeros((x.size, 2 * base), dtype=np.float64),
            "base": base,
            "positions": np.zeros(x.size, dtype=np.int64),
            "counts": np.zeros(x.size, dtype=np.int64),
        }
        state["buffer"] = tracker
    tree = tracker["tree"]
    base = tracker["base"]
    positions = tracker["positions"]
    counts = tracker["counts"]
    finite = ~np.isnan(x)
    cells = np.flatnonzero(finite)
    nodes = base + positions[cells]
    tree[cells, nodes] = x[cells]
    with np.errstate(over="ignore", invalid="ignore"):
        while nodes.size and nodes[0] > 1:
            nodes //= 2
            tree[cells, nodes] = tree[cells, 2 * nodes] + tree[cells, 2 * nodes + 1]
    positions[cells] = (positions[cells] + 1) % window
    counts[cells] = np.minimum(counts[cells] + 1, window)
    total = tree[:, 1].copy()
    return np.where(np.isfinite(total), total, np.nan), counts


KERNELS: dict[str, Kernel] = {
    "ring-resum": _ring_resum,
    "segment-tree": _segment_tree,
    "chunk-prefix-suffix": window_valid_sum,
}


def _execute(kernel: Kernel, values: Array, window: int) -> Array:
    state: dict[str, Any] = {}
    output = np.empty_like(values)
    for i, row in enumerate(values):
        total, count = kernel(state, row, window)
        output[i] = np.where(count == window, total, np.nan)
    return output


def _state_mib(kernel: Kernel, values: Array, window: int) -> float:
    state: dict[str, Any] = {}
    for row in values:
        kernel(state, row, window)

    def array_bytes(value: Any) -> int:
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(array_bytes(item) for item in value.values())
        return 0

    return array_bytes(state) / 2**20


def _reference(values: Array, window: int) -> Array:
    histories: list[list[float]] = [[] for _ in range(values.shape[1])]
    output = np.full_like(values, np.nan)
    for i, row in enumerate(values):
        for cell, value in enumerate(row):
            if not math.isnan(float(value)):
                histories[cell].append(float(value))
                histories[cell] = histories[cell][-window:]
            if len(histories[cell]) == window:
                try:
                    output[i, cell] = math.fsum(histories[cell])
                except OverflowError:
                    output[i, cell] = np.nan
    return output


def _check_accuracy() -> None:
    rng = np.random.default_rng(0)
    length, symbols, window = 4_000, 7, 63
    time_axis = np.arange(length, dtype=np.float64)[:, None]
    values = (
        1e12
        + 0.1 * np.sin(0.013 * time_axis + np.arange(symbols))
        + rng.normal(scale=0.01, size=(length, symbols))
    )
    values[rng.random(values.shape) < 0.13] = np.nan
    reference = _reference(values, window)
    finite = np.isfinite(reference)
    print("accuracy (large level + asynchronous NaNs)")
    for name, kernel in KERNELS.items():
        got = _execute(kernel, values, window)
        max_abs = np.max(np.abs(got[finite] - reference[finite]))
        max_rel = np.max(
            np.abs(got[finite] - reference[finite]) / np.abs(reference[finite])
        )
        mismatched_nan = np.count_nonzero(np.isnan(got) != np.isnan(reference))
        assert max_rel <= window * np.finfo(np.float64).eps
        assert mismatched_nan == 0
        print(
            f"  {name:22s} max_abs={max_abs:.6g} max_rel={max_rel:.3g} "
            f"bad_nan={mismatched_nan}"
        )

    recovery = np.ones((2 * window + 2, symbols), dtype=np.float64)
    recovery[0] = 1e308
    recovery[1] = 1e308
    recovery[::11, 1::2] = np.nan
    reference = _reference(recovery, window)
    print("accuracy (overflow followed by recovery)")
    for name, kernel in KERNELS.items():
        got = _execute(kernel, recovery, window)
        np.testing.assert_allclose(got, reference, rtol=2e-15, equal_nan=True)
        print(f"  {name:22s} pass")


def _median_time(fn: Callable[[], object], repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _bench_case(length: int, symbols: int, window: int, missing: float) -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(loc=100.0, scale=3.0, size=(length, symbols))
    values[rng.random(values.shape) < missing] = np.nan
    print(f"speed T={length}, S={symbols}, window={window}, missing={missing:.0%}")
    for name, kernel in KERNELS.items():
        elapsed = _median_time(lambda kernel=kernel: _execute(kernel, values, window))
        state_mib = _state_mib(kernel, values, window)
        print(f"  {name:22s} {elapsed * 1e3:8.1f} ms  state={state_mib:5.2f} MiB")


def main() -> None:
    _check_accuracy()
    print()
    _bench_case(length=2_000, symbols=64, window=16, missing=0.0)
    _bench_case(length=2_000, symbols=512, window=64, missing=0.0)
    _bench_case(length=2_000, symbols=512, window=64, missing=0.2)
    _bench_case(length=1_000, symbols=1_024, window=256, missing=0.2)


if __name__ == "__main__":
    main()
