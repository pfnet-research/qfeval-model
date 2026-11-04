"""Compare stable fixed-capacity streaming kernels for ``window_sum``.

The existing two-stack aggregate queue is reported as a baseline. The three
implementation candidates are:

1. pairwise-resumming the complete fixed ring on every timestamp (O(window));
2. updating a fixed pairwise segment tree (O(log(window)));
3. merging a previous-chunk suffix with a current-chunk prefix (amortized O(1)).

All candidates bound accumulation to the active window and therefore avoid the
series-length-dependent error of subtracting two global cumulative sums.

Run with::

    uv run python benchmarks/bench_rolling_sum_candidates.py
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from qfeval_model.zoo._window import SumAggregate, _aggregate_queue, window_sum

Array = npt.NDArray[np.float64]
Kernel = Callable[[dict[str, Any], Array, int], SumAggregate | None]
LocalSumAggregate = tuple[Array, Array, npt.NDArray[np.int64]]


def _sum_singleton(x: Array) -> LocalSumAggregate:
    finite = ~np.isnan(x)
    return (
        np.where(finite, x, 0.0),
        np.zeros(x.shape, dtype=np.float64),
        finite.astype(np.int64),
    )


def _sum_merge(left: LocalSumAggregate, right: LocalSumAggregate) -> LocalSumAggregate:
    ref_left, relative_left, count_left = left
    ref_right, relative_right, count_right = right
    reference = np.where(count_left > 0, ref_left, ref_right)
    with np.errstate(over="ignore", invalid="ignore"):
        relative = (
            relative_left
            + (ref_left - reference) * count_left
            + relative_right
            + (ref_right - reference) * count_right
        )
    return reference, relative, count_left + count_right


def _two_stack(state: dict[str, Any], x: Array, window: int) -> SumAggregate | None:
    local = _aggregate_queue(
        state, x.copy(), window, "buffer", _sum_singleton, _sum_merge
    )
    if local is None:
        return None
    reference, relative, count = cast(LocalSumAggregate, local)
    with np.errstate(over="ignore", invalid="ignore"):
        total = reference * count + relative
    return np.where(np.isfinite(total), total, np.nan), count


def _ring_resum(state: dict[str, Any], x: Array, window: int) -> SumAggregate | None:
    tracker = state.get("buffer")
    if tracker is None:
        tracker = {
            "values": np.zeros((x.size, window), dtype=np.float64),
            "valid": np.zeros((x.size, window), dtype=np.bool_),
            "position": 0,
            "size": 0,
        }
        state["buffer"] = tracker

    position = tracker["position"]
    finite = ~np.isnan(x)
    slot = tracker["values"][:, position]
    slot.fill(0.0)
    np.copyto(slot, x, where=finite)
    tracker["valid"][:, position] = finite
    tracker["position"] = (position + 1) % window
    tracker["size"] = min(tracker["size"] + 1, window)
    if tracker["size"] < window:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        total = tracker["values"].sum(axis=1)
    count = tracker["valid"].sum(axis=1, dtype=np.int64)
    return np.where(np.isfinite(total), total, np.nan), count


def _segment_tree(state: dict[str, Any], x: Array, window: int) -> SumAggregate | None:
    tracker = state.get("buffer")
    if tracker is None:
        base = 1 << (window - 1).bit_length()
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "sums": np.zeros((2 * base, x.size), dtype=np.float64),
            "counts": np.zeros((2 * base, x.size), dtype=count_dtype),
            "base": base,
            "position": 0,
            "size": 0,
        }
        state["buffer"] = tracker

    position = tracker["position"]
    node = tracker["base"] + position
    finite = ~np.isnan(x)
    leaf = tracker["sums"][node]
    leaf.fill(0.0)
    np.copyto(leaf, x, where=finite)
    tracker["counts"][node] = finite
    with np.errstate(over="ignore", invalid="ignore"):
        while node > 1:
            node //= 2
            np.add(
                tracker["sums"][2 * node],
                tracker["sums"][2 * node + 1],
                out=tracker["sums"][node],
            )
            np.add(
                tracker["counts"][2 * node],
                tracker["counts"][2 * node + 1],
                out=tracker["counts"][node],
            )
    tracker["position"] = (position + 1) % window
    tracker["size"] = min(tracker["size"] + 1, window)
    if tracker["size"] < window:
        return None
    total = tracker["sums"][1]
    return np.where(np.isfinite(total), total, np.nan), tracker["counts"][1]


CANDIDATES: dict[str, Kernel] = {
    "ring-resum": _ring_resum,
    "segment-tree": _segment_tree,
    "chunk-prefix-suffix": window_sum,
}
KERNELS = {"previous-two-stack": _two_stack, **CANDIDATES}


def _execute(kernel: Kernel, values: Array, window: int) -> tuple[Array, Array]:
    state: dict[str, Any] = {}
    totals = np.full_like(values, np.nan)
    counts = np.full_like(values, np.nan)
    for index, row in enumerate(values):
        aggregate = kernel(state, row, window)
        if index >= window - 1 and aggregate is not None:
            totals[index], counts[index] = aggregate
    return totals, counts


def _reference(values: Array, window: int) -> tuple[Array, Array]:
    totals = np.full_like(values, np.nan)
    counts = np.full_like(values, np.nan)
    for end in range(window - 1, values.shape[0]):
        block = values[end - window + 1 : end + 1]
        for cell in range(values.shape[1]):
            finite = block[:, cell][~np.isnan(block[:, cell])]
            counts[end, cell] = finite.size
            try:
                totals[end, cell] = math.fsum(map(float, finite))
            except OverflowError:
                totals[end, cell] = np.nan
    return totals, counts


def _state_mib(kernel: Kernel, values: Array, window: int) -> float:
    state: dict[str, Any] = {}
    for row in values:
        kernel(state, row, window)

    seen: set[int] = set()

    def array_bytes(value: Any) -> int:
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            return value.nbytes
        if isinstance(value, dict):
            return sum(array_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(array_bytes(item) for item in value)
        return 0

    return array_bytes(state) / 2**20


def _check_accuracy() -> None:
    rng = np.random.default_rng(4)
    length, symbols, window = 4_000, 7, 63
    axis = np.arange(length, dtype=np.float64)[:, None]
    values = (
        1e12
        + 0.1 * np.sin(0.013 * axis + np.arange(symbols))
        + rng.normal(scale=0.01, size=(length, symbols))
    )
    values[rng.random(values.shape) < 0.13] = np.nan
    reference, reference_counts = _reference(values, window)
    finite = np.isfinite(reference)
    print("accuracy (large level + NaNs)")
    for name, kernel in KERNELS.items():
        got, counts = _execute(kernel, values, window)
        max_relative = np.max(
            np.abs(got[finite] - reference[finite]) / np.abs(reference[finite])
        )
        bad_nan = np.count_nonzero(np.isnan(got) != np.isnan(reference))
        np.testing.assert_array_equal(counts, reference_counts)
        assert max_relative <= window * np.finfo(np.float64).eps
        assert bad_nan == 0
        print(f"  {name:22s} max_rel={max_relative:.3g} bad_nan={bad_nan}")

    recovery = np.ones((2 * window + 2, symbols), dtype=np.float64)
    recovery[0] = 1e308
    recovery[1] = 1e308
    recovery[::11, 1::2] = np.nan
    reference, reference_counts = _reference(recovery, window)
    print("accuracy (overflow followed by recovery)")
    for name, kernel in KERNELS.items():
        got, counts = _execute(kernel, recovery, window)
        np.testing.assert_array_equal(counts, reference_counts)
        bad_nan = np.count_nonzero(np.isnan(got) != np.isnan(reference))
        if name in CANDIDATES:
            np.testing.assert_allclose(got, reference, rtol=2e-15, equal_nan=True)
        print(f"  {name:22s} bad_nan={bad_nan}")


def _median_time(fn: Callable[[], object], repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _bench_case(length: int, symbols: int, window: int, missing: float) -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(loc=100.0, scale=3.0, size=(length, symbols))
    values[rng.random(values.shape) < missing] = np.nan
    print(f"speed T={length}, S={symbols}, window={window}, missing={missing:.0%}")
    for name, kernel in KERNELS.items():
        elapsed = _median_time(partial(_execute, kernel, values, window))
        state_mib = _state_mib(kernel, values, window)
        print(f"  {name:22s} {elapsed * 1e3:8.1f} ms  state={state_mib:6.2f} MiB")


def main() -> None:
    _check_accuracy()
    print()
    _bench_case(length=2_000, symbols=64, window=16, missing=0.0)
    _bench_case(length=2_000, symbols=512, window=64, missing=0.0)
    _bench_case(length=2_000, symbols=512, window=64, missing=0.2)
    _bench_case(length=1_000, symbols=1_024, window=256, missing=0.2)


if __name__ == "__main__":
    main()
