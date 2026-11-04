"""Private kernels for strict trailing windows.

Reducible statistics use fixed ``window``-length chunks: every window is a
previous chunk suffix plus a current chunk prefix, both accumulated over at
most ``window`` values. Moment statistics locally center those chunks and join
them with Chan's stable formula. Extrema that need tie-resolving absolute
indices use the streaming equivalent, a two-stack aggregate queue.

Checkpoint state contains only dictionaries, lists, scalars, and owned
arrays. Statistics whose order cannot be reduced this way (quantile,
median, rank, and path-dependent drawdown) use :func:`full_block` instead.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any, Literal, cast, overload

import numpy as np
import numpy.typing as npt


def _index(value: int, name: str) -> int:
    """Return an integer parameter, rejecting floats and booleans explicitly."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def validate_window(window: int, minimum: int = 1) -> None:
    """Reject a non-integer window shorter than ``minimum`` steps."""
    validate_integer(window, "window", minimum)


def validate_integer(value: int, name: str, minimum: int = 0) -> None:
    """Reject a non-integer value or one below ``minimum``."""
    value = _index(value, name)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


QUANTILE_METHODS = ("linear", "inverted_cdf")


def validate_quantile_method(method: str) -> Literal["linear", "inverted_cdf"]:
    """Validate and narrow a quantile ``method`` to the supported pair."""
    if method not in QUANTILE_METHODS:
        raise ValueError(f"method must be one of {QUANTILE_METHODS}, got {method!r}")
    return cast('Literal["linear", "inverted_cdf"]', method)


def validate_ddof(window: int, ddof: int) -> None:
    """Reject invalid integer ``window`` / ``ddof`` combinations."""
    validate_integer(ddof, "ddof")
    if _index(window, "window") <= ddof:
        raise ValueError(f"window must be > ddof ({ddof}), got {window}")


def full_block(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> npt.NDArray[Any]:
    """Push ``x`` and expose its NaN-padded window from a duplicated ring.

    The second copy turns every wrapped window into one contiguous slice.  It
    costs one extra cell-by-window buffer, but avoids allocating and copying a
    complete block at every timestamp. Initializing the ring with NaN also
    makes the same slice naturally left-padded during warm-up.
    """
    tracker = state.get(key)
    if tracker is None:
        tracker = {
            "values": np.full((2 * window, *x.shape), np.nan, dtype=np.float64),
            "position": 0,
        }
        state[key] = tracker
    values = cast(npt.NDArray[Any], tracker["values"])
    position = cast(int, tracker["position"])
    values[position] = x
    values[position + window] = x
    position = (position + 1) % window
    tracker["position"] = position
    return values[position : position + window]


Aggregate = Any
RawValue = Any
Entry = tuple[RawValue, Aggregate]


def _aggregate_queue(
    state: dict[str, Any],
    raw: RawValue,
    window: int,
    key: str,
    singleton: Callable[[RawValue], Aggregate],
    merge: Callable[[Aggregate, Aggregate], Aggregate],
) -> Aggregate:
    """Push one raw value and return the current trailing aggregate.

    ``front[-1]`` is the oldest queue item and contains the aggregate from
    that item through the end of the front stack. ``back[-1]`` is the newest
    item and contains the aggregate from the start of the back stack through
    that item. Their merge is the complete chronological window.
    """
    tracker = state.get(key)
    if tracker is None:
        tracker = {"front": [], "back": []}
        state[key] = tracker
    front = cast(list[Entry], tracker["front"])
    back = cast(list[Entry], tracker["back"])

    if len(front) + len(back) == window:
        if not front:
            while back:
                moved_raw, _ = back.pop()
                aggregate = singleton(moved_raw)
                if front:
                    aggregate = merge(aggregate, front[-1][1])
                front.append((moved_raw, aggregate))
        front.pop()

    aggregate = singleton(raw)
    if back:
        aggregate = merge(back[-1][1], aggregate)
    back.append((raw, aggregate))
    if front:
        return merge(front[-1][1], back[-1][1])
    return back[-1][1]


SumAggregate = tuple[npt.NDArray[Any], npt.NDArray[np.int64]]


def window_sum(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> SumAggregate:
    """Return a stable rolling finite-value sum/count in amortized O(1) time.

    Timestamps are split into chunks of ``window`` rows. A rolling window is
    the previous chunk's suffix plus the current chunk's prefix, so no expired
    value is subtracted from a long-lived total. Prefix sums are updated per
    timestamp and all suffix sums are built at a chunk boundary. Each partial
    sum therefore spans at most ``window`` values and its error cannot grow
    with the full series length.

    State consists only of fixed cell-by-window arrays. Counts use the smallest
    unsigned dtype capable of holding ``window``; their public working output
    remains int64. Dense input needs only the two sum buffers. Missing-data
    buffers are allocated at most once, on the first NaN, and then reused.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "values": np.zeros((window, cell_count), dtype=np.float64),
            "valid": None,
            "right": np.zeros((window + 1, cell_count), dtype=np.float64),
            "right_counts": None,
            "left": np.zeros(cell_count, dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "totals": np.zeros(cell_count, dtype=np.float64),
            "counts": np.zeros(cell_count, dtype=np.int64),
            "position": 0,
            "size": 0,
            "dense_counts": True,
        }
        state[key] = tracker

    values = cast(npt.NDArray[Any], tracker["values"])
    valid = cast(npt.NDArray[np.bool_] | None, tracker["valid"])
    right = cast(npt.NDArray[Any], tracker["right"])
    right_counts = cast(npt.NDArray[Any] | None, tracker["right_counts"])
    left = cast(npt.NDArray[Any], tracker["left"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    totals = cast(npt.NDArray[Any], tracker["totals"])
    counts = cast(npt.NDArray[np.int64], tracker["counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    slot = values[position]
    dense_counts = cast(bool, tracker["dense_counts"])
    if dense_counts and finite.all():
        slot[:] = flat_x
    else:
        if dense_counts:
            count_dtype = left_counts.dtype
            dense_right_counts = np.arange(window, -1, -1, dtype=count_dtype).reshape(
                -1, 1
            )
            valid = np.ones_like(values, dtype=np.bool_)
            if tracker["size"] == window:
                right_counts = np.broadcast_to(dense_right_counts, right.shape).copy()
            else:
                right_counts = np.zeros(right.shape, dtype=count_dtype)
            tracker["valid"] = valid
            tracker["right_counts"] = right_counts
            left_counts.fill(position)
            tracker["dense_counts"] = False
            dense_counts = False
        valid_array = cast(npt.NDArray[np.bool_], valid)
        slot.fill(0.0)
        np.copyto(slot, flat_x, where=finite)
        valid_array[position] = finite
    with np.errstate(over="ignore", invalid="ignore"):
        np.add(left, slot, out=left)
        np.add(left, right[position + 1], out=totals)
    if not dense_counts:
        right_count_array = cast(npt.NDArray[Any], right_counts)
        np.add(left_counts, finite, out=left_counts)
        np.add(left_counts, right_count_array[position + 1], out=counts)

    size = min(cast(int, tracker["size"]) + 1, window)
    tracker["size"] = size
    if dense_counts:
        counts.fill(size)
    if position == window - 1:
        # The arrays are (window, cells), so every add scans one contiguous
        # symbol row. This is faster in NumPy than a strided axis-0 cumsum and
        # is exactly the same reverse cumulative sum.
        with np.errstate(over="ignore", invalid="ignore"):
            if dense_counts:
                for index in range(window - 1, -1, -1):
                    np.add(values[index], right[index + 1], out=right[index])
            else:
                valid_array = cast(npt.NDArray[np.bool_], valid)
                right_count_array = cast(npt.NDArray[Any], right_counts)
                for index in range(window - 1, -1, -1):
                    np.add(values[index], right[index + 1], out=right[index])
                    np.add(
                        valid_array[index],
                        right_count_array[index + 1],
                        out=right_count_array[index],
                    )
        left.fill(0.0)
        left_counts.fill(0)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1

    return (
        np.where(np.isfinite(totals), totals, np.nan).reshape(x.shape),
        counts.reshape(x.shape),
    )


BooleanCountAggregate = tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    int,
]


def window_boolean_count(
    state: dict[str, Any],
    truth: npt.NDArray[np.bool_],
    window: int,
    key: str = "buffer",
    *,
    valid: npt.NDArray[np.bool_] | None = None,
) -> BooleanCountAggregate:
    """Count true and valid cells with an exact boolean ring.

    Unlike :func:`window_sum`, count-only callers need no float64 value or
    stable-summation buffers.  Expiration is exact integer ``new - old`` and
    state is one (or, with ``valid``, two) compact boolean rings. When
    ``valid`` is omitted, the second result is the true count as well.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = truth.size
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "truth": np.zeros((window, cell_count), dtype=np.bool_),
            "valid": (
                None
                if valid is None
                else np.zeros((window, cell_count), dtype=np.bool_)
            ),
            "truth_counts": np.zeros(cell_count, dtype=count_dtype),
            "valid_counts": (
                None if valid is None else np.zeros(cell_count, dtype=count_dtype)
            ),
            "position": 0,
            "size": 0,
        }
        state[key] = tracker
    truth_ring = cast(npt.NDArray[np.bool_], tracker["truth"])
    valid_ring = cast(npt.NDArray[np.bool_] | None, tracker["valid"])
    truth_counts = cast(npt.NDArray[Any], tracker["truth_counts"])
    valid_counts = cast(npt.NDArray[Any] | None, tracker["valid_counts"])
    position = cast(int, tracker["position"])
    size = cast(int, tracker["size"])

    flat_truth = truth.reshape(-1)
    if valid is not None:
        flat_valid = valid.reshape(-1)
        flat_truth = flat_truth & flat_valid
    else:
        flat_valid = None
    if size == window:
        np.subtract(truth_counts, truth_ring[position], out=truth_counts)
        if valid_ring is not None and valid_counts is not None:
            np.subtract(valid_counts, valid_ring[position], out=valid_counts)
    else:
        size += 1
        tracker["size"] = size

    truth_ring[position] = flat_truth
    np.add(truth_counts, flat_truth, out=truth_counts)
    if valid_ring is not None and valid_counts is not None and flat_valid is not None:
        valid_ring[position] = flat_valid
        np.add(valid_counts, flat_valid, out=valid_counts)
    tracker["position"] = (position + 1) % window

    return (
        truth_counts.reshape(truth.shape),
        (truth_counts if valid_counts is None else valid_counts).reshape(truth.shape),
        size,
    )


def window_multi_sum(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> SumAggregate:
    """Return sums of jointly-valid fields and one shared finite count.

    ``x`` has shape ``(fields, *cell_shape)``.  If any field is NaN, the
    entire observation is missing for that cell.  The prefix/suffix layout
    matches :func:`window_sum`, while counts are stored only once per cell.
    """
    fields = x.shape[0]
    cell_shape = x.shape[1:]
    cell_count = math.prod(cell_shape)
    tracker = state.get(key)
    if tracker is None:
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "values": np.zeros((window, fields, cell_count), dtype=np.float64),
            "valid": np.zeros((window, cell_count), dtype=np.bool_),
            "right": np.zeros((window + 1, fields, cell_count), dtype=np.float64),
            "right_counts": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "left": np.zeros((fields, cell_count), dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "totals": np.zeros((fields, cell_count), dtype=np.float64),
            "counts": np.zeros(cell_count, dtype=np.int64),
            "position": 0,
            "size": 0,
        }
        state[key] = tracker

    values = cast(npt.NDArray[Any], tracker["values"])
    valid_ring = cast(npt.NDArray[np.bool_], tracker["valid"])
    right = cast(npt.NDArray[Any], tracker["right"])
    right_counts = cast(npt.NDArray[Any], tracker["right_counts"])
    left = cast(npt.NDArray[Any], tracker["left"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    totals = cast(npt.NDArray[Any], tracker["totals"])
    counts = cast(npt.NDArray[np.int64], tracker["counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(fields, cell_count)
    finite = ~np.isnan(flat_x).any(axis=0)
    slot = values[position]
    slot.fill(0.0)
    np.copyto(slot, flat_x, where=finite[np.newaxis])
    valid_ring[position] = finite
    with np.errstate(over="ignore", invalid="ignore"):
        np.add(left, slot, out=left)
        np.add(left, right[position + 1], out=totals)
    np.add(left_counts, finite, out=left_counts)
    np.add(left_counts, right_counts[position + 1], out=counts)

    size = min(cast(int, tracker["size"]) + 1, window)
    tracker["size"] = size
    if position == window - 1:
        with np.errstate(over="ignore", invalid="ignore"):
            for index in range(window - 1, -1, -1):
                np.add(values[index], right[index + 1], out=right[index])
                np.add(
                    valid_ring[index],
                    right_counts[index + 1],
                    out=right_counts[index],
                )
        left.fill(0.0)
        left_counts.fill(0)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1

    return (
        np.where(np.isfinite(totals), totals, np.nan).reshape((fields, *cell_shape)),
        counts.reshape(cell_shape),
    )


def window_valid_sum(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> SumAggregate:
    """Return the sum/count of each cell's latest ``window`` finite values.

    Each cell's finite observations are split into chunks of ``window`` values.
    A rolling window is the previous chunk's suffix plus the current chunk's
    prefix, so it can be returned without subtracting an expired value from a
    long-lived total. Prefix sums are updated online; suffix sums are built in
    one reverse cumulative sum when a chunk fills. Work is O(1) amortized per
    value, roundoff is confined to the active window, and a past extreme value
    cannot affect the result after it leaves.

    ``values`` and ``right`` are fixed cell-by-window buffers. Dense aligned
    input writes ``right`` directly; asynchronous missingness gets one lazily
    allocated fixed-size scratch buffer which is then reused at every boundary.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        tracker = {
            "values": np.zeros((cell_count, window), dtype=np.float64),
            "right": np.zeros((cell_count, window + 1), dtype=np.float64),
            "scratch": None,
            "positions": np.zeros(cell_count, dtype=np.int64),
            "counts": np.zeros(cell_count, dtype=np.int64),
            "left": np.zeros(cell_count, dtype=np.float64),
            "totals": np.zeros(cell_count, dtype=np.float64),
            "synchronized": True,
        }
        state[key] = tracker

    values = cast(npt.NDArray[Any], tracker["values"])
    right = cast(npt.NDArray[Any], tracker["right"])
    positions = cast(npt.NDArray[np.int64], tracker["positions"])
    counts = cast(npt.NDArray[np.int64], tracker["counts"])
    left = cast(npt.NDArray[Any], tracker["left"])
    totals = cast(npt.NDArray[Any], tracker["totals"])
    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)

    # Until cells first diverge due to missingness, all of them share one
    # position. This is the common dense path and avoids indexing or scratch.
    if tracker["synchronized"]:
        if finite.all() and flat_x.size:
            position = int(positions[0])
            values[:, position] = flat_x
            with np.errstate(over="ignore", invalid="ignore"):
                left += flat_x
                totals[:] = left + right[:, position + 1]
            np.add(counts, 1, out=counts)
            np.minimum(counts, window, out=counts)
            if position == window - 1:
                with np.errstate(over="ignore", invalid="ignore"):
                    np.cumsum(
                        values[:, ::-1],
                        axis=1,
                        out=right[:, window - 1 :: -1],
                    )
                left.fill(0.0)
                positions.fill(0)
            else:
                positions += 1
            return (
                np.where(np.isfinite(totals), totals, np.nan).reshape(x.shape),
                counts.reshape(x.shape),
            )
        if not finite.any():
            return (
                np.where(np.isfinite(totals), totals, np.nan).reshape(x.shape),
                counts.reshape(x.shape),
            )
        tracker["synchronized"] = False

    # Missing cells do not advance. Boundary columns can therefore be
    # different on every call, so gather them into one persistent workspace
    # and let NumPy perform each reverse scan as a single vectorized operation.
    cells = np.flatnonzero(finite)
    current_positions = positions[cells]
    values[cells, current_positions] = flat_x[cells]
    with np.errstate(over="ignore", invalid="ignore"):
        left[cells] += flat_x[cells]
        totals[cells] = left[cells] + right[cells, current_positions + 1]
    counts[cells] = np.minimum(counts[cells] + 1, window)

    completed = cells[current_positions == window - 1]
    if completed.size:
        scratch = cast(npt.NDArray[Any] | None, tracker["scratch"])
        if scratch is None:
            scratch = np.zeros_like(values)
            tracker["scratch"] = scratch
        work = scratch[: completed.size]
        np.take(values, completed, axis=0, out=work)
        with np.errstate(over="ignore", invalid="ignore"):
            np.cumsum(work[:, ::-1], axis=1, out=work[:, ::-1])
        right[completed, :window] = work
        left[completed] = 0.0

    positions[cells] = (current_positions + 1) % window
    return (
        np.where(np.isfinite(totals), totals, np.nan).reshape(x.shape),
        counts.reshape(x.shape),
    )


def _accumulate_suffixes(window: int, *arrays: npt.NDArray[Any]) -> None:
    """Turn chunk slots into suffix aggregates: an in-place reverse cumsum.

    Each array has ``window + 1`` rows whose last row is a permanent zero
    sentinel, so row ``window - 1`` is already its own suffix.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(window - 2, -1, -1):
            for array in arrays:
                np.add(array[index], array[index + 1], out=array[index])


def _merge_moments(
    left_sum: npt.NDArray[Any],
    left_square: npt.NDArray[Any],
    left_counts: npt.NDArray[Any],
    left_reference: npt.NDArray[Any],
    right_sum: npt.NDArray[Any],
    right_square: npt.NDArray[Any],
    right_counts: npt.NDArray[Any],
    right_reference: npt.NDArray[Any],
    total_counts: npt.NDArray[Any] | int,
) -> tuple[npt.NDArray[Any], ...]:
    """Join two locally centered chunk moments with Chan's formula."""
    with np.errstate(over="ignore", invalid="ignore"):
        left_relative = np.divide(
            left_sum,
            left_counts,
            out=np.zeros_like(left_sum),
            where=left_counts > 0,
        )
        right_relative = np.divide(
            right_sum,
            right_counts,
            out=np.zeros_like(right_sum),
            where=right_counts > 0,
        )
        left_m2 = left_square - left_sum * left_relative
        right_m2 = right_square - right_sum * right_relative
        delta = (left_reference - right_reference) + (left_relative - right_relative)
        left_float = left_counts.astype(np.float64)
        right_float = right_counts.astype(np.float64)
        cross_weight = np.divide(
            left_float * right_float,
            total_counts,
            out=np.zeros_like(left_sum),
            where=total_counts > 0,
        )
        m2 = left_m2 + right_m2 + delta * delta * cross_weight
        left_weight = np.divide(
            left_float,
            total_counts,
            out=np.zeros_like(left_sum),
            where=total_counts > 0,
        )
        mean = right_reference + right_relative + delta * left_weight
        left_mean = left_reference + left_relative
    mean = np.where(right_counts > 0, mean, left_mean)
    m2 = np.where(right_counts > 0, m2, left_m2)
    mean[~np.isfinite(mean)] = np.nan
    m2[~np.isfinite(m2)] = np.nan
    np.maximum(m2, 0.0, out=m2)
    return mean, m2, left_relative, right_relative, delta, cross_weight


MomentAggregate = tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64]]


def window_moments(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> MomentAggregate:
    """Return rolling ``(mean, M2, finite_count)`` from local chunks.

    Prefixes and suffixes are centered on a value contained in that part, so
    all accumulated values are local deviations instead of the possibly huge
    series level.  Their deviation sums and squared-deviation sums give each
    part's M2; the two parts are joined with Chan's formula.  The raw prefix
    buffer becomes the next suffix sum in place at a chunk boundary.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "sums": np.zeros((window + 1, cell_count), dtype=np.float64),
            "squares": np.zeros((window + 1, cell_count), dtype=np.float64),
            "counts": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "left_reference": np.zeros(cell_count, dtype=np.float64),
            "right_reference": np.zeros(cell_count, dtype=np.float64),
            "left_sum": np.zeros(cell_count, dtype=np.float64),
            "left_square": np.zeros(cell_count, dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "total_counts": np.zeros(cell_count, dtype=np.int64),
            "position": 0,
        }
        state[key] = tracker

    sums = cast(npt.NDArray[Any], tracker["sums"])
    squares = cast(npt.NDArray[Any], tracker["squares"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    left_reference = cast(npt.NDArray[Any], tracker["left_reference"])
    right_reference = cast(npt.NDArray[Any], tracker["right_reference"])
    left_sum = cast(npt.NDArray[Any], tracker["left_sum"])
    left_square = cast(npt.NDArray[Any], tracker["left_square"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    sums[position] = flat_x
    counts[position] = finite
    if position == 0:
        np.copyto(left_reference, flat_x)
        left_sum.fill(0.0)
        left_square.fill(0.0)
        left_counts[:] = finite
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            deviation = np.where(finite, flat_x - left_reference, 0.0)
            left_sum += deviation
            left_square += deviation * deviation
        left_counts += finite

    np.add(left_counts, counts[position + 1], out=total_counts)
    mean, m2, *_ = _merge_moments(
        left_sum,
        left_square,
        left_counts,
        left_reference,
        sums[position + 1],
        squares[position + 1],
        counts[position + 1],
        right_reference,
        total_counts,
    )
    result = (
        mean.reshape(x.shape),
        m2.reshape(x.shape),
        total_counts.reshape(x.shape),
    )

    if position == window - 1:
        np.copyto(right_reference, sums[window - 1])
        with np.errstate(over="ignore", invalid="ignore"):
            np.subtract(
                sums[:window],
                right_reference,
                out=sums[:window],
            )
            sums[:window][counts[:window] == 0] = 0.0
            np.square(sums[:window], out=squares[:window])
        _accumulate_suffixes(window, sums, squares, counts)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1
    return result


def window_valid_moments(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> MomentAggregate:
    """Return ``(mean, M2, finite_count)`` of each cell's latest ``window``
    finite observations.

    The per-cell chunk layout of :func:`window_valid_sum` applied to the
    centered statistics of :func:`window_moments`: each cell's finite
    observations form chunks of ``window`` values, a window is the previous
    chunk's suffix plus the current chunk's prefix, and both parts accumulate
    deviations from a value they contain (the chunk's last and first value)
    before joining with Chan's formula. Roundoff therefore stays local to the
    active window and an overflowed chunk cannot contaminate windows after it
    expires. NaN input leaves a cell's state and its last aggregate untouched;
    counts saturate at ``window``.

    Only cells with finite input advance, so boundary columns differ per call;
    completed chunks rebuild their suffix arrays through two fixed lazily
    allocated scratch buffers.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        tracker = {
            "values": np.zeros((cell_count, window), dtype=np.float64),
            "right_sums": np.zeros((cell_count, window + 1), dtype=np.float64),
            "right_squares": np.zeros((cell_count, window + 1), dtype=np.float64),
            "scratch_sums": None,
            "scratch_squares": None,
            "left_reference": np.zeros(cell_count, dtype=np.float64),
            "right_reference": np.zeros(cell_count, dtype=np.float64),
            "left_sum": np.zeros(cell_count, dtype=np.float64),
            "left_square": np.zeros(cell_count, dtype=np.float64),
            "positions": np.zeros(cell_count, dtype=np.int64),
            "counts": np.zeros(cell_count, dtype=np.int64),
            "means": np.full(cell_count, np.nan, dtype=np.float64),
            "m2s": np.full(cell_count, np.nan, dtype=np.float64),
        }
        state[key] = tracker

    values = cast(npt.NDArray[Any], tracker["values"])
    right_sums = cast(npt.NDArray[Any], tracker["right_sums"])
    right_squares = cast(npt.NDArray[Any], tracker["right_squares"])
    left_reference = cast(npt.NDArray[Any], tracker["left_reference"])
    right_reference = cast(npt.NDArray[Any], tracker["right_reference"])
    left_sum = cast(npt.NDArray[Any], tracker["left_sum"])
    left_square = cast(npt.NDArray[Any], tracker["left_square"])
    positions = cast(npt.NDArray[np.int64], tracker["positions"])
    counts = cast(npt.NDArray[np.int64], tracker["counts"])
    means = cast(npt.NDArray[Any], tracker["means"])
    m2s = cast(npt.NDArray[Any], tracker["m2s"])

    flat_x = x.reshape(-1)
    cells = np.flatnonzero(~np.isnan(flat_x))
    if cells.size:
        current_positions = positions[cells]
        values[cells, current_positions] = flat_x[cells]
        starting_cells = cells[current_positions == 0]
        left_reference[starting_cells] = flat_x[starting_cells]
        left_sum[starting_cells] = 0.0
        left_square[starting_cells] = 0.0
        continuing_cells = cells[current_positions > 0]
        with np.errstate(over="ignore", invalid="ignore"):
            deviation = flat_x[continuing_cells] - left_reference[continuing_cells]
            left_sum[continuing_cells] += deviation
            left_square[continuing_cells] += deviation * deviation
        counts[cells] = np.minimum(counts[cells] + 1, window)

        left_counts = (current_positions + 1).astype(np.float64)
        mean, m2, *_ = _merge_moments(
            left_sum[cells],
            left_square[cells],
            left_counts,
            left_reference[cells],
            right_sums[cells, current_positions + 1],
            right_squares[cells, current_positions + 1],
            window - left_counts,
            right_reference[cells],
            window,
        )
        means[cells] = mean
        m2s[cells] = m2

        completed = cells[current_positions == window - 1]
        if completed.size:
            scratch_sums = cast(npt.NDArray[Any] | None, tracker["scratch_sums"])
            scratch_squares = cast(npt.NDArray[Any] | None, tracker["scratch_squares"])
            if scratch_sums is None or scratch_squares is None:
                scratch_sums = np.zeros_like(values)
                scratch_squares = np.zeros_like(values)
                tracker["scratch_sums"] = scratch_sums
                tracker["scratch_squares"] = scratch_squares
            work = scratch_sums[: completed.size]
            square = scratch_squares[: completed.size]
            np.take(values, completed, axis=0, out=work)
            reference = flat_x[completed]
            with np.errstate(over="ignore", invalid="ignore"):
                work -= reference[:, np.newaxis]
                np.square(work, out=square)
                np.cumsum(work[:, ::-1], axis=1, out=work[:, ::-1])
                np.cumsum(square[:, ::-1], axis=1, out=square[:, ::-1])
            right_sums[completed, :window] = work
            right_squares[completed, :window] = square
            right_reference[completed] = reference
        positions[cells] = (current_positions + 1) % window

    return (
        means.reshape(x.shape),
        m2s.reshape(x.shape),
        counts.reshape(x.shape),
    )


BivariateAggregate = tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[np.int64],
]


def window_bivariate_moments(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> BivariateAggregate:
    """Return stable bivariate moments from two local chunk statistics."""
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        buffer_shape = (window + 1, cell_count)
        tracker = {
            "sum_x": np.zeros(buffer_shape, dtype=np.float64),
            "sum_y": np.zeros(buffer_shape, dtype=np.float64),
            "square_x": np.zeros(buffer_shape, dtype=np.float64),
            "square_y": np.zeros(buffer_shape, dtype=np.float64),
            "cross": np.zeros(buffer_shape, dtype=np.float64),
            "counts": np.zeros(buffer_shape, dtype=count_dtype),
            "left_reference_x": np.zeros(cell_count, dtype=np.float64),
            "left_reference_y": np.zeros(cell_count, dtype=np.float64),
            "right_reference_x": np.zeros(cell_count, dtype=np.float64),
            "right_reference_y": np.zeros(cell_count, dtype=np.float64),
            "left_sum_x": np.zeros(cell_count, dtype=np.float64),
            "left_sum_y": np.zeros(cell_count, dtype=np.float64),
            "left_square_x": np.zeros(cell_count, dtype=np.float64),
            "left_square_y": np.zeros(cell_count, dtype=np.float64),
            "left_cross": np.zeros(cell_count, dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "total_counts": np.zeros(cell_count, dtype=np.int64),
            "position": 0,
        }
        state[key] = tracker

    sum_x = cast(npt.NDArray[Any], tracker["sum_x"])
    sum_y = cast(npt.NDArray[Any], tracker["sum_y"])
    square_x = cast(npt.NDArray[Any], tracker["square_x"])
    square_y = cast(npt.NDArray[Any], tracker["square_y"])
    cross = cast(npt.NDArray[Any], tracker["cross"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    left_reference_x = cast(npt.NDArray[Any], tracker["left_reference_x"])
    left_reference_y = cast(npt.NDArray[Any], tracker["left_reference_y"])
    right_reference_x = cast(npt.NDArray[Any], tracker["right_reference_x"])
    right_reference_y = cast(npt.NDArray[Any], tracker["right_reference_y"])
    left_sum_x = cast(npt.NDArray[Any], tracker["left_sum_x"])
    left_sum_y = cast(npt.NDArray[Any], tracker["left_sum_y"])
    left_square_x = cast(npt.NDArray[Any], tracker["left_square_x"])
    left_square_y = cast(npt.NDArray[Any], tracker["left_square_y"])
    left_cross = cast(npt.NDArray[Any], tracker["left_cross"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    finite = ~np.isnan(flat_x) & ~np.isnan(flat_y)
    sum_x[position] = flat_x
    sum_y[position] = flat_y
    counts[position] = finite
    if position == 0:
        np.copyto(left_reference_x, flat_x)
        np.copyto(left_reference_y, flat_y)
        left_sum_x.fill(0.0)
        left_sum_y.fill(0.0)
        left_square_x.fill(0.0)
        left_square_y.fill(0.0)
        left_cross.fill(0.0)
        left_counts[:] = finite
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            deviation_x = np.where(finite, flat_x - left_reference_x, 0.0)
            deviation_y = np.where(finite, flat_y - left_reference_y, 0.0)
            left_sum_x += deviation_x
            left_sum_y += deviation_y
            left_square_x += deviation_x * deviation_x
            left_square_y += deviation_y * deviation_y
            left_cross += deviation_x * deviation_y
        left_counts += finite

    right_counts = counts[position + 1]
    np.add(left_counts, right_counts, out=total_counts)
    mean_x, m2x, _, _, delta_x, cross_weight = _merge_moments(
        left_sum_x,
        left_square_x,
        left_counts,
        left_reference_x,
        sum_x[position + 1],
        square_x[position + 1],
        right_counts,
        right_reference_x,
        total_counts,
    )
    mean_y, m2y, left_relative_y, right_relative_y, delta_y, _ = _merge_moments(
        left_sum_y,
        left_square_y,
        left_counts,
        left_reference_y,
        sum_y[position + 1],
        square_y[position + 1],
        right_counts,
        right_reference_y,
        total_counts,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        left_comoment = left_cross - left_sum_x * left_relative_y
        right_comoment = cross[position + 1] - sum_x[position + 1] * right_relative_y
        comoment = left_comoment + right_comoment + delta_x * delta_y * cross_weight
    comoment = np.where(right_counts > 0, comoment, left_comoment)
    comoment[~np.isfinite(comoment)] = np.nan
    result = (
        mean_x.reshape(x.shape),
        mean_y.reshape(x.shape),
        m2x.reshape(x.shape),
        m2y.reshape(x.shape),
        comoment.reshape(x.shape),
        total_counts.reshape(x.shape),
    )

    if position == window - 1:
        np.copyto(right_reference_x, sum_x[window - 1])
        np.copyto(right_reference_y, sum_y[window - 1])
        with np.errstate(over="ignore", invalid="ignore"):
            np.subtract(sum_x[:window], right_reference_x, out=sum_x[:window])
            np.subtract(sum_y[:window], right_reference_y, out=sum_y[:window])
            invalid = counts[:window] == 0
            sum_x[:window][invalid] = 0.0
            sum_y[:window][invalid] = 0.0
            np.square(sum_x[:window], out=square_x[:window])
            np.square(sum_y[:window], out=square_y[:window])
            np.multiply(sum_x[:window], sum_y[:window], out=cross[:window])
        _accumulate_suffixes(window, sum_x, sum_y, square_x, square_y, cross, counts)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1
    return result


def window_product(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Return a rolling product from chunk-local log/sign/zero sums.

    Each transformed field is additive, so the previous-chunk suffix and the
    current-chunk prefix are maintained in fixed buffers.  No aggregate tuple
    or input array is allocated per timestamp, and an overflowed logarithmic
    sum cannot contaminate a later window after its chunk suffix expires.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "logabs": np.zeros((window + 1, cell_count), dtype=np.float64),
            "negative": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "zero": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "counts": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "left_logabs": np.zeros(cell_count, dtype=np.float64),
            "left_negative": np.zeros(cell_count, dtype=count_dtype),
            "left_zero": np.zeros(cell_count, dtype=count_dtype),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "total_logabs": np.zeros(cell_count, dtype=np.float64),
            "total_negative": np.zeros(cell_count, dtype=count_dtype),
            "total_zero": np.zeros(cell_count, dtype=count_dtype),
            "total_counts": np.zeros(cell_count, dtype=np.int64),
            "product": np.zeros(cell_count, dtype=np.float64),
            "position": 0,
        }
        state[key] = tracker

    logabs = cast(npt.NDArray[Any], tracker["logabs"])
    negative = cast(npt.NDArray[Any], tracker["negative"])
    zero = cast(npt.NDArray[Any], tracker["zero"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    left_logabs = cast(npt.NDArray[Any], tracker["left_logabs"])
    left_negative = cast(npt.NDArray[Any], tracker["left_negative"])
    left_zero = cast(npt.NDArray[Any], tracker["left_zero"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    total_logabs = cast(npt.NDArray[Any], tracker["total_logabs"])
    total_negative = cast(npt.NDArray[Any], tracker["total_negative"])
    total_zero = cast(npt.NDArray[Any], tracker["total_zero"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    product = cast(npt.NDArray[Any], tracker["product"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    nonzero = finite & (flat_x != 0.0)
    log_slot = logabs[position]
    log_slot.fill(0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.abs(flat_x, out=log_slot, where=nonzero)
        np.log(log_slot, out=log_slot, where=nonzero)
    negative[position] = finite & (flat_x < 0.0)
    zero[position] = finite & (flat_x == 0.0)
    counts[position] = finite

    with np.errstate(over="ignore", invalid="ignore"):
        np.add(left_logabs, log_slot, out=left_logabs)
        np.add(left_logabs, logabs[position + 1], out=total_logabs)
    np.add(left_negative, negative[position], out=left_negative)
    np.add(left_zero, zero[position], out=left_zero)
    np.add(left_counts, counts[position], out=left_counts)
    np.add(left_negative, negative[position + 1], out=total_negative)
    np.add(left_zero, zero[position + 1], out=total_zero)
    np.add(left_counts, counts[position + 1], out=total_counts)

    if position == window - 1:
        _accumulate_suffixes(window, logabs, negative, zero, counts)
        left_logabs.fill(0.0)
        left_negative.fill(0)
        left_zero.fill(0)
        left_counts.fill(0)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1

    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(total_logabs, out=product)
        np.negative(product, out=product, where=(total_negative % 2) != 0)
    product[total_zero > 0] = 0.0
    product[~np.isfinite(product)] = np.nan
    return product.reshape(x.shape), total_counts.reshape(x.shape)


def window_max_drawdown(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Return the best future/peak ratio for a positive trailing window.

    The chronological aggregate ``(max, min, best_ratio, valid_count)`` is
    associative: when ``left`` precedes ``right``, the combined best ratio is
    the minimum of each side's ratio and ``right.min / left.max``.  Fixed
    chunk prefixes/suffixes therefore replace a full-window path scan and keep
    amortized work O(1) per timestamp.
    """
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        shape = (window + 1, cell_count)
        tracker = {
            "maximum": np.full(shape, -np.inf, dtype=np.float64),
            "minimum": np.full(shape, np.inf, dtype=np.float64),
            "ratio": np.full(shape, np.inf, dtype=np.float64),
            "counts": np.zeros(shape, dtype=count_dtype),
            "left_maximum": np.full(cell_count, -np.inf, dtype=np.float64),
            "left_minimum": np.full(cell_count, np.inf, dtype=np.float64),
            "left_ratio": np.full(cell_count, np.inf, dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "total_ratio": np.empty(cell_count, dtype=np.float64),
            "total_counts": np.empty(cell_count, dtype=np.int64),
            "scratch": np.empty(cell_count, dtype=np.float64),
            "position": 0,
        }
        state[key] = tracker

    maximum = cast(npt.NDArray[Any], tracker["maximum"])
    minimum = cast(npt.NDArray[Any], tracker["minimum"])
    ratio = cast(npt.NDArray[Any], tracker["ratio"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    left_maximum = cast(npt.NDArray[Any], tracker["left_maximum"])
    left_minimum = cast(npt.NDArray[Any], tracker["left_minimum"])
    left_ratio = cast(npt.NDArray[Any], tracker["left_ratio"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    total_ratio = cast(npt.NDArray[Any], tracker["total_ratio"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    scratch = cast(npt.NDArray[Any], tracker["scratch"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    maximum[position].fill(-np.inf)
    minimum[position].fill(np.inf)
    ratio[position].fill(np.inf)
    np.copyto(maximum[position], flat_x, where=finite)
    np.copyto(minimum[position], flat_x, where=finite)
    np.copyto(ratio[position], 1.0, where=finite)
    counts[position] = finite
    if position == 0:
        np.copyto(left_maximum, maximum[position])
        np.copyto(left_minimum, minimum[position])
        np.copyto(left_ratio, ratio[position])
        left_counts[:] = finite
    else:
        scratch.fill(np.inf)
        present = (left_counts > 0) & finite
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(flat_x, left_maximum, out=scratch, where=present)
        np.minimum(left_ratio, scratch, out=left_ratio)
        np.maximum(left_maximum, maximum[position], out=left_maximum)
        np.minimum(left_minimum, minimum[position], out=left_minimum)
        np.add(left_counts, finite, out=left_counts)

    right_maximum = maximum[position + 1]
    right_ratio = ratio[position + 1]
    right_counts = counts[position + 1]
    np.minimum(right_ratio, left_ratio, out=total_ratio)
    both = (right_counts > 0) & (left_counts > 0)
    scratch.fill(np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(left_minimum, right_maximum, out=scratch, where=both)
    np.minimum(total_ratio, scratch, out=total_ratio)
    np.add(right_counts, left_counts, out=total_counts)
    result = total_ratio.reshape(x.shape), total_counts.reshape(x.shape)

    if position == window - 1:
        for index in range(window - 2, -1, -1):
            scratch.fill(np.inf)
            both = (counts[index] > 0) & (counts[index + 1] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(
                    minimum[index + 1],
                    maximum[index],
                    out=scratch,
                    where=both,
                )
            np.minimum(ratio[index], ratio[index + 1], out=ratio[index])
            np.minimum(ratio[index], scratch, out=ratio[index])
            np.maximum(maximum[index], maximum[index + 1], out=maximum[index])
            np.minimum(minimum[index], minimum[index + 1], out=minimum[index])
            np.add(counts[index], counts[index + 1], out=counts[index])
        left_maximum.fill(-np.inf)
        left_minimum.fill(np.inf)
        left_ratio.fill(np.inf)
        left_counts.fill(0)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1
    return result


ExtremeAggregate = tuple[
    npt.NDArray[Any],
    npt.NDArray[np.int64],
    npt.NDArray[Any],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]
IndexedExtremeAggregate = tuple[
    npt.NDArray[Any],
    npt.NDArray[np.int64],
    npt.NDArray[Any],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    int,
]


def _extreme_singleton(
    raw: tuple[npt.NDArray[Any], int],
) -> ExtremeAggregate:
    x, index = raw
    finite = ~np.isnan(x)
    indices = np.full(x.shape, index, dtype=np.int64)
    return (
        np.where(finite, x, np.inf),
        indices,
        np.where(finite, x, -np.inf),
        indices,
        finite.astype(np.int64),
    )


def _extreme_merge(left: ExtremeAggregate, right: ExtremeAggregate) -> ExtremeAggregate:
    lmin, lmini, lmax, lmaxi, lcount = left
    rmin, rmini, rmax, rmaxi, rcount = right
    right_min = rmin <= lmin
    right_max = rmax >= lmax
    return (
        np.where(right_min, rmin, lmin),
        np.where(right_min, rmini, lmini),
        np.where(right_max, rmax, lmax),
        np.where(right_max, rmaxi, lmaxi),
        lcount + rcount,
    )


def window_indexed_extrema(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
) -> IndexedExtremeAggregate:
    """Return extrema, their latest absolute indices, count, and current index."""
    tracker = state.get(key)
    index = 0 if tracker is None else cast(int, tracker["next_index"])
    aggregate = _aggregate_queue(
        state,
        (x.copy(), index),
        window,
        key,
        _extreme_singleton,
        _extreme_merge,
    )
    cast(dict[str, Any], state[key])["next_index"] = index + 1
    return (*cast(ExtremeAggregate, aggregate), index)


@overload
def window_extrema(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
    *,
    direction: Literal["min"],
) -> tuple[npt.NDArray[Any], None, npt.NDArray[np.int64]]: ...


@overload
def window_extrema(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
    *,
    direction: Literal["max"],
) -> tuple[None, npt.NDArray[Any], npt.NDArray[np.int64]]: ...


@overload
def window_extrema(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
    *,
    direction: Literal["both"] = "both",
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[np.int64]]: ...


def window_extrema(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
    *,
    direction: Literal["min", "max", "both"] = "both",
) -> tuple[
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
    npt.NDArray[np.int64],
]:
    """Return requested extrema and finite counts using prefix/suffix chunks."""
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        tracker = {
            "minimum": (
                np.full((window + 1, cell_count), np.inf, dtype=np.float64)
                if direction in ("min", "both")
                else None
            ),
            "maximum": (
                np.full((window + 1, cell_count), -np.inf, dtype=np.float64)
                if direction in ("max", "both")
                else None
            ),
            "counts": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "left_minimum": (
                np.empty(cell_count, dtype=np.float64)
                if direction in ("min", "both")
                else None
            ),
            "left_maximum": (
                np.empty(cell_count, dtype=np.float64)
                if direction in ("max", "both")
                else None
            ),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "total_minimum": (
                np.empty(cell_count, dtype=np.float64)
                if direction in ("min", "both")
                else None
            ),
            "total_maximum": (
                np.empty(cell_count, dtype=np.float64)
                if direction in ("max", "both")
                else None
            ),
            "total_counts": np.empty(cell_count, dtype=np.int64),
            "position": 0,
        }
        state[key] = tracker

    minimum = cast(npt.NDArray[Any] | None, tracker["minimum"])
    maximum = cast(npt.NDArray[Any] | None, tracker["maximum"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    left_minimum = cast(npt.NDArray[Any] | None, tracker["left_minimum"])
    left_maximum = cast(npt.NDArray[Any] | None, tracker["left_maximum"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    total_minimum = cast(npt.NDArray[Any] | None, tracker["total_minimum"])
    total_maximum = cast(npt.NDArray[Any] | None, tracker["total_maximum"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    if minimum is not None:
        minimum[position] = flat_x
    if maximum is not None:
        maximum[position] = flat_x
    counts[position] = finite
    if position == 0:
        if left_minimum is not None:
            np.copyto(left_minimum, flat_x)
        if left_maximum is not None:
            np.copyto(left_maximum, flat_x)
        left_counts[:] = finite
    else:
        if left_minimum is not None:
            np.minimum(left_minimum, flat_x, out=left_minimum)
        if left_maximum is not None:
            np.maximum(left_maximum, flat_x, out=left_maximum)
        left_counts += finite

    if minimum is not None and left_minimum is not None:
        minimum_result = cast(npt.NDArray[Any], total_minimum)
        np.minimum(left_minimum, minimum[position + 1], out=minimum_result)
    if maximum is not None and left_maximum is not None:
        maximum_result = cast(npt.NDArray[Any], total_maximum)
        np.maximum(left_maximum, maximum[position + 1], out=maximum_result)
    np.add(left_counts, counts[position + 1], out=total_counts)
    result = (
        None if total_minimum is None else total_minimum.reshape(x.shape),
        None if total_maximum is None else total_maximum.reshape(x.shape),
        total_counts.reshape(x.shape),
    )

    if position == window - 1:
        if minimum is not None:
            np.minimum.accumulate(
                minimum[window - 1 :: -1],
                axis=0,
                out=minimum[window - 1 :: -1],
            )
        if maximum is not None:
            np.maximum.accumulate(
                maximum[window - 1 :: -1],
                axis=0,
                out=maximum[window - 1 :: -1],
            )
        _accumulate_suffixes(window, counts)
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1
    return result


LinearAggregate = tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[np.int64],
]


def window_linear_stats(
    state: dict[str, Any],
    x: npt.NDArray[Any],
    window: int,
    key: str = "buffer",
    *,
    with_m2: bool = False,
) -> LinearAggregate:
    """Return rolling sum, time comoment, optional source M2, and count."""
    tracker = state.get(key)
    if tracker is None:
        cell_count = x.size
        count_dtype = np.min_scalar_type(window)
        index_dtype = np.min_scalar_type(window * (window - 1) // 2)
        tracker = {
            "sums": np.zeros((window + 1, cell_count), dtype=np.float64),
            "squares": (
                np.zeros((window + 1, cell_count), dtype=np.float64)
                if with_m2
                else None
            ),
            "weighted": np.zeros((window + 1, cell_count), dtype=np.float64),
            "counts": np.zeros((window + 1, cell_count), dtype=count_dtype),
            "index_sums": np.zeros((window + 1, cell_count), dtype=index_dtype),
            "left_reference": np.zeros(cell_count, dtype=np.float64),
            "right_reference": np.zeros(cell_count, dtype=np.float64),
            "left_sum": np.zeros(cell_count, dtype=np.float64),
            "left_square": (
                np.zeros(cell_count, dtype=np.float64) if with_m2 else None
            ),
            "left_weighted": np.zeros(cell_count, dtype=np.float64),
            "left_counts": np.zeros(cell_count, dtype=count_dtype),
            "left_index_sums": np.zeros(cell_count, dtype=index_dtype),
            "total_counts": np.zeros(cell_count, dtype=np.int64),
            "position": 0,
        }
        state[key] = tracker
    sums = cast(npt.NDArray[Any], tracker["sums"])
    squares = cast(npt.NDArray[Any] | None, tracker["squares"])
    weighted = cast(npt.NDArray[Any], tracker["weighted"])
    counts = cast(npt.NDArray[Any], tracker["counts"])
    index_sums = cast(npt.NDArray[Any], tracker["index_sums"])
    left_reference = cast(npt.NDArray[Any], tracker["left_reference"])
    right_reference = cast(npt.NDArray[Any], tracker["right_reference"])
    left_sum = cast(npt.NDArray[Any], tracker["left_sum"])
    left_square = cast(npt.NDArray[Any] | None, tracker["left_square"])
    left_weighted = cast(npt.NDArray[Any], tracker["left_weighted"])
    left_counts = cast(npt.NDArray[Any], tracker["left_counts"])
    left_index_sums = cast(npt.NDArray[Any], tracker["left_index_sums"])
    total_counts = cast(npt.NDArray[np.int64], tracker["total_counts"])
    position = cast(int, tracker["position"])

    flat_x = x.reshape(-1)
    finite = ~np.isnan(flat_x)
    sums[position] = flat_x
    counts[position] = finite
    if position == 0:
        np.copyto(left_reference, flat_x)
        left_sum.fill(0.0)
        if left_square is not None:
            left_square.fill(0.0)
        left_weighted.fill(0.0)
        left_counts[:] = finite
        left_index_sums.fill(0)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            deviation = np.where(finite, flat_x - left_reference, 0.0)
            left_sum += deviation
            if left_square is not None:
                left_square += deviation * deviation
            left_weighted += position * deviation
        left_counts += finite
        np.add(
            left_index_sums,
            position,
            out=left_index_sums,
            where=finite,
        )

    right_counts = counts[position + 1]
    right_sum = sums[position + 1]
    right_weighted = weighted[position + 1]
    right_index_sums = index_sums[position + 1]
    np.add(left_counts, right_counts, out=total_counts)
    with np.errstate(over="ignore", invalid="ignore"):
        reference = np.where(right_counts > 0, right_reference, left_reference)
        right_shift = np.where(right_counts > 0, right_reference - reference, 0.0)
        left_shift = np.where(left_counts > 0, left_reference - reference, 0.0)
        right_relative = right_sum + right_shift * right_counts
        left_relative = left_sum + left_shift * left_counts
        right_index_float = right_index_sums.astype(np.float64)
        left_index_float = left_index_sums.astype(np.float64)
        right_weight = right_weighted + right_shift * right_index_float
        left_weight = left_weighted + left_shift * left_index_float
        right_length = window - position - 1
        relative = right_relative + left_relative
        total_weighted = right_weight + left_weight + right_length * left_relative
        total = reference * total_counts + relative
        center = (window - 1.0) / 2.0
        centered = total_weighted - center * relative
        total_index = (
            right_index_float
            + left_index_float
            + right_length * left_counts.astype(np.float64)
        )
        centered += reference * (total_index - center * total_counts)
    if squares is not None and left_square is not None:
        _, m2, *_ = _merge_moments(
            left_sum,
            left_square,
            left_counts,
            left_reference,
            right_sum,
            squares[position + 1],
            right_counts,
            right_reference,
            total_counts,
        )
        shaped_m2: npt.NDArray[Any] | None = m2.reshape(x.shape)
    else:
        shaped_m2 = None
    total[~np.isfinite(total)] = np.nan
    centered[~np.isfinite(centered)] = np.nan
    result = (
        total.reshape(x.shape),
        centered.reshape(x.shape),
        shaped_m2,
        total_counts.reshape(x.shape),
    )

    if position == window - 1:
        np.copyto(right_reference, sums[window - 1])
        with np.errstate(over="ignore", invalid="ignore"):
            np.subtract(sums[:window], right_reference, out=sums[:window])
            sums[:window][counts[:window] == 0] = 0.0
            if squares is not None:
                np.square(sums[:window], out=squares[:window])
            weighted[:window].fill(0.0)
            index_sums[:window].fill(0)
            for index in range(window - 2, -1, -1):
                np.add(
                    weighted[index + 1],
                    sums[index + 1],
                    out=weighted[index],
                )
                np.add(
                    index_sums[index + 1],
                    counts[index + 1],
                    out=index_sums[index],
                )
                np.add(sums[index], sums[index + 1], out=sums[index])
                if squares is not None:
                    np.add(squares[index], squares[index + 1], out=squares[index])
                np.add(counts[index], counts[index + 1], out=counts[index])
        tracker["position"] = 0
    else:
        tracker["position"] = position + 1
    return result
