"""Private helpers for the verification harness: comparison and execution.

This module holds :class:`~qfeval_model.core.Features` comparison,
concatenation, slicing, and the fit/predict drivers (one-shot and streamed).
Inputs are normalized once into the wide ``{input: frame}`` dict and sliced
by their timestamp index, so anonymous trailing axes (MultiIndex columns) pass
through untouched. It deliberately does not use :meth:`Model.run` (verify
drives fit/predict itself).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from qfeval_model.core import (
    DEFAULT_FEATURE,
    TIMESTAMP,
    Features,
    Model,
    RawData,
    TimeOrderError,
    as_wide,
)
from qfeval_model.core.normalize import (
    normalize_timestamp_series,
    split_side_frame,
    to_utc_ns,
)
from qfeval_model.verify.errors import VerificationError

WideInputs = dict[str, pd.DataFrame]
"""The normalized wide ``{input: frame}`` inputs the drivers slice."""

CallPack = tuple[WideInputs, pd.DataFrame | None]
"""One side of a partitioned call: ``(inputs, side_data)``."""


def _index_ns(frame: pd.DataFrame) -> npt.NDArray[Any]:
    """Return a wide frame's index as tz-naive UTC ``datetime64[ns]`` values."""
    return normalize_timestamp_series(frame.index)


def slice_inputs(
    inputs: WideInputs,
    *,
    start: np.datetime64 | None = None,
    stop: np.datetime64 | None = None,
) -> WideInputs:
    """Return the rows with ``start <= timestamp < stop`` (``None`` = open)."""
    out: WideInputs = {}
    for name, frame in inputs.items():
        ts = _index_ns(frame)
        mask = np.ones(len(frame), dtype=bool)
        if start is not None:
            mask &= ts >= start
        if stop is not None:
            mask &= ts < stop
        out[name] = frame[mask]
    return out


def has_rows(inputs: WideInputs) -> bool:
    """Return whether any input frame has at least one row."""
    return any(len(frame) > 0 for frame in inputs.values())


def all_timestamps(inputs: WideInputs) -> npt.NDArray[np.datetime64]:
    """Return the sorted unique timestamps across all input frames."""
    if not inputs:
        return np.empty(0, dtype="datetime64[ns]")
    return np.unique(np.concatenate([_index_ns(f) for f in inputs.values()]))


def _is_float(dtype: np.dtype) -> bool:
    return bool(
        np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating)
    )


def _position_label(
    position: tuple[int, ...],
    timestamps: npt.NDArray[np.datetime64],
    symbols: npt.NDArray[np.str_] | None,
) -> str:
    """Describe a mismatching cell position.

    The leading axis is the timestamp; the main output's second axis carries
    the symbol labels and its trailing axes are anonymous (reported as plain
    indices). Side outputs have unnamed trailing axes throughout.
    """
    parts = [f"timestamp {timestamps[position[0]]!s}"]
    for axis, index in enumerate(position[1:], start=1):
        if axis == 1 and symbols is not None:
            parts.append(f"symbol {symbols[index]!s}")
        else:
            parts.append(f"axis {axis} index {index}")
    return ", ".join(parts)


def _compare_array(
    label: str,
    expected: npt.NDArray[Any],
    actual: npt.NDArray[Any],
    timestamps: npt.NDArray[np.datetime64],
    symbols: npt.NDArray[np.str_] | None,
    rtol: float,
    atol: float,
) -> None:
    # The shape check must come first: broadcast-compatible shapes would
    # otherwise slip through the element comparison (and mis-index the
    # mismatch position).
    if expected.shape != actual.shape:
        raise VerificationError(
            f"{label} shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
    if expected.dtype != actual.dtype:
        raise VerificationError(
            f"{label} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        )
    if _is_float(expected.dtype):
        close = np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
    elif expected.dtype.kind in "mM":
        # NaT != NaT elementwise; like NaN above, matching missing values in
        # a datetime/timedelta side output must compare as equal.
        close = (expected == actual) | (np.isnat(expected) & np.isnat(actual))
    else:
        close = expected == actual
    if bool(np.all(close)):
        return
    position = tuple(int(i) for i in np.argwhere(~close)[0])
    where = _position_label(position, timestamps, symbols)
    raise VerificationError(
        f"{label} mismatch at {where}: "
        f"expected {expected[position]!r}, got {actual[position]!r}"
    )


def compare_outputs(a: Features, b: Features, *, rtol: float, atol: float) -> None:
    """Assert that two outputs are equal (raises ``VerificationError``).

    Timestamps and symbols must match exactly; outputs and side outputs are
    compared with ``np.allclose`` (float) or ``np.array_equal`` (otherwise).
    ``info`` is not compared.
    """
    if not np.array_equal(a.timestamps, b.timestamps):
        raise VerificationError(
            f"timestamps differ: {a.timestamps!r} vs {b.timestamps!r}"
        )
    if not np.array_equal(a.symbols, b.symbols):
        raise VerificationError(f"symbols differ: {a.symbols!r} vs {b.symbols!r}")
    _compare_array("output", a.array, b.array, a.timestamps, a.symbols, rtol, atol)
    if set(a.side_data) != set(b.side_data):
        raise VerificationError(
            f"side_data keys differ: {sorted(a.side_data)} vs {sorted(b.side_data)}"
        )
    for name in a.side_data:
        _compare_array(
            f"side_data[{name!r}]",
            a.side_data[name],
            b.side_data[name],
            a.timestamps,
            None,
            rtol,
            atol,
        )


def _trailing(shape: tuple[int, ...], lead: int) -> tuple[int, ...]:
    """Return the axes after the first ``lead`` (the anonymous trailing shape)."""
    return shape[lead:]


def _check_concat_consistency(preds: list[Features]) -> None:
    """Diagnose the drift the lock-free engine no longer rejects per call.

    The reference is ``preds[0]``. Each check reports the first offending index
    with both actual values so a cross-call output/side-output mismatch is
    legible at the join point rather than surfacing as a raw ``KeyError`` or a
    numpy ``concatenate`` failure.
    """
    base = preds[0]
    for index, pred in enumerate(preds[1:], start=1):
        if pred.array.dtype != base.array.dtype:
            raise VerificationError(
                f"output dtype differs: outputs[0] has {base.array.dtype}, "
                f"outputs[{index}] has {pred.array.dtype}"
            )
        base_trailing = _trailing(base.array.shape, 2)
        pred_trailing = _trailing(pred.array.shape, 2)
        if pred_trailing != base_trailing:
            raise VerificationError(
                f"output trailing shape differs: outputs[0] has "
                f"{base_trailing}, outputs[{index}] has {pred_trailing}"
            )
    base_keys = set(base.side_data)
    for index, pred in enumerate(preds[1:], start=1):
        keys = set(pred.side_data)
        if keys != base_keys:
            raise VerificationError(
                f"side_data keys differ: outputs[{index}] lacks "
                f"{base_keys - keys or set()}, outputs[0] lacks "
                f"{keys - base_keys or set()}"
            )
        for name in base_keys:
            base_arr = base.side_data[name]
            pred_arr = pred.side_data[name]
            if pred_arr.dtype != base_arr.dtype:
                raise VerificationError(
                    f"side_data[{name!r}] dtype differs: outputs[0] has "
                    f"{base_arr.dtype}, outputs[{index}] has {pred_arr.dtype}"
                )
            base_st = _trailing(base_arr.shape, 1)
            pred_st = _trailing(pred_arr.shape, 1)
            if pred_st != base_st:
                raise VerificationError(
                    f"side_data[{name!r}] trailing shape differs: "
                    f"outputs[0] has {base_st}, outputs[{index}] has "
                    f"{pred_st}"
                )


def concat_outputs(preds: list[Features]) -> Features:
    """Concatenate outputs along the timestamp axis (validates monotonicity).

    This is the library's single join point for multiple calls, so it is where
    cross-call output/side-output drift (dtype, trailing shape, side-output key
    set) is diagnosed — the lock-free engine no longer rejects it per call.
    Diagnostic ``info`` is dropped from the reconstructed output; the verify
    assertions never compare it.
    """
    if not preds:
        raise VerificationError("cannot concatenate an empty list of outputs")
    symbols = preds[0].symbols
    for pred in preds[1:]:
        if not np.array_equal(pred.symbols, symbols):
            raise VerificationError("cannot concatenate outputs with differing symbols")
    timestamps = np.concatenate([pred.timestamps for pred in preds])
    if len(timestamps) > 1 and not bool((timestamps[1:] > timestamps[:-1]).all()):
        raise TimeOrderError("concatenated outputs have non-increasing timestamps")
    _check_concat_consistency(preds)
    side_keys = set(preds[0].side_data)
    return Features(
        timestamps=timestamps,
        symbols=symbols,
        data={DEFAULT_FEATURE: np.concatenate([pred.array for pred in preds], axis=0)},
        side_data={
            name: np.concatenate([pred.side_data[name] for pred in preds], axis=0)
            for name in side_keys
        },
    )


def slice_output(pred: Features, mask: npt.NDArray[Any]) -> Features:
    """Return the sub-output selected by a boolean timestamp ``mask``.

    Diagnostic ``info`` is dropped from the reconstructed output; the
    verify assertions never compare it.
    """
    return Features(
        timestamps=pred.timestamps[mask],
        symbols=pred.symbols,
        data={DEFAULT_FEATURE: pred.array[mask]},
        side_data={n: a[mask] for n, a in pred.side_data.items()},
    )


def partition_call(
    inputs: WideInputs,
    side_data: pd.DataFrame | None,
    at: np.datetime64,
) -> tuple[CallPack, CallPack]:
    """Partition a call at ``at`` into ``(before, from)`` packs.

    Each pack is ``(inputs, side_data)``; the first covers
    ``timestamp < at`` and the second ``timestamp >= at`` (an empty side frame
    becomes ``None``, matching the omitted-argument convention).
    """
    side_before, side_onward = split_side_frame(side_data, at)
    return (
        (slice_inputs(inputs, stop=at), side_before),
        (slice_inputs(inputs, start=at), side_onward),
    )


def execute(
    model: Model,
    inputs: WideInputs,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    symbols: Sequence[str] | None = None,
) -> Features:
    """Fit the rows before ``test_from`` and predict the rest in one call.

    ``symbols`` rides the run's first call only (it is a model-lifetime
    setting).
    """
    if test_from is None:
        return model.predict(
            inputs,
            side_data=side_data,
            symbols=symbols,
        )
    train, test = partition_call(inputs, side_data, to_utc_ns(test_from))
    if has_rows(train[0]):
        model.fit(
            train[0],
            side_data=train[1],
            symbols=symbols,
        )
        symbols = None
    if not has_rows(test[0]):
        raise VerificationError("no rows remain at or after test_from for output")
    return model.predict(
        test[0],
        side_data=test[1],
        symbols=symbols,
    )


_NO_ROWS = np.empty(0, dtype=np.intp)


def _positions_by_timestamp(
    ts: npt.NDArray[Any],
) -> dict[np.datetime64, npt.NDArray[Any]]:
    """Group row positions by timestamp (original order kept within a group)."""
    order = np.argsort(ts, kind="stable")
    sorted_ts = ts[order]
    if len(sorted_ts) == 0:
        return {}
    starts = np.flatnonzero(np.r_[True, sorted_ts[1:] != sorted_ts[:-1]])
    bounds = np.append(starts, len(sorted_ts))
    return {
        sorted_ts[start]: order[start:stop]
        for start, stop in zip(starts, bounds[1:], strict=True)
    }


def _stream_chunks(
    inputs: WideInputs,
    side_data: pd.DataFrame | None,
) -> Iterator[CallPack]:
    """Yield one ``(inputs, side)`` pack per timestamp, in time order.

    Row groups are precomputed once per frame, so streaming T timestamps
    costs time proportional to the total number of rows instead of the
    O(T x rows) of re-masking every frame at every timestamp.
    """
    input_groups = {
        name: (frame, _positions_by_timestamp(_index_ns(frame)))
        for name, frame in inputs.items()
    }

    def long_groups(
        df: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, dict[np.datetime64, npt.NDArray[Any]]] | None:
        if df is None:
            return None
        return df, _positions_by_timestamp(normalize_timestamp_series(df[TIMESTAMP]))

    side_groups = long_groups(side_data)

    def long_at(
        groups: tuple[pd.DataFrame, dict[np.datetime64, npt.NDArray[Any]]] | None,
        ts: np.datetime64,
    ) -> pd.DataFrame | None:
        if groups is None:
            return None
        frame, positions = groups
        rows = positions.get(ts)
        return None if rows is None else frame.iloc[rows]

    for ts in all_timestamps(inputs):
        chunk = {
            name: frame.iloc[positions.get(ts, _NO_ROWS)]
            for name, (frame, positions) in input_groups.items()
        }
        yield chunk, long_at(side_groups, ts)


def execute_streamed(
    model: Model,
    inputs: WideInputs,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    symbols: Sequence[str] | None = None,
) -> Features:
    """Fit/predict one timestamp at a time and concatenate the outputs.

    Applies the same ``symbols`` convention as :func:`execute` (the first
    call only).
    """
    if test_from is not None:
        train, test_pack = partition_call(inputs, side_data, to_utc_ns(test_from))
        if has_rows(train[0]):
            for chunk, side in _stream_chunks(*train):
                model.fit(
                    chunk,
                    side_data=side,
                    symbols=symbols,
                )
                symbols = None
    else:
        test_pack = (inputs, side_data)
    if not has_rows(test_pack[0]):
        raise VerificationError("no rows remain for output")
    preds = []
    for chunk, side in _stream_chunks(*test_pack):
        preds.append(
            model.predict(
                chunk,
                side_data=side,
                symbols=symbols,
            )
        )
        symbols = None
    return concat_outputs(preds)


def assert_streamed_match(
    make_model: Callable[[], Model],
    inputs: RawData,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    symbols: Sequence[str] | None = None,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert a one-shot run agrees with the per-timestamp (streamed) run.

    For a batch model this also detects chunk-local look-ahead when the
    one-shot output differs from chunk-size-1 execution.
    """
    inputs = as_wide(inputs)
    model_a = make_model()
    model_b = make_model()
    if model_a is model_b or model_a.started or model_b.started:
        raise ValueError(
            "make_model must return a fresh, distinct, unstarted model each call"
        )
    one_shot = execute(
        model_a,
        inputs,
        side_data=side_data,
        test_from=test_from,
        symbols=symbols,
    )
    streamed = execute_streamed(
        model_b,
        inputs,
        side_data=side_data,
        test_from=test_from,
        symbols=symbols,
    )
    compare_outputs(one_shot, streamed, rtol=rtol, atol=atol)
