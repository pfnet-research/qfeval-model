"""Input acceptance: shape coercion and DataFrame -> ``Features`` normalization.

Wide format — one DataFrame per feature with ``index=timestamp`` and
``columns=symbol`` — is the package's basic input format; the long format
(``timestamp`` / ``symbol`` / feature columns) is the transport format.
``fit`` / ``predict`` / ``run`` / ``split`` accept either. Every accepted
shape is first normalized into canonical wide input frames (``as_wide``) or
the canonical long frame (``as_frame``); ``normalize_wide_inputs`` is the
single DataFrame -> array path.

The input has two fixed axes: timestamp and symbol. A wide frame with a flat
column ``Index`` carries plain ``(T, S)`` values. ``MultiIndex`` columns carry
anonymous trailing axes: level 0 is the symbol, and each deeper level is one
trailing axis whose sorted unique values fix the axis order and size for the
call — the values themselves are not retained. Long inputs carry 2-D inputs
only. Cells absent from an input's cross product become NaN (float only).

Normalization follows the DataFrame contract of DESIGN §The input
boundary: UTC timestamp normalization, string-label validation,
last-non-NaN duplicate aggregation, input expansion, and reindexing to the
fixed symbol universe. Feature names, symbol labels, and side-input keys
must be non-empty strings and are never implicitly converted; feature names may
not be ``timestamp`` / ``symbol`` (the long-format key columns).
pandas nullable
dtypes (``Float64`` / ``Int64`` / ``boolean``) are explicitly converted to
``float64`` with ``NaN`` for missing values at this boundary. All produced
arrays are fresh copies that do not share memory with the input DataFrame.

The functions are laid out in dataflow order: validation / timestamp
primitives, RawData -> canonical wide (``as_wide``), canonical wide ->
canonical long (``as_frame``), canonical wide -> arrays
(``normalize_wide_inputs``), side inputs, and finally the ``normalize_call``
assembly. This module depends one-way on ``features`` (it builds and returns
:class:`Features`); ``features`` never imports back.

``as_wide`` / ``as_frame`` are the public RawData conversions (the input-side
counterparts of ``Features.to_wide`` / ``Features.to_frame``); ``to_utc_ns`` /
``split_side_frame`` are package-internal SPI used by the ``verify`` /
``metrics`` packages.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from qfeval_model.core.constants import DEFAULT_FEATURE, SYMBOL, TIMESTAMP
from qfeval_model.core.features import Features, _allocate, _melt_wide

RawData = pd.DataFrame | Mapping[str, pd.DataFrame]
"""Any accepted inputs shape (wide dict, wide frame, or long frame)."""

_RESERVED = (TIMESTAMP, SYMBOL)

# --- validation & timestamp primitives ---------------------------------------


def _validate_name(value: Any, what: str, *, reserved: bool = False) -> str:
    """Validate a single name against the shared name contract.

    A name must be a non-empty ``str``; when ``reserved`` is set (feature
    names), the axis keywords (``timestamp`` / ``symbol``) are additionally
    rejected. This is the one place the name rule lives, shared by feature
    names, symbol labels, side keys, and ``symbols=`` entries.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} must be str, got {type(value).__name__}: {value!r}")
    if not value:
        raise ValueError(f"{what} must be a non-empty string")
    if reserved and value in _RESERVED:
        raise ValueError(f"{what} {value!r} is reserved")
    return value


def _validate_names(
    values: Iterable[Any], what: str, *, reserved: bool = False
) -> list[str]:
    """Validate a name collection: all valid names (see :func:`_validate_name`),
    no duplicates within the collection."""
    names = [_validate_name(value, what, reserved=reserved) for value in values]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {what}: {name!r}")
        seen.add(name)
    return names


def to_utc_ns(value: Any) -> np.datetime64:
    """Convert a scalar timestamp to a tz-naive ``datetime64[ns]`` in UTC.

    This is package-internal SPI shared with the ``verify`` package; it is
    not part of the user-facing API.

    Args:
        value: Anything :class:`pandas.Timestamp` accepts. tz-aware values are
            converted to UTC and stripped of their timezone.

    Returns:
        A ``datetime64[ns]`` value (never ``NaT``).

    Raises:
        ValueError: If ``value`` is missing (``None`` / ``NaT``), a bare
            number, or cannot be parsed as a timestamp.
    """
    if isinstance(value, bool | int | float | np.number):
        # pd.Timestamp reads a bare number as epoch *nanoseconds*, so a
        # plausible input mistake like 20240101 silently becomes a 1970
        # timestamp; require an explicit datetime or a parseable string.
        raise ValueError(
            f"expected a timestamp-like value, got the number {value!r}; "
            "numbers are read as epoch nanoseconds, which is never intended "
            "— pass a datetime or a string like '2024-01-01'"
        )
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(
            f"expected a timestamp-like value, got {value!r}: {exc}"
        ) from exc
    if pd.isna(ts):
        raise ValueError(
            f"expected a concrete timestamp, got a missing value: {value!r}"
        )
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.as_unit("ns").to_datetime64()


def _to_utc_ns_array(values: Any, what: str) -> npt.NDArray[np.datetime64]:
    """The vector timestamp primitive: parse timestamp-like values to a
    tz-naive UTC ``datetime64[ns]`` array with no ``NaT``.

    This is the single point that defines what a core timestamp axis is. The
    scalar (:func:`to_utc_ns`) and dtype-guarded (:func:`_normalize_wide_index`)
    paths are thin layers over it.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    array = index.to_numpy(dtype="datetime64[ns]")
    if bool(np.isnat(array).any()):
        raise ValueError(f"{what} must not contain NaT")
    return array


def normalize_timestamp_series(values: Any) -> npt.NDArray[np.datetime64]:
    """Parse timestamp-like values to a tz-naive UTC ``datetime64[ns]`` array.

    The lenient series counterpart of :func:`to_utc_ns` (strings are parsed),
    used by auxiliary alignment paths (``run`` / ``verify`` / ``metrics``). It
    shares the core timestamp contract: ``NaT`` is rejected on every path.
    Long-input columns keep the stricter ``_normalize_timestamp_column``
    (datetime dtype required).
    """
    return _to_utc_ns_array(values, "timestamps")


def split_side_frame(
    df: pd.DataFrame | None, at: np.datetime64
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Split a long frame by its ``timestamp`` column into ``(before, from)``.

    The side-frame counterpart of :func:`~qfeval_model.core.split.split`,
    shared by ``Model.run`` and the ``verify`` package (package-internal SPI).
    Each half is ``None`` when empty (or when ``df`` is ``None``), matching
    the omitted-argument convention of ``side_data=``. The ``timestamp``
    column is checked with the same strict rule as ``normalize_side_frame``
    (datetime dtype required), so an invalid side frame fails at the split
    instead of inside the fit/predict that follows.
    """
    if df is None:
        return None, None
    if TIMESTAMP not in df.columns:
        raise ValueError("side_data must have a 'timestamp' column")
    ts = _normalize_timestamp_column(df)
    before, onward = df[ts < at], df[ts >= at]
    return (
        before if len(before) > 0 else None,
        onward if len(onward) > 0 else None,
    )


def _normalize_wide_index(index: pd.Index, what: str) -> npt.NDArray[Any]:
    """Return a wide frame's index as tz-naive UTC ``datetime64[ns]`` values.

    The dtype guard over the timestamp primitive: a wide index must already be
    a datetime index (its dtype is not inferred).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"{what} must have a datetime index (any unit, tz-aware allowed); "
            f"got dtype {index.dtype}"
        )
    return _to_utc_ns_array(index, what)


def _normalize_timestamp_column(inputs: pd.DataFrame) -> npt.NDArray[Any]:
    ts = inputs[TIMESTAMP]
    if not is_datetime64_any_dtype(ts):
        raise ValueError(
            "'timestamp' column must be datetime64 (any unit, tz-aware allowed); "
            f"got dtype {ts.dtype}"
        )
    return _normalize_wide_index(pd.DatetimeIndex(ts), "'timestamp' column")


def _to_numpy_1d(column: pd.Series, what: str) -> npt.NDArray[Any]:
    """Convert one column to a 1-D numpy array (the single conversion point).

    pandas nullable numeric / boolean dtypes are explicitly converted to
    ``float64`` with ``NaN`` as the missing value; other dtypes keep their
    natural numpy representation. Conversion failures are normalized to
    :class:`ValueError`.
    """
    dtype = column.dtype
    try:
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            if isinstance(dtype, pd.DatetimeTZDtype):
                converted = cast(pd.Series, column.dt.tz_convert("UTC"))
                converted = cast(pd.Series, converted.dt.tz_localize(None))
                return converted.to_numpy(dtype="datetime64[ns]")
            if dtype.kind in "iufb":
                return column.to_numpy(dtype=np.float64, na_value=np.nan)
        return column.to_numpy()
    except Exception as exc:
        raise ValueError(
            f"{what} could not be converted to a numpy representation "
            f"(dtype {dtype}): {exc}"
        ) from exc


# --- RawData -> canonical wide -----------------------------------------------


def _convert_value_columns(
    df: pd.DataFrame, cols: list[str], what: str
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, np.dtype]]:
    """Convert each of ``cols`` to a 1-D array, recording its dtype.

    The value-column reader shared by long inputs and side inputs: name
    validation and the groupby key differ per caller, this is the common
    ``_to_numpy_1d`` + dtype-record step (nullable dtypes accepted here once).
    """
    arrays: dict[str, npt.NDArray[Any]] = {}
    dtypes: dict[str, np.dtype] = {}
    for c in cols:
        values = _to_numpy_1d(df[c], f"{what} {c!r}")
        dtypes[c] = values.dtype
        arrays[c] = values
    return arrays, dtypes


def _pivot_long(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot a canonical long frame into wide ``{feature: frame}``.

    Each value column becomes one wide feature; the ``(timestamp, symbol)`` key
    is UTC-normalized and duplicate cells keep their last non-NaN value. Cells
    absent from the cross product become NaN (float only). Requires at least
    one value column.
    """
    input_cols = _validate_names(
        [c for c in frame.columns if c not in _RESERVED],
        "long-format feature name",
        reserved=True,
    )
    if not input_cols:
        raise ValueError("inputs must have at least one feature")
    _validate_names(pd.unique(frame[SYMBOL]), "long-format symbol label")
    arrays, dtypes = _convert_value_columns(frame, input_cols, "feature")
    long = pd.DataFrame(
        {
            TIMESTAMP: _normalize_timestamp_column(frame),
            SYMBOL: frame[SYMBOL].to_numpy(),
            **arrays,
        }
    )
    agg = long.groupby([TIMESTAMP, SYMBOL], sort=True).last()
    result: dict[str, pd.DataFrame] = {}
    for c in input_cols:
        sub = agg[c].unstack(SYMBOL)
        # The pivot fills cells absent from its cross product with NaN, which
        # only an inexact input can represent — the same coverage rule the
        # wide path enforces at materialization (an object cell's NaN would
        # otherwise slip through the astype below).
        if len(agg) != sub.size and not np.issubdtype(dtypes[c], np.inexact):
            raise ValueError(
                f"non-float feature {c!r} has missing "
                "(timestamp, symbol) cells that cannot be NaN-filled"
            )
        sub = sub.astype(dtypes[c])
        result[c] = sub.rename_axis(index=TIMESTAMP, columns=SYMBOL)
    return result


def _canonical_wide_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Apply the row-side DataFrame contract to one wide frame.

    The index is normalized to tz-naive UTC ``datetime64[ns]`` (named
    ``timestamp``), duplicate timestamps keep the last non-NaN value per cell,
    and the rows are sorted. The columns are validated (string symbol labels,
    no duplicates) but otherwise left untouched except that the symbol level
    (level 0) is named ``symbol`` so the frame melts back to canonical long
    form.
    """
    timestamps = _normalize_wide_index(frame.index, f"wide index of feature {name!r}")
    frame = frame.set_axis(pd.DatetimeIndex(timestamps, name=TIMESTAMP))
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.set_names([SYMBOL, *frame.columns.names[1:]])
    else:
        frame = frame.rename_axis(columns=SYMBOL)
    _wide_frame_columns(frame, name)
    if pd.Index(timestamps).has_duplicates:
        frame = frame.groupby(level=0).last()
    else:
        frame = frame.sort_index()  # already sorted in the common case: a no-op
    return frame


def as_wide(data: RawData) -> dict[str, pd.DataFrame]:
    """Normalize any accepted shape into canonical wide ``{input: frame}``.

    This is the single long -> wide conversion point; every accepted shape
    passes through here before :func:`normalize_wide_inputs` turns the wide
    frames into feature arrays. It is the input-side counterpart of
    :meth:`Features.to_wide` and part of the public API.

    Args:
        data: A wide mapping, a flat wide DataFrame, or a long DataFrame. The
            data may equally be a rendered :class:`Features` (its ``to_wide``
            frame), since the two share this normalization.

    Returns:
        A mapping of feature name to a canonical wide DataFrame (index=timestamp,
        columns=symbol). Every frame — wide or pivoted from long — carries the
        DataFrame contract: UTC timestamp normalization, string-symbol
        validation, last-non-NaN duplicate aggregation, sorted rows, and
        per-input dtype preservation (cells absent from a pivot's cross product
        become NaN, float only). The conversion is idempotent.

    Raises:
        ValueError: On an unrecognised or ambiguous shape, or a
            DataFrame-contract violation.
    """
    if isinstance(data, Mapping):
        return _canonicalize_wide_mapping(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "inputs must be a DataFrame or a mapping of feature name to "
            f"wide DataFrame; got {type(data).__name__}"
        )
    # A bare DataFrame is one of three shapes: long (both key columns),
    # ambiguous (exactly one key column), or a flat wide frame (neither).
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError(
            "inputs with MultiIndex columns are not accepted; pass wide data "
            "as a {feature_name: frame} mapping instead"
        )
    has_timestamp = TIMESTAMP in data.columns
    has_symbol = SYMBOL in data.columns
    if has_timestamp and has_symbol:
        return _pivot_long(data)
    if has_timestamp or has_symbol:
        present = TIMESTAMP if has_timestamp else SYMBOL
        raise ValueError(
            f"ambiguous inputs: column {present!r} is present but a long "
            "frame needs both 'timestamp' and 'symbol'; pass wide data as "
            "{feature_name: frame} to be explicit"
        )
    return {DEFAULT_FEATURE: _canonical_wide_frame(data, DEFAULT_FEATURE)}


def _canonicalize_wide_mapping(
    data: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Canonicalize an explicit ``{feature: wide frame}`` mapping.

    Input names share the reserved-aware name contract; every value must be a
    DataFrame and is put through the row-side contract (:func:`_canonical_wide_frame`).
    """
    _validate_names(data.keys(), "wide feature name", reserved=True)
    if not data:
        raise ValueError("inputs must have at least one feature")
    result: dict[str, pd.DataFrame] = {}
    for name, frame in data.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(
                f"wide inputs feature {name!r} must be a "
                f"DataFrame; got {type(frame).__name__}"
            )
        result[name] = _canonical_wide_frame(frame, name)
    return result


# --- canonical wide -> canonical long ----------------------------------------


def as_frame(data: RawData) -> pd.DataFrame:
    """Normalize any accepted shape into a canonical long DataFrame.

    The counterpart of :func:`as_wide`: every input is first aligned onto the
    call's dense ``timestamp x symbol`` grid — the sorted union of the inputs'
    timestamps and symbols, exactly what a model call sees — and then rendered
    into one long frame with a ``timestamp`` / ``symbol`` key and one value
    column per feature. For 2-D inputs (the long format's supported domain),
    sharing the model call's normalization makes the conversion idempotent:
    ``as_frame(as_frame(x)) == as_frame(x)``, with rows sorted by the key
    columns.

    Inputs with trailing axes are rejected. Keep them in the wide
    ``MultiIndex`` layout (the format produced by ``Features.to_wide()``).

    Args:
        data: A wide mapping, a flat wide DataFrame, or a long DataFrame (or a
            rendered :class:`Features` frame).

    Returns:
        A long DataFrame; every cell of the dense grid is emitted (``NaN``
        kept). Value columns keep their dtype. Grid cells an input does not
        cover on the timestamp/symbol axes become NaN, so they require an
        inexact dtype — the same coverage rule the model input path enforces.

    Raises:
        ValueError: On an unrecognised or ambiguous shape, a
            DataFrame-contract violation, trailing axes, or a non-float
            feature with missing grid cells.
    """
    # The exact input pipeline of a model call (extract -> resolve shared
    # axes -> materialize), so as_frame is canonical by construction.
    wide = as_wide(data)
    if any(isinstance(frame.columns, pd.MultiIndex) for frame in wide.values()):
        raise ValueError(
            "as_frame() does not support trailing axes; keep inputs wide "
            "(Features.to_wide() is the round-trip format)"
        )
    specs = {name: _extract_wide_frame(frame, name) for name, frame in wide.items()}
    all_ts, symbols = _resolve_call_axes(specs.values(), None)
    index = pd.DatetimeIndex(all_ts, name=TIMESTAMP)
    merged: pd.DataFrame | None = None
    keys = [TIMESTAMP, SYMBOL]
    for name, spec in specs.items():
        values = _materialize_wide_input(spec, name, all_ts, symbols)
        columns = pd.Index(symbols, name=SYMBOL)
        dense = pd.DataFrame(
            values.reshape(len(index), len(columns)), index=index, columns=columns
        )
        long = _melt_wide(dense, name)
        if merged is None:
            merged = long
        else:
            # pandas 2.1 preserves the left keys and appends right-only keys
            # for an outer merge unless sort is explicit. The public long
            # form promises key order independent of input order/version.
            merged = merged.merge(long, on=keys, how="outer", sort=True)
    assert merged is not None
    return merged


# --- canonical wide -> arrays ------------------------------------------------


def _wide_frame_columns(frame: pd.DataFrame, name: str) -> list[tuple[Any, ...]]:
    """Read a wide frame's column label tuples.

    Flat columns carry symbols only; ``MultiIndex`` columns carry the symbol
    on level 0 and one anonymous trailing axis per deeper level.
    """
    multi = isinstance(frame.columns, pd.MultiIndex)
    col_tuples = [tuple(key) if multi else (key,) for key in frame.columns]
    for key in col_tuples:
        _validate_name(key[0], f"wide symbol column label of feature {name!r}")
    if len(set(col_tuples)) != len(col_tuples):
        raise ValueError(
            f"wide columns of feature {name!r} must not contain duplicates"
        )
    return col_tuples


@dataclass(frozen=True, slots=True)
class _WideFrameSpec:
    """One canonical wide frame read into named arrays, ready to materialize.

    ``timestamps`` is sorted unique ``datetime64[ns]``; ``columns`` are the
    validated per-column label tuples (symbol first); ``trailing_labels`` holds
    the sorted unique labels of each anonymous trailing axis; ``values`` is the
    ``(T_frame, N_frame)`` value array (duplicate timestamps already reduced to
    their last non-NaN value per cell by ``as_wide``).
    """

    timestamps: npt.NDArray[np.datetime64]
    columns: list[tuple[Any, ...]]
    trailing_labels: list[list[Any]]
    values: npt.NDArray[Any]


def _extract_wide_frame(frame: pd.DataFrame, name: str) -> _WideFrameSpec:
    """Read one canonical wide frame into a :class:`_WideFrameSpec`."""
    col_tuples = _wide_frame_columns(frame, name)
    n_levels = frame.columns.nlevels
    trailing_labels: list[list[Any]] = []
    for level in range(1, n_levels):
        level_values = {key[level] for key in col_tuples}
        try:
            trailing_labels.append(sorted(level_values))
        except TypeError as exc:
            raise ValueError(
                f"trailing axis {level} of feature {name!r} has labels that "
                f"cannot be sorted into one order: {exc}"
            ) from exc
    # ``frame`` is already canonical (``as_wide`` normalized, deduplicated, and
    # sorted the index), so the timestamps read straight off it.
    timestamps = frame.index.to_numpy(dtype="datetime64[ns]", copy=True)
    if col_tuples:
        converted = []
        for position, key in enumerate(col_tuples):
            label = key[0] if len(key) == 1 else key
            converted.append(
                _to_numpy_1d(
                    frame.iloc[:, position], f"feature {name!r} column {label!r}"
                )
            )
        try:
            values = np.column_stack(converted)
        except TypeError as exc:
            # Mixed column dtypes (e.g. a datetime column next to floats)
            # have no common dtype; normalize NumPy's promotion error into
            # the input boundary's ValueError like every other contract
            # violation. Other failures (notably MemoryError) keep their
            # original meaning instead of being mislabeled as bad input.
            raise ValueError(
                f"columns of feature {name!r} have no common dtype: {exc}"
            ) from exc
        if values.dtype.kind == "O" and any(a.dtype.kind != "O" for a in converted):
            raise ValueError(
                f"columns of feature {name!r} mix object and non-object dtypes; "
                "use one dtype for a named feature"
            )
    else:
        values = np.empty((len(timestamps), 0))
    return _WideFrameSpec(timestamps, col_tuples, trailing_labels, values)


def _flat_column_positions(
    label_lists: list[list[Any]], col_tuples: list[tuple[Any, ...]]
) -> tuple[list[int], npt.NDArray[Any]]:
    """Map column label tuples to flat positions in the axes' cross product."""
    sizes = [len(labels) for labels in label_lists]
    index_of = [{label: i for i, label in enumerate(labels)} for labels in label_lists]
    multi = [
        [index_of[level][label] for level, label in enumerate(key)]
        for key in col_tuples
    ]
    positions: npt.NDArray[Any] = (
        np.asarray(np.ravel_multi_index(np.array(multi, dtype=np.intp).T, sizes))
        if multi
        else np.empty(0, dtype=np.intp)
    )
    return sizes, positions


def _resolve_declared_symbols(
    symbols: Sequence[str] | npt.NDArray[np.str_] | None,
) -> list[str] | None:
    """Validate a declared symbol universe into a sorted label list."""
    if symbols is None:
        return None
    if isinstance(symbols, str):
        # A bare string iterates per character ("ABC" -> ['A', 'B', 'C']),
        # silently mangling a single multi-character symbol.
        raise ValueError(
            f"symbols= must be a sequence of symbol labels, got the string "
            f"{symbols!r}; pass a single symbol as [{symbols!r}]"
        )
    resolved = sorted(_validate_names(symbols, "symbols= entry"))
    if not resolved:
        raise ValueError("symbols= must declare at least one symbol")
    return resolved


def _resolve_symbols(
    present: list[str],
    declared: list[str] | None,
) -> list[str]:
    """Resolve the symbol universe for this call.

    The universe is the declared one (a ``symbols=`` declaration or the
    model's fixed universe) or, absent a declaration, the labels present in
    the input; input labels outside a declared universe are rejected.
    """
    if declared is not None:
        unknown = sorted(set(present) - set(declared))
        if unknown:
            raise ValueError(
                f"input symbols {unknown} are not in the declared or fixed "
                f"symbol universe {declared}; the universe is fixed for the "
                "model's lifetime — declare a wider universe on the first "
                "call (symbols=) or build a new instance"
            )
        return declared
    if not present:
        raise ValueError("inputs must contain at least one symbol label")
    return sorted(present)


def _resolve_call_axes(
    specs: Iterable[_WideFrameSpec],
    declared_symbols: list[str] | None,
) -> tuple[npt.NDArray[np.datetime64], list[str]]:
    """Resolve the call's shared timestamp axis and symbol universe.

    The timestamp axis is the sorted union of every input's timestamps; the
    universe is the declared one or, absent a declaration, the labels present
    across the inputs.
    """
    specs = list(specs)
    # One concatenate + unique keeps the union O(FT log(FT)) instead of the
    # O(F^2 T log(FT)) of a growing per-input union.
    all_ts: npt.NDArray[np.datetime64] = np.unique(
        np.concatenate([spec.timestamps for spec in specs])
    )
    if all_ts.shape[0] == 0:
        raise ValueError("inputs must have at least one row")
    present: set[str] = set()
    for spec in specs:
        present.update(key[0] for key in spec.columns)
    symbols = _resolve_symbols(sorted(present), declared_symbols)
    return all_ts, symbols


def _materialize_wide_input(
    spec: _WideFrameSpec,
    name: str,
    all_ts: npt.NDArray[np.datetime64],
    symbols: list[str],
) -> npt.NDArray[Any]:
    """Reindex one input's values onto the call's ``(T, S, *trailing)`` grid.

    Columns map to positions in the cross product of the universe and the
    input's own trailing labels; cells the input does not cover become NaN
    (which requires an inexact dtype — a non-float feature with missing cells is
    rejected here).
    """
    sizes, col_pos = _flat_column_positions(
        [symbols, *spec.trailing_labels], spec.columns
    )
    n_timestamps = len(all_ts)
    # math.prod keeps huge cross products exact (np.prod would overflow
    # int64); an unallocatable size then fails as numpy's own MemoryError.
    n_cells = math.prod(sizes)
    covers_all = (
        spec.timestamps.shape[0] == n_timestamps and len(spec.columns) == n_cells
    )
    if covers_all:
        flat = np.empty((n_timestamps, n_cells), dtype=spec.values.dtype)
        flat[:, col_pos] = spec.values
    else:
        if not np.issubdtype(spec.values.dtype, np.inexact):
            raise ValueError(
                f"non-float feature {name!r} has missing "
                "(timestamp, symbol) cells that cannot be NaN-filled"
            )
        row_pos = np.searchsorted(all_ts, spec.timestamps)
        flat = np.full((n_timestamps, n_cells), np.nan, dtype=spec.values.dtype)
        flat[np.ix_(row_pos, col_pos)] = spec.values
    return flat.reshape(n_timestamps, *sizes)


def normalize_wide_inputs(
    data: RawData,
    *,
    declared_symbols: list[str] | None = None,
) -> tuple[
    npt.NDArray[np.datetime64], npt.NDArray[np.str_], dict[str, npt.NDArray[Any]]
]:
    """Normalize any accepted inputs shape into feature arrays (the single path).

    Runs :func:`as_wide`, reads each canonical wide frame into a
    :class:`_WideFrameSpec`, resolves the shared timestamp axis and symbol
    universe (:func:`_resolve_call_axes`), then materializes each input onto
    that grid (:func:`_materialize_wide_input`).

    Args:
        data: Inputs in any accepted shape.
        declared_symbols: Validated symbol universe, or ``None``.

    Returns:
        A ``(timestamps, symbols, inputs)`` tuple with fresh arrays.

    Raises:
        ValueError: On a DataFrame-contract violation, or symbols
            outside the declared universe.
    """
    specs = {
        name: _extract_wide_frame(frame, name) for name, frame in as_wide(data).items()
    }
    all_ts, symbols = _resolve_call_axes(specs.values(), declared_symbols)
    arrays = {
        name: _materialize_wide_input(spec, name, all_ts, symbols)
        for name, spec in specs.items()
    }
    return all_ts, np.array(symbols, dtype=str), arrays


# --- side inputs -------------------------------------------------------------


def normalize_side_frame(
    df: pd.DataFrame | None,
    call_timestamps: npt.NDArray[np.datetime64],
) -> dict[str, npt.NDArray[Any]]:
    """Normalize a side-input DataFrame into ``(T,)`` per-name arrays.

    Each column keeps its own dtype: object columns hold arbitrary Python
    objects (the transport for what were once ``metadata`` values), numeric and
    boolean columns keep their numpy dtype. Timestamps a column does not cover
    become the dtype's missing value — ``NaN`` for float / complex, ``NaT``
    for datetime / timedelta, and ``None`` for object (present-but-missing
    cells are also normalized to ``None``); a dtype that cannot represent a
    missing value (int, bool, ...) is rejected unless the column covers every
    call timestamp. ``df=None`` means the call has no side inputs (an empty
    mapping).

    Args:
        df: Long-format DataFrame with a ``timestamp`` column and side columns,
            or ``None``.
        call_timestamps: The call's ``(T,)`` timestamps (the alignment axis).

    Returns:
        A mapping of side-input name to a fresh ``(T,)`` array.
    """
    result: dict[str, npt.NDArray[Any]] = {}
    if df is None:
        return result
    if TIMESTAMP not in df.columns:
        raise ValueError("side_data must have a 'timestamp' column")
    value_cols = _validate_names(
        [c for c in df.columns if c != TIMESTAMP], "side_inputs column name"
    )
    arrays, declared = _convert_value_columns(df, value_cols, "side_inputs column")
    frame = pd.DataFrame({TIMESTAMP: _normalize_timestamp_column(df), **arrays})
    agg = frame.groupby(TIMESTAMP, sort=True).last()
    # ``call_timestamps`` is sorted unique, so searchsorted maps each covered
    # timestamp to its row position in one pass; a position whose call
    # timestamp does not equal the side timestamp means it is not covered.
    agg_ts = agg.index.to_numpy()
    positions = np.searchsorted(call_timestamps, agg_ts)
    covered = positions < len(call_timestamps)
    covered[covered] = call_timestamps[positions[covered]] == agg_ts[covered]
    if not bool(covered.all()):
        raise ValueError(
            "side_inputs timestamps must be a subset of the inputs' timestamps"
        )
    n = len(call_timestamps)
    for c, dtype in declared.items():
        if dtype.kind == "O":
            out = np.full(n, None, dtype=dtype)
            out[positions] = [
                None if _is_missing(value) else value for value in agg[c].tolist()
            ]
        elif np.issubdtype(dtype, np.inexact) or dtype.kind in "mM":
            out = _allocate((n,), dtype)
            out[positions] = agg[c].to_numpy()
        elif len(agg) == n:
            out = np.empty(n, dtype=dtype)
            out[positions] = agg[c].to_numpy()
        else:
            raise ValueError(
                f"non-float side_inputs column {c!r} does not cover every "
                "input timestamp; its missing cells cannot be NaN-filled "
                "(use a float or pandas nullable dtype for sparse columns)"
            )
        result[c] = out
    return result


def _is_missing(value: Any) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


# --- assembly ----------------------------------------------------------------


def normalize_call(
    inputs: RawData,
    side_data: pd.DataFrame | None,
    *,
    symbols: Sequence[str] | npt.NDArray[np.str_] | None = None,
) -> Features:
    """Normalize the ``(inputs, side_data)`` pair into a :class:`Features`.

    Inputs of any accepted shape are normalized into ``data`` arrays
    (:func:`normalize_wide_inputs`, via :func:`as_wide`). Omitted
    ``side_data`` is treated as empty.

    Args:
        inputs: Inputs in any accepted shape.
        side_data: Optional long-format side-input DataFrame.
        symbols: The symbol universe (a ``symbols=`` declaration or the
            model's fixed universe), or ``None``.

    Returns:
        A validated :class:`Features`.
    """
    timestamps, symbols_array, input_arrays = normalize_wide_inputs(
        inputs,
        declared_symbols=_resolve_declared_symbols(symbols),
    )
    side = normalize_side_frame(side_data, timestamps)
    return Features(
        timestamps=timestamps,
        symbols=symbols_array,
        data=input_arrays,
        side_data=side,
    )
