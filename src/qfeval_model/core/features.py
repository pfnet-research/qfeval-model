"""``Features``: the (timestamp x symbol) panel that flows in and out of a call.

One type describes both sides of a call (DESIGN §Selected design
decisions). The
engine normalizes a public call's inputs into a :class:`Features` and hands it
to the model function; the model function writes into a second
:class:`Features` (the call's result) step by step, and ``predict`` returns it
unchanged. Input and output are therefore symmetric inside the engine. The
public entry points accept DataFrames (``RawData``), not ``Features``, so a
result feeds the next model as ``pred.to_wide()``.

The attributes are ordinary arrays and dicts. There are no defensive copies or
read-only wrappers; mutating them (or slicing views) is undefined behaviour.
The main output is real floating-point (unassigned cells default to ``NaN``);
unassigned side-output cells default to the dtype's natural default
(``NaN`` / ``NaT`` / ``0``).

The two fixed axes are timestamp (the execution axis) and symbol. Every
``data`` array is shaped ``(T, S, *trailing)``; the trailing axes are
anonymous — they carry no names and no labels, only a shape. The symbol
universe is fixed by a model's first successful call (``Model.symbols``);
later calls are reindexed to it during normalization (DESIGN §The input
boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from qfeval_model.core.constants import (
    DEFAULT_FEATURE,
    SYMBOL,
    TIMESTAMP,
)


def _allocate(shape: tuple[int, ...], dtype: np.dtype) -> npt.NDArray[Any]:
    """Allocate a default-filled buffer: ``NaN`` for float/complex, ``NaT``
    for datetime/timedelta, else ``0`` (the dtype's missing-value contract,
    shared with the side-input fill in ``normalize``)."""
    if np.issubdtype(dtype, np.inexact):
        return np.full(shape, np.nan, dtype=dtype)
    if dtype.kind in "mM":
        return np.full(shape, "NaT", dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _melt_wide(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Flatten one flat wide frame into long ``(timestamp, symbol)`` rows.

    Every cell is emitted in the frame's row-major order (``NaN`` kept) and
    the value column keeps its dtype. Trailing axes intentionally have no long
    representation; their round-trip format is :meth:`Features.to_wide`.

    This is the leaf-side melt primitive: :meth:`Features.to_frame` renders its
    own ``to_wide`` frame with it, and :func:`~qfeval_model.core.normalize.as_frame`
    imports it for the input-side wide -> long conversion.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(
            "long format does not support trailing axes; use to_wide() instead"
        )
    series = cast(pd.Series, frame.stack(future_stack=True))
    series.index.names = [TIMESTAMP, SYMBOL]
    return series.rename(value_name).reset_index()


@dataclass(eq=False)
class Features:
    """A panel of named arrays over ``(timestamp x symbol)`` plus side channels.

    The same type is a whole call and a single step of it (see
    :meth:`__getitem__`). Values follow NumPy indexing: ``data`` always keeps
    its symbol axis and remains a dict of ndarrays, while integer row indexing
    narrows ``timestamps`` and trailing-less ``side_data`` values to scalars.

    Plain data — no defensive copies or read-only wrappers. The arrays are the
    engine's own buffers; mutating them (or the dicts) is undefined behaviour.

    Attributes:
        timestamps: ``(T,)`` timestamps of the call's effective rows (a
            scalar ``np.datetime64`` for a single step obtained by
            ``pred[i]``). ``T=0`` (empty) is valid; it occurs when a fit
            call is fully replay-skipped.
        symbols: ``(S,)`` symbol labels (sorted ascending, unique).
        data: Mapping of name to a ``(T, S, *trailing)`` array; the trailing
            axes are anonymous. On the input side this holds the call's named
            inputs; on the output side it holds the single main output under
            :data:`DEFAULT_FEATURE` (read key-free via :attr:`array`). Only
            names actually present/written are here.
        side_data: Mapping of name to a ``(T, *shape)`` array (trailing axes
            unnamed). Object-dtype arrays carry arbitrary Python objects
            (missing entries are ``None``); numeric/boolean arrays keep dtype.
        info: The call's output-only diagnostic dict, created empty at the
            public call boundary, shared by every step, and updated in place
            as steps run.
    """

    # These two fields follow NumPy integer indexing in ``__getitem__`` and
    # can therefore hold scalars on a row view. ``Any`` keeps that normal
    # public use honest without adding a second row-view type to the API.
    timestamps: Any
    symbols: npt.NDArray[np.str_]
    data: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    side_data: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    # --- single-array alias ------------------------------------------------

    @property
    def array(self) -> npt.NDArray[Any]:
        """The sole ``data`` array, for a Features that holds exactly one.

        A result carries a single main output, so ``pred.array`` reads it
        without naming a key — ``(T, S, *trailing)`` for a whole call,
        ``(S, *trailing)`` for a step. In-place edits write through the
        returned view; replace the array via ``pred.data[name] = ...``. Raises
        ``ValueError`` unless ``data`` has exactly one entry (e.g. on a
        multi-input Features).
        """
        if len(self.data) != 1:
            raise ValueError(
                f"array requires data to hold exactly one entry, "
                f"but it has {sorted(self.data)}"
            )
        (value,) = self.data.values()
        return value

    def to_frame(self) -> pd.DataFrame:
        """Return the long-format DataFrame with one row per main-output cell.

        Columns are ``timestamp, symbol, close`` (``close`` is
        :data:`DEFAULT_FEATURE`). Every cell is emitted (``NaN`` rows are
        kept) in the output's row-major order, keeping the output dtype.

        Rendered by :func:`_melt_wide` on :meth:`to_wide` (whose output is
        already canonical, so it needs no ``as_wide`` re-normalization).

        Outputs with trailing axes are rejected; :meth:`to_wide` is their
        round-trip display and storage format.
        """
        return _melt_wide(self.to_wide(), DEFAULT_FEATURE)

    def to_wide(self) -> pd.DataFrame:
        """Return the main output as a wide DataFrame (index=timestamp, columns=symbol).

        Trailing axes add column levels: the columns become a ``MultiIndex``
        whose level 0 is the symbol and each deeper level holds one trailing
        axis's integer positions (``0..n-1``) — the same layout the input
        boundary accepts, so the frame round-trips as a wide input. A step
        view (``pred[i]``, 0-d timestamp) renders as a length-1 call.
        """
        output = self.array
        # A step view squeezed the T axis away; put it back for rendering.
        timestamps = np.atleast_1d(self.timestamps)
        trailing = output.shape[2:] if self.timestamps.ndim else output.shape[1:]
        if trailing:
            columns: pd.Index = pd.MultiIndex.from_product(
                [self.symbols, *(range(size) for size in trailing)],
                names=[SYMBOL, *[None] * len(trailing)],
            )
        else:
            columns = pd.Index(self.symbols, name=SYMBOL)
        return pd.DataFrame(
            output.reshape(timestamps.shape[0], len(columns)),
            index=pd.DatetimeIndex(timestamps, name=TIMESTAMP),
            columns=columns,
        )

    def select_trailing(self, *indices: int) -> Features:
        """Select one position from every anonymous trailing axis.

        Returns a ``Features`` view with the timestamp and symbol axes left
        intact. This is the supported path from a multi-horizon (or otherwise
        trailing-axis) result to the plain ``(T, S)`` result accepted by
        metrics. Pass one integer index per trailing axis; negative indices
        follow normal NumPy indexing.

        The selected output and the other fields share their underlying data
        with ``self``, like :meth:`__getitem__`; mutating either view is
        undefined behaviour.
        """
        output = self.array
        fixed_axes = 2 if self.timestamps.ndim else 1
        trailing = output.ndim - fixed_axes
        if trailing == 0:
            raise ValueError("select_trailing() requires at least one trailing axis")
        if len(indices) != trailing:
            raise ValueError(
                f"select_trailing() needs {trailing} index(es), got {len(indices)}"
            )
        (name,) = self.data
        selected = output[(slice(None),) * fixed_axes + indices]
        return replace(self, data={name: selected})

    # --- row slicing -------------------------------------------------------

    def __getitem__(self, rows: int | slice) -> Features:
        """Return the sub-panel for a step (int) or row range (slice).

        Follows ndarray indexing on the ``T`` axis:

        - ``pred[i]`` (int) squeezes the ``T`` axis, giving one step's
          ``Features``: ``data`` arrays become ``(S, *trailing)``,
          ``timestamps`` becomes a scalar ``np.datetime64``, and each
          ``side_data`` value becomes its per-row value — an array for a
          trailing shape, otherwise a NumPy scalar or arbitrary Python
          object. The field annotations describe the whole-call shapes; a
          row view narrows them to these scalars.
        - ``pred[a:b]`` (slice) is a row-range view.

        ``timestamps`` / arrays are views; ``symbols`` and ``info`` (the same
        dict) are shared. Mutating a view is undefined behaviour.
        """
        return Features(
            timestamps=self.timestamps[rows],
            symbols=self.symbols,
            data={name: array[rows] for name, array in self.data.items()},
            side_data={name: array[rows] for name, array in self.side_data.items()},
            info=self.info,
        )

    # --- engine/Step write surface -----------------------------------------

    def _slot(self, name: str | None) -> tuple[dict[str, npt.NDArray[Any]], str]:
        """The (dict, key) the slot ``name`` writes into.

        ``None`` is the main output (``data[DEFAULT_FEATURE]``); a string is a
        side output (``side_data[name]``).
        """
        if name is None:
            return self.data, DEFAULT_FEATURE
        return self.side_data, name

    def _assign(self, name: str | None, rows: slice, value: npt.NDArray[Any]) -> None:
        """Write an ndarray ``value`` into the slot ``name`` at the rows ``rows``.

        ``value`` arrives already ndarray-shaped with its ``T`` axis (the
        caller — ``Step._write`` — owns that). The consistency reference is the
        slot's already-allocated buffer: the first write of a call fixes its
        dtype and trailing shape (lazily allocating an all-default ``(T, ...)``
        buffer), and later writes in the *same call* must match it. There is no
        cross-call lock.
        """
        label = "output" if name is None else f"side output {name!r}"
        if name is None and not np.issubdtype(value.dtype, np.floating):
            # Restricting the main output to real floating dtypes keeps NaN
            # the one missing value (an int/bool buffer would render unwritten
            # rows as plausible 0/False values) and gives every metric one
            # well-defined input dtype.
            raise ValueError(
                "output must be a real floating-point array (unwritten cells "
                f"default to NaN), got dtype {value.dtype}"
            )
        if value.dtype == object:
            raise ValueError(f"{label} must not be an object-dtype array")
        store, key = self._slot(name)
        # The fixed axes between ``T`` and the trailing shape: ``S`` for the
        # main output, none for a side output. ``rows`` is always an
        # engine-produced positive step slice.
        num_rows = rows.stop - rows.start
        lead = (num_rows, len(self.symbols)) if name is None else (num_rows,)
        if value.shape[: len(lead)] != lead:
            lead_desc = ", ".join(str(n) for n in lead)
            raise ValueError(
                f"{label} shape mismatch: expected ({lead_desc}, *trailing), "
                f"got {value.shape}"
            )
        buffer = store.get(key)
        if buffer is None:
            buffer = _allocate(
                (len(self.timestamps), *lead[1:], *value.shape[len(lead) :]),
                value.dtype,
            )
            store[key] = buffer
        else:
            if value.dtype != buffer.dtype:
                raise ValueError(
                    f"{label} dtype mismatch: this call already wrote "
                    f"{buffer.dtype}, got {value.dtype}"
                )
            expected = (*lead, *buffer.shape[len(lead) :])
            if value.shape != expected:
                raise ValueError(
                    f"{label} shape mismatch: expected {expected}, got {value.shape}"
                )
        buffer[rows] = value

    def _view(self, name: str | None, key: int | slice) -> npt.NDArray[Any]:
        """Return a writable view of slot ``name`` (must have been assigned).

        ``key`` follows ndarray indexing: an ``int`` squeezes the ``T`` axis
        (one single-mode step), a ``slice`` keeps it (a batch chunk).
        """
        store, slot = self._slot(name)
        buffer = store.get(slot)
        if buffer is None:
            label = "step.output" if name is None else f"step.side_outputs[{name!r}]"
            raise RuntimeError(
                f"{label} was read before anything was written to it this "
                f"call; assign it first ({label} = ...) — the first "
                "assignment fixes the dtype and trailing shape"
            )
        return buffer[key]

    def _finalize(self) -> None:
        """Fill in a default main output if the call never wrote one.

        A call that never assigned the main output gets an all-``NaN``
        ``float64`` ``(T, S)`` array under :data:`DEFAULT_FEATURE`. Side
        outputs that were never written stay absent from :attr:`side_data`.
        """
        if DEFAULT_FEATURE not in self.data:
            self.data[DEFAULT_FEATURE] = _allocate(
                (len(self.timestamps), len(self.symbols)), np.dtype(np.float64)
            )
