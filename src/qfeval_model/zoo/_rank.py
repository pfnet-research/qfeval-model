"""Vectorized average ranks shared by cross-sectional and rolling models."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def average_ranks_axis0(
    values: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64]]:
    """Return one-based average ranks down axis 0 and per-column finite counts.

    Every trailing column is independent.  A single stable axis-wise sort is
    followed by vectorized tie-run start/end scans and one scatter back to the
    original row order; there is no per-column Python callback.  NaN cells
    receive placement ranks after every valid cell; callers mask them with
    the returned counts or their own NaN mask.
    """
    rows = values.shape[0]
    matrix = np.asarray(values, dtype=np.float64).reshape(rows, -1)
    counts = (~np.isnan(matrix)).sum(axis=0, dtype=np.int64)
    order = np.argsort(matrix, axis=0, kind="stable")
    sorted_values = np.take_along_axis(matrix, order, axis=0)

    positions = np.arange(rows, dtype=np.int64)[:, np.newaxis]
    ends_group = np.ones(matrix.shape, dtype=bool)
    ends_group[:-1] = sorted_values[:-1] != sorted_values[1:]
    starts = np.maximum.accumulate(
        np.where(np.roll(ends_group, 1, axis=0), positions, 0), axis=0
    )
    ends = np.minimum.accumulate(np.where(ends_group, positions, rows)[::-1], axis=0)[
        ::-1
    ]

    ranks = np.empty(matrix.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, (starts + ends) / 2.0 + 1.0, axis=0)
    return ranks.reshape(values.shape), counts
