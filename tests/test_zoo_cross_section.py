"""Goldens and validation for the cross-sectional module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import trailing_wide, wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")


def test_cross_sectional_rank_golden() -> None:
    frame = wide(
        [
            [3.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],  # tie -> averaged rank
            [NAN, 5.0, 7.0],
            [NAN, NAN, 4.0],  # single valid symbol -> 0.5
            [NAN, NAN, NAN],  # all-NaN row stays all NaN
        ],
        symbols=("A", "B", "C"),
    )
    pred = zoo.cross_sectional_rank("close").predict(frame)
    expected = [
        [1.0, 0.0, 0.5],
        [0.25, 0.25, 1.0],
        [NAN, 0.0, 1.0],
        [NAN, NAN, 0.5],
        [NAN, NAN, NAN],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_rank_trailing_axes_are_independent() -> None:
    # horizon 0 is ascending A<B<C, horizon 1 is descending.
    frame = trailing_wide([[1.0, 9.0, 2.0, 5.0, 3.0, 1.0]], symbols=("A", "B", "C"))
    pred = zoo.cross_sectional_rank("close").predict({"close": frame})
    expected = [[[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_rank_matches_columnwise_reference_with_ties() -> None:
    rng = np.random.default_rng(20260819)
    values = rng.integers(-3, 4, size=(19, 23)).astype(np.float64)
    values[rng.random(values.shape) < 0.2] = np.nan
    symbols = tuple(f"S{index:02d}" for index in range(values.shape[1]))
    got = (
        zoo.cross_sectional_rank("close")
        .predict(wide(values.tolist(), symbols=symbols))
        .array
    )
    expected = np.full_like(values, np.nan)
    for timestamp, row in enumerate(values):
        valid = ~np.isnan(row)
        count = int(valid.sum())
        if count == 1:
            expected[timestamp, valid] = 0.5
        elif count > 1:
            for symbol in np.flatnonzero(valid):
                below = np.sum(row[valid] < row[symbol])
                equal = np.sum(row[valid] == row[symbol])
                expected[timestamp, symbol] = (below + (equal - 1) / 2.0) / (count - 1)
    np.testing.assert_array_equal(got, expected)


def test_cross_sectional_demean_golden() -> None:
    frame = wide(
        [[1.0, 2.0, NAN], [1.0, 2.0, 3.0], [NAN, NAN, NAN]],
        symbols=("A", "B", "C"),
    )
    pred = zoo.cross_sectional_demean("close").predict(frame)
    expected = [[-0.5, 0.5, NAN], [-1.0, 0.0, 1.0], [NAN, NAN, NAN]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_demean_trailing_axes_are_independent() -> None:
    frame = trailing_wide([[1.0, 9.0, 2.0, 5.0, 3.0, 1.0]], symbols=("A", "B", "C"))
    pred = zoo.cross_sectional_demean("close").predict({"close": frame})
    expected = [[[-1.0, 4.0], [0.0, 0.0], [1.0, -4.0]]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_zscore_golden() -> None:
    frame = wide(
        [
            [1.0, 2.0, 3.0],
            [1.0, NAN, 3.0],
            [5.0, 5.0, 5.0],  # zero std -> NaN
            [NAN, NAN, 7.0],  # single valid -> NaN (no std with ddof=1)
        ],
        symbols=("A", "B", "C"),
    )
    pred = zoo.cross_sectional_zscore("close").predict(frame)
    z = 1.0 / np.sqrt(2.0)
    expected = [
        [-1.0, 0.0, 1.0],
        [-z, NAN, z],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_zscore_ddof_zero_single_valid_is_nan_on_zero_std() -> None:
    frame = wide([[NAN, 7.0]], symbols=("A", "B"))
    pred = zoo.cross_sectional_zscore("close", ddof=0).predict(frame)
    # One valid value has mean == value, std == 0 -> NaN.
    np.testing.assert_array_equal(pred.array, [[NAN, NAN]])


def test_cross_sectional_scale_golden() -> None:
    frame = wide(
        [
            [1.0, -3.0, NAN],
            [0.0, 0.0, 0.0],  # zero gross exposure -> NaN
            [NAN, NAN, NAN],
        ],
        symbols=("A", "B", "C"),
    )
    pred = zoo.cross_sectional_scale("close", gross=2.0).predict(frame)
    expected = [
        [0.5, -1.5, NAN],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_cross_sectional_winsorize_linear_golden() -> None:
    frame = wide(
        [[0.0, 1.0, 2.0, 3.0, 100.0]],
        symbols=("A", "B", "C", "D", "E"),
    )
    pred = zoo.cross_sectional_winsorize("close", 0.25, 0.75).predict(frame)
    np.testing.assert_allclose(pred.array, [[1.0, 1.0, 2.0, 3.0, 3.0]])


def test_cross_sectional_winsorize_inverted_cdf_golden() -> None:
    frame = wide(
        [[0.0, 1.0, 2.0, 3.0, 100.0]],
        symbols=("A", "B", "C", "D", "E"),
    )
    pred = zoo.cross_sectional_winsorize(
        "close", 0.25, 0.75, method="inverted_cdf"
    ).predict(frame)
    np.testing.assert_allclose(pred.array, [[1.0, 1.0, 2.0, 3.0, 3.0]])


def test_cross_sectional_winsorize_keeps_nan_and_all_nan_rows() -> None:
    frame = wide([[1.0, NAN, 3.0], [NAN, NAN, NAN]], symbols=("A", "B", "C"))
    pred = zoo.cross_sectional_winsorize("close", 0.0, 1.0).predict(frame)
    expected = [[1.0, NAN, 3.0], [NAN, NAN, NAN]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.cross_sectional_zscore("close", ddof=-1),
            "ddof must be >= 0",
            id="zscore-ddof-negative",
        ),
        pytest.param(
            lambda: zoo.cross_sectional_scale("close", gross=0.0),
            "gross must be a positive finite number",
            id="scale-gross-0",
        ),
        pytest.param(
            lambda: zoo.cross_sectional_winsorize("close", 0.9, 0.1),
            "0 <= lower <= upper <= 1",
            id="winsorize-crossed",
        ),
        pytest.param(
            lambda: zoo.cross_sectional_winsorize("close", 0.1, 0.9, method="nearest"),
            "method must be one of",
            id="winsorize-bad-method",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(wide([[1.0, 2.0]]))
