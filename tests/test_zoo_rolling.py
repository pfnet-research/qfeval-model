"""Goldens and validation for the rolling module."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo
from qfeval_model.zoo import _window as window_kernels

NAN = float("nan")

PRICES = wide(
    [
        [1.0, 10.0],
        [2.0, NAN],
        [4.0, 30.0],
        [8.0, 40.0],
    ]
)


def test_rolling_mean_golden() -> None:
    pred = zoo.rolling_mean("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [1.5, NAN], [3.0, NAN], [6.0, 35.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_sum_golden() -> None:
    pred = zoo.rolling_sum("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [3.0, NAN], [6.0, NAN], [12.0, 70.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_sum_recovers_as_soon_as_an_overflowing_value_leaves() -> None:
    frame = wide([[1e308], [1e308], [1.0], [1.0]], symbols=("A",))
    pred = zoo.rolling_sum("close", window=3).predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, NAN, NAN, 1e308])


def test_rolling_sum_extreme_does_not_taint_a_later_tiny_window() -> None:
    tiny = 1e-100
    frame = wide([[1e308], [tiny], [tiny], [tiny]], symbols=("A",))
    pred = zoo.rolling_sum("close", window=3).predict(frame)
    np.testing.assert_allclose(pred.array[-1, 0], 3 * tiny, rtol=2e-15)


def test_window_sum_matches_fsum_and_finite_counts() -> None:
    rng = np.random.default_rng(23)
    length, cells, window = 257, 5, 17
    values = 1e12 + rng.normal(scale=0.1, size=(length, cells))
    rows, columns = np.indices(values.shape)
    missing = np.asarray((5 * rows + 3 * columns) % 19 < 4, dtype=np.bool_)
    missing[: 2 * window] = False
    values[missing] = np.nan
    state: dict[str, Any] = {}

    for end, row in enumerate(values):
        total, count = window_kernels.window_sum(state, row, window)
        block = values[max(0, end - window + 1) : end + 1]
        expected_count = np.sum(~np.isnan(block), axis=0)
        expected_total = np.array(
            [math.fsum(column[~np.isnan(column)]) for column in block.T]
        )
        np.testing.assert_array_equal(count, expected_count)
        np.testing.assert_allclose(
            total,
            expected_total,
            rtol=window * np.finfo(np.float64).eps,
        )


def test_window_sum_reuses_fixed_chunk_buffers() -> None:
    window, cells = 65, 6
    dense_state: dict[str, Any] = {}
    for timestamp in range(2 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        window_kernels.window_sum(dense_state, row, window)
    assert dense_state["buffer"]["valid"] is None
    assert dense_state["buffer"]["right_counts"] is None

    state: dict[str, Any] = {}
    for timestamp in range(3 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[timestamp % cells] = np.nan
        window_kernels.window_sum(state, row, window)

    tracker = state["buffer"]
    assert tracker["values"].shape == (window, cells)
    assert tracker["valid"].shape == (window, cells)
    assert tracker["right"].shape == (window + 1, cells)
    assert tracker["right_counts"].shape == (window + 1, cells)
    assert tracker["right_counts"].dtype == np.min_scalar_type(window)
    names = ("values", "valid", "right", "right_counts")
    buffers = tuple(id(tracker[name]) for name in names)

    for timestamp in range(3 * window, 6 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[(2 * timestamp) % cells] = np.nan
        window_kernels.window_sum(state, row, window)

    assert tuple(id(tracker[name]) for name in names) == buffers


def test_rolling_valid_sum_uses_latest_finite_values_per_cell() -> None:
    pred = zoo.rolling_valid_sum("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [3.0, NAN], [6.0, 40.0], [12.0, 70.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_valid_sum_does_not_advance_on_nan() -> None:
    frame = wide([[1.0], [2.0], [NAN], [4.0]], symbols=("A",))
    pred = zoo.rolling_valid_sum("close", window=2).predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, 3.0, 3.0, 6.0])


def test_rolling_valid_sum_recovers_after_an_unrepresentable_sum() -> None:
    frame = wide([[1e308], [1e308], [-1e308]], symbols=("A",))
    pred = zoo.rolling_valid_sum("close", window=2).predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, NAN, 0.0])


def test_rolling_valid_sum_rebases_when_a_large_value_leaves() -> None:
    frame = wide([[1e308], [1.0], [1.0], [1.0]], symbols=("A",))
    pred = zoo.rolling_valid_sum("close", window=3).predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, NAN, 1e308, 3.0])


@pytest.mark.parametrize("window", [1, 2, 3, 7])
def test_rolling_valid_sum_matches_per_cell_finite_history(window: int) -> None:
    """Cell-local chunk positions may diverge at every timestamp."""
    rng = np.random.default_rng(12)
    values = rng.integers(-100, 101, size=(73, 5)).astype(np.float64)
    rows, cells = np.indices(values.shape)
    values[(3 * rows + 5 * cells) % 11 < 3] = np.nan

    histories: list[list[float]] = [[] for _ in range(values.shape[1])]
    expected = np.full_like(values, np.nan)
    for row, current in enumerate(values):
        for cell, value in enumerate(current):
            if not np.isnan(value):
                histories[cell].append(float(value))
                histories[cell] = histories[cell][-window:]
            if len(histories[cell]) == window:
                expected[row, cell] = math.fsum(histories[cell])

    pred = zoo.rolling_valid_sum("close", window=window).predict(
        wide(values.tolist(), symbols=tuple("ABCDE"))
    )
    np.testing.assert_array_equal(pred.array, expected)


def test_rolling_valid_sum_extreme_does_not_taint_a_later_tiny_window() -> None:
    tiny = 1e-100
    frame = wide(
        [
            [1e308, NAN],
            [tiny, 1e308],
            [tiny, tiny],
            [tiny, NAN],
            [tiny, tiny],
            [tiny, tiny],
        ],
        symbols=("A", "B"),
    )
    pred = zoo.rolling_valid_sum("close", window=3).predict(frame)
    np.testing.assert_allclose(pred.array[-1], 3 * tiny, rtol=2e-15)


def test_window_valid_sum_reuses_fixed_chunk_buffers() -> None:
    window, cells = 5, 6
    dense_state: dict[str, Any] = {}
    for timestamp in range(2 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        window_kernels.window_valid_sum(dense_state, row, window)
    assert dense_state["buffer"]["scratch"] is None

    state: dict[str, Any] = {}
    for timestamp in range(4 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[timestamp % cells] = np.nan
        window_kernels.window_valid_sum(state, row, window)

    tracker = state["buffer"]
    assert tracker["values"].shape == (cells, window)
    assert tracker["right"].shape == (cells, window + 1)
    assert tracker["scratch"].shape == (cells, window)
    buffers = tuple(id(tracker[name]) for name in ("values", "right", "scratch"))

    for timestamp in range(4 * window, 8 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[(2 * timestamp) % cells] = np.nan
        window_kernels.window_valid_sum(state, row, window)

    assert (
        tuple(id(tracker[name]) for name in ("values", "right", "scratch")) == buffers
    )


def test_rolling_valid_mean_golden() -> None:
    pred = zoo.rolling_valid_mean("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [1.5, NAN], [3.0, 20.0], [6.0, 35.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_valid_mean_equals_valid_sum_over_window() -> None:
    frame = wide([[1.0, NAN], [2.0, 20.0], [NAN, 30.0], [4.0, NAN], [8.0, 50.0]])
    mean = zoo.rolling_valid_mean("close", window=3).predict(frame).array
    total = zoo.rolling_valid_sum("close", window=3).predict(frame).array
    np.testing.assert_array_equal(mean, total / 3.0)


def test_rolling_valid_var_uses_latest_finite_values_per_cell() -> None:
    pred = zoo.rolling_valid_var("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [0.5, NAN], [2.0, 200.0], [8.0, 50.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_valid_std_ddof_zero_golden() -> None:
    pred = zoo.rolling_valid_std("close", window=2, ddof=0).predict(PRICES)
    expected = [[NAN, NAN], [0.5, NAN], [1.0, 10.0], [2.0, 5.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_valid_var_does_not_advance_on_nan() -> None:
    frame = wide([[1.0], [2.0], [NAN], [4.0]], symbols=("A",))
    pred = zoo.rolling_valid_var("close", window=2).predict(frame)
    np.testing.assert_allclose(pred.array[:, 0], [NAN, 0.5, 0.5, 2.0])


def test_rolling_valid_std_squared_equals_valid_var() -> None:
    frame = wide([[1.0, NAN], [2.0, 20.0], [NAN, 30.0], [4.0, NAN], [8.0, 50.0]])
    std = zoo.rolling_valid_std("close", window=2).predict(frame).array
    variance = zoo.rolling_valid_var("close", window=2).predict(frame).array
    np.testing.assert_allclose(std**2, variance, rtol=1e-12)


@pytest.mark.parametrize(("window", "ddof"), [(1, 0), (2, 0), (2, 1), (3, 1), (7, 1)])
def test_rolling_valid_moments_match_per_cell_finite_history(
    window: int, ddof: int
) -> None:
    """Cell-local chunk positions may diverge at every timestamp."""
    rng = np.random.default_rng(12)
    values = rng.integers(-100, 101, size=(73, 5)).astype(np.float64)
    rows, cells = np.indices(values.shape)
    values[(3 * rows + 5 * cells) % 11 < 3] = np.nan

    histories: list[list[float]] = [[] for _ in range(values.shape[1])]
    expected_mean = np.full_like(values, np.nan)
    expected_var = np.full_like(values, np.nan)
    for row, current in enumerate(values):
        for cell, value in enumerate(current):
            if not np.isnan(value):
                histories[cell].append(float(value))
                histories[cell] = histories[cell][-window:]
            if len(histories[cell]) == window:
                expected_mean[row, cell] = np.mean(histories[cell])
                expected_var[row, cell] = np.var(histories[cell], ddof=ddof)

    frame = wide(values.tolist(), symbols=tuple("ABCDE"))
    mean = zoo.rolling_valid_mean("close", window=window).predict(frame).array
    variance = (
        zoo.rolling_valid_var("close", window=window, ddof=ddof).predict(frame).array
    )
    std = zoo.rolling_valid_std("close", window=window, ddof=ddof).predict(frame).array
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance, expected_var, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(std, np.sqrt(expected_var), rtol=1e-12, atol=1e-12)


def test_rolling_valid_var_recovers_after_an_extreme_value_leaves() -> None:
    frame = wide([[1e308], [1.0], [1.0], [1.0]], symbols=("A",))
    pred = zoo.rolling_valid_var("close", window=3).predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, NAN, NAN, 0.0])


def test_rolling_valid_var_extreme_does_not_taint_a_later_tiny_window() -> None:
    tiny = 1e-100
    frame = wide(
        [
            [1e308, NAN],
            [tiny, 1e308],
            [2 * tiny, tiny],
            [tiny, NAN],
            [tiny, 2 * tiny],
            [tiny, tiny],
        ],
        symbols=("A", "B"),
    )
    pred = zoo.rolling_valid_var("close", window=3).predict(frame)
    # A's last window is three equal tinies (exactly 0); B's is {tiny, 2*tiny,
    # tiny} whose ddof=1 variance is tiny**2 / 3 — no 1e308 residue in either.
    np.testing.assert_allclose(
        pred.array[-1], [0.0, tiny * tiny / 3.0], rtol=2e-15, atol=0.0
    )


def test_rolling_valid_var_resumes_after_a_long_missing_gap() -> None:
    frame = wide([[1.0], [2.0]] + [[NAN]] * 10 + [[4.0]], symbols=("A",))
    pred = zoo.rolling_valid_var("close", window=3).predict(frame)
    assert np.isnan(pred.array[:-1, 0]).all()
    np.testing.assert_allclose(pred.array[-1, 0], 7.0 / 3.0)


def test_rolling_valid_variance_is_stable_at_a_large_level() -> None:
    length, window = 512, 63
    index = np.arange(length, dtype=np.float64)
    values = np.stack(
        [
            1e12 + 0.1 * np.sin(0.31 * index) + 0.03 * np.cos(0.07 * index),
            -1e12 + 0.2 * np.cos(0.13 * index),
        ],
        axis=1,
    )
    values[5::7, 0] = np.nan
    values[3::5, 1] = np.nan
    frame = wide(values.tolist(), symbols=("A", "B"))
    got = zoo.rolling_valid_var("close", window, ddof=0).predict(frame).array

    expected = np.full_like(values, np.nan)
    histories: list[list[float]] = [[] for _ in range(values.shape[1])]
    for row, current in enumerate(values):
        for cell, value in enumerate(current):
            if not np.isnan(value):
                histories[cell].append(float(value))
                histories[cell] = histories[cell][-window:]
            if len(histories[cell]) == window:
                relative = np.array(histories[cell]) - histories[cell][0]
                expected[row, cell] = np.mean(relative**2) - np.mean(relative) ** 2
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=1e-18)


def test_window_valid_moments_reuses_fixed_chunk_buffers() -> None:
    window, cells = 5, 6
    state: dict[str, Any] = {}
    for timestamp in range(4 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[timestamp % cells] = np.nan
        window_kernels.window_valid_moments(state, row, window)

    tracker = state["buffer"]
    assert tracker["values"].shape == (cells, window)
    assert tracker["right_sums"].shape == (cells, window + 1)
    assert tracker["right_squares"].shape == (cells, window + 1)
    assert tracker["scratch_sums"].shape == (cells, window)
    assert tracker["scratch_squares"].shape == (cells, window)
    names = ("values", "right_sums", "right_squares", "scratch_sums", "scratch_squares")
    buffers = tuple(id(tracker[name]) for name in names)

    for timestamp in range(4 * window, 8 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[(2 * timestamp) % cells] = np.nan
        window_kernels.window_valid_moments(state, row, window)

    assert tuple(id(tracker[name]) for name in names) == buffers


def test_rolling_var_golden() -> None:
    pred = zoo.rolling_var("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [0.5, NAN], [2.0, NAN], [8.0, 50.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_std_golden() -> None:
    pred = zoo.rolling_std("close", window=2).predict(PRICES)
    root_half = np.sqrt(0.5)
    expected = [
        [NAN, NAN],
        [root_half, NAN],  # std([1, 2], ddof=1)
        [np.sqrt(2.0), NAN],  # std([2, 4], ddof=1)
        [np.sqrt(8.0), np.sqrt(50.0)],  # std([4, 8]) / std([30, 40])
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_std_ddof_zero_golden() -> None:
    pred = zoo.rolling_std("close", window=2, ddof=0).predict(PRICES)
    expected = [
        [NAN, NAN],
        [0.5, NAN],  # std([1, 2], ddof=0)
        [1.0, NAN],
        [2.0, 5.0],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_zscore_golden() -> None:
    pred = zoo.rolling_zscore("close", window=2).predict(PRICES)
    z = np.sqrt(0.5)  # every doubling pair scores (x - mean) / std = sqrt(1/2)
    expected = [[NAN, NAN], [z, NAN], [z, NAN], [z, z]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_zscore_ddof_zero_golden() -> None:
    pred = zoo.rolling_zscore("close", window=2, ddof=0).predict(PRICES)
    expected = [[NAN, NAN], [1.0, NAN], [1.0, NAN], [1.0, 1.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_zscore_zero_std_window_is_nan() -> None:
    pred = zoo.rolling_zscore("close", window=2).predict(wide([[5.0, 1.0], [5.0, 2.0]]))
    expected = [[NAN, NAN], [NAN, np.sqrt(0.5)]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_min_max_range_goldens() -> None:
    frame = wide([[3.0, 1.0], [1.0, NAN], [2.0, 4.0]])
    lo = zoo.rolling_min("close", window=2).predict(frame).array
    hi = zoo.rolling_max("close", window=2).predict(frame).array
    span = zoo.rolling_range("close", window=2).predict(frame).array
    np.testing.assert_array_equal(lo, [[NAN, NAN], [1.0, NAN], [1.0, NAN]])
    np.testing.assert_array_equal(hi, [[NAN, NAN], [3.0, NAN], [2.0, NAN]])
    np.testing.assert_array_equal(span, [[NAN, NAN], [2.0, NAN], [1.0, NAN]])


def test_rolling_count_counts_finite_after_strict_warmup() -> None:
    frame = wide([[1.0, NAN], [NAN, NAN], [3.0, 5.0]])
    pred = zoo.rolling_count("close", window=2).predict(frame)
    expected = [[NAN, NAN], [1.0, 0.0], [1.0, 1.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_rolling_median_golden() -> None:
    frame = wide([[1.0, 2.0], [5.0, 4.0], [3.0, NAN]])
    pred = zoo.rolling_median("close", window=3).predict(frame)
    expected = [[NAN, NAN], [NAN, NAN], [3.0, NAN]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_rolling_median_even_window_averages_middle_two() -> None:
    frame = wide([[1.0, 10.0], [4.0, 20.0]])
    pred = zoo.rolling_median("close", window=2).predict(frame)
    expected = [[NAN, NAN], [2.5, 15.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_product_golden() -> None:
    pred = zoo.rolling_product("close", window=2).predict(PRICES)
    expected = [[NAN, NAN], [2.0, NAN], [8.0, NAN], [32.0, 1200.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_product_recovers_after_overflow_and_nan_leave() -> None:
    frame = wide(
        [[1e308], [1e308], [NAN], [1e-308], [1e308]],
        symbols=("A",),
    )
    got = zoo.rolling_product("close", window=2).predict(frame).array[:, 0]
    np.testing.assert_allclose(
        got,
        [NAN, NAN, NAN, NAN, 1.0],
        rtol=2e-15,
        equal_nan=True,
    )


def test_rolling_argmin_argmax_steps_back_with_latest_tie() -> None:
    frame = wide([[2.0, 5.0], [1.0, 5.0], [1.0, 3.0], [4.0, NAN]])
    amin = zoo.rolling_argmin("close", window=3).predict(frame).array
    amax = zoo.rolling_argmax("close", window=3).predict(frame).array
    # A: min of [2,1,1] is tied -> latest (0 back); then [1,1,4] -> 1 back.
    np.testing.assert_array_equal(amin[:, 0], [NAN, NAN, 0.0, 1.0])
    np.testing.assert_array_equal(amax[:, 0], [NAN, NAN, 2.0, 0.0])
    # B: NaN inside the window poisons the output.
    np.testing.assert_array_equal(amin[:, 1], [NAN, NAN, 0.0, NAN])
    np.testing.assert_array_equal(amax[:, 1], [NAN, NAN, 1.0, NAN])


def test_recursive_ewm_mean_golden_with_nan_carry() -> None:
    pred = zoo.recursive_ewm_mean("close", alpha=0.5).predict(PRICES)
    expected = [
        [1.0, 10.0],  # initialized at the first finite value
        [1.5, 10.0],  # NaN input carries the previous average
        [2.75, 20.0],
        [5.375, 30.0],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_recursive_ewm_mean_alpha_one_tracks_latest_value() -> None:
    pred = zoo.recursive_ewm_mean("close", alpha=1.0).predict(PRICES)
    expected = [[1.0, 10.0], [2.0, 10.0], [4.0, 30.0], [8.0, 40.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_linear_weighted_moving_average_weight_golden() -> None:
    frame = wide([[1.0, 30.0], [2.0, 20.0], [3.0, 10.0]])
    pred = zoo.linear_weighted_moving_average("close", window=3).predict(frame)
    # weights 1,2,3 oldest->newest: (1*1+2*2+3*3)/6 and (30+40+30)/6.
    expected = [[NAN, NAN], [NAN, NAN], [14.0 / 6.0, 100.0 / 6.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_moving_averages_are_identity_on_constant_series() -> None:
    frame = wide([[7.0, 7.0]] * 5)
    factories: list[Callable[[], qm.Model]] = [
        lambda: zoo.rolling_mean("close", window=3),
        lambda: zoo.linear_weighted_moving_average("close", window=3),
        lambda: zoo.recursive_ewm_mean("close", alpha=0.3),
    ]
    for factory in factories:
        tail = factory().predict(frame).array[2:]
        np.testing.assert_allclose(tail, 7.0)


def test_reducible_window_kernels_reuse_fixed_chunk_buffers() -> None:
    window, cells = 17, 6
    cases: list[tuple[Callable[..., Any], dict[str, Any]]] = [
        (window_kernels.window_moments, {}),
        (window_kernels.window_product, {}),
        (window_kernels.window_linear_stats, {}),
    ]
    bivariate_state: dict[str, Any] = {}
    extrema_state: dict[str, Any] = {}

    for timestamp in range(3 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[(2 * timestamp) % cells] = np.nan
        for kernel, state in cases:
            kernel(state, row, window)
        window_kernels.window_bivariate_moments(bivariate_state, row, -row, window)
        window_kernels.window_extrema(extrema_state, row, window)

    states = [state for _, state in cases] + [bivariate_state, extrema_state]
    buffer_ids = [
        tuple(
            id(value)
            for value in state["buffer"].values()
            if isinstance(value, np.ndarray)
        )
        for state in states
    ]
    for state in states:
        arrays = [
            value for value in state["buffer"].values() if isinstance(value, np.ndarray)
        ]
        assert arrays
        assert all(array.base is None for array in arrays)
        assert all(array.ndim == 1 or array.shape[0] <= window + 1 for array in arrays)

    for timestamp in range(3 * window, 7 * window):
        row = np.arange(cells, dtype=np.float64) + timestamp
        row[(3 * timestamp) % cells] = np.nan
        for kernel, state in cases:
            kernel(state, row, window)
        window_kernels.window_bivariate_moments(bivariate_state, row, -row, window)
        window_kernels.window_extrema(extrema_state, row, window)

    assert buffer_ids == [
        tuple(
            id(value)
            for value in state["buffer"].values()
            if isinstance(value, np.ndarray)
        )
        for state in states
    ]


def test_count_kernel_uses_compact_boolean_state() -> None:
    model = zoo.rolling_count("close", window=17)
    model.predict(PRICES)
    tracker = model.state["buffer"]
    assert tracker["truth"].dtype == np.bool_
    assert tracker["truth"].shape == (17, 2)
    assert tracker["valid"] is None
    assert not any(
        isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 2
        for value in tracker.values()
    )


def test_one_sided_extrema_omit_the_unused_window_matrix() -> None:
    minimum_state: dict[str, Any] = {}
    maximum_state: dict[str, Any] = {}
    for row in PRICES.to_numpy():
        window_kernels.window_extrema(minimum_state, row, 3, direction="min")
        window_kernels.window_extrema(maximum_state, row, 3, direction="max")
    assert minimum_state["buffer"]["maximum"] is None
    assert minimum_state["buffer"]["left_maximum"] is None
    assert maximum_state["buffer"]["minimum"] is None
    assert maximum_state["buffer"]["left_minimum"] is None


def test_full_block_uses_a_fixed_duplicated_ring() -> None:
    window, cells = 5, 3
    state: dict[str, Any] = {}
    block = None
    for timestamp in range(2 * window + 2):
        row = np.arange(cells, dtype=np.float64) + 10 * timestamp
        block = window_kernels.full_block(state, row, window)

    assert block is not None
    expected = np.array(
        [
            np.arange(cells, dtype=np.float64) + 10 * timestamp
            for timestamp in range(window + 2, 2 * window + 2)
        ]
    )
    np.testing.assert_array_equal(block, expected)
    values = state["buffer"]["values"]
    assert values.shape == (2 * window, cells)
    assert values.base is None
    assert np.shares_memory(block, values)
    buffer_id = id(values)
    for timestamp in range(2 * window + 2, 4 * window):
        window_kernels.full_block(
            state,
            np.arange(cells, dtype=np.float64) + 10 * timestamp,
            window,
        )
    assert id(state["buffer"]["values"]) == buffer_id


def test_rolling_variance_is_stable_at_a_large_level() -> None:
    length, window = 1024, 127
    index = np.arange(length, dtype=np.float64)
    values = 1e12 + 0.1 * np.sin(0.31 * index) + 0.03 * np.cos(0.07 * index)
    frame = wide([[float(value)] for value in values], symbols=("A",))
    got = zoo.rolling_var("close", window, ddof=0).predict(frame).array[:, 0]

    expected = np.full(length, np.nan)
    for end in range(window - 1, length):
        block = values[end - window + 1 : end + 1]
        relative = block - block[0]
        expected[end] = np.mean(relative**2) - np.mean(relative) ** 2
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=1e-18)


def test_local_chunk_statistics_recover_after_nan_endpoint_leaves() -> None:
    frame = wide([[1.0], [2.0], [NAN], [3.0], [4.0], [5.0]], symbols=("A",))
    variance = zoo.rolling_var("close", window=3, ddof=1).predict(frame).array[:, 0]
    slope = zoo.rolling_linear_trend_slope("close", window=3).predict(frame).array[:, 0]
    span = zoo.rolling_range("close", window=3).predict(frame).array[:, 0]
    assert variance[-1] == pytest.approx(1.0)
    assert slope[-1] == pytest.approx(1.0)
    assert span[-1] == pytest.approx(2.0)


def test_composition_lag_of_rolling_mean_golden() -> None:
    pred = zoo.lag(zoo.rolling_mean("close", 2)).predict(PRICES)
    expected = [[NAN, NAN], [NAN, NAN], [1.5, NAN], [3.0, NAN]]
    np.testing.assert_allclose(pred.array, np.array(expected))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_mean("close", 0),
            "window must be >= 1",
            id="rolling_mean-window-0",
        ),
        pytest.param(
            lambda: zoo.rolling_mean("close", 2.5),  # type: ignore[arg-type]
            "window must be an integer",
            id="rolling_mean-window-float",
        ),
        pytest.param(
            lambda: zoo.rolling_mean("close", True),
            "window must be an integer",
            id="rolling_mean-window-bool",
        ),
        pytest.param(
            lambda: zoo.rolling_valid_sum("close", 0),
            "window must be >= 1",
            id="rolling_valid_sum-window-0",
        ),
        pytest.param(
            lambda: zoo.rolling_valid_mean("close", 0),
            "window must be >= 1",
            id="rolling_valid_mean-window-0",
        ),
        pytest.param(
            lambda: zoo.rolling_valid_var("close", 1),
            "window must be > ddof",
            id="rolling_valid_var-window-1",
        ),
        pytest.param(
            lambda: zoo.rolling_valid_std("close", 2, ddof=-1),
            "ddof must be >= 0",
            id="rolling_valid_std-ddof-negative",
        ),
        pytest.param(
            lambda: zoo.rolling_std("close", 1),
            "window must be > ddof",
            id="rolling_std-window-1",
        ),
        pytest.param(
            lambda: zoo.rolling_std("close", 2, ddof=-1),
            "ddof must be >= 0",
            id="rolling_std-ddof-negative",
        ),
        pytest.param(
            lambda: zoo.rolling_var("close", 2, ddof=2),
            "window must be > ddof",
            id="rolling_var-window-eq-ddof",
        ),
        pytest.param(
            lambda: zoo.rolling_zscore("close", 1),
            "window must be > ddof",
            id="rolling_zscore-window-1",
        ),
        pytest.param(
            lambda: zoo.recursive_ewm_mean("close", 0.0),
            "alpha must be in",
            id="recursive_ewm_mean-alpha-0",
        ),
        pytest.param(
            lambda: zoo.recursive_ewm_mean("close", 1.1),
            "alpha must be in",
            id="recursive_ewm_mean-alpha-1.1",
        ),
        pytest.param(
            lambda: zoo.linear_weighted_moving_average("close", 0),
            "window must be >= 1",
            id="lwma-window-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
