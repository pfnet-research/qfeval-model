"""Goldens and validation for the pairwise module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

# y = 2x + 1 on symbol A (exact fit), y independent of x on symbol B.
INPUTS = {
    "x": wide([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0], [4.0, 2.0]]),
    "y": wide([[3.0, 5.0], [5.0, 5.0], [7.0, 5.0], [9.0, 5.0]]),
}


def test_rolling_cov_golden() -> None:
    pred = zoo.rolling_cov("x", "y", window=3).predict(INPUTS)
    # A: cov(x, 2x+1) = 2 * var(x) = 2 * 1 = 2; B: y constant -> 0.
    expected = [[NAN, NAN], [NAN, NAN], [2.0, 0.0], [2.0, 0.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_cov_matches_pandas() -> None:
    got = zoo.rolling_cov("x", "y", window=3).predict(INPUTS).array
    ref = INPUTS["x"].rolling(3).cov(INPUTS["y"]).to_numpy()
    # pandas computes the rolling covariance incrementally and can leave
    # ~1e-15 residue where the exact answer is 0.
    np.testing.assert_allclose(got, ref, equal_nan=True, atol=1e-12)


def test_rolling_corr_golden() -> None:
    pred = zoo.rolling_corr("x", "y", window=3).predict(INPUTS)
    # A is an exact linear fit -> 1; B has a zero-variance y -> NaN.
    expected = [[NAN, NAN], [NAN, NAN], [1.0, NAN], [1.0, NAN]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_beta_golden() -> None:
    pred = zoo.rolling_beta("y", "x", window=3).predict(INPUTS)
    # A: y = 2x + 1 -> beta 2; B: y constant -> cov 0 -> beta 0.
    expected = [[NAN, NAN], [NAN, NAN], [2.0, 0.0], [2.0, 0.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_beta_zero_benchmark_variance_is_nan() -> None:
    inputs = {
        "x": wide([[1.0], [2.0], [3.0]], symbols=("A",)),
        "b": wide([[5.0], [5.0], [5.0]], symbols=("A",)),
    }
    pred = zoo.rolling_beta("x", "b", window=3).predict(inputs)
    assert np.isnan(pred.array).all()


def test_rolling_tracking_error_golden() -> None:
    pred = zoo.rolling_tracking_error("y", "x", window=3).predict(INPUTS)
    # A: active return x+1 has std(x) = 1 over the window; B: std([4,3,4]) etc.
    expected = [
        [NAN, NAN],
        [NAN, NAN],
        [1.0, np.std([4.0, 3.0, 4.0], ddof=1)],
        [1.0, np.std([3.0, 4.0, 3.0], ddof=1)],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_nan_in_either_window_is_nan() -> None:
    inputs = {
        "x": wide([[1.0], [NAN], [3.0], [4.0], [5.0]], symbols=("A",)),
        "y": wide([[2.0], [4.0], [6.0], [8.0], [NAN]], symbols=("A",)),
    }
    got = zoo.rolling_cov("x", "y", window=2).predict(inputs).array
    np.testing.assert_array_equal(got, [[NAN], [NAN], [NAN], [1.0], [NAN]])


def test_bivariate_moments_are_stable_at_large_independent_levels() -> None:
    length, window = 700, 63
    index = np.arange(length, dtype=np.float64)
    x = 1e12 + 0.1 * np.sin(0.31 * index) + 0.03 * np.cos(0.07 * index)
    y = -3e12 + 0.2 * np.cos(0.19 * index) - 0.04 * np.sin(0.11 * index)
    inputs = {
        "x": wide([[float(value)] for value in x], symbols=("A",)),
        "y": wide([[float(value)] for value in y], symbols=("A",)),
    }
    got = zoo.rolling_cov("x", "y", window, ddof=0).predict(inputs).array[:, 0]

    expected = np.full(length, np.nan)
    for end in range(window - 1, length):
        local_x = x[end - window + 1 : end + 1] - x[end - window + 1]
        local_y = y[end - window + 1 : end + 1] - y[end - window + 1]
        expected[end] = np.mean((local_x - local_x.mean()) * (local_y - local_y.mean()))
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=1e-18)


def test_bivariate_moments_recover_after_nan_endpoint_leaves() -> None:
    inputs = {
        "x": wide([[1.0], [2.0], [NAN], [3.0], [4.0], [5.0]], symbols=("A",)),
        "y": wide([[2.0], [4.0], [6.0], [6.0], [8.0], [10.0]], symbols=("A",)),
    }
    got = zoo.rolling_cov("x", "y", window=3).predict(inputs).array[:, 0]
    assert got[-1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_cov("x", "y", window=1),
            "window must be > ddof",
            id="cov-window-1",
        ),
        pytest.param(
            lambda: zoo.rolling_corr("x", "y", window=1),
            "window must be >= 2",
            id="corr-window-1",
        ),
        pytest.param(
            lambda: zoo.rolling_beta("x", "y", window=2, ddof=-1),
            "ddof must be >= 0",
            id="beta-ddof-negative",
        ),
        pytest.param(
            lambda: zoo.rolling_tracking_error("x", "y", window=1),
            "window must be > ddof",
            id="te-window-1",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(INPUTS)
