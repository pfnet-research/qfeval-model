"""Goldens and validation for the rolling linear-trend regression module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

# A is an exact line x = 2t + 1; B is a hand-checkable non-line.
FRAME = wide(
    [
        [1.0, 1.0],
        [3.0, 2.0],
        [5.0, 1.0],
        [7.0, 4.0],
    ]
)


def test_trend_slope_recovers_exact_line() -> None:
    pred = zoo.rolling_linear_trend_slope("close", window=3).predict(FRAME)
    np.testing.assert_allclose(pred.array[2:, 0], [2.0, 2.0])
    # B window [1,2,1]: slope = cov(t, x) / var(t) = 0.
    assert pred.array[2, 1] == pytest.approx(0.0)
    # B window [2,1,4]: cov = (0*(-1) + 0 + 1*2 - ... ) hand value 1.0.
    assert pred.array[3, 1] == pytest.approx(1.0)


def test_trend_intercept_is_fit_at_oldest_step() -> None:
    pred = zoo.rolling_linear_trend_intercept("close", window=3).predict(FRAME)
    # A window [1,3,5]: line x = 1 + 2t -> intercept 1; [3,5,7] -> 3.
    np.testing.assert_allclose(pred.array[2:, 0], [1.0, 3.0])
    # B window [1,2,1]: flat fit at the window mean 4/3.
    assert pred.array[2, 1] == pytest.approx(4.0 / 3.0)


def test_trend_r2_one_on_exact_line_and_zero_on_uncorrelated() -> None:
    pred = zoo.rolling_linear_trend_r2("close", window=3).predict(FRAME)
    np.testing.assert_allclose(pred.array[2:, 0], [1.0, 1.0])
    assert pred.array[2, 1] == pytest.approx(0.0)


def test_trend_r2_constant_window_is_nan() -> None:
    pred = zoo.rolling_linear_trend_r2("close", window=3).predict(
        wide([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    )
    assert np.isnan(pred.array[2, 0])
    assert pred.array[2, 1] == pytest.approx(1.0)


def test_trend_r2_uses_one_integrated_window_state() -> None:
    model = zoo.rolling_linear_trend_r2("close", window=3)
    model.predict(FRAME)
    assert set(model.state) == {"shape", "buffer"}
    tracker = model.state["buffer"]
    assert "weighted" in tracker
    assert "squares" in tracker


def test_trend_residual_zero_on_exact_line() -> None:
    pred = zoo.rolling_linear_trend_residual("close", window=3).predict(FRAME)
    np.testing.assert_allclose(pred.array[2:, 0], [0.0, 0.0], atol=1e-12)
    # B window [1,2,1]: fitted current = mean = 4/3 -> residual 1 - 4/3.
    assert pred.array[2, 1] == pytest.approx(1.0 - 4.0 / 3.0)


def test_trend_slope_is_stable_at_a_large_level() -> None:
    length, window = 700, 63
    index = np.arange(length, dtype=np.float64)
    values = 1e12 + 0.1 * np.sin(0.31 * index) + 0.03 * np.cos(0.07 * index)
    frame = wide([[float(value)] for value in values], symbols=("A",))
    got = zoo.rolling_linear_trend_slope("close", window).predict(frame).array[:, 0]

    weights = np.arange(window, dtype=np.float64)
    weights -= weights.mean()
    denominator = np.sum(weights**2)
    expected = np.full(length, np.nan)
    for end in range(window - 1, length):
        block = values[end - window + 1 : end + 1]
        expected[end] = np.sum(weights * (block - block[0])) / denominator
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=1e-18)


def test_trend_warm_up_and_nan_window() -> None:
    frame = wide([[1.0, 1.0], [2.0, NAN], [3.0, 3.0], [4.0, 4.0]])
    pred = zoo.rolling_linear_trend_slope("close", window=3).predict(frame)
    assert np.isnan(pred.array[:2]).all()
    assert np.isnan(pred.array[2:, 1]).all()  # NaN inside both B windows


def test_rolling_autocorr_perfect_on_linear_series() -> None:
    frame = wide([[float(t), float(-t)] for t in range(6)])
    pred = zoo.rolling_autocorr("close", window=3, lag=1).predict(frame)
    # A linear series is perfectly correlated with its lag on both sides.
    np.testing.assert_allclose(pred.array[3:], 1.0)
    assert np.isnan(pred.array[:3]).all()  # needs window + lag observations


def test_rolling_autocorr_alternating_series_is_negative() -> None:
    frame = wide([[1.0], [-1.0], [1.0], [-1.0], [1.0]], symbols=("A",))
    pred = zoo.rolling_autocorr("close", window=4, lag=1).predict(frame)
    assert pred.array[4, 0] == pytest.approx(-1.0)


def test_rolling_autocorr_constant_series_is_nan() -> None:
    frame = wide([[2.0]] * 5, symbols=("A",))
    pred = zoo.rolling_autocorr("close", window=3, lag=1).predict(frame)
    assert np.isnan(pred.array).all()


def test_rolling_autocorr_matches_pandas_reference() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(size=30)
    frame = wide([[float(v)] for v in series], symbols=("A",))
    window, lag = 5, 2
    got = zoo.rolling_autocorr("close", window=window, lag=lag).predict(frame).array
    for t in range(window + lag - 1, 30):
        recent = series[t - window + 1 : t + 1]
        lagged = series[t - window + 1 - lag : t + 1 - lag]
        expected = np.corrcoef(recent, lagged)[0, 1]
        assert got[t, 0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_linear_trend_slope("close", 1),
            "window must be >= 2",
            id="slope-window-1",
        ),
        pytest.param(
            lambda: zoo.rolling_autocorr("close", 3, lag=0),
            "lag must be >= 1",
            id="autocorr-lag-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(FRAME)
