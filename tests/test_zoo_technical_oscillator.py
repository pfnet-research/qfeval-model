"""Goldens and validation for technical oscillator indicators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")


def test_rsi_all_up_is_100_all_down_is_0_flat_is_50() -> None:
    # Columns: A rises, B falls, C is flat (symbols stay in sorted order).
    frame = wide(
        [[1.0, 9.0, 5.0], [2.0, 8.0, 5.0], [3.0, 7.0, 5.0], [4.0, 6.0, 5.0]],
        symbols=("A", "B", "C"),
    )
    pred = zoo.rolling_relative_strength_index("close", window=3).predict(frame)
    np.testing.assert_allclose(pred.array[3], [100.0, 0.0, 50.0])
    assert np.isnan(pred.array[:3]).all()  # needs window + 1 prices


def test_rsi_mixed_golden() -> None:
    # Deltas: +2, -1, +1, -2 -> window 4: gains mean 3/4, losses mean 3/4.
    frame = wide([[10.0], [12.0], [11.0], [12.0], [10.0]], symbols=("A",))
    pred = zoo.rolling_relative_strength_index("close", window=4).predict(frame)
    assert pred.array[4, 0] == pytest.approx(50.0)
    # Window 2 at t2: deltas +2, -1 -> RSI = 100 * 2 / (2 + 1).
    pred2 = zoo.rolling_relative_strength_index("close", window=2).predict(frame)
    assert pred2.array[2, 0] == pytest.approx(100.0 * 2.0 / 3.0)


def test_rsi_nan_delta_poisons_window() -> None:
    frame = wide([[1.0], [NAN], [3.0], [4.0], [5.0]], symbols=("A",))
    pred = zoo.rolling_relative_strength_index("close", window=2).predict(frame)
    # Deltas: NaN, NaN, NaN(3-NaN), 1, 1 -> first clean window at t4.
    assert np.isnan(pred.array[:4]).all()
    assert pred.array[4, 0] == pytest.approx(100.0)


OHLC = {
    "high": wide([[10.0], [12.0], [11.0], [13.0]], symbols=("A",)),
    "low": wide([[8.0], [9.0], [9.5], [10.0]], symbols=("A",)),
    "close": wide([[9.0], [11.0], [10.0], [12.5]], symbols=("A",)),
}


def test_stochastic_percent_k_golden() -> None:
    pred = zoo.stochastic_percent_k(window=3).predict(OHLC)
    # t2: hh=12, ll=8 -> 100*(10-8)/4 = 50; t3: hh=13, ll=9 -> 100*3.5/4.
    assert pred.array[2, 0] == pytest.approx(50.0)
    assert pred.array[3, 0] == pytest.approx(100.0 * 3.5 / 4.0)
    assert np.isnan(pred.array[:2]).all()


def test_stochastic_percent_k_range_endpoints() -> None:
    inputs = {
        "high": wide([[10.0], [11.0], [12.0]], symbols=("A",)),
        "low": wide([[9.0], [10.0], [10.5]], symbols=("A",)),
        "close": wide([[9.5], [10.5], [12.0]], symbols=("A",)),
    }
    pred = zoo.stochastic_percent_k(window=3).predict(inputs)
    assert pred.array[2, 0] == pytest.approx(100.0)  # close at the window high
    low_close = {
        "high": inputs["high"],
        "low": inputs["low"],
        "close": wide([[9.5], [10.5], [9.0]], symbols=("A",)),
    }
    pred = zoo.stochastic_percent_k(window=3).predict(low_close)
    assert pred.array[2, 0] == pytest.approx(0.0)  # close at the window low


def test_stochastic_flat_range_is_nan() -> None:
    inputs = {
        "high": wide([[10.0], [10.0]], symbols=("A",)),
        "low": wide([[10.0], [10.0]], symbols=("A",)),
        "close": wide([[10.0], [10.0]], symbols=("A",)),
    }
    pred = zoo.stochastic_percent_k(window=2).predict(inputs)
    assert np.isnan(pred.array[1, 0])


def test_stochastic_percent_d_is_rolling_mean_of_percent_k() -> None:
    k = zoo.stochastic_percent_k(window=2).predict(OHLC).array
    d = zoo.stochastic_percent_d(window=2, smooth=2).predict(OHLC).array
    expected = pd.DataFrame(k).rolling(2, min_periods=2).mean().to_numpy()
    np.testing.assert_allclose(d, expected, equal_nan=True)


def test_slow_stochastic_percent_d_two_stage_smoothing() -> None:
    d = zoo.stochastic_percent_d(window=2, smooth=2).predict(OHLC).array
    slow = (
        zoo.slow_stochastic_percent_d(window=2, smooth_k=2, smooth_d=2)
        .predict(OHLC)
        .array
    )
    expected = pd.DataFrame(d).rolling(2, min_periods=2).mean().to_numpy()
    np.testing.assert_allclose(slow, expected, equal_nan=True)


def test_psychological_line_golden() -> None:
    frame = wide([[1.0], [2.0], [2.0], [1.0], [2.0]], symbols=("A",))
    pred = zoo.psychological_line("close", window=3).predict(frame)
    # A flat step counts in the denominator but not as an up step.
    assert pred.array[3, 0] == pytest.approx(100.0 / 3.0)  # deltas +1, 0, -1
    assert pred.array[4, 0] == pytest.approx(100.0 / 3.0)  # deltas 0, -1, +1
    assert np.isnan(pred.array[:3]).all()  # needs window + 1 prices


def test_rci_extremes_and_ties() -> None:
    # Columns: A rises, B falls, C has ties (symbols stay in sorted order).
    frame = wide(
        [[1.0, 9.0, 5.0], [2.0, 8.0, 5.0], [3.0, 7.0, 6.0], [4.0, 6.0, 5.0]],
        symbols=("A", "B", "C"),
    )
    pred = zoo.rank_correlation_index("close", window=4).predict(frame)
    assert pred.array[3, 0] == pytest.approx(100.0)  # strictly rising
    assert pred.array[3, 1] == pytest.approx(-100.0)  # strictly falling
    # C: prices [5,5,6,5] -> tied ranks [2,2,4,2], rho computable and small.
    time_ranks = np.array([1.0, 2.0, 3.0, 4.0])
    price_ranks = np.array([2.0, 2.0, 4.0, 2.0])
    expected = 100.0 * np.corrcoef(price_ranks, time_ranks)[0, 1]
    assert pred.array[3, 2] == pytest.approx(expected)


def test_rci_constant_window_is_nan() -> None:
    frame = wide([[5.0]] * 4, symbols=("A",))
    pred = zoo.rank_correlation_index("close", window=3).predict(frame)
    assert np.isnan(pred.array).all()


def test_moving_average_deviation_percent_exact_scale() -> None:
    frame = wide([[100.0], [100.0], [110.0]], symbols=("A",))
    pred = zoo.moving_average_deviation_percent("close", window=2).predict(frame)
    # t2: mean(100, 110) = 105 -> 100 * (110/105 - 1).
    assert pred.array[2, 0] == pytest.approx(100.0 * (110.0 / 105.0 - 1.0))


def test_moving_average_deviation_zero_mean_is_nan() -> None:
    frame = wide([[-1.0], [1.0]], symbols=("A",))
    pred = zoo.moving_average_deviation_percent("close", window=2).predict(frame)
    assert np.isnan(pred.array[1, 0])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_relative_strength_index("close", window=0),
            "window must be >= 1",
            id="rsi-window-0",
        ),
        pytest.param(
            lambda: zoo.rank_correlation_index("close", window=1),
            "window must be >= 2",
            id="rci-window-1",
        ),
        pytest.param(
            lambda: zoo.stochastic_percent_d(window=3, smooth=0),
            "window must be >= 1",
            id="stochastic-smooth-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(wide([[1.0, 2.0]]))
