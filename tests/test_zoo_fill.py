"""Goldens and validation for the fill module (lag, ffill, stale_age, with_history)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import trailing_wide, wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

# T=4, S=2 hand-checkable price panel (B has a missing value at t2).
PRICES = wide(
    [
        [1.0, 10.0],
        [2.0, NAN],
        [4.0, 30.0],
        [8.0, 40.0],
    ]
)


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        (1, [[NAN, NAN], [1.0, 10.0], [2.0, NAN], [4.0, 30.0]]),
        (2, [[NAN, NAN], [NAN, NAN], [1.0, 10.0], [2.0, NAN]]),
    ],
)
def test_lag_shifts_by_periods(periods: int, expected: list[list[float]]) -> None:
    pred = zoo.lag("close", periods=periods).predict(PRICES)
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_lag_of_ffill_carries_last_valid_lagged_value() -> None:
    pred = zoo.lag(zoo.ffill("close"), periods=1).predict(PRICES)
    expected = [[NAN, NAN], [1.0, 10.0], [2.0, 10.0], [4.0, 30.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_ffill_is_locf_including_today() -> None:
    frame = wide([[1.0, NAN], [2.0, 20.0], [NAN, NAN], [8.0, 40.0]])
    pred = zoo.ffill("close").predict(frame)
    expected = [[1.0, NAN], [2.0, 20.0], [2.0, 20.0], [8.0, 40.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_stale_age_counts_steps_since_last_finite() -> None:
    frame = wide([[NAN, 10.0], [1.0, NAN], [NAN, NAN], [NAN, 40.0]])
    pred = zoo.stale_age("close").predict(frame)
    expected = [[NAN, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


@pytest.mark.parametrize(
    ("factory", "key"),
    [(zoo.ffill, "carry"), (zoo.stale_age, "age")],
)
def test_fill_state_array_is_reused_across_calls(
    factory: Callable[[str], qm.Model], key: str
) -> None:
    model = factory("close")
    model.predict(PRICES.iloc[:2])
    state_id = id(model.state[key])
    model.predict(PRICES.iloc[2:])
    assert id(model.state[key]) == state_id


def test_with_history_outputs_current_value_and_padded_buffer() -> None:
    pred = zoo.with_history("close", window=2).predict(PRICES)
    np.testing.assert_array_equal(pred.array, PRICES.to_numpy())
    expected = [
        [[NAN, NAN], [1.0, 10.0]],
        [[1.0, 10.0], [2.0, NAN]],
        [[2.0, NAN], [4.0, 30.0]],
        [[4.0, 30.0], [8.0, 40.0]],
    ]
    np.testing.assert_array_equal(pred.side_data["history"], np.array(expected))


def test_with_history_supports_trailing_axes() -> None:
    frame = trailing_wide([[1.0, 10.0, 2.0, 20.0], [2.0, 11.0, 3.0, 30.0]])
    pred = zoo.with_history("close", window=2).predict({"close": frame})
    hist = pred.side_data["history"]
    assert hist.shape == (2, 2, 2, 2)  # (T, window, S, k)
    assert np.isnan(hist[0, 0]).all()
    np.testing.assert_array_equal(hist[0, 1], [[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_array_equal(hist[1, 0], [[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_array_equal(hist[1, 1], [[2.0, 11.0], [3.0, 30.0]])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.lag("close", periods=0),
            "periods must be >= 1",
            id="lag-periods-0",
        ),
        pytest.param(
            lambda: zoo.with_history("close", 0),
            "window must be >= 1",
            id="with_history-window-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
