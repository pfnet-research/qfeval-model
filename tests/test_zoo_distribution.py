"""Goldens and validation for the distribution module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

FRAME = wide(
    [
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, NAN],
    ]
)


def test_rolling_quantile_linear_golden() -> None:
    pred = zoo.rolling_quantile("close", window=3, q=0.5, method="linear").predict(
        FRAME
    )
    expected = [[NAN, NAN], [NAN, NAN], [2.0, 3.0], [3.0, NAN]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_quantile_linear_interpolates() -> None:
    pred = zoo.rolling_quantile("close", window=3, q=0.25, method="linear").predict(
        FRAME
    )
    # sorted [1,2,3]: linear q=0.25 -> 1.5.
    assert pred.array[2, 0] == pytest.approx(1.5)


def test_rolling_quantile_inverted_cdf_picks_sample() -> None:
    pred = zoo.rolling_quantile(
        "close", window=3, q=0.25, method="inverted_cdf"
    ).predict(FRAME)
    # inverted_cdf returns an order statistic, never an interpolated value.
    assert pred.array[2, 0] == 1.0


def test_rolling_rank_golden() -> None:
    frame = wide([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    pred = zoo.rolling_rank("close", window=3).predict(frame)
    # A: current 2 within [3,1,2] -> 1 below, no tie -> 0.5.
    # B: all ties -> average position 1 of positions 0..2 -> 0.5.
    np.testing.assert_allclose(pred.array[2], [0.5, 0.5])
    assert np.isnan(pred.array[:2]).all()


def test_rolling_rank_extremes_and_window_one() -> None:
    frame = wide([[1.0, 9.0], [2.0, 5.0], [3.0, 1.0]])
    pred = zoo.rolling_rank("close", window=3).predict(frame)
    np.testing.assert_allclose(pred.array[2], [1.0, 0.0])
    single = zoo.rolling_rank("close", window=1).predict(frame)
    np.testing.assert_allclose(single.array, 0.5)


def test_threshold_rates_golden() -> None:
    below = zoo.rolling_below_threshold_rate("close", window=3, threshold=2.5)
    above = zoo.rolling_above_threshold_rate("close", window=3, threshold=2.5)
    got_below = below.predict(FRAME).array
    got_above = above.predict(FRAME).array
    # A window [1,2,3]: 2 below, 1 above; [2,3,4]: 1 below, 2 above.
    np.testing.assert_allclose(got_below[2:, 0], [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(got_above[2:, 0], [1.0 / 3.0, 2.0 / 3.0])
    # Values equal to the threshold count on neither side.
    eq = zoo.rolling_below_threshold_rate("close", window=1, threshold=5.0)
    np.testing.assert_allclose(eq.predict(wide([[5.0, 4.0]])).array, [[0.0, 1.0]])
    # NaN in the window poisons the cell.
    assert np.isnan(got_below[3, 1])


def test_rolling_lower_partial_moment_golden() -> None:
    frame = wide([[0.02, -0.01], [-0.03, -0.02], [0.01, 0.03]])
    pred = zoo.rolling_lower_partial_moment(
        "close", window=3, threshold=0.0, order=1
    ).predict(frame)
    np.testing.assert_allclose(pred.array[2], [0.01, 0.01])
    squared = zoo.rolling_lower_partial_moment(
        "close", window=3, threshold=0.0, order=2
    ).predict(frame)
    np.testing.assert_allclose(
        squared.array[2], [0.0009 / 3.0, (0.0001 + 0.0004) / 3.0]
    )


def test_rolling_range_position_golden() -> None:
    pred = zoo.rolling_range_position("close", window=3).predict(FRAME)
    # A: 3 in [1,3] -> 1.0 then 4 in [2,4] -> 1.0; B: 2 in [2,4] -> 0.0.
    np.testing.assert_allclose(pred.array[2], [1.0, 0.0])
    assert np.isnan(pred.array[3, 1])


def test_rolling_range_position_flat_window_is_nan() -> None:
    pred = zoo.rolling_range_position("close", window=2).predict(
        wide([[5.0, 1.0], [5.0, 2.0]])
    )
    np.testing.assert_allclose(pred.array[1], [NAN, 1.0])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_quantile("close", 3, q=1.5, method="linear"),
            r"q must be in \[0, 1\]",
            id="quantile-q-out-of-range",
        ),
        pytest.param(
            lambda: zoo.rolling_quantile("close", 3, q=0.5, method="nearest"),
            "method must be one of",
            id="quantile-bad-method",
        ),
        pytest.param(
            lambda: zoo.rolling_rank("close", 0),
            "window must be >= 1",
            id="rank-window-0",
        ),
        pytest.param(
            lambda: zoo.rolling_below_threshold_rate("close", 3, float("nan")),
            "threshold must be finite",
            id="below-nan-threshold",
        ),
        pytest.param(
            lambda: zoo.rolling_lower_partial_moment("close", 3, order=0),
            "order must be >= 1",
            id="lpm-order-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(FRAME)
