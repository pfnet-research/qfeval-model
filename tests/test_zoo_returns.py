"""Goldens and validation for simple_return / log_return."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

PRICES = wide(
    [
        [1.0, 10.0],
        [2.0, NAN],
        [4.0, 30.0],
        [8.0, 40.0],
    ]
)


def test_simple_return_golden() -> None:
    pred = zoo.simple_return().predict(PRICES)
    expected = [
        [NAN, NAN],
        [1.0, NAN],
        [1.0, NAN],
        [1.0, 40.0 / 30.0 - 1.0],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_simple_return_multi_period_golden() -> None:
    pred = zoo.simple_return(periods=2).predict(PRICES)
    expected = [
        [NAN, NAN],
        [NAN, NAN],
        [3.0, 2.0],
        [3.0, NAN],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_simple_return_zero_denominator_is_nan() -> None:
    frame = wide([[0.0, -1.0], [3.0, 2.0]])
    pred = zoo.simple_return().predict(frame)
    expected = [[NAN, NAN], [NAN, -3.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_returns_map_finite_input_overflow_to_nan() -> None:
    frame = wide([[1e-308], [1e308]], symbols=("A",))
    simple = zoo.simple_return("close").predict(frame).array
    logged = zoo.log_return("close").predict(frame).array
    assert np.isnan(simple[1, 0])
    assert np.isnan(logged[1, 0])


def test_log_return_golden() -> None:
    pred = zoo.log_return().predict(PRICES)
    expected = [
        [NAN, NAN],
        [np.log(2.0), NAN],
        [np.log(2.0), NAN],
        [np.log(2.0), np.log(40.0 / 30.0)],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_log_return_non_positive_ratio_is_nan() -> None:
    frame = wide([[0.0, -1.0], [3.0, 2.0], [6.0, -4.0]])
    pred = zoo.log_return().predict(frame)
    expected = [[NAN, NAN], [NAN, NAN], [np.log(2.0), NAN]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_returns_accept_another_source_column() -> None:
    pred = zoo.simple_return(source="px").predict({"px": PRICES})
    reference = zoo.simple_return().predict(PRICES)
    np.testing.assert_array_equal(pred.array, reference.array)


def test_log_return_sums_to_multi_period_log_return() -> None:
    frame = wide([[100.0, 50.0], [110.0, 45.0], [121.0, 54.0], [133.1, 27.0]])
    single = zoo.log_return().predict(frame).array
    double = zoo.log_return(periods=2).predict(frame).array
    np.testing.assert_allclose(double[2:], single[1:-1] + single[2:])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.simple_return(periods=0),
            "periods must be >= 1",
            id="simple_return-periods-0",
        ),
        pytest.param(
            lambda: zoo.log_return(periods=0),
            "periods must be >= 1",
            id="log_return-periods-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
