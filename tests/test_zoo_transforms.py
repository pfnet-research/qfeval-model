"""Goldens and validation for the transforms module (elementwise + algebra)."""

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


def test_diff_golden() -> None:
    pred = zoo.diff("close").predict(PRICES)
    expected = [[NAN, NAN], [1.0, NAN], [2.0, NAN], [4.0, 10.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_diff_multi_period_golden() -> None:
    pred = zoo.diff("close", periods=2).predict(PRICES)
    expected = [[NAN, NAN], [NAN, NAN], [3.0, 20.0], [6.0, NAN]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_fill_constant_replaces_only_nan() -> None:
    pred = zoo.fill_constant("close", value=-1.0).predict(PRICES)
    expected = [[1.0, 10.0], [2.0, -1.0], [4.0, 30.0], [8.0, 40.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_finite_indicator_golden() -> None:
    pred = zoo.finite_indicator("close").predict(PRICES)
    expected = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_coalesce_takes_first_finite_source() -> None:
    inputs = {
        "a": wide([[NAN, 1.0], [NAN, NAN]]),
        "b": wide([[2.0, 9.0], [NAN, 3.0]]),
    }
    pred = zoo.coalesce(["a", "b"]).predict(inputs)
    expected = [[2.0, 1.0], [NAN, 3.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_clip_golden_and_nan_kept() -> None:
    pred = zoo.clip("close", lower=2.0, upper=30.0).predict(PRICES)
    expected = [[2.0, 10.0], [2.0, NAN], [4.0, 30.0], [8.0, 30.0]]
    np.testing.assert_array_equal(pred.array, np.array(expected))


def test_binary_algebra_with_model_operand() -> None:
    lagged = zoo.lag("close")
    got = zoo.add("close", lagged).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[NAN, NAN], [3.0, NAN], [6.0, NAN], [12.0, 70.0]]
    )
    got = zoo.subtract("close", zoo.lag("close")).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[NAN, NAN], [1.0, NAN], [2.0, NAN], [4.0, 10.0]]
    )
    got = zoo.multiply("close", zoo.lag("close")).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[NAN, NAN], [2.0, NAN], [8.0, NAN], [32.0, 1200.0]]
    )
    got = zoo.divide("close", zoo.lag("close")).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[NAN, NAN], [2.0, NAN], [2.0, NAN], [2.0, 4.0 / 3.0]]
    )


def test_binary_algebra_with_scalar_operand() -> None:
    got = zoo.multiply("close", 2.0).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[2.0, 20.0], [4.0, NAN], [8.0, 60.0], [16.0, 80.0]]
    )
    got = zoo.minimum("close", 4.0).predict(PRICES).array
    np.testing.assert_array_equal(got, [[1.0, 4.0], [2.0, NAN], [4.0, 4.0], [4.0, 4.0]])
    got = zoo.maximum("close", 4.0).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[4.0, 10.0], [4.0, NAN], [4.0, 30.0], [8.0, 40.0]]
    )
    got = zoo.power("close", 2.0).predict(PRICES).array
    np.testing.assert_array_equal(
        got, [[1.0, 100.0], [4.0, NAN], [16.0, 900.0], [64.0, 1600.0]]
    )


def test_divide_by_zero_is_nan_not_inf() -> None:
    inputs = {
        "num": wide([[1.0, 0.0]]),
        "den": wide([[0.0, 0.0]]),
    }
    pred = zoo.divide("num", "den").predict(inputs)
    np.testing.assert_array_equal(pred.array, [[NAN, NAN]])


def test_overflow_is_nan_not_inf() -> None:
    huge = wide([[1e308, -1e308]])
    np.testing.assert_array_equal(
        zoo.add("close", zoo.ffill("close")).predict(huge).array, [[NAN, NAN]]
    )
    np.testing.assert_array_equal(
        zoo.exp("close").predict(wide([[710.0]], symbols=("A",))).array, [[NAN]]
    )
    diff = zoo.diff("close").predict(wide([[-1e308], [1e308]], symbols=("A",)))
    assert np.isnan(diff.array[1, 0])


def test_power_domain_error_is_nan() -> None:
    pred = zoo.power("close", 0.5).predict(wide([[-4.0, 9.0]]))
    np.testing.assert_array_equal(pred.array, [[NAN, 3.0]])


def test_unary_algebra_goldens() -> None:
    frame = wide([[-2.0, 0.0], [3.0, NAN]])
    np.testing.assert_array_equal(
        zoo.negate("close").predict(frame).array, [[2.0, -0.0], [-3.0, NAN]]
    )
    np.testing.assert_array_equal(
        zoo.absolute("close").predict(frame).array, [[2.0, 0.0], [3.0, NAN]]
    )
    np.testing.assert_array_equal(
        zoo.sign("close").predict(frame).array, [[-1.0, 0.0], [1.0, NAN]]
    )


def test_log_non_positive_is_nan() -> None:
    pred = zoo.log("close").predict(wide([[-1.0, 0.0], [np.e, 1.0]]))
    np.testing.assert_allclose(pred.array, [[NAN, NAN], [1.0, 0.0]])


def test_exp_golden() -> None:
    pred = zoo.exp("close").predict(wide([[0.0, 1.0]]))
    np.testing.assert_allclose(pred.array, [[1.0, np.e]])


def test_where_selects_by_condition_and_propagates_nan_condition() -> None:
    inputs = {
        "cond": wide([[1.0, 0.0], [NAN, -2.0]]),
        "x": wide([[10.0, 20.0], [30.0, 40.0]]),
        "y": wide([[-1.0, -2.0], [-3.0, -4.0]]),
    }
    pred = zoo.where("cond", "x", "y").predict(inputs)
    np.testing.assert_array_equal(pred.array, [[10.0, -2.0], [NAN, 40.0]])


def test_where_accepts_scalar_branches() -> None:
    pred = zoo.where(zoo.diff("close"), 1.0, 0.0).predict(PRICES)
    np.testing.assert_array_equal(
        pred.array, [[NAN, NAN], [1.0, NAN], [1.0, NAN], [1.0, 1.0]]
    )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.diff("close", periods=0),
            "periods must be >= 1",
            id="diff-periods-0",
        ),
        pytest.param(
            lambda: zoo.fill_constant("close", value=float("inf")),
            "value must be finite",
            id="fill_constant-inf",
        ),
        pytest.param(
            lambda: zoo.coalesce([]),
            "at least one source",
            id="coalesce-empty",
        ),
        pytest.param(
            lambda: zoo.clip("close", lower=float("nan")),
            "lower must be finite",
            id="clip-nan-bound",
        ),
        pytest.param(
            lambda: zoo.clip("close", lower=2.0, upper=1.0),
            "lower must be <= upper",
            id="clip-crossed-bounds",
        ),
        pytest.param(
            lambda: zoo.add("close", float("inf")),
            "scalar operands must be finite",
            id="add-inf-scalar",
        ),
        pytest.param(
            lambda: zoo.where("close", float("nan"), 0.0),
            "scalar operands must be finite",
            id="where-nan-scalar",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
