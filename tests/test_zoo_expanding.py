"""Goldens and validation for the expanding module."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

from qfeval_model import zoo
from qfeval_model.core import Model

NAN = float("nan")

# A has a NaN gap; B starts with NaN (delayed first observation).
FRAME = wide(
    [
        [1.0, NAN],
        [2.0, 10.0],
        [NAN, 20.0],
        [4.0, NAN],
    ]
)


def test_expanding_sum_skips_nan() -> None:
    pred = zoo.expanding_sum("close").predict(FRAME)
    expected = [[1.0, NAN], [3.0, 10.0], [3.0, 30.0], [7.0, 30.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_expanding_mean_skips_nan() -> None:
    pred = zoo.expanding_mean("close").predict(FRAME)
    expected = [[1.0, NAN], [1.5, 10.0], [1.5, 15.0], [7.0 / 3.0, 15.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_expanding_mean_avoids_overflow_for_opposite_extremes() -> None:
    largest = np.finfo(np.float64).max
    frame = wide([[largest], [-largest], [1.0]], symbols=("A",))
    pred = zoo.expanding_mean("close").predict(frame)
    np.testing.assert_array_equal(pred.array[:, 0], [largest, 0.0, 1.0 / 3.0])


def test_expanding_var_and_std_match_pandas() -> None:
    var = zoo.expanding_var("close").predict(FRAME).array
    std = zoo.expanding_std("close").predict(FRAME).array
    var_ref = FRAME.expanding().var(ddof=1).to_numpy()
    std_ref = FRAME.expanding().std(ddof=1).to_numpy()
    np.testing.assert_allclose(var, var_ref, equal_nan=True)
    np.testing.assert_allclose(std, std_ref, equal_nan=True)


def test_expanding_var_ddof_zero_golden() -> None:
    pred = zoo.expanding_var("close", ddof=0).predict(FRAME)
    expected = [[0.0, NAN], [0.25, 0.0], [0.25, 25.0], [14.0 / 9.0, 25.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_expanding_zscore_scores_current_value() -> None:
    frame = wide([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    pred = zoo.expanding_zscore("close").predict(frame)
    # A at t2: mean([1,3,5])=3, std=2 -> (5-3)/2 = 1; B is constant -> NaN.
    assert pred.array[2, 0] == pytest.approx(1.0)
    assert np.isnan(pred.array[:, 1]).all()
    assert np.isnan(pred.array[0, 0])  # a single observation has no std


def test_expanding_zscore_nan_input_is_nan() -> None:
    pred = zoo.expanding_zscore("close").predict(FRAME)
    assert np.isnan(pred.array[2, 0])


def test_expanding_min_max_goldens() -> None:
    lo = zoo.expanding_min("close").predict(FRAME).array
    hi = zoo.expanding_max("close").predict(FRAME).array
    np.testing.assert_array_equal(
        lo, [[1.0, NAN], [1.0, 10.0], [1.0, 10.0], [1.0, 10.0]]
    )
    np.testing.assert_array_equal(
        hi, [[1.0, NAN], [2.0, 10.0], [2.0, 20.0], [4.0, 20.0]]
    )


@pytest.mark.parametrize(
    ("factory", "keys"),
    [
        (zoo.expanding_sum, ("sum", "count")),
        (zoo.expanding_mean, ("mean", "count")),
        (zoo.expanding_var, ("mean", "m2", "count")),
        (zoo.expanding_min, ("min",)),
        (zoo.expanding_max, ("max",)),
    ],
)
def test_expanding_state_arrays_are_reused_across_calls(
    factory: Callable[[str], Model], keys: tuple[str, ...]
) -> None:
    model = factory("close")
    model.predict(FRAME.iloc[:2])
    assert set(model.state) == {"shape", *keys}
    state_ids = {key: id(model.state[key]) for key in keys}
    model.predict(FRAME.iloc[2:])
    assert {key: id(model.state[key]) for key in keys} == state_ids


@pytest.mark.parametrize("name", ["expanding_var", "expanding_std", "expanding_zscore"])
def test_negative_ddof_raises(name: str) -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        getattr(zoo, name)("close", ddof=-1).fit(FRAME)
