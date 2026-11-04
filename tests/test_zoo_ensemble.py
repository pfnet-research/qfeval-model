"""Goldens and validation for the ensemble module."""

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


def test_mean_ensemble_golden_and_nan_propagation() -> None:
    pred = zoo.mean_ensemble([zoo.lag("close"), zoo.rolling_mean("close", 2)]).predict(
        PRICES
    )
    expected = [[NAN, NAN], [1.25, NAN], [2.5, NAN], [5.0, 32.5]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_mean_ensemble_single_child_is_identity() -> None:
    pred = zoo.mean_ensemble([zoo.lag("close")]).predict(PRICES)
    reference = zoo.lag("close").predict(PRICES)
    np.testing.assert_array_equal(pred.array, reference.array)


def test_nanmean_ensemble_uses_valid_children_only() -> None:
    pred = zoo.nanmean_ensemble(
        [zoo.lag("close"), zoo.rolling_mean("close", 2)]
    ).predict(PRICES)
    # Where lag is NaN (t0) both children are NaN -> NaN; where only one is
    # finite the mean falls back to it.
    expected = [[NAN, NAN], [1.25, 10.0], [2.5, NAN], [5.0, 32.5]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_nanmean_ensemble_min_children_masks_thin_cells() -> None:
    pred = zoo.nanmean_ensemble(
        [zoo.lag("close"), zoo.rolling_mean("close", 2)], min_children=2
    ).predict(PRICES)
    expected = [[NAN, NAN], [1.25, NAN], [2.5, NAN], [5.0, 32.5]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_linear_combination_golden() -> None:
    pred = zoo.linear_combination(
        [zoo.lag("close"), zoo.rolling_mean("close", 2)], weights=[2.0, -1.0]
    ).predict(PRICES)
    # 2 * lag - rolling_mean; no implicit normalization.
    expected = [[NAN, NAN], [0.5, NAN], [1.0, NAN], [2.0, 25.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.mean_ensemble([]),
            "at least one child",
            id="mean_ensemble-empty",
        ),
        pytest.param(
            lambda: zoo.nanmean_ensemble([], min_children=1),
            "at least one child",
            id="nanmean_ensemble-empty",
        ),
        pytest.param(
            lambda: zoo.nanmean_ensemble([zoo.lag("close")], min_children=0),
            "min_children must be >= 1",
            id="nanmean_ensemble-min-0",
        ),
        pytest.param(
            lambda: zoo.linear_combination([zoo.lag("close")], weights=[1.0, 2.0]),
            "same length",
            id="linear_combination-length",
        ),
        pytest.param(
            lambda: zoo.linear_combination([zoo.lag("close")], weights=[float("inf")]),
            "weights must be finite",
            id="linear_combination-inf",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
