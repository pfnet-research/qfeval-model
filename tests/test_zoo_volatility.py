"""Goldens and validation for annualized_historical_volatility."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

RETURNS = wide(
    [
        [0.01, 0.05],
        [-0.02, 0.00],
        [0.03, -0.05],
        [0.00, 0.10],
    ]
)


def test_volatility_is_rolling_std_times_sqrt_periods() -> None:
    vol = (
        zoo.annualized_historical_volatility("close", window=3, periods_per_year=252.0)
        .predict(RETURNS)
        .array
    )
    std = zoo.rolling_std("close", window=3).predict(RETURNS).array
    np.testing.assert_allclose(vol, std * np.sqrt(252.0), equal_nan=True)


def test_volatility_annualization_scales_with_sqrt_time() -> None:
    daily = zoo.annualized_historical_volatility(
        "close", window=3, periods_per_year=252.0
    ).predict(RETURNS)
    weekly = zoo.annualized_historical_volatility(
        "close", window=3, periods_per_year=52.0
    ).predict(RETURNS)
    ratio = daily.array[2:] / weekly.array[2:]
    np.testing.assert_allclose(ratio, np.sqrt(252.0 / 52.0))


def test_volatility_composes_with_log_return() -> None:
    prices = wide([[100.0], [110.0], [99.0], [104.0]], symbols=("A",))
    vol = (
        zoo.annualized_historical_volatility(
            zoo.log_return(), window=3, periods_per_year=252.0
        )
        .predict(prices)
        .array
    )
    returns = np.diff(np.log([100.0, 110.0, 99.0, 104.0]))
    expected = np.std(returns, ddof=1) * np.sqrt(252.0)
    assert vol[3, 0] == pytest.approx(expected)
    assert np.isnan(vol[:3, 0]).all()  # 1 return NaN + 2 warm-up steps


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.annualized_historical_volatility(
                "close", window=3, periods_per_year=0.0
            ),
            "periods_per_year must be a positive finite number",
            id="ppy-0",
        ),
        pytest.param(
            lambda: zoo.annualized_historical_volatility(
                "close", window=3, periods_per_year=float("inf")
            ),
            "periods_per_year must be a positive finite number",
            id="ppy-inf",
        ),
        pytest.param(
            lambda: zoo.annualized_historical_volatility(
                "close", window=1, periods_per_year=252.0
            ),
            "window must be > ddof",
            id="window-1",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(RETURNS)
