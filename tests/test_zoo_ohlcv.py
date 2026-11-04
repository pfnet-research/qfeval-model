"""Goldens and validation for the OHLCV module."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo
from qfeval_model.core import Step
from qfeval_model.zoo._ohlcv import read_ohlcv

NAN = float("nan")

OHLCV: dict[str, pd.DataFrame] = {
    "open": wide([[10.0, 101.0], [11.0, NAN], [12.0, 99.0]]),
    "high": wide([[12.0, 103.0], [13.0, 104.0], [12.5, 102.0]]),
    "low": wide([[9.0, 99.0], [10.5, 100.0], [10.0, 97.0]]),
    "close": wide([[11.0, 102.0], [12.5, 103.0], [10.5, 98.0]]),
    "volume": wide([[100.0, 0.0], [50.0, 200.0], [0.0, 300.0]]),
}


def test_read_ohlcv_dense_path_preserves_source_views() -> None:
    high = np.array([2.0, 3.0])
    low = np.array([1.0, 2.0])
    step = cast(Step, SimpleNamespace(inputs={"high": high, "low": low}))
    values = read_ohlcv(step, {}, {"high": "high", "low": "low"})
    assert np.shares_memory(values["high"], high)
    assert np.shares_memory(values["low"], low)


def test_true_range_golden() -> None:
    pred = zoo.true_range().predict(OHLCV)
    # t0 has no previous close; true_range never reads "open", so B's NaN
    # open at t1 does not blank its high/low/close tuple.
    expected = [
        [NAN, NAN],
        [2.5, 4.0],  # A: max(13-10.5, |13-11|, |10.5-11|)
        [2.5, 6.0],  # A: max(2.5, 0, 2.5); B: max(5, |102-103|, |97-103|)
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_true_range_joint_nan_blanks_the_stored_close() -> None:
    inputs = {
        "high": wide([[12.0], [NAN], [12.0]], symbols=("A",)),
        "low": wide([[9.0], [10.0], [10.0]], symbols=("A",)),
        "close": wide([[11.0], [11.5], [11.0]], symbols=("A",)),
    }
    pred = zoo.true_range().predict(inputs)
    # t1's NaN high blanks the whole tuple, so t2 has no previous close.
    np.testing.assert_array_equal(pred.array, [[NAN], [NAN], [NAN]])


def test_typical_price_and_weighted_close_goldens() -> None:
    tp = zoo.typical_price().predict(OHLCV).array
    wc = zoo.weighted_close().predict(OHLCV).array
    assert tp[0, 0] == pytest.approx((12.0 + 9.0 + 11.0) / 3.0)
    assert wc[0, 0] == pytest.approx((12.0 + 9.0 + 2 * 11.0) / 4.0)
    # Neither model reads "open", so B's t1 stays finite.
    assert tp[1, 1] == pytest.approx((104.0 + 100.0 + 103.0) / 3.0)
    assert wc[1, 1] == pytest.approx((104.0 + 100.0 + 2 * 103.0) / 4.0)


def test_rolling_volume_weighted_mean_golden() -> None:
    pred = zoo.rolling_volume_weighted_mean(window=2).predict(OHLCV)
    # A t1: (11*100 + 12.5*50) / 150.
    assert pred.array[1, 0] == pytest.approx((11.0 * 100.0 + 12.5 * 50.0) / 150.0)
    # A t2: (12.5*50 + 10.5*0) / 50.
    assert pred.array[2, 0] == pytest.approx(12.5 * 50.0 / 50.0)


def test_rolling_volume_weighted_mean_zero_volume_window_is_nan() -> None:
    inputs = {
        "close": wide([[10.0], [11.0]], symbols=("A",)),
        "volume": wide([[0.0], [0.0]], symbols=("A",)),
    }
    pred = zoo.rolling_volume_weighted_mean(window=2).predict(inputs)
    assert np.isnan(pred.array[1, 0])


def test_amihud_illiquidity_golden() -> None:
    inputs = {
        "close": wide([[100.0], [110.0], [99.0]], symbols=("A",)),
        "volume": wide([[10.0], [20.0], [10.0]], symbols=("A",)),
    }
    pred = zoo.amihud_illiquidity(zoo.simple_return(), window=2).predict(inputs)
    r1, r2 = 0.10, 99.0 / 110.0 - 1.0
    expected = (abs(r1) / (110.0 * 20.0) + abs(r2) / (99.0 * 10.0)) / 2.0
    assert pred.array[2, 0] == pytest.approx(expected)


def test_amihud_illiquidity_zero_traded_value_is_nan() -> None:
    inputs = {
        "close": wide([[100.0], [110.0]], symbols=("A",)),
        "volume": wide([[10.0], [0.0]], symbols=("A",)),
    }
    pred = zoo.amihud_illiquidity(zoo.simple_return(), window=1).predict(inputs)
    assert np.isnan(pred.array[1, 0])


def test_candle_geometry_goldens() -> None:
    signed = zoo.candle_signed_body().predict(OHLCV).array
    real = zoo.candle_real_body().predict(OHLCV).array
    upper = zoo.candle_upper_shadow().predict(OHLCV).array
    lower = zoo.candle_lower_shadow().predict(OHLCV).array
    candle_span = zoo.candle_range().predict(OHLCV).array
    assert signed[0, 0] == pytest.approx(1.0)  # 11 - 10
    assert signed[2, 0] == pytest.approx(-1.5)  # 10.5 - 12
    assert real[2, 0] == pytest.approx(1.5)
    assert upper[0, 0] == pytest.approx(1.0)  # 12 - max(10, 11)
    assert lower[0, 0] == pytest.approx(1.0)  # min(10, 11) - 9
    assert candle_span[0, 0] == pytest.approx(3.0)
    assert np.isnan(signed[1, 1])  # NaN open blanks the tuple


def test_inverted_bar_raises() -> None:
    inputs = {
        "high": wide([[1.0]], symbols=("A",)),
        "low": wide([[2.0]], symbols=("A",)),
    }
    with pytest.raises(ValueError, match="low > high"):
        zoo.candle_range().predict(inputs)


def test_negative_candle_geometry_raises() -> None:
    inputs = {
        "open": wide([[10.0]], symbols=("A",)),
        "high": wide([[9.5]], symbols=("A",)),
        "low": wide([[9.0]], symbols=("A",)),
        "close": wide([[9.2]], symbols=("A",)),
    }
    with pytest.raises(ValueError, match=r"high < max\(open, close\)"):
        zoo.candle_upper_shadow().predict(inputs)
    inputs = {
        "open": wide([[10.0]], symbols=("A",)),
        "low": wide([[10.5]], symbols=("A",)),
        "high": wide([[11.0]], symbols=("A",)),
        "close": wide([[10.8]], symbols=("A",)),
    }
    with pytest.raises(ValueError, match=r"low > min\(open, close\)"):
        zoo.candle_lower_shadow().predict(inputs)


def test_negative_volume_raises() -> None:
    inputs = {
        "close": wide([[10.0]], symbols=("A",)),
        "volume": wide([[-1.0]], symbols=("A",)),
    }
    with pytest.raises(ValueError, match="volume < 0"):
        zoo.rolling_volume_weighted_mean(window=1).predict(inputs)


def test_model_sources_equal_column_sources() -> None:
    via_column = zoo.typical_price().predict(OHLCV)
    via_model = zoo.typical_price(
        high=zoo.ffill("high"), low=zoo.ffill("low"), close=zoo.ffill("close")
    ).predict(OHLCV)
    np.testing.assert_array_equal(via_column.array, via_model.array)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_volume_weighted_mean(window=0),
            "window must be >= 1",
            id="vwm-window-0",
        ),
        pytest.param(
            lambda: zoo.amihud_illiquidity(zoo.simple_return(), window=0),
            "window must be >= 1",
            id="amihud-window-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(OHLCV)
