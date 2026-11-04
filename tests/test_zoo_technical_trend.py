"""Goldens and validation for technical trend indicators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
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


def test_bollinger_bands_equal_mean_plus_minus_deviations_std() -> None:
    upper = zoo.bollinger_upper_band("close", window=2).predict(PRICES).array
    lower = zoo.bollinger_lower_band("close", window=2).predict(PRICES).array
    mean = zoo.rolling_mean("close", window=2).predict(PRICES).array
    std = zoo.rolling_std("close", window=2).predict(PRICES).array
    np.testing.assert_allclose(upper, mean + 2.0 * std, equal_nan=True)
    np.testing.assert_allclose(lower, mean - 2.0 * std, equal_nan=True)


def test_bollinger_constant_series_collapses_to_middle() -> None:
    frame = wide([[5.0, 5.0]] * 3)
    upper = zoo.bollinger_upper_band("close", window=3).predict(frame).array
    lower = zoo.bollinger_lower_band("close", window=3).predict(frame).array
    np.testing.assert_allclose(upper[2], [5.0, 5.0])
    np.testing.assert_allclose(lower[2], [5.0, 5.0])
    # Zero width: bandwidth is 0/middle = 0, percent-b is 0/0 -> NaN.
    bandwidth = zoo.bollinger_bandwidth("close", window=3).predict(frame).array
    percent_b = zoo.bollinger_percent_b("close", window=3).predict(frame).array
    np.testing.assert_allclose(bandwidth[2], [0.0, 0.0])
    assert np.isnan(percent_b[2]).all()


def test_bollinger_bandwidth_golden() -> None:
    got = zoo.bollinger_bandwidth("close", window=2).predict(PRICES).array
    mean = zoo.rolling_mean("close", window=2).predict(PRICES).array
    std = zoo.rolling_std("close", window=2).predict(PRICES).array
    np.testing.assert_allclose(got, 4.0 * std / mean, equal_nan=True)


def test_bollinger_bandwidth_zero_middle_is_nan() -> None:
    frame = wide([[-1.0], [1.0]], symbols=("A",))
    got = zoo.bollinger_bandwidth("close", window=2).predict(frame).array
    assert np.isnan(got[1, 0])


def test_bollinger_percent_b_is_half_at_middle_band() -> None:
    got = zoo.bollinger_percent_b("close", window=2, deviations=2.0).predict(PRICES)
    # With window 2 the current value sits mean + std/sqrt(2)... hand value:
    # x=[1,2]: mean 1.5, std ~0.7071; lower = 1.5-2*std; %B=(2-lower)/(4*std).
    std = np.sqrt(0.5)
    expected = (2.0 - (1.5 - 2.0 * std)) / (4.0 * std)
    assert got.array[1, 0] == pytest.approx(expected)


def test_percent_b_not_clamped_outside_bands() -> None:
    frame = wide([[1.0], [1.1], [5.0]], symbols=("A",))
    got = zoo.bollinger_percent_b("close", window=3, deviations=1.0).predict(frame)
    assert got.array[2, 0] > 1.0  # a spike sits above the +1-sigma band


def test_envelope_golden() -> None:
    upper = (
        zoo.moving_average_envelope_upper("close", window=2, width=0.05)
        .predict(PRICES)
        .array
    )
    lower = (
        zoo.moving_average_envelope_lower("close", window=2, width=0.05)
        .predict(PRICES)
        .array
    )
    mean = zoo.rolling_mean("close", window=2).predict(PRICES).array
    np.testing.assert_allclose(upper, mean * 1.05, equal_nan=True)
    np.testing.assert_allclose(lower, mean * 0.95, equal_nan=True)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.bollinger_upper_band("close", window=1),
            "window must be > ddof",
            id="bollinger-window-1",
        ),
        pytest.param(
            lambda: zoo.bollinger_lower_band("close", window=20, deviations=0.0),
            "deviations must be a positive finite number",
            id="bollinger-deviations-0",
        ),
        pytest.param(
            lambda: zoo.moving_average_envelope_upper("close", width=1.0),
            r"width must be a fraction in \[0, 1\)",
            id="envelope-width-1",
        ),
        pytest.param(
            lambda: zoo.moving_average_envelope_lower("close", width=-0.1),
            r"width must be a fraction in \[0, 1\)",
            id="envelope-width-negative",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)


# ---------------------------------------------------------------------------
# Ichimoku (causal alignment)
# ---------------------------------------------------------------------------

ICHIMOKU_INPUTS = {
    "high": wide([[3.0], [5.0], [4.0], [6.0], [7.0]], symbols=("A",)),
    "low": wide([[1.0], [2.0], [3.0], [3.0], [5.0]], symbols=("A",)),
}


def test_ichimoku_conversion_and_base_are_window_midpoints() -> None:
    conversion = zoo.ichimoku_conversion_line(window=2).predict(ICHIMOKU_INPUTS).array
    # t1: (max(3,5) + min(1,2)) / 2; t2: (max(5,4) + min(2,3)) / 2.
    np.testing.assert_allclose(conversion[1:3, 0], [3.0, 3.5])
    assert np.isnan(conversion[0, 0])
    base = zoo.ichimoku_base_line(window=2).predict(ICHIMOKU_INPUTS).array
    np.testing.assert_array_equal(base, conversion)  # same formula, same window


def test_ichimoku_leading_spans_are_displaced_past_values() -> None:
    now = zoo.ichimoku_leading_span_a(
        conversion_window=2, base_window=3, displacement=0
    ).predict(ICHIMOKU_INPUTS)
    displaced = zoo.ichimoku_leading_span_a(
        conversion_window=2, base_window=3, displacement=2
    ).predict(ICHIMOKU_INPUTS)
    np.testing.assert_array_equal(displaced.array[2:], now.array[:-2])
    assert np.isnan(displaced.array[:2]).all()

    now_b = zoo.ichimoku_leading_span_b(window=2, displacement=0).predict(
        ICHIMOKU_INPUTS
    )
    displaced_b = zoo.ichimoku_leading_span_b(window=2, displacement=1).predict(
        ICHIMOKU_INPUTS
    )
    np.testing.assert_array_equal(displaced_b.array[1:], now_b.array[:-1])


def test_ichimoku_span_a_golden() -> None:
    pred = zoo.ichimoku_leading_span_a(
        conversion_window=2, base_window=2, displacement=1
    ).predict(ICHIMOKU_INPUTS)
    # With equal windows span A is the midpoint itself, displaced by 1.
    conversion = zoo.ichimoku_conversion_line(window=2).predict(ICHIMOKU_INPUTS)
    np.testing.assert_array_equal(pred.array[1:], conversion.array[:-1])


def test_ichimoku_future_rows_do_not_change_past_outputs() -> None:
    full = zoo.ichimoku_leading_span_a(
        conversion_window=2, base_window=2, displacement=1
    ).predict(ICHIMOKU_INPUTS)
    head = {name: frame.iloc[:3] for name, frame in ICHIMOKU_INPUTS.items()}
    partial = zoo.ichimoku_leading_span_a(
        conversion_window=2, base_window=2, displacement=1
    ).predict(head)
    np.testing.assert_array_equal(partial.array, full.array[:3])


# ---------------------------------------------------------------------------
# Rolling-sum DMI
# ---------------------------------------------------------------------------


def _trend_bars(direction: float) -> dict[str, pd.DataFrame]:
    """Steadily trending bars: +1 per step when direction=+1, -1 when -1."""
    base = np.arange(4.0) * direction
    return {
        "high": wide([[10.0 + b] for b in base], symbols=("A",)),
        "low": wide([[9.0 + b] for b in base], symbols=("A",)),
        "close": wide([[9.5 + b] for b in base], symbols=("A",)),
    }


def test_dmi_uptrend_golden() -> None:
    inputs = _trend_bars(+1.0)
    plus = zoo.rolling_positive_directional_indicator(window=2).predict(inputs).array
    minus = zoo.rolling_negative_directional_indicator(window=2).predict(inputs).array
    # Each step: +DM = 1, -DM = 0, TR = max(1, 1.5, 0.5) = 1.5.
    np.testing.assert_allclose(plus[2:, 0], 100.0 * 2.0 / 3.0)
    np.testing.assert_allclose(minus[2:, 0], 0.0)
    assert np.isnan(plus[:2, 0]).all()  # needs window + 1 bars
    dx = zoo.rolling_directional_index(window=2).predict(inputs).array
    np.testing.assert_allclose(dx[2:, 0], 100.0)
    adx = zoo.rolling_average_directional_index(window=2).predict(inputs).array
    np.testing.assert_allclose(adx[3:, 0], 100.0)
    assert np.isnan(adx[:3, 0]).all()  # one window for DX, one to average


def test_dmi_downtrend_mirrors_uptrend() -> None:
    up = _trend_bars(+1.0)
    down = _trend_bars(-1.0)
    plus_up = zoo.rolling_positive_directional_indicator(window=2).predict(up).array
    minus_down = (
        zoo.rolling_negative_directional_indicator(window=2).predict(down).array
    )
    np.testing.assert_allclose(plus_up, minus_down, equal_nan=True)


def test_dmi_tied_moves_count_as_zero() -> None:
    # The range widens symmetrically: up move == down move == 1 -> both DM 0.
    inputs = {
        "high": wide([[10.0], [11.0], [12.0]], symbols=("A",)),
        "low": wide([[9.0], [8.0], [7.0]], symbols=("A",)),
        "close": wide([[9.5], [9.5], [9.5]], symbols=("A",)),
    }
    plus = zoo.rolling_positive_directional_indicator(window=2).predict(inputs).array
    minus = zoo.rolling_negative_directional_indicator(window=2).predict(inputs).array
    np.testing.assert_allclose(plus[2, 0], 0.0)
    np.testing.assert_allclose(minus[2, 0], 0.0)
    # Both DI are zero, so DX has a zero denominator.
    dx = zoo.rolling_directional_index(window=2).predict(inputs).array
    assert np.isnan(dx[2, 0])


def test_dmi_zero_true_range_is_nan() -> None:
    inputs = {
        "high": wide([[10.0], [10.0]], symbols=("A",)),
        "low": wide([[10.0], [10.0]], symbols=("A",)),
        "close": wide([[10.0], [10.0]], symbols=("A",)),
    }
    plus = zoo.rolling_positive_directional_indicator(window=1).predict(inputs).array
    assert np.isnan(plus[1, 0])


# ---------------------------------------------------------------------------
# MACD (golden table fixes the adopted EWM seed / warm-up contract)
# ---------------------------------------------------------------------------


def test_macd_line_golden_table() -> None:
    # fast_span=1 (alpha 1: tracks price), slow_span=2 (alpha 2/3).
    # Hand-computed slow EWM over [2, 4, 8]: 2, 10/3, 58/9.
    frame = wide([[2.0], [4.0], [8.0]], symbols=("A",))
    line = zoo.macd_line("close", fast_span=1, slow_span=2).predict(frame).array
    # slow_span=2 finite prices are needed before the line shows.
    np.testing.assert_allclose(
        line[:, 0], [NAN, 4.0 - 10.0 / 3.0, 8.0 - 58.0 / 9.0], equal_nan=True
    )


def test_macd_signal_and_histogram_golden_table() -> None:
    frame = wide([[2.0], [4.0], [8.0]], symbols=("A",))
    signal = (
        zoo.macd_signal("close", fast_span=1, slow_span=2, signal_span=2)
        .predict(frame)
        .array
    )
    histogram = (
        zoo.macd_histogram("close", fast_span=1, slow_span=2, signal_span=2)
        .predict(frame)
        .array
    )
    # Signal EWM (alpha 2/3) over the line [NaN, 2/3, 14/9]:
    # seeds at 2/3, then (2/3)*(14/9) + (1/3)*(2/3) = 34/27; it shows only
    # after signal_span=2 finite line values.
    np.testing.assert_allclose(signal[:, 0], [NAN, NAN, 34.0 / 27.0], equal_nan=True)
    np.testing.assert_allclose(
        histogram[:, 0], [NAN, NAN, 14.0 / 9.0 - 34.0 / 27.0], equal_nan=True
    )


def test_macd_constant_series_line_and_histogram_are_zero() -> None:
    frame = wide([[5.0, 5.0]] * 8)
    line = zoo.macd_line("close", fast_span=2, slow_span=4).predict(frame).array
    histogram = (
        zoo.macd_histogram("close", fast_span=2, slow_span=4, signal_span=3)
        .predict(frame)
        .array
    )
    np.testing.assert_allclose(line[4:], 0.0)
    np.testing.assert_allclose(histogram[6:], 0.0)


def test_macd_missing_price_carries_the_line() -> None:
    frame = wide([[2.0], [4.0], [NAN], [8.0]], symbols=("A",))
    line = zoo.macd_line("close", fast_span=1, slow_span=2).predict(frame).array
    # t2's NaN price does not poison the line: both EWMs carry, and the
    # finite-price count stays at 2 (>= slow_span).
    assert line[1, 0] == pytest.approx(4.0 - 10.0 / 3.0)
    assert line[2, 0] == pytest.approx(4.0 - 10.0 / 3.0)  # carried, not NaN
    assert np.isfinite(line[3, 0])


def test_macd_warm_up_counts_finite_prices_only() -> None:
    frame = wide([[NAN], [2.0], [NAN], [4.0], [8.0]], symbols=("A",))
    line = zoo.macd_line("close", fast_span=1, slow_span=2).predict(frame).array
    # Finite prices arrive at t1 and t3, so the line first shows at t3.
    assert np.isnan(line[:3, 0]).all()
    assert np.isfinite(line[3, 0])


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.macd_line("close", fast_span=3, slow_span=3),
            "1 <= fast_span < slow_span",
            id="macd-equal-spans",
        ),
        pytest.param(
            lambda: zoo.macd_line("close", fast_span=0, slow_span=2),
            "1 <= fast_span < slow_span",
            id="macd-fast-0",
        ),
        pytest.param(
            lambda: zoo.macd_signal("close", signal_span=0),
            "signal_span must be >= 1",
            id="macd-signal-0",
        ),
    ],
)
def test_macd_invalid_spans_raise(factory: Callable[[], qm.Model], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(PRICES)
