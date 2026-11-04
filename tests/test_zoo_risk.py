"""Goldens, invariants, and validation for the risk module."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from conftest import wide

import qfeval_model as qm
from qfeval_model import zoo

NAN = float("nan")

WEALTH = wide(
    [
        [100.0, 50.0],
        [120.0, 40.0],
        [90.0, NAN],
        [110.0, 60.0],
        [130.0, 30.0],
    ]
)


def test_rolling_compounded_return_golden() -> None:
    frame = wide([[0.10, -0.50], [0.10, 1.00], [0.10, 0.00]])
    pred = zoo.rolling_compounded_return("close", window=3).predict(frame)
    np.testing.assert_allclose(pred.array[2], [1.1**3 - 1.0, 0.0])


def test_rolling_worst_loss_golden() -> None:
    frame = wide([[0.02, 0.01], [-0.05, 0.03], [0.01, 0.02]])
    pred = zoo.rolling_worst_loss("close", window=3).predict(frame)
    # A's worst step lost 5%; B never lost, so the "worst loss" is negative.
    np.testing.assert_allclose(pred.array[2], [0.05, -0.01])


def test_drawdown_golden() -> None:
    pred = zoo.drawdown("close").predict(WEALTH)
    expected = [
        [0.0, 0.0],
        [0.0, 0.2],
        [0.25, NAN],  # B's NaN stays NaN without resetting the peak
        [1.0 - 110.0 / 120.0, 0.0],  # B's 60 is a fresh peak
        [0.0, 0.5],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_drawdown_non_positive_peak_is_nan() -> None:
    pred = zoo.drawdown("close").predict(wide([[-5.0, 0.0], [-4.0, 1.0]]))
    np.testing.assert_array_equal(pred.array[:, 0], [NAN, NAN])
    np.testing.assert_array_equal(pred.array[:, 1], [NAN, 0.0])


def test_rolling_drawdown_uses_window_peak_only() -> None:
    frame = wide([[100.0], [80.0], [90.0], [95.0]], symbols=("A",))
    pred = zoo.rolling_drawdown("close", window=2).predict(frame)
    expected = [[NAN], [0.2], [0.0], [0.0]]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_max_drawdown_golden() -> None:
    frame = wide([[100.0], [120.0], [90.0], [105.0]], symbols=("A",))
    pred = zoo.rolling_max_drawdown("close", window=3).predict(frame)
    # [100,120,90]: worst is 120 -> 90 = 0.25; [120,90,105]: still 0.25.
    np.testing.assert_allclose(pred.array[2:, 0], [0.25, 0.25])


def test_rolling_max_drawdown_matches_naive_scan_with_nan_windows() -> None:
    rng = np.random.default_rng(20260819)
    values = np.exp(rng.normal(size=(80, 5)))
    values[rng.random(values.shape) < 0.08] = np.nan
    window = 11
    got = (
        zoo.rolling_max_drawdown("close", window)
        .predict(wide(values.tolist(), symbols=tuple("ABCDE")))
        .array
    )
    expected = np.full_like(values, np.nan)
    for end in range(window - 1, len(values)):
        block = values[end - window + 1 : end + 1]
        peaks = np.maximum.accumulate(block, axis=0)
        expected[end] = np.max(1.0 - block / peaks, axis=0)
    np.testing.assert_allclose(got, expected, equal_nan=True, atol=2e-15)


def test_rolling_max_drawdown_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="must be positive or NaN"):
        zoo.rolling_max_drawdown("close", 2).predict(
            wide([[1.0], [0.0]], symbols=("A",))
        )


def test_drawdown_duration_golden() -> None:
    pred = zoo.drawdown_duration("close").predict(WEALTH)
    expected = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 0.0],  # B's 60 sets a fresh peak
        [0.0, 1.0],
    ]
    np.testing.assert_allclose(pred.array, np.array(expected))


def test_rolling_downside_deviation_is_sqrt_of_order2_lpm() -> None:
    frame = wide([[0.02, -0.01], [-0.03, -0.02], [0.01, 0.03]])
    dd = zoo.rolling_downside_deviation("close", window=3).predict(frame).array
    lpm = (
        zoo.rolling_lower_partial_moment("close", window=3, order=2)
        .predict(frame)
        .array
    )
    np.testing.assert_allclose(dd, np.sqrt(lpm), equal_nan=True)


# ---------------------------------------------------------------------------
# Historical VaR / Expected Shortfall
# ---------------------------------------------------------------------------

# One symbol, window 4, returns +2%, -1%, +3%, -4% -> losses -2%, 1%, -3%, 4%.
RETURNS = wide(
    [[0.02], [-0.01], [0.03], [-0.04]],
    symbols=("A",),
)


def var_at(confidence: float, frame: pd.DataFrame | None = None) -> float:
    m = zoo.rolling_historical_value_at_risk("close", 4, confidence)
    return float(m.predict(RETURNS if frame is None else frame).array[3, 0])


def es_at(confidence: float, frame: pd.DataFrame | None = None) -> float:
    m = zoo.rolling_historical_expected_shortfall("close", 4, confidence)
    return float(m.predict(RETURNS if frame is None else frame).array[3, 0])


def test_var_is_the_inverted_cdf_loss_quantile() -> None:
    # Sorted losses: [-0.03, -0.02, 0.01, 0.04] (positive = loss).
    assert var_at(0.25) == pytest.approx(-0.03)  # ceil(1)/4 mass -> 1st loss
    assert var_at(0.5) == pytest.approx(-0.02)
    assert var_at(0.51) == pytest.approx(0.01)  # just past the 2/4 boundary
    assert var_at(0.75) == pytest.approx(0.01)
    assert var_at(0.9) == pytest.approx(0.04)


def test_var_profit_only_window_is_negative_not_clamped() -> None:
    profits = wide([[0.01], [0.02], [0.03], [0.04]], symbols=("A",))
    assert var_at(0.75, profits) < 0.0


def test_es_fractional_boundary_mass_golden() -> None:
    # c=0.75, n=4: tail mass 1 -> exactly the worst loss.
    assert es_at(0.75) == pytest.approx(0.04)
    # c=0.5, n=4: tail mass 2 -> mean of the two worst losses.
    assert es_at(0.5) == pytest.approx((0.04 + 0.01) / 2.0)
    # c=0.6, n=4: ceil(2.4)=3 -> full weight on 0.04, 0.6 weight on 0.01,
    # divided by the 1.6-sample tail mass.
    assert es_at(0.6) == pytest.approx((0.04 + 0.6 * 0.01) / 1.6)


def test_es_dominates_var_and_is_monotone_in_confidence() -> None:
    grid = [0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]
    for confidence in grid:
        assert es_at(confidence) >= var_at(confidence) - 1e-12
    for lo, hi in itertools.pairwise(grid):
        assert var_at(lo) <= var_at(hi) + 1e-12
        assert es_at(lo) <= es_at(hi) + 1e-12


def test_es_equals_var_on_constant_losses() -> None:
    constant = wide([[-0.02]] * 4, symbols=("A",))
    for confidence in (0.25, 0.5, 0.9):
        assert es_at(confidence, constant) == pytest.approx(0.02)
        assert var_at(confidence, constant) == pytest.approx(0.02)


def test_es_partition_matches_sorted_reference_with_fractional_tail() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(37, 6))
    window, confidence = 13, 0.83
    got = (
        zoo.rolling_historical_expected_shortfall("close", window, confidence)
        .predict(wide(values.tolist(), symbols=tuple("ABCDEF")))
        .array
    )
    expected = np.full_like(values, np.nan)
    tail_mass = window * (1.0 - confidence)
    full = window - math.ceil(confidence * window)
    fraction = math.ceil(confidence * window) - confidence * window
    for end in range(window - 1, len(values)):
        losses = np.sort(-values[end - window + 1 : end + 1], axis=0)[::-1]
        expected[end] = (
            losses[:full].sum(axis=0) + fraction * losses[full]
        ) / tail_mass
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("risk_at", [var_at, es_at])
def test_risk_translation_and_positive_homogeneity(
    risk_at: Callable[..., float],
) -> None:
    base = risk_at(0.6)
    shifted = RETURNS + 0.01
    assert risk_at(0.6, shifted) == pytest.approx(base - 0.01)
    scaled = RETURNS * 3.0
    assert risk_at(0.6, scaled) == pytest.approx(3.0 * base)


def test_var_and_es_mixed_loss_only_windows() -> None:
    losses = wide([[-0.01], [-0.02], [-0.03], [-0.04]], symbols=("A",))
    assert var_at(0.5, losses) == pytest.approx(0.02)
    assert es_at(0.9, losses) == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        pytest.param(
            lambda: zoo.rolling_historical_value_at_risk("close", 4, 0.0),
            "confidence must be in",
            id="var-confidence-0",
        ),
        pytest.param(
            lambda: zoo.rolling_historical_value_at_risk("close", 4, 95.0),
            "confidence must be in",
            id="var-confidence-percent",
        ),
        pytest.param(
            lambda: zoo.rolling_historical_expected_shortfall("close", 4, 1.0),
            "confidence must be in",
            id="es-confidence-1",
        ),
        pytest.param(
            lambda: zoo.rolling_downside_deviation("close", 3, float("inf")),
            "threshold must be finite",
            id="downside-inf-threshold",
        ),
        pytest.param(
            lambda: zoo.rolling_compounded_return("close", 0),
            "window must be >= 1",
            id="compounded-window-0",
        ),
    ],
)
def test_invalid_parameters_raise_value_error_at_fit(
    factory: Callable[[], qm.Model], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory().fit(RETURNS)
