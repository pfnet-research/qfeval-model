"""Evaluation metrics: forward_returns / IC / hit_rate / long_short_pnl.

All numbers are hand-computed goldens on small panels so they can be
verified by inspection, independent of any reference implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, long_of, wide

import qfeval_model as qm
from qfeval_model import metrics

NAN = float("nan")


@qm.batch_model
def emit(ctx: qm.Context, values: Sequence[Any]) -> None:
    """Output the given literal (T, S, ...) values, ignoring the inputs."""
    for step in ctx:
        step.output = np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------

# T=4, S=2: close doubles every step for A; B moves 10 -> 20 -> 10 -> 40.
CLOSE = wide([[1.0, 10.0], [2.0, 20.0], [4.0, 10.0], [8.0, 40.0]])


def expected_long(cells: list[list[float]]) -> pd.DataFrame:
    """(T, S) literal cells -> the long frame forward_returns should emit."""
    return pd.DataFrame(
        {
            "timestamp": np.repeat(days(4).as_unit("ns").to_numpy(), 2),
            "symbol": ["A", "B"] * 4,
            "forward_return": np.array(cells, dtype=np.float64).reshape(-1),
        }
    )


def test_forward_returns_golden_periods_1() -> None:
    result = metrics.forward_returns(CLOSE, periods=1)
    expected = expected_long([[1.0, 1.0], [1.0, -0.5], [1.0, 3.0], [NAN, NAN]])
    pd.testing.assert_frame_equal(result, expected)


def test_forward_returns_golden_periods_2() -> None:
    result = metrics.forward_returns(CLOSE, periods=2)
    expected = expected_long([[3.0, 0.0], [3.0, 1.0], [NAN, NAN], [NAN, NAN]])
    pd.testing.assert_frame_equal(result, expected)


def test_forward_returns_accepts_all_input_shapes() -> None:
    from_flat = metrics.forward_returns(CLOSE)
    from_dict = metrics.forward_returns({"close": CLOSE})
    from_long = metrics.forward_returns(long_of(CLOSE))
    pd.testing.assert_frame_equal(from_dict, from_flat)
    pd.testing.assert_frame_equal(from_long, from_flat)


def test_forward_periods_ignore_rows_with_no_price_in_wide_and_long() -> None:
    # A second feature can make the canonical long form denser than the price
    # frame. Rows with no price are not observations and must not consume a
    # forward period merely because the transport representation changed.
    close = wide([[100.0, 100.0], [110.0, 90.0], [121.0, 81.0]]).set_axis(days(5)[::2])
    volume = wide([[1.0, 2.0]] * 5)
    inputs = {"close": close, "volume": volume}

    from_wide = metrics.forward_returns(inputs)
    from_long = metrics.forward_returns(qm.as_frame(inputs))

    pd.testing.assert_frame_equal(from_long, from_wide)
    np.testing.assert_allclose(
        from_wide["forward_return"].to_numpy()[:4], [0.1, -0.1, 0.1, -0.1]
    )


@pytest.mark.parametrize("periods", [0, -1])
@pytest.mark.parametrize(
    "call",
    [
        lambda periods: metrics.forward_returns(CLOSE, periods=periods),
        lambda periods: metrics.information_coefficient(
            emit([[1.0, 2.0]] * 4).predict(CLOSE), CLOSE, periods=periods
        ),
        lambda periods: metrics.hit_rate(
            emit([[1.0, 2.0]] * 4).predict(CLOSE), CLOSE, periods=periods
        ),
        lambda periods: metrics.long_short_pnl(
            emit([[1.0, 2.0]] * 4).predict(CLOSE), CLOSE, periods=periods
        ),
    ],
    ids=["forward_returns", "information_coefficient", "hit_rate", "long_short_pnl"],
)
def test_metrics_reject_non_forward_periods(
    call: Callable[[int], object], periods: int
) -> None:
    # periods=0 correlates a price with itself and a negative value reads the
    # past while still being called "forward"; the shared forward_wide entry
    # rejects both for every metric.
    with pytest.raises(ValueError, match="periods must be >= 1"):
        call(periods)


def test_trailing_axis_price_input_is_rejected() -> None:
    # A price table carrying anonymous trailing axes (e.g. multiple
    # horizons) has no single financial meaning; every metric shares the
    # wide_price gate, so it must refuse instead of stacking/reindexing
    # into silent NaN or a confusing TypeError.
    multi = pd.DataFrame(
        [[1.0, 2.0, 10.0, 20.0]] * 4,
        index=days(4),
        columns=pd.MultiIndex.from_product([["A", "B"], [0, 1]]),
    )
    with pytest.raises(ValueError, match="trailing axes"):
        metrics.forward_returns({"close": multi})
    result = emit([[1.0, 10.0]] * 4).predict(CLOSE)
    with pytest.raises(ValueError, match="trailing axes"):
        metrics.information_coefficient(result, {"close": multi})


def test_non_numeric_price_input_is_rejected() -> None:
    labels = pd.DataFrame(
        [["x", "y"]] * 4, index=days(4), columns=["A", "B"], dtype=object
    )
    with pytest.raises(ValueError, match="real numbers"):
        metrics.forward_returns({"close": labels})


@pytest.mark.parametrize(
    ("bad", "match"),
    [(0.0, "zero"), (np.inf, "infinity"), (-np.inf, "infinity")],
)
def test_invalid_price_values_are_rejected(bad: float, match: str) -> None:
    prices = CLOSE.copy()
    prices.iloc[1, 0] = bad
    with pytest.raises(ValueError, match=match):
        metrics.forward_returns(prices)


def test_mixed_nullable_numeric_prices_are_normalized() -> None:
    # A nullable column mixed with an ordinary NumPy column becomes an object
    # array under DataFrame.to_numpy(); normalize it once at the shared price
    # boundary so np.isnan-based metrics receive float64 + NaN, not pd.NA.
    prices = pd.DataFrame(
        {
            "A": pd.array([1.0, 2.0, None, 8.0], dtype="Float64"),
            "B": np.array([10.0, 20.0, 40.0, 80.0]),
        },
        index=days(4),
    )
    forward = metrics.forward_returns(prices)
    assert forward["forward_return"].dtype == np.float64
    result = emit([[1.0, 1.0]] * 4).predict(CLOSE)
    assert metrics.hit_rate(result, prices) == 1.0
    assert len(metrics.information_coefficient(result, prices)) == 4
    assert len(metrics.long_short_pnl(result, prices)) == 4


def test_forward_returns_rejects_missing_price_column() -> None:
    with pytest.raises(ValueError, match="close"):
        metrics.forward_returns({"px": CLOSE})


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------

# S=3 close whose forward returns are [1, 2, 3] at t0 (ascending),
# [-1/2, -2/3, -3/4] at t1 (descending), and [1, 3, 7] at t2 (ascending).
IC_CLOSE = wide(
    [[1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0], [2.0, 4.0, 8.0]],
    symbols=("A", "B", "C"),
)
IC3 = IC_CLOSE.iloc[:3]  # forward returns: t0 ascending, t1 descending, t2 NaN
ASCENDING = [[1.0, 2.0, 3.0]] * 3


def test_information_coefficient_spearman_golden() -> None:
    result = emit(ASCENDING).predict(IC3)
    ic = metrics.information_coefficient(result, IC3)
    expected = pd.Series(
        [1.0, -1.0, NAN],
        index=days(3).as_unit("ns").rename("timestamp"),
        name="information_coefficient",
        dtype=np.float64,
    )
    pd.testing.assert_series_equal(ic, expected, check_freq=False)


def test_information_coefficient_reversed_output_is_minus_one() -> None:
    result = emit([[3.0, 2.0, 1.0]] * 3).predict(IC3)
    ic = metrics.information_coefficient(result, IC3)
    assert ic.iloc[0] == -1.0
    assert ic.iloc[1] == 1.0


def test_information_coefficient_pearson_exact_value() -> None:
    result = emit([[1.0, 2.0, 4.0]] * 3).predict(IC3)
    ic = metrics.information_coefficient(result, IC3, method="pearson")
    # pred [1, 2, 4] vs forward [1, 2, 3]: r = 9 / sqrt(84).
    np.testing.assert_allclose(ic.iloc[0], 9.0 / np.sqrt(84.0))


def test_information_coefficient_rejects_unknown_method() -> None:
    result = emit(ASCENDING).predict(IC3)
    with pytest.raises(ValueError, match="kendall"):
        # The Literal annotation catches this statically; the ignore lets the
        # test exercise the runtime check behind it.
        metrics.information_coefficient(result, IC3, method="kendall")  # type: ignore[arg-type]


def test_information_coefficient_fewer_than_two_valid_pairs_is_nan() -> None:
    result = emit([[NAN, NAN, 1.0]] * 3).predict(IC3)
    ic = metrics.information_coefficient(result, IC3)
    assert np.isnan(ic.iloc[0])


def test_information_coefficient_constant_output_is_nan() -> None:
    result = emit([[2.0, 2.0, 2.0]] * 3).predict(IC3)
    ic = metrics.information_coefficient(result, IC3)
    assert np.isnan(ic.iloc[0])


def test_information_coefficient_scores_with_future_extending_inputs() -> None:
    # The inputs extend one step past the output: the last output row is
    # scored against that future price instead of being NaN.
    result = emit(ASCENDING).predict(IC_CLOSE.iloc[:3])
    ic = metrics.information_coefficient(result, IC_CLOSE)
    assert list(ic.index) == list(days(3))
    np.testing.assert_allclose(ic.to_numpy(), [1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

# forward returns: t0 [1.0, -0.5], t1 [1.0, 3.0], t2 NaN.
HIT_CLOSE = wide([[1.0, 2.0], [2.0, 1.0], [4.0, 4.0]])


def test_hit_rate_golden_excluding_zero_and_nan() -> None:
    # (0,A) + vs + hit; (0,B) - vs - hit; (1,A) zero output excluded;
    # (1,B) - vs + miss; t2 forward NaN excluded -> 2/3.
    result = emit([[1.0, -1.0], [0.0, -2.0], [1.0, 1.0]]).predict(HIT_CLOSE)
    assert metrics.hit_rate(result, HIT_CLOSE) == pytest.approx(2.0 / 3.0)


def test_hit_rate_nan_output_cells_are_excluded() -> None:
    result = emit([[1.0, NAN], [NAN, -2.0], [1.0, 1.0]]).predict(HIT_CLOSE)
    assert metrics.hit_rate(result, HIT_CLOSE) == pytest.approx(0.5)


def test_hit_rate_with_no_valid_cells_is_nan() -> None:
    result = emit([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).predict(HIT_CLOSE)
    assert np.isnan(metrics.hit_rate(result, HIT_CLOSE))


def test_hit_rate_zero_forward_return_is_a_miss() -> None:
    # The contract: only zero *predictions* are excluded; a forward return of
    # exactly zero stays in the denominator, and no nonzero prediction can
    # match its sign, so the cell scores as a miss (directional accuracy).
    flat_a = wide([[1.0, 1.0], [1.0, 2.0], [1.0, 4.0]])  # A returns 0, 0, NaN
    result = emit([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).predict(flat_a)
    # Valid cells: A t0/t1 (fwd 0 -> miss), B t0/t1 (fwd + -> hit) = 2/4.
    assert metrics.hit_rate(result, flat_a) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# long_short_pnl
# ---------------------------------------------------------------------------

# forward returns: t0 [1, 3, 7], t1 [0.5, -0.25, -0.625], t2 NaN.
PNL_CLOSE = wide(
    [[1.0, 1.0, 1.0], [2.0, 4.0, 8.0], [3.0, 3.0, 3.0]],
    symbols=("A", "B", "C"),
)


def test_long_short_pnl_golden() -> None:
    # t0 pred [1,2,3]: ranks [1,2,3] -> demeaned [-1,0,1] -> gross 2
    # -> weights [-0.5, 0, 0.5]; pnl = -0.5*1 + 0.5*7 = 3.0.
    # t1 all-equal pred -> gross 0 -> NaN; t2 forward NaN -> NaN.
    result = emit([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1.0, 2.0, 3.0]]).predict(
        PNL_CLOSE
    )
    pnl = metrics.long_short_pnl(result, PNL_CLOSE)
    expected = pd.Series(
        [3.0, NAN, NAN],
        index=days(3).as_unit("ns").rename("timestamp"),
        name="long_short_pnl",
        dtype=np.float64,
    )
    pd.testing.assert_series_equal(pnl, expected, check_freq=False)


def test_long_short_pnl_partially_missing_forward_return_is_nan() -> None:
    # C's future price is missing, so its long leg cannot be evaluated;
    # scoring only the remaining legs would be neither gross-1 nor
    # market-neutral, so the whole timestamp is NaN. A symbol with no
    # prediction (t1's C) is not in the book and does not force NaN.
    close = wide(
        [[1.0, 1.0, 1.0], [2.0, 4.0, NAN], [3.0, 3.0, 3.0]],
        symbols=("A", "B", "C"),
    )
    result = emit([[1.0, 2.0, 3.0], [1.0, 2.0, NAN], [1.0, 2.0, 3.0]]).predict(close)
    pnl = metrics.long_short_pnl(result, close)
    # t0: C's forward return needs t1's missing price -> NaN.
    assert np.isnan(pnl.iloc[0])
    # t1: the book is [A, B] only; pnl = -0.5*(3/2-1) + 0.5*(3/4-1) = -0.375.
    assert pnl.iloc[1] == pytest.approx(-0.375)
    # t2: the last timestamp has no forward return at all -> NaN.
    assert np.isnan(pnl.iloc[2])


def test_long_short_pnl_is_market_neutral() -> None:
    # close_b's forward returns are close_a's plus a constant 1; weights sum
    # to zero, so the PnL is unchanged.
    close_a = wide([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0]], symbols=("A", "B", "C"))
    close_b = wide([[1.0, 1.0, 1.0], [3.0, 5.0, 9.0]], symbols=("A", "B", "C"))
    pred = [[3.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
    pnl_a = metrics.long_short_pnl(emit(pred).predict(close_a), close_a)
    pnl_b = metrics.long_short_pnl(emit(pred).predict(close_b), close_b)
    assert pnl_a.iloc[0] == pytest.approx(-1.0)
    assert pnl_a.iloc[0] == pnl_b.iloc[0]


# ---------------------------------------------------------------------------
# trailing-axis outputs are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        metrics.information_coefficient,
        metrics.hit_rate,
        metrics.long_short_pnl,
    ],
    ids=["information_coefficient", "hit_rate", "long_short_pnl"],
)
def test_metrics_reject_trailing_axis_outputs(
    metric: Callable[..., object],
) -> None:
    values = [[[1.0, 2.0]] * 2] * 3  # (T=3, S=2, K=2)
    result = emit(values).predict(HIT_CLOSE)
    with pytest.raises(ValueError, match="select_trailing"):
        metric(result, HIT_CLOSE)


def test_select_trailing_makes_output_evaluable() -> None:
    values = [[[1.0, -1.0]] * 2] * 3  # (T=3, S=2, H=2)
    result = emit(values).predict(HIT_CLOSE)
    selected = result.select_trailing(0)
    assert selected.array.shape == (3, 2)
    assert metrics.hit_rate(selected, HIT_CLOSE) == 0.75
