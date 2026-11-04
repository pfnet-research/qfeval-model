"""The blanket zoo contract, parametrized over every public zoo model.

``FACTORIES`` maps every ``zoo.__all__`` name to one or more ready-to-run
factories (a coverage test keeps the mapping complete). Running each through
``verify.check_all`` — on plain and trailing-axis inputs — is the blanket
guarantee that their state designs are deterministic, stream-consistent, and
checkpoint-ready; the other parametrized tests pin the shared source
contract: complex/infinity rejection, the shape lock, string-vs-model source
equivalence, and no retained input views.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import pytest
from conftest import days, trailing_wide, wide

import qfeval_model as qm
from qfeval_model import verify, zoo

NAN = float("nan")

CHECK_CLOSE = wide(
    [
        [100.0, 200.0, 50.0],
        [101.0, NAN, 51.0],
        [102.0, 202.0, NAN],
        [103.0, 203.0, 53.0],
        [NAN, 204.0, 54.0],
        [105.0, 205.0, 55.0],
        [106.0, NAN, 56.0],
        [107.0, 207.0, 57.0],
    ],
    symbols=("A", "B", "C"),
)


def _ohlcv_panel(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A consistent OHLCV panel derived from a positive close panel."""
    return {
        "close": close,
        "open": close * 1.01,
        "high": close * 1.02,
        "low": close * 0.98,
        "volume": close * 100.0,
    }


CHECK_INPUTS = _ohlcv_panel(CHECK_CLOSE)

_EXTREME_CLOSE = wide(
    [[1e308 if t % 2 == 0 else -1e308] * 3 for t in range(8)],
    symbols=("A", "B", "C"),
)
EXTREME_INPUTS = {
    "close": _EXTREME_CLOSE,
    "open": _EXTREME_CLOSE,
    "high": wide([[1e308] * 3] * 8, symbols=("A", "B", "C")),
    "low": wide([[-1e308] * 3] * 8, symbols=("A", "B", "C")),
    "volume": wide([[1e308] * 3] * 8, symbols=("A", "B", "C")),
}

# One or more ready-to-run factories per public zoo name; multi-source
# entries feed a child model so a single "close" panel exercises them.
FACTORIES: dict[str, list[Callable[[], qm.Model]]] = {
    "lag": [
        lambda: zoo.lag("close"),
        lambda: zoo.lag(zoo.ffill("close"), periods=2),
        lambda: zoo.lag(zoo.rolling_mean("close", 2)),
    ],
    "ffill": [lambda: zoo.ffill("close")],
    "stale_age": [lambda: zoo.stale_age("close")],
    "with_history": [lambda: zoo.with_history("close", window=3)],
    "diff": [lambda: zoo.diff("close", periods=2)],
    "fill_constant": [lambda: zoo.fill_constant("close", value=1.0)],
    "finite_indicator": [lambda: zoo.finite_indicator("close")],
    "coalesce": [lambda: zoo.coalesce(["close", zoo.ffill("close")])],
    "clip": [lambda: zoo.clip("close", lower=52.0, upper=204.0)],
    "add": [lambda: zoo.add("close", zoo.lag("close"))],
    "subtract": [lambda: zoo.subtract("close", 1.0)],
    "multiply": [lambda: zoo.multiply("close", zoo.lag("close"))],
    "divide": [lambda: zoo.divide("close", zoo.lag("close"))],
    "minimum": [lambda: zoo.minimum("close", zoo.lag("close"))],
    "maximum": [lambda: zoo.maximum("close", 100.0)],
    "power": [lambda: zoo.power("close", 2.0)],
    "negate": [lambda: zoo.negate("close")],
    "absolute": [lambda: zoo.absolute("close")],
    "sign": [lambda: zoo.sign(zoo.diff("close"))],
    "log": [lambda: zoo.log("close")],
    "exp": [lambda: zoo.exp(zoo.log_return("close"))],
    "where": [lambda: zoo.where(zoo.diff("close"), "close", 0.0)],
    "simple_return": [lambda: zoo.simple_return()],
    "log_return": [lambda: zoo.log_return()],
    "rolling_mean": [lambda: zoo.rolling_mean("close", window=3)],
    "rolling_sum": [lambda: zoo.rolling_sum("close", window=3)],
    "rolling_valid_sum": [lambda: zoo.rolling_valid_sum("close", window=3)],
    "rolling_valid_mean": [lambda: zoo.rolling_valid_mean("close", window=3)],
    "rolling_valid_var": [lambda: zoo.rolling_valid_var("close", window=3)],
    "rolling_valid_std": [lambda: zoo.rolling_valid_std("close", window=3)],
    "rolling_var": [lambda: zoo.rolling_var("close", window=3)],
    "rolling_std": [lambda: zoo.rolling_std("close", window=3)],
    "rolling_zscore": [lambda: zoo.rolling_zscore("close", window=3)],
    "rolling_min": [lambda: zoo.rolling_min("close", window=3)],
    "rolling_max": [lambda: zoo.rolling_max("close", window=3)],
    "rolling_range": [lambda: zoo.rolling_range("close", window=3)],
    "rolling_count": [lambda: zoo.rolling_count("close", window=3)],
    "rolling_median": [lambda: zoo.rolling_median("close", window=3)],
    "rolling_product": [lambda: zoo.rolling_product(zoo.sign("close"), window=3)],
    "rolling_argmin": [lambda: zoo.rolling_argmin("close", window=3)],
    "rolling_argmax": [lambda: zoo.rolling_argmax("close", window=3)],
    "recursive_ewm_mean": [lambda: zoo.recursive_ewm_mean("close", alpha=0.5)],
    "linear_weighted_moving_average": [
        lambda: zoo.linear_weighted_moving_average("close", window=3)
    ],
    "expanding_sum": [lambda: zoo.expanding_sum("close")],
    "expanding_mean": [lambda: zoo.expanding_mean("close")],
    "expanding_var": [lambda: zoo.expanding_var("close")],
    "expanding_std": [lambda: zoo.expanding_std("close")],
    "expanding_zscore": [lambda: zoo.expanding_zscore("close")],
    "expanding_min": [lambda: zoo.expanding_min("close")],
    "expanding_max": [lambda: zoo.expanding_max("close")],
    "cross_sectional_rank": [lambda: zoo.cross_sectional_rank("close")],
    "cross_sectional_demean": [lambda: zoo.cross_sectional_demean("close")],
    "cross_sectional_zscore": [lambda: zoo.cross_sectional_zscore("close")],
    "cross_sectional_scale": [lambda: zoo.cross_sectional_scale("close")],
    "cross_sectional_winsorize": [
        lambda: zoo.cross_sectional_winsorize("close", 0.25, 0.75)
    ],
    "rolling_quantile": [
        lambda: zoo.rolling_quantile("close", window=3, q=0.25, method="linear"),
        lambda: zoo.rolling_quantile("close", window=3, q=0.25, method="inverted_cdf"),
    ],
    "rolling_rank": [lambda: zoo.rolling_rank("close", window=3)],
    "rolling_below_threshold_rate": [
        lambda: zoo.rolling_below_threshold_rate("close", window=3, threshold=55.0)
    ],
    "rolling_above_threshold_rate": [
        lambda: zoo.rolling_above_threshold_rate("close", window=3, threshold=55.0)
    ],
    "rolling_lower_partial_moment": [
        lambda: zoo.rolling_lower_partial_moment(
            zoo.simple_return(), window=3, threshold=0.0, order=2
        )
    ],
    "rolling_range_position": [lambda: zoo.rolling_range_position("close", window=3)],
    "rolling_compounded_return": [
        lambda: zoo.rolling_compounded_return(zoo.simple_return(), window=3)
    ],
    "rolling_worst_loss": [
        lambda: zoo.rolling_worst_loss(zoo.simple_return(), window=3)
    ],
    "drawdown": [lambda: zoo.drawdown("close")],
    "rolling_drawdown": [lambda: zoo.rolling_drawdown("close", window=3)],
    "rolling_max_drawdown": [lambda: zoo.rolling_max_drawdown("close", window=3)],
    "drawdown_duration": [lambda: zoo.drawdown_duration("close")],
    "rolling_downside_deviation": [
        lambda: zoo.rolling_downside_deviation(zoo.simple_return(), window=3)
    ],
    "rolling_historical_value_at_risk": [
        lambda: zoo.rolling_historical_value_at_risk(
            zoo.simple_return(), window=4, confidence=0.75
        )
    ],
    "rolling_historical_expected_shortfall": [
        lambda: zoo.rolling_historical_expected_shortfall(
            zoo.simple_return(), window=4, confidence=0.75
        )
    ],
    "rolling_linear_trend_slope": [
        lambda: zoo.rolling_linear_trend_slope("close", window=3)
    ],
    "rolling_linear_trend_intercept": [
        lambda: zoo.rolling_linear_trend_intercept("close", window=3)
    ],
    "rolling_linear_trend_r2": [lambda: zoo.rolling_linear_trend_r2("close", window=3)],
    "rolling_linear_trend_residual": [
        lambda: zoo.rolling_linear_trend_residual("close", window=3)
    ],
    "rolling_autocorr": [lambda: zoo.rolling_autocorr("close", window=3, lag=1)],
    "true_range": [lambda: zoo.true_range()],
    "typical_price": [lambda: zoo.typical_price()],
    "weighted_close": [lambda: zoo.weighted_close()],
    "rolling_volume_weighted_mean": [
        lambda: zoo.rolling_volume_weighted_mean(window=3)
    ],
    "amihud_illiquidity": [
        lambda: zoo.amihud_illiquidity(zoo.simple_return(), window=3)
    ],
    "candle_signed_body": [lambda: zoo.candle_signed_body()],
    "candle_real_body": [lambda: zoo.candle_real_body()],
    "candle_upper_shadow": [lambda: zoo.candle_upper_shadow()],
    "candle_lower_shadow": [lambda: zoo.candle_lower_shadow()],
    "candle_range": [lambda: zoo.candle_range()],
    "annualized_historical_volatility": [
        lambda: zoo.annualized_historical_volatility(
            zoo.simple_return(), window=3, periods_per_year=252.0
        )
    ],
    "bollinger_upper_band": [lambda: zoo.bollinger_upper_band("close", window=3)],
    "bollinger_lower_band": [lambda: zoo.bollinger_lower_band("close", window=3)],
    "bollinger_bandwidth": [lambda: zoo.bollinger_bandwidth("close", window=3)],
    "bollinger_percent_b": [lambda: zoo.bollinger_percent_b("close", window=3)],
    "moving_average_envelope_upper": [
        lambda: zoo.moving_average_envelope_upper("close", window=3)
    ],
    "moving_average_envelope_lower": [
        lambda: zoo.moving_average_envelope_lower("close", window=3)
    ],
    "rolling_relative_strength_index": [
        lambda: zoo.rolling_relative_strength_index("close", window=3)
    ],
    "stochastic_percent_k": [lambda: zoo.stochastic_percent_k(window=3)],
    "stochastic_percent_d": [lambda: zoo.stochastic_percent_d(window=3, smooth=2)],
    "slow_stochastic_percent_d": [
        lambda: zoo.slow_stochastic_percent_d(window=3, smooth_k=2, smooth_d=2)
    ],
    "psychological_line": [lambda: zoo.psychological_line("close", window=3)],
    "rank_correlation_index": [lambda: zoo.rank_correlation_index("close", window=3)],
    "moving_average_deviation_percent": [
        lambda: zoo.moving_average_deviation_percent("close", window=3)
    ],
    "macd_line": [lambda: zoo.macd_line("close", fast_span=2, slow_span=4)],
    "macd_signal": [
        lambda: zoo.macd_signal("close", fast_span=2, slow_span=4, signal_span=2)
    ],
    "macd_histogram": [
        lambda: zoo.macd_histogram("close", fast_span=2, slow_span=4, signal_span=2)
    ],
    "ichimoku_conversion_line": [lambda: zoo.ichimoku_conversion_line(window=3)],
    "ichimoku_base_line": [lambda: zoo.ichimoku_base_line(window=4)],
    "ichimoku_leading_span_a": [
        lambda: zoo.ichimoku_leading_span_a(
            conversion_window=2, base_window=3, displacement=2
        )
    ],
    "ichimoku_leading_span_b": [
        lambda: zoo.ichimoku_leading_span_b(window=4, displacement=2)
    ],
    "rolling_positive_directional_indicator": [
        lambda: zoo.rolling_positive_directional_indicator(window=3)
    ],
    "rolling_negative_directional_indicator": [
        lambda: zoo.rolling_negative_directional_indicator(window=3)
    ],
    "rolling_directional_index": [lambda: zoo.rolling_directional_index(window=3)],
    "rolling_average_directional_index": [
        lambda: zoo.rolling_average_directional_index(window=3)
    ],
    "rolling_cov": [lambda: zoo.rolling_cov("close", zoo.lag("close"), window=3)],
    "rolling_corr": [lambda: zoo.rolling_corr("close", zoo.lag("close"), window=3)],
    "rolling_beta": [lambda: zoo.rolling_beta("close", zoo.lag("close"), window=3)],
    "rolling_tracking_error": [
        lambda: zoo.rolling_tracking_error("close", zoo.lag("close"), window=3)
    ],
    "mean_ensemble": [
        lambda: zoo.mean_ensemble([zoo.lag("close"), zoo.rolling_mean("close", 2)])
    ],
    "nanmean_ensemble": [
        lambda: zoo.nanmean_ensemble(
            [zoo.lag("close"), zoo.rolling_mean("close", 2)], min_children=2
        )
    ],
    "linear_combination": [
        lambda: zoo.linear_combination(
            [zoo.lag("close"), zoo.rolling_mean("close", 2)], weights=[0.7, 0.3]
        )
    ],
}

ALL_FACTORIES = [factory for factories in FACTORIES.values() for factory in factories]
ALL_IDS = [
    name if len(factories) == 1 else f"{name}-{index}"
    for name, factories in FACTORIES.items()
    for index, factory in enumerate(factories)
]
all_zoo = pytest.mark.parametrize("factory", ALL_FACTORIES, ids=ALL_IDS)


def test_factories_cover_every_public_zoo_model() -> None:
    assert set(FACTORIES) == set(zoo.__all__)


@all_zoo
def test_every_zoo_model_passes_check_all(factory: Callable[[], qm.Model]) -> None:
    report = verify.check_all(factory, CHECK_INPUTS, test_from="2020-01-05")
    assert report.all_passed, str(report)


TRAILING_CHECK_INPUTS = _ohlcv_panel(
    trailing_wide(
        [
            [1.0, 10.0, 2.0, 20.0],
            [2.0, NAN, 3.0, 30.0],
            [NAN, 12.0, 4.0, NAN],
            [4.0, 13.0, 5.0, 50.0],
            [5.0, 14.0, NAN, 60.0],
            [6.0, 15.0, 7.0, 70.0],
        ]
    )
)


@all_zoo
def test_every_zoo_model_passes_check_all_on_trailing_axis_inputs(
    factory: Callable[[], qm.Model],
) -> None:
    report = verify.check_all(factory, TRAILING_CHECK_INPUTS, test_from="2020-01-04")
    assert report.all_passed, str(report)


@all_zoo
def test_every_zoo_model_rejects_complex_sources(
    factory: Callable[[], qm.Model],
) -> None:
    complex_inputs = {
        name: frame.astype(np.complex128) + 10j for name, frame in CHECK_INPUTS.items()
    }
    with pytest.raises(ValueError, match="real numeric"):
        factory().predict(complex_inputs)


@all_zoo
def test_every_zoo_model_rejects_infinite_sources(
    factory: Callable[[], qm.Model],
) -> None:
    if factory().fn.__name__ == "finite_indicator":
        pytest.skip("finite_indicator is the explicit non-finite reader")
    infinite = {name: frame.copy() for name, frame in CHECK_INPUTS.items()}
    for frame in infinite.values():
        frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        factory().predict(infinite)


@all_zoo
def test_every_zoo_model_handles_extreme_finite_sources_without_infinity(
    factory: Callable[[], qm.Model],
) -> None:
    # Finite values at the dtype boundary are valid sources. Intermediate
    # overflow must neither escape as +/-inf nor emit a warning (warnings are
    # errors suite-wide); an unrepresentable result is reported as NaN.
    if factory().fn.__name__ == "rolling_max_drawdown":
        with pytest.raises(ValueError, match="must be positive or NaN"):
            factory().predict(EXTREME_INPUTS)
        return
    prediction = factory().predict(EXTREME_INPUTS)
    assert not np.isinf(prediction.array).any()


@all_zoo
def test_zoo_state_does_not_retain_call_input_views(
    factory: Callable[[], qm.Model],
) -> None:
    # Step inputs/child outputs are views into the whole call's arrays; a
    # view stored in state would keep the full (T, S) input alive after the
    # call. Owned copies have no base.
    m = factory()
    m.predict(CHECK_INPUTS, seed=0)
    models = [m]
    while models:
        current = models.pop()
        models.extend((current.children or {}).values())
        for array in _state_arrays(current.state):
            assert array.base is None, f"state of {current} retains a view"


def _state_arrays(obj: object) -> list[np.ndarray]:
    """All ndarrays reachable from a state value (dicts, sequences, deques)."""
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return [a for item in obj for a in _state_arrays(item)]
    return []


# ---------------------------------------------------------------------------
# source polymorphism: column-name source == model source
# ---------------------------------------------------------------------------

PRICES = wide(
    [
        [1.0, 10.0],
        [2.0, NAN],
        [4.0, 30.0],
        [8.0, 40.0],
    ]
)

SOURCE_FACTORIES: dict[str, Callable[[str | qm.Model], qm.Model]] = {
    "lag": zoo.lag,
    "ffill": zoo.ffill,
    "stale_age": zoo.stale_age,
    "with_history": lambda source: zoo.with_history(source, window=2),
    "diff": zoo.diff,
    "fill_constant": zoo.fill_constant,
    "finite_indicator": zoo.finite_indicator,
    "clip": lambda source: zoo.clip(source, lower=2.0, upper=30.0),
    "negate": zoo.negate,
    "absolute": zoo.absolute,
    "sign": zoo.sign,
    "log": zoo.log,
    "exp": zoo.exp,
    "simple_return": zoo.simple_return,
    "log_return": zoo.log_return,
    "rolling_mean": lambda source: zoo.rolling_mean(source, window=2),
    "rolling_sum": lambda source: zoo.rolling_sum(source, window=2),
    "rolling_valid_sum": lambda source: zoo.rolling_valid_sum(source, window=2),
    "rolling_valid_mean": lambda source: zoo.rolling_valid_mean(source, window=2),
    "rolling_valid_var": lambda source: zoo.rolling_valid_var(source, window=2),
    "rolling_valid_std": lambda source: zoo.rolling_valid_std(source, window=2),
    "rolling_var": lambda source: zoo.rolling_var(source, window=2),
    "rolling_std": lambda source: zoo.rolling_std(source, window=2),
    "rolling_zscore": lambda source: zoo.rolling_zscore(source, window=2),
    "rolling_min": lambda source: zoo.rolling_min(source, window=2),
    "rolling_max": lambda source: zoo.rolling_max(source, window=2),
    "rolling_range": lambda source: zoo.rolling_range(source, window=2),
    "rolling_count": lambda source: zoo.rolling_count(source, window=2),
    "rolling_median": lambda source: zoo.rolling_median(source, window=2),
    "rolling_product": lambda source: zoo.rolling_product(source, window=2),
    "rolling_argmin": lambda source: zoo.rolling_argmin(source, window=2),
    "rolling_argmax": lambda source: zoo.rolling_argmax(source, window=2),
    "recursive_ewm_mean": lambda source: zoo.recursive_ewm_mean(source, alpha=0.5),
    "linear_weighted_moving_average": (
        lambda source: zoo.linear_weighted_moving_average(source, window=2)
    ),
    "expanding_sum": zoo.expanding_sum,
    "expanding_mean": zoo.expanding_mean,
    "expanding_var": zoo.expanding_var,
    "expanding_std": zoo.expanding_std,
    "expanding_zscore": zoo.expanding_zscore,
    "expanding_min": zoo.expanding_min,
    "expanding_max": zoo.expanding_max,
    "cross_sectional_rank": zoo.cross_sectional_rank,
    "cross_sectional_demean": zoo.cross_sectional_demean,
    "cross_sectional_zscore": zoo.cross_sectional_zscore,
    "cross_sectional_scale": zoo.cross_sectional_scale,
    "cross_sectional_winsorize": (
        lambda source: zoo.cross_sectional_winsorize(source, 0.25, 0.75)
    ),
    "rolling_quantile": (
        lambda source: zoo.rolling_quantile(source, window=2, q=0.5, method="linear")
    ),
    "rolling_rank": lambda source: zoo.rolling_rank(source, window=2),
    "rolling_below_threshold_rate": (
        lambda source: zoo.rolling_below_threshold_rate(source, window=2, threshold=3.0)
    ),
    "rolling_above_threshold_rate": (
        lambda source: zoo.rolling_above_threshold_rate(source, window=2, threshold=3.0)
    ),
    "rolling_lower_partial_moment": (
        lambda source: zoo.rolling_lower_partial_moment(
            source, window=2, threshold=5.0, order=2
        )
    ),
    "rolling_range_position": (
        lambda source: zoo.rolling_range_position(source, window=2)
    ),
    "rolling_compounded_return": (
        lambda source: zoo.rolling_compounded_return(source, window=2)
    ),
    "rolling_worst_loss": lambda source: zoo.rolling_worst_loss(source, window=2),
    "drawdown": zoo.drawdown,
    "rolling_drawdown": lambda source: zoo.rolling_drawdown(source, window=2),
    "rolling_max_drawdown": lambda source: zoo.rolling_max_drawdown(source, window=2),
    "drawdown_duration": zoo.drawdown_duration,
    "rolling_downside_deviation": (
        lambda source: zoo.rolling_downside_deviation(source, window=2, threshold=5.0)
    ),
    "rolling_historical_value_at_risk": (
        lambda source: zoo.rolling_historical_value_at_risk(
            source, window=2, confidence=0.6
        )
    ),
    "rolling_historical_expected_shortfall": (
        lambda source: zoo.rolling_historical_expected_shortfall(
            source, window=2, confidence=0.6
        )
    ),
    "rolling_linear_trend_slope": (
        lambda source: zoo.rolling_linear_trend_slope(source, window=2)
    ),
    "rolling_linear_trend_intercept": (
        lambda source: zoo.rolling_linear_trend_intercept(source, window=2)
    ),
    "rolling_linear_trend_r2": (
        lambda source: zoo.rolling_linear_trend_r2(source, window=2)
    ),
    "rolling_linear_trend_residual": (
        lambda source: zoo.rolling_linear_trend_residual(source, window=2)
    ),
    "rolling_autocorr": lambda source: zoo.rolling_autocorr(source, window=2),
    "annualized_historical_volatility": (
        lambda source: zoo.annualized_historical_volatility(
            source, window=2, periods_per_year=252.0
        )
    ),
    "bollinger_upper_band": lambda source: zoo.bollinger_upper_band(source, window=2),
    "bollinger_lower_band": lambda source: zoo.bollinger_lower_band(source, window=2),
    "bollinger_bandwidth": lambda source: zoo.bollinger_bandwidth(source, window=2),
    "bollinger_percent_b": lambda source: zoo.bollinger_percent_b(source, window=2),
    "moving_average_envelope_upper": (
        lambda source: zoo.moving_average_envelope_upper(source, window=2)
    ),
    "moving_average_envelope_lower": (
        lambda source: zoo.moving_average_envelope_lower(source, window=2)
    ),
    "rolling_relative_strength_index": (
        lambda source: zoo.rolling_relative_strength_index(source, window=2)
    ),
    "psychological_line": lambda source: zoo.psychological_line(source, window=2),
    "rank_correlation_index": (
        lambda source: zoo.rank_correlation_index(source, window=2)
    ),
    "moving_average_deviation_percent": (
        lambda source: zoo.moving_average_deviation_percent(source, window=2)
    ),
    "macd_line": lambda source: zoo.macd_line(source, fast_span=1, slow_span=2),
    "macd_signal": (
        lambda source: zoo.macd_signal(source, fast_span=1, slow_span=2, signal_span=2)
    ),
    "macd_histogram": (
        lambda source: zoo.macd_histogram(
            source, fast_span=1, slow_span=2, signal_span=2
        )
    ),
}


@pytest.mark.parametrize(
    "factory", SOURCE_FACTORIES.values(), ids=SOURCE_FACTORIES.keys()
)
def test_model_source_equals_precomputed_column_source(
    factory: Callable[[str | qm.Model], qm.Model],
) -> None:
    lagged = PRICES.shift(1)
    via_model = factory(zoo.lag("close")).predict(PRICES)
    via_column = factory("sig").predict({"close": PRICES, "sig": lagged})
    np.testing.assert_array_equal(via_model.array, via_column.array)
    assert via_model.side_data.keys() == via_column.side_data.keys()
    for name in via_model.side_data:
        np.testing.assert_array_equal(
            via_model.side_data[name], via_column.side_data[name]
        )


def test_missing_column_source_raises_key_error() -> None:
    with pytest.raises(KeyError, match="nope"):
        zoo.lag("nope").predict(PRICES)


# ---------------------------------------------------------------------------
# checkpoint identity and the shape lock
# ---------------------------------------------------------------------------


def test_zoo_checkpoint_identity_is_pinned_to_the_package() -> None:
    # The zoo's public surface is qfeval_model.zoo and its submodules are an
    # implementation detail, so checkpoints must record zoo models as
    # ``qfeval_model.zoo:<name>`` — moving a function between submodules must
    # not orphan existing checkpoints (identity golden).
    for name in zoo.__all__:
        assert getattr(zoo, name).fn.__module__ == "qfeval_model.zoo"
    m = zoo.lag(zoo.rolling_mean("close", 2))
    m.predict(PRICES)
    envelope = pickle.loads(m.dumps())
    assert envelope["ident"] == "qfeval_model.zoo:lag"
    (child,) = envelope["children"]
    assert child["model"]["ident"] == "qfeval_model.zoo:rolling_mean"


def _trailing_next_day() -> dict[str, pd.DataFrame]:
    """A (S=2, K=2) trailing-axis close panel one day after 2020-01-01."""
    return {
        "close": pd.DataFrame(
            [[3.0, 4.0, 5.0, 6.0]],
            index=days(1, start="2020-01-02"),
            columns=pd.MultiIndex.from_product([["A", "B"], range(2)]),
            dtype=np.float64,
        )
    }


def test_stateful_zoo_rejects_source_shape_change_across_calls() -> None:
    # A stateful transform holds prior-shape arrays in state; a differently
    # shaped source on a later call could silently NumPy-broadcast against
    # them, so the shared source reader rejects the change explicitly.
    m = zoo.ffill("close")
    m.fit(wide([[1.0, 2.0]]), symbols=("A", "B"))
    with pytest.raises(ValueError, match="source shape changed"):
        m.predict(_trailing_next_day())


def test_stateful_zoo_shape_lock_survives_checkpoint() -> None:
    m = zoo.rolling_mean("close", window=2)
    m.fit(wide([[1.0, 2.0]]), symbols=("A", "B"))
    restored = zoo.rolling_mean("close", window=2).loads(m.dumps())
    with pytest.raises(ValueError, match="source shape changed"):
        restored.predict(_trailing_next_day())


def test_multi_source_shape_mismatch_is_rejected() -> None:
    # Two sources of one model must agree in shape exactly; broadcasting a
    # flat panel against a trailing-axis panel is refused.
    inputs: dict[str, pd.DataFrame] = {
        "close": wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "other": trailing_wide(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]
        ),
    }
    with pytest.raises(ValueError, match="source shape changed"):
        zoo.rolling_cov("close", "other", window=2).predict(inputs)


def test_finite_indicator_flags_infinity_instead_of_rejecting() -> None:
    infinite = wide([[1.0, NAN], [np.inf, 2.0]])
    pred = zoo.finite_indicator("close").predict(infinite)
    np.testing.assert_array_equal(pred.array, [[1.0, 0.0], [0.0, 1.0]])
