"""``qm.zoo``: built-in models and composable building blocks.

The zoo is the landing place for conveniences that would otherwise grow the
core: every entry is an ordinary ``@qm.model`` you could have written
yourself. All zoo models are deterministic, checkpoint-ready at every step
boundary, and pass ``verify.check_all``.

API conventions:

- Most entries take a first argument ``source`` that accepts a feature
  name (``str``) or a child model (``qm.Model``), so zoo models compose
  freely: ``lag("close")`` lags an input feature, ``lag(momentum())`` lags a
  child model's output. The ensembles take a list of models instead.
- Argument names follow pandas (``periods``, ``window``, ``alpha``,
  ``ddof``).
- Sources must be real numeric and finite-or-NaN; complex and infinite
  inputs are rejected at the shared source boundary (``finite_indicator``
  is the one deliberate exception). Multiple sources of one model must have
  exactly the same shape — implicit NumPy broadcasting is rejected.
- ``window`` counts observation steps; windows trail and include the
  current step; warm-up is strict (``min_periods=window``), and a NaN in a
  window propagates to that cell's output unless the model is an explicit
  missing-data model (``rolling_count``, the ``rolling_valid_*`` family,
  ``stale_age``, ...).
- Only backward-looking transforms are admitted; forward-return computation
  lives in :mod:`qfeval_model.metrics` and is deliberately not composable as a
  model input (this structurally closes one leak path).
- One module per category. Modules depend only on the core's public surface
  (:mod:`qfeval_model.core`) and on sibling zoo modules, so the zoo can grow
  without widening the core.
"""

from qfeval_model.zoo.cross_section import (
    cross_sectional_demean,
    cross_sectional_rank,
    cross_sectional_scale,
    cross_sectional_winsorize,
    cross_sectional_zscore,
)
from qfeval_model.zoo.distribution import (
    rolling_above_threshold_rate,
    rolling_below_threshold_rate,
    rolling_lower_partial_moment,
    rolling_quantile,
    rolling_range_position,
    rolling_rank,
)
from qfeval_model.zoo.ensemble import (
    linear_combination,
    mean_ensemble,
    nanmean_ensemble,
)
from qfeval_model.zoo.expanding import (
    expanding_max,
    expanding_mean,
    expanding_min,
    expanding_std,
    expanding_sum,
    expanding_var,
    expanding_zscore,
)
from qfeval_model.zoo.fill import ffill, lag, stale_age, with_history
from qfeval_model.zoo.ohlcv import (
    amihud_illiquidity,
    candle_lower_shadow,
    candle_range,
    candle_real_body,
    candle_signed_body,
    candle_upper_shadow,
    rolling_volume_weighted_mean,
    true_range,
    typical_price,
    weighted_close,
)
from qfeval_model.zoo.pairwise import (
    rolling_beta,
    rolling_corr,
    rolling_cov,
    rolling_tracking_error,
)
from qfeval_model.zoo.regression import (
    rolling_autocorr,
    rolling_linear_trend_intercept,
    rolling_linear_trend_r2,
    rolling_linear_trend_residual,
    rolling_linear_trend_slope,
)
from qfeval_model.zoo.returns import log_return, simple_return
from qfeval_model.zoo.risk import (
    drawdown,
    drawdown_duration,
    rolling_compounded_return,
    rolling_downside_deviation,
    rolling_drawdown,
    rolling_historical_expected_shortfall,
    rolling_historical_value_at_risk,
    rolling_max_drawdown,
    rolling_worst_loss,
)
from qfeval_model.zoo.rolling import (
    linear_weighted_moving_average,
    recursive_ewm_mean,
    rolling_argmax,
    rolling_argmin,
    rolling_count,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_product,
    rolling_range,
    rolling_std,
    rolling_sum,
    rolling_valid_mean,
    rolling_valid_std,
    rolling_valid_sum,
    rolling_valid_var,
    rolling_var,
    rolling_zscore,
)
from qfeval_model.zoo.technical_oscillator import (
    moving_average_deviation_percent,
    psychological_line,
    rank_correlation_index,
    rolling_relative_strength_index,
    slow_stochastic_percent_d,
    stochastic_percent_d,
    stochastic_percent_k,
)
from qfeval_model.zoo.technical_trend import (
    bollinger_bandwidth,
    bollinger_lower_band,
    bollinger_percent_b,
    bollinger_upper_band,
    ichimoku_base_line,
    ichimoku_conversion_line,
    ichimoku_leading_span_a,
    ichimoku_leading_span_b,
    macd_histogram,
    macd_line,
    macd_signal,
    moving_average_envelope_lower,
    moving_average_envelope_upper,
    rolling_average_directional_index,
    rolling_directional_index,
    rolling_negative_directional_indicator,
    rolling_positive_directional_indicator,
)
from qfeval_model.zoo.transforms import (
    absolute,
    add,
    clip,
    coalesce,
    diff,
    divide,
    exp,
    fill_constant,
    finite_indicator,
    log,
    maximum,
    minimum,
    multiply,
    negate,
    power,
    sign,
    subtract,
    where,
)
from qfeval_model.zoo.volatility import annualized_historical_volatility

__all__ = [
    "absolute",
    "add",
    "amihud_illiquidity",
    "annualized_historical_volatility",
    "bollinger_bandwidth",
    "bollinger_lower_band",
    "bollinger_percent_b",
    "bollinger_upper_band",
    "candle_lower_shadow",
    "candle_range",
    "candle_real_body",
    "candle_signed_body",
    "candle_upper_shadow",
    "clip",
    "coalesce",
    "cross_sectional_demean",
    "cross_sectional_rank",
    "cross_sectional_scale",
    "cross_sectional_winsorize",
    "cross_sectional_zscore",
    "diff",
    "divide",
    "drawdown",
    "drawdown_duration",
    "exp",
    "expanding_max",
    "expanding_mean",
    "expanding_min",
    "expanding_std",
    "expanding_sum",
    "expanding_var",
    "expanding_zscore",
    "ffill",
    "fill_constant",
    "finite_indicator",
    "ichimoku_base_line",
    "ichimoku_conversion_line",
    "ichimoku_leading_span_a",
    "ichimoku_leading_span_b",
    "lag",
    "linear_combination",
    "linear_weighted_moving_average",
    "log",
    "log_return",
    "macd_histogram",
    "macd_line",
    "macd_signal",
    "maximum",
    "mean_ensemble",
    "minimum",
    "moving_average_deviation_percent",
    "moving_average_envelope_lower",
    "moving_average_envelope_upper",
    "multiply",
    "nanmean_ensemble",
    "negate",
    "power",
    "psychological_line",
    "rank_correlation_index",
    "recursive_ewm_mean",
    "rolling_above_threshold_rate",
    "rolling_argmax",
    "rolling_argmin",
    "rolling_autocorr",
    "rolling_average_directional_index",
    "rolling_below_threshold_rate",
    "rolling_beta",
    "rolling_compounded_return",
    "rolling_corr",
    "rolling_count",
    "rolling_cov",
    "rolling_directional_index",
    "rolling_downside_deviation",
    "rolling_drawdown",
    "rolling_historical_expected_shortfall",
    "rolling_historical_value_at_risk",
    "rolling_linear_trend_intercept",
    "rolling_linear_trend_r2",
    "rolling_linear_trend_residual",
    "rolling_linear_trend_slope",
    "rolling_lower_partial_moment",
    "rolling_max",
    "rolling_max_drawdown",
    "rolling_mean",
    "rolling_median",
    "rolling_min",
    "rolling_negative_directional_indicator",
    "rolling_positive_directional_indicator",
    "rolling_product",
    "rolling_quantile",
    "rolling_range",
    "rolling_range_position",
    "rolling_rank",
    "rolling_relative_strength_index",
    "rolling_std",
    "rolling_sum",
    "rolling_tracking_error",
    "rolling_valid_mean",
    "rolling_valid_std",
    "rolling_valid_sum",
    "rolling_valid_var",
    "rolling_var",
    "rolling_volume_weighted_mean",
    "rolling_worst_loss",
    "rolling_zscore",
    "sign",
    "simple_return",
    "slow_stochastic_percent_d",
    "stale_age",
    "stochastic_percent_d",
    "stochastic_percent_k",
    "subtract",
    "true_range",
    "typical_price",
    "weighted_close",
    "where",
    "with_history",
]

# The zoo's public surface is this package, and its submodules are an
# implementation detail — so checkpoints must identify zoo models as
# ``qfeval_model.zoo:<name>``, not by the submodule a function happens to
# live in. Pinning ``__module__`` here keeps checkpoints readable across
# internal file reshuffles (tested by the identity golden test).
for _name in __all__:
    _entry = globals()[_name]
    _entry.__module__ = _entry.fn.__module__ = __name__
del _name, _entry
