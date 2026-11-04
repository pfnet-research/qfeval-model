"""Public-surface snapshots: exported names, constants, exceptions, layering.

Constants such as ``DEFAULT_FEATURE`` / ``TIMESTAMP`` / ``SYMBOL`` are
wire contracts (checkpoint and DataFrame column names), so their literal
values are pinned, not just their existence.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import types

import qfeval_model as qm
import qfeval_model.core as core
from qfeval_model.verify import VerificationError


def _public_names(module: types.ModuleType) -> tuple[set[str], set[str]]:
    """Return (public dir() names, the subset that are modules)."""
    public = {name for name in dir(module) if not name.startswith("_")}
    modules = {
        name for name in public if isinstance(getattr(module, name), types.ModuleType)
    }
    return public, modules


def test_facade_all_names_exist() -> None:
    for name in qm.__all__:
        assert getattr(qm, name) is not None


def test_public_api_name_snapshots() -> None:
    """Make every public-name change an intentional review diff."""
    assert set(qm.__all__) == {
        "DEFAULT_FEATURE",
        "SYMBOL",
        "TIMESTAMP",
        "CheckpointError",
        "Context",
        "Features",
        "Mode",
        "Model",
        "ModelFactory",
        "QfevalModelWarning",
        "RawData",
        "Step",
        "TimeOrderError",
        "TrainTest",
        "__version__",
        "as_frame",
        "as_wide",
        "batch_model",
        "metrics",
        "model",
        "split",
        "verify",
        "zoo",
    }
    assert set(qm.zoo.__all__) == {
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
    }
    assert set(qm.metrics.__all__) == {
        "forward_returns",
        "hit_rate",
        "information_coefficient",
        "long_short_pnl",
    }
    assert set(qm.verify.__all__) == {
        "CheckReport",
        "CheckResult",
        "VerificationError",
        "assert_checkpoint_consistent",
        "assert_deterministic",
        "assert_stream_consistent",
        "check_all",
    }


def test_facade_all_matches_public_dir() -> None:
    public, modules = _public_names(qm)
    # Every public non-module name is exported, and vice versa.
    assert public - modules == set(qm.__all__) - {
        "__version__",
        "metrics",
        "verify",
        "zoo",
    }
    # The re-exported subpackages plus core are the only module attributes.
    assert modules == {"core", "metrics", "verify", "zoo"}


def test_core_all_names_exist() -> None:
    for name in core.__all__:
        assert getattr(core, name) is not None


def test_core_all_matches_public_dir() -> None:
    public, modules = _public_names(core)
    assert public - modules == set(core.__all__)
    # Any remaining module attributes are core's own submodules.
    submodules = {
        "constants",
        "context",
        "errors",
        "features",
        "model",
        "normalize",
        "split",
    }
    assert modules <= submodules


def test_version_matches_package_metadata() -> None:
    assert qm.__version__ == importlib.metadata.version("qfeval-model")


def test_wire_constants_are_fixed() -> None:
    assert qm.DEFAULT_FEATURE == "close"
    assert qm.TIMESTAMP == "timestamp"
    assert qm.SYMBOL == "symbol"


def test_model_public_surface_names_exist() -> None:
    # The compatibility policy promises exactly these documented methods and
    # attributes on Model; engine-created constructors are not supported API,
    # so instances come from factories (see docs/explanation/compatibility.md).
    m = qm.zoo.rolling_mean("close", window=2)
    for name in (
        "fit",
        "predict",
        "run",
        "dumps",
        "loads",
        "save",
        "load",
        "clone",
        "state",
        "seed",
        "symbols",
        "cursor",
        "mode",
        "started",
        "children",
    ):
        assert hasattr(m, name)


def test_model_call_vocabulary_is_inputs_and_has_no_info_argument() -> None:
    for method in (qm.Model.fit, qm.Model.predict, qm.Model.run):
        parameters = inspect.signature(method).parameters
        assert "inputs" in parameters
        assert "data" not in parameters
        assert "info" not in parameters


def test_exception_hierarchy() -> None:
    # The promise is catchability, not the exact bases: a common base class
    # may be inserted later without breaking compatibility.
    assert issubclass(qm.TimeOrderError, Exception)
    assert issubclass(qm.CheckpointError, Exception)
    assert issubclass(qm.QfevalModelWarning, UserWarning)
    assert qm.verify.VerificationError is VerificationError
    assert issubclass(VerificationError, Exception)
