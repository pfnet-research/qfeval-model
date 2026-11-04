# %%
"""# 09 — technical indicators on one OHLCV dataset

Every indicator line is its own zoo factory — Bollinger's upper band, the
stochastic %K, Ichimoku's conversion line — so any line can feed another
model's `source`. Indicators keep their conventional market scales (RSI
and stochastics 0-100), and none of them emit buy/sell decisions:
thresholding is your model's job.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# One deterministic OHLCV panel for a single symbol over 16 bars.
days = pd.date_range("2024-01-01", periods=16)
base = 100.0 + np.cumsum(
    [0.0, 1.5, 1.0, -0.5, 2.0, 1.0, -3.0, -2.0, 1.0, 2.5, 1.5, -1.0, 0.5, 2.0, 1.0, 0.5]
)
inputs = {
    "open": pd.DataFrame({"AAA": base + 0.2}, index=days),
    "high": pd.DataFrame({"AAA": base + 1.0}, index=days),
    "low": pd.DataFrame({"AAA": base - 1.0}, index=days),
    "close": pd.DataFrame({"AAA": base + 0.5}, index=days),
    "volume": pd.DataFrame({"AAA": 1000.0 + 10.0 * base}, index=days),
}

# %%
# Bollinger bands: one factory per line, sample std (ddof=1) by default.
upper = qm.zoo.bollinger_upper_band("close", window=5).predict(inputs, seed=0)
lower = qm.zoo.bollinger_lower_band("close", window=5).predict(inputs, seed=0)
percent_b = qm.zoo.bollinger_percent_b("close", window=5).predict(inputs, seed=0)
bands = pd.DataFrame(
    {
        "close": inputs["close"]["AAA"],
        "upper": upper.to_wide()["AAA"],
        "lower": lower.to_wide()["AAA"],
        "percent_b": percent_b.to_wide()["AAA"],
    }
)
print("== Bollinger bands (window 5, 2 std) ==")
print(bands.round(3).to_string())
print("percent_b is a fraction: 0.5 at the middle band, unclamped outside")

# %%
# RSI (simple rolling-mean variant) and the stochastic pair, both 0-100.
rsi = qm.zoo.rolling_relative_strength_index("close", window=5)
percent_k = qm.zoo.stochastic_percent_k(window=5)
percent_d = qm.zoo.stochastic_percent_d(window=5, smooth=3)
oscillators = pd.DataFrame(
    {
        "rsi": rsi.predict(inputs, seed=0).to_wide()["AAA"],
        "percent_k": percent_k.predict(inputs, seed=0).to_wide()["AAA"],
        "percent_d": percent_d.predict(inputs, seed=0).to_wide()["AAA"],
    }
)
print("== RSI and stochastics (0-100 scales) ==")
print(oscillators.round(2).to_string())
print("the names say which variant: rolling means, not Wilder smoothing")

# %%
# Ichimoku's causal alignment: the span value shown at t was computed
# displacement steps earlier — nothing is written into the future.
conversion = qm.zoo.ichimoku_conversion_line(window=3).predict(inputs, seed=0)
span_a = qm.zoo.ichimoku_leading_span_a(
    conversion_window=3, base_window=5, displacement=3
).predict(inputs, seed=0)
ichimoku = pd.DataFrame(
    {
        "conversion": conversion.to_wide()["AAA"],
        "span_a": span_a.to_wide()["AAA"],
    }
)
print("== Ichimoku conversion line and causal leading span A ==")
print(ichimoku.round(3).to_string())
print("the lagging span is absent by design: it would need a future close")

# %%
# Historical volatility on log returns, annualized explicitly.
vol = qm.zoo.annualized_historical_volatility(
    qm.zoo.log_return(), window=5, periods_per_year=252
)
print("== annualized historical volatility (window 5) ==")
print(vol.predict(inputs, seed=0).to_wide().round(4).to_string())
print("every window is strict: NaN until enough bars have been observed")
