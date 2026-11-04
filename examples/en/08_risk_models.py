# %%
"""# 08 — historical VaR, expected shortfall, and drawdowns

The zoo's risk entries are backward-looking statistics with explicit sign
conventions: returns are positive for profit, VaR / expected shortfall
report losses as positive numbers, and drawdowns read a positive wealth
or price level. Nothing is annualized, clamped, or aggregated across the
portfolio — those decisions stay in your models.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# A deterministic price path with a visible crash and recovery.
days = pd.date_range("2024-01-01", periods=12)
close = pd.DataFrame(
    {
        "AAA": [
            100.0,
            103.0,
            101.0,
            106.0,
            99.0,
            92.0,
            95.0,
            98.0,
            103.0,
            105.0,
            104.0,
            108.0,
        ],
        "BBB": [50.0, 50.5, 51.5, 51.0, 52.0, 52.5, 51.5, 52.0, 53.0, 53.5, 54.0, 54.5],
    },
    index=days,
)

# %%
# Historical VaR and expected shortfall at 75% over an 8-step window.
# Positive output = loss. confidence is a fraction in (0, 1) — never 95.
returns = qm.zoo.simple_return()
var = qm.zoo.rolling_historical_value_at_risk(returns, window=8, confidence=0.75)
es = qm.zoo.rolling_historical_expected_shortfall(returns, window=8, confidence=0.75)
var_out = var.predict(close, seed=0).to_wide()
es_out = es.predict(close, seed=0).to_wide()
print("== 8-step historical VaR at 75% (positive = loss) ==")
print(var_out.round(4).to_string())
print("== 8-step expected shortfall at 75% ==")
print(es_out.round(4).to_string())
dominates = ((es_out >= var_out - 1e-12) | var_out.isna()).all().all()
print("ES >= VaR holds cellwise:", bool(dominates))
print("BBB rarely loses, so its VaR can be negative — no clamping to zero")

# %%
# Drawdowns read the price level itself, not returns.
dd = qm.zoo.drawdown("close").predict(close, seed=0).to_wide()
duration = qm.zoo.drawdown_duration("close").predict(close, seed=0).to_wide()
print("== drawdown 1 - close / running_max ==")
print(dd.round(4).to_string())
print("== steps since the running peak was last matched ==")
print(duration.to_string())
print("AAA's crash shows as a deep drawdown; the peak match resets it to 0")

# %%
# Downside deviation: sqrt of the order-2 lower partial moment.
downside = qm.zoo.rolling_downside_deviation(returns, window=8)
lpm = qm.zoo.rolling_lower_partial_moment(returns, window=8, order=2)
downside_out = downside.predict(close, seed=0).array
lpm_out = lpm.predict(close, seed=0).array
print("== rolling_downside_deviation == sqrt(order-2 lower partial moment) ==")
print(bool(np.allclose(downside_out, np.sqrt(lpm_out), equal_nan=True)))
print("a semivariance alias is unnecessary — it is exactly the order-2 LPM")

# %%
# Annualized volatility needs an explicit return series and frequency.
vol = qm.zoo.annualized_historical_volatility(
    qm.zoo.log_return(), window=8, periods_per_year=252
)
print("== annualized historical volatility (log returns, 252 periods/yr) ==")
print(vol.predict(close, seed=0).to_wide().round(4).to_string())
print("periods_per_year is required — bar frequency is never assumed")
