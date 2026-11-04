# %%
"""# 04 — backtesting with `Model.run`, `qm.split`, and `qm.metrics`

`model.run` fits everything before `test_from` (inclusive start of the
test side, like `range(start, stop)`), predicts the rest, and returns the
test-period `Features`. It advances the model like any other call —
clone first if you want to keep the original fresh. `qm.metrics` owns
forward-return computation, so evaluation can never leak into the model.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# AAA trends up, BBB trends down, CCC wobbles — deterministic toy data.
# The wide frame (index=timestamp, columns=symbol) is passed to split, run,
# and metrics as-is; it is read as the single feature "close".
days = pd.date_range("2024-01-01", periods=10)
close = pd.DataFrame(
    {
        "AAA": 100.0 * 1.01 ** np.arange(10),
        "BBB": 100.0 * 0.99 ** np.arange(10),
        "CCC": 100.0 + np.tile([0.0, 1.0], 5),
    },
    index=days,
)

# %%
# split normalizes to the wide dict: a flat wide frame becomes {'close': frame}.
train, test = qm.split(close, test_from="2024-01-06")
print("== qm.split(close, test_from='2024-01-06') — train (wide) ==")
print(train["close"].round(2).to_string())
print("== test (wide) ==")
print(test["close"].round(2).to_string())
print("train is the half-open interval before test_from; test starts at it")

# %%
# Momentum: yesterday's return, per the zoo. run() advances `model` itself.
model = qm.zoo.lag(qm.zoo.simple_return(), periods=1)
output = model.run(close, test_from="2024-01-06", seed=0)
print("== output (lagged 1-day return as a momentum signal) ==")
print(output.to_wide().round(4).to_string())

# %%
ic = qm.metrics.information_coefficient(output, close)
print("== information coefficient per test day ==")
print(ic.round(3).to_string())
print(f"mean IC {ic.mean():+.3f}; momentum ranks tomorrow's returns well here")

# %%
hit = qm.metrics.hit_rate(output, close)
pnl = qm.metrics.long_short_pnl(output, close)
print("== hit rate and long-short PnL ==")
print(f"hit_rate: {hit:.2f}, cumulative pnl: {pnl.sum():+.4f}")
print("metrics computes forward returns internally — never write shift(-1)")
