# %%
"""# Cookbook — multi-horizon outputs (anonymous trailing axes)

A input has two fixed axes — timestamp and symbol — and outputs may carry
any number of **anonymous trailing axes** beyond them. This example predicts
backward returns over several horizons at once: a `(T, S)` input producing a
`(T, S, H)` output.

Nothing needs to be declared up front: the output's trailing shape is locked
by the first `step.output` assignment of each call. The trailing axis
carries no labels — what each position means (here: which horizon) lives in
your code.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
HORIZONS = (1, 2, 5)  # lookback steps; the trailing axis follows this order


@qm.model
def multi_horizon_return(ctx: qm.Context) -> None:
    """Backward return over each horizon — a (S, H) output per step."""
    history = ctx.state.setdefault("history", [])
    for step in ctx:
        close = np.asarray(step.inputs["close"], dtype=np.float64)  # (S,)
        out = np.full((close.shape[0], len(HORIZONS)), np.nan)  # (S, H)
        for column, horizon in enumerate(HORIZONS):
            if len(history) >= horizon:
                out[:, column] = close / history[-horizon] - 1.0
        step.output = out
        history.append(close.copy())
        del history[: -max(HORIZONS)]


# %%
inputs = pd.DataFrame(
    {
        "AAA": [100.0, 102.0, 101.0, 105.0, 107.0, 104.0, 108.0, 110.0],
        "BBB": [50.0, 49.0, 51.0, 52.0, 50.0, 53.0, 54.0, 52.0],
    },
    index=pd.date_range("2024-01-01", periods=8),
)
pred = multi_horizon_return().predict(inputs)
print(f"array: {pred.array.shape}  # (T, symbol, horizon)")

# %%
# The trailing axis is anonymous, so to_wide's columns use integer positions
# in a MultiIndex (symbol, position) — the same layout the input boundary
# accepts. This is the only DataFrame rendering for trailing axes; long format
# stays 2-D. Your own axis legend (HORIZONS) maps positions back to meanings.
print(pred.to_wide().round(4).to_string())
print(pred.array[-1].round(4))  # last timestamp, (S, H)

# %%
# The verify gates accept trailing-axis models unchanged.
report = qm.verify.check_all(
    multi_horizon_return,
    inputs,
    test_from=str(inputs.index[len(inputs) // 2].date()),
)
print(report)

# %%
# Metrics are defined over the (T, S) plane and refuse trailing axes rather
# than silently averaging them. select_trailing is the supported path for
# evaluating one horizon at a time without constructing Features directly.
one_step = pred.select_trailing(HORIZONS.index(1))
print(f"1-step hit rate: {qm.metrics.hit_rate(one_step, inputs):.3f}")
