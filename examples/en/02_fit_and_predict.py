# %%
"""# 02 — `fit` / `predict` and the train/test phases

`fit` supplies train steps, `predict` supplies test steps; the model
function is the same in both, and `step.is_train` / `step.is_test` tell
the phases apart. Learned parameters live in `ctx.state`, which persists
across calls and is visible afterwards as `model.state`.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def symbol_mean(ctx: qm.Context) -> None:
    """Learn each symbol's mean close during fit; predict that mean."""
    for step in ctx:
        close = step.inputs["close"]
        if step.is_train:
            ctx.state["total"] = ctx.state.get("total", 0.0) + close
            ctx.state["count"] = ctx.state.get("count", 0) + 1
        if ctx.state.get("count", 0) > 0:
            step.output = ctx.state["total"] / ctx.state["count"]
        else:
            step.output = np.full(close.shape, np.nan)


# %%
# Data stays wide throughout: qm.split cuts the timestamp axis at test_from
# and returns the normalized {feature: wide frame} dict per side, ready for
# fit/predict.
close = pd.DataFrame(
    [[100.0, 200.0], [110.0, 190.0], [120.0, 180.0], [999.0, 999.0]],
    index=pd.date_range("2024-01-01", periods=4),
    columns=["AAA", "BBB"],
)
train, test = qm.split(close, test_from="2024-01-04")
print("== train['close'] (wide) ==")
print(train["close"].to_string())
print("split returns {feature: wide frame} dicts, cut at test_from")

# %%
model = symbol_mean()
model.fit(train, seed=0)
print("== learned state after fit ==")
print({k: np.round(v, 2) for k, v in model.state.items()})
print("ctx.state persisted across the fit call; model.state reads it back")

# %%
output = model.predict(test)
print("== output on the test day ==")
print(output.to_wide().to_string())
print("the test day's crazy close (999) is ignored: the phase was 'test'")

np.testing.assert_allclose(output.array[0], [110.0, 190.0])
