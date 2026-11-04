# %%
"""# 03 — composing models: `ctx.child`, named children, and the zoo

A model can drive other models. Children are registered by name in the
`ctx.children` dict (`ctx.children["fast"] = m`); `ctx.child = m` is
shorthand for `ctx.children["child"] = m`. Registration is always
explicit — a model passed as a function parameter is not registered until
you put it into the dict. `qm.zoo` provides ready-made building blocks
(`lag`, `rolling_mean`, `simple_return`...).
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def ma_crossover(ctx: qm.Context) -> None:
    """Fast moving average minus slow moving average, both borrowed from the zoo."""
    ctx.children["fast"] = qm.zoo.rolling_mean("close", window=2)
    ctx.children["slow"] = qm.zoo.rolling_mean("close", window=4)
    for step in ctx:
        step.output = step.children["fast"].array - step.children["slow"].array


# %%
@qm.model
def smoothed(ctx: qm.Context, child: qm.Model, window: int = 3) -> None:
    """Rolling mean of any child model's output."""
    ctx.child = child
    for step in ctx:
        buffer = [*ctx.state.get("buffer", []), step.child.array.copy()][-window:]
        ctx.state["buffer"] = buffer
        step.output = np.mean(buffer, axis=0)


# %%
# A flat wide frame (index=timestamp, columns=symbol) is the package's basic
# input format; it is read as the single feature "close" — exactly what the
# zoo children below ask for.
close = pd.DataFrame(
    [[100.0], [104.0], [96.0], [108.0], [112.0], [105.0]],
    index=pd.date_range("2024-01-01", periods=6),
    columns=["AAA"],
)

# %%
print("== ma_crossover (named zoo children) ==")
print(ma_crossover().predict(close, seed=0).to_wide().to_string())
print("positive = short-term average above long-term (upward momentum)")

# %%
print("== smoothed(simple_return) (child passed as a parameter) ==")
model = smoothed(qm.zoo.simple_return(), window=2)
print(model.predict(close, seed=0).to_wide().round(4).to_string())
print("ctx.child = child registered the name 'child', read back as step.child")
