# %%
"""# Cookbook — an online learner: recursive least squares with forgetting

A model that keeps learning inside `fit`: it regresses tomorrow-agnostic
targets (today's return) on yesterday's return, updating per-symbol RLS state
each train step. All learner state lives in `ctx.state`, so it checkpoints.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def rls_momentum(ctx: qm.Context, forgetting: float = 0.95) -> None:
    ctx.child = qm.zoo.simple_return()
    for step in ctx:
        r = step.child.array  # today's return (the target for yesterday's x)
        x = ctx.state.get("last_r")  # yesterday's return (the feature)
        beta = ctx.state.get("beta", np.zeros(r.shape))
        p = ctx.state.get("p", np.full(r.shape, 100.0))
        if step.is_train and x is not None:
            valid = ~np.isnan(x) & ~np.isnan(r)
            gain = p * x / (forgetting + x * p * x)
            beta = np.where(valid, beta + gain * (r - x * beta), beta)
            p = np.where(valid, (p - gain * x * p) / forgetting, p)
            ctx.state["beta"], ctx.state["p"] = beta, p
        ctx.state["last_r"] = r.copy()
        step.output = beta * np.nan_to_num(r)  # predict next return


# %%
days = pd.date_range("2024-01-01", periods=8)
close = pd.DataFrame(
    {"UP": 100.0 * 1.02 ** np.arange(8), "DOWN": 100.0 * 0.98 ** np.arange(8)},
    index=days,
)

# %%
model = rls_momentum()
output = model.run(close, test_from="2024-01-07", seed=0)
print("== RLS beta after fitting 6 days ==")
print({k: np.round(v, 3) for k, v in model.state.items() if k == "beta"})
print("positive betas: both series are (log-)trending, so momentum fits")
print("== test-day outputs ==")
print(output.to_wide().round(5).to_string())
