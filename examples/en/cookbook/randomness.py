# %%
"""# Cookbook — randomness: `step.rng` and the determinism guarantee

`step.rng` is derived from the model seed and the step's timestamp, so
the same seed always reproduces the same run — regardless of how the calls
are chunked.
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def noisy_zero(ctx: qm.Context) -> None:
    for step in ctx:
        step.output = step.rng.normal(size=step.symbols.shape[0])


# %%
close = pd.DataFrame(
    [[1.0, 1.0], [1.0, 1.0]],
    index=pd.date_range("2024-01-01", periods=2),
    columns=["AAA", "BBB"],
)

# %%
a = noisy_zero().predict(close, seed=7).array
b = noisy_zero().predict(close, seed=7).array
c = noisy_zero().predict(close, seed=8).array
print("== three runs ==")
print(f"seed=7 twice identical: {(a == b).all()}")
print(f"seed=8 differs:         {not (a == c).all()}")
print("pass seed= to the first call; never seed global numpy state")
