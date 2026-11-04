# %%
"""# Cookbook — what-if analysis with `clone`

`model.clone()` snapshots a started model at its current stable point. Each
clone advances independently, so you can branch one fitted model into several
futures without re-fitting.
"""

# %%
import pandas as pd

import qfeval_model as qm

# %%
days = pd.date_range("2024-01-01", periods=4)
close = pd.DataFrame([[100.0], [104.0], [96.0], [108.0]], index=days, columns=["AAA"])
history, _ = qm.split(close, test_from="2024-01-04")  # {feature: wide frame}

# %%
base = qm.zoo.recursive_ewm_mean("close", alpha=0.5)
base.predict(history, seed=0)  # advance through the shared history

# %%
scenarios = {"crash": 80.0, "flat": 96.0, "rally": 120.0}
print("== EMA after each hypothetical day 4 ==")
for name, price in scenarios.items():
    branch = base.clone()
    frame = pd.DataFrame({"AAA": [price]}, index=[days[3]])
    value = branch.predict(frame).array[0, 0]
    print(f"{name:>6}: close={price:6.1f} -> ema={value:.2f}")
print("each branch starts from the same fitted state; `base` is untouched")
