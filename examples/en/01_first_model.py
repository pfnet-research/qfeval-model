# %%
"""# 01 — the smallest model, and how to read a `Features`

A model is a plain function decorated with `@qm.model`: it iterates over
time steps and writes one output per symbol. Everything that is fixed for the
model's lifetime — the symbol universe, the seed — is an argument of the
*first* `fit` / `predict` / `run` call.
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def echo_close(ctx: qm.Context) -> None:
    """Output today's close unchanged — the identity model."""
    for step in ctx:
        step.output = step.inputs["close"]


# %%
# Your data lives wide (index=timestamp, columns=symbol) — and that is
# exactly what the package takes: a flat wide frame is treated as a single
# feature named "close".
close = pd.DataFrame(
    [[100.0, 200.0], [101.0, 202.0], [102.0, 198.0]],
    index=pd.date_range("2024-01-01", periods=3),
    columns=["AAA", "BBB"],
)
print("== close (wide format) ==")
print(close.to_string())
print("one row per timestamp, one column per symbol; passed to predict as-is")

# %%
# Model-lifetime settings are arguments of the first call:
#   symbols= (the universe), seed= (randomness).
model = echo_close()
output = model.predict(close, symbols=("AAA", "BBB"), seed=0)

print("== output.to_wide() ==")
print(output.to_wide().to_string())
print("one output per (timestamp, symbol), on the declared symbol axis")

# %%
# There is also a long *transport* format — one row per (timestamp, symbol) —
# handy for files and databases. `to_frame()` renders it, and long DataFrames
# with timestamp/symbol columns are accepted as input too.
print("== output.to_frame().head(2) ==")
print(output.to_frame().head(2).to_string(index=False))
print("the same outputs in long format; .array gives the raw (T, S) numpy")

assert output.array.shape == (3, 2)
