# %%
"""# 07 — the fixed symbol universe

The symbol axis is fixed by the first call: either declared up front with
`symbols=` or inferred from the first input. Later calls may omit symbols
(missing cells become NaN) but may never introduce new ones — an unknown
symbol raises `ValueError` instead of silently reshaping your outputs.
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
def day(date: str, closes: dict[str, float]) -> pd.DataFrame:
    """One day of closes as a one-row wide frame (index=timestamp, columns=symbol)."""
    return pd.DataFrame(closes, index=[pd.Timestamp(date)])


# %%
model = qm.zoo.ffill("close")  # as-of value: bridges missing rows

# Declare three symbols even though day 1 only has data for two.
first = model.predict(
    day("2024-01-01", {"AAA": 100.0, "BBB": 200.0}),
    symbols=("AAA", "BBB", "CCC"),
    seed=0,
)
print("== day 1: universe declared as (AAA, BBB, CCC) ==")
print(first.to_wide().to_string())
print("CCC has no data yet, so its cell is NaN on the fixed 3-symbol axis")

# %%
second = model.predict(day("2024-01-02", {"AAA": 101.0, "CCC": 50.0}))
print("== day 2: BBB missing, CCC appears ==")
print(second.to_wide().to_string())
print("ffill bridges BBB with day 1's close; the axis never changes shape")

# %%
try:
    model.predict(day("2024-01-03", {"AAA": 102.0, "ZZZ": 1.0}))
except ValueError as error:
    print("== day 3: unknown symbol ZZZ ==")
    print(f"ValueError: {error}")
    print("new symbols need a new model (or declare them on day 1)")
