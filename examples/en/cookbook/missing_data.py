# %%
"""# Cookbook — missing data: NaN propagation, `ffill`, and composition

`zoo.lag` propagates gaps as-is (like `shift`); `zoo.ffill` is the as-of
value including today; `lag(ffill(source))` is the last valid value up to
`periods` ago (`ffill().shift()`).
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
close = pd.DataFrame(
    [[100.0], [np.nan], [np.nan], [104.0]],
    index=pd.date_range("2024-01-01", periods=4),
    columns=["AAA"],
)

# %%
table = pd.DataFrame(
    {
        "close": close["AAA"].to_numpy(),
        "lag": qm.zoo.lag("close").predict(close).array[:, 0],
        "lag_ffill": qm.zoo.lag(qm.zoo.ffill("close")).predict(close).array[:, 0],
        "ffill": qm.zoo.ffill("close").predict(close).array[:, 0],
    },
    index=close.index,
)
print("== gap handling, column by column ==")
print(table.to_string())
print("lag keeps the gaps; lag_ffill bridges yesterday; ffill includes today")
