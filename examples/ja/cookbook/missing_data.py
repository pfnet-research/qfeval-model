# %%
"""# Cookbook — 欠損データ: NaN の保持、`ffill`、モデル合成

`zoo.lag` は `shift` と同様に、欠損値を補完せずそのまま残します。
`zoo.ffill` は当日までの直近の有効値、`lag(ffill(source))` は `periods` 日前
までの直近の有効値(`ffill().shift()`)です。
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
print("== 欠損値の扱いを列ごとに比較 ==")
print(table.to_string())
print("lag は欠損をそのまま残す。lag_ffill は前日、ffill は当日までの有効値で補完")
