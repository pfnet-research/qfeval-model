# %%
"""# Cookbook — `clone` による what-if 分析

`model.clone()` は、実行を開始したモデルをその時点の安定した状態のまま
複製します。各 clone は独立して状態を更新するため、再学習することなく、
1 つの学習済みモデルから複数のシナリオに分岐できます。
"""

# %%
import pandas as pd

import qfeval_model as qm

# %%
days = pd.date_range("2024-01-01", periods=4)
close = pd.DataFrame([[100.0], [104.0], [96.0], [108.0]], index=days, columns=["AAA"])
history, _ = qm.split(close, test_from="2024-01-04")  # {特徴量名: ワイドフレーム}

# %%
base = qm.zoo.recursive_ewm_mean("close", alpha=0.5)
base.predict(history, seed=0)  # 共通の履歴を処理し、あらかじめ state を更新しておく

# %%
scenarios = {"crash": 80.0, "flat": 96.0, "rally": 120.0}
print("== 4 日目を各シナリオとした場合の EMA ==")
for name, price in scenarios.items():
    branch = base.clone()
    frame = pd.DataFrame({"AAA": [price]}, index=[days[3]])
    value = branch.predict(frame).array[0, 0]
    print(f"{name:>6}: close={price:6.1f} -> ema={value:.2f}")
print("各分岐は同じ更新済み state から始まり、`base` の状態は変わらない")
