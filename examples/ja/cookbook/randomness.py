# %%
"""# Cookbook — 乱数: `step.rng` と決定性の保証

`step.rng` はセッションのシードとそのステップのタイムスタンプから導出
されるため、同じシードを使えば常に同じ実行結果を再現できます。呼び出しを
どのように分割しても結果は変わりません。
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
print("== 3 回の実行結果 ==")
print(f"seed=7 での 2 回の実行は一致: {(a == b).all()}")
print(f"seed=8 では異なる:             {not (a == c).all()}")
print("seed= は最初の呼び出しで指定する。NumPy のグローバル乱数には設定しない")
