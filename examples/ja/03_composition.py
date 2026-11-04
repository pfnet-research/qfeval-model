# %%
"""# 03 — モデルの合成: `ctx.child`、名前付き子モデル、zoo

モデルから別のモデルを実行できます。子モデルは `ctx.children` dict に
名前を付けて入れることで登録されます(`ctx.children["fast"] = m`)。
`ctx.child = m` は `ctx.children["child"] = m` の短縮形です。登録は
常に明示的で、関数引数として渡しただけのモデルは登録されません。`qm.zoo`
には `lag`、`rolling_mean`、`simple_return` など、すぐに使える部品が用意されています。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def ma_crossover(ctx: qm.Context) -> None:
    """zoo の部品を使い、短期移動平均から長期移動平均を引く。"""
    ctx.children["fast"] = qm.zoo.rolling_mean("close", window=2)
    ctx.children["slow"] = qm.zoo.rolling_mean("close", window=4)
    for step in ctx:
        step.output = step.children["fast"].array - step.children["slow"].array


# %%
@qm.model
def smoothed(ctx: qm.Context, child: qm.Model, window: int = 3) -> None:
    """任意の子モデルの出力の移動平均。"""
    ctx.child = child
    for step in ctx:
        buffer = [*ctx.state.get("buffer", []), step.child.array.copy()][-window:]
        ctx.state["buffer"] = buffer
        step.output = np.mean(buffer, axis=0)


# %%
# フラットなワイド形式(index=timestamp、columns=symbol)がこのパッケージの
# 基本入力形式です。単一の特徴量 "close" として解釈され、下の zoo の子モデルが
# 要求する形式とちょうど一致します。
close = pd.DataFrame(
    [[100.0], [104.0], [96.0], [108.0], [112.0], [105.0]],
    index=pd.date_range("2024-01-01", periods=6),
    columns=["AAA"],
)

# %%
print("== ma_crossover(名前付きの zoo 子モデル)==")
print(ma_crossover().predict(close, seed=0).to_wide().to_string())
print("正の値 = 短期平均が長期平均を上回っている(上昇モメンタム)")

# %%
print("== smoothed(simple_return)(子モデルを引数として渡す)==")
model = smoothed(qm.zoo.simple_return(), window=2)
print(model.predict(close, seed=0).to_wide().round(4).to_string())
print("ctx.child = child で登録した名前 'child' を step.child で参照できる")
