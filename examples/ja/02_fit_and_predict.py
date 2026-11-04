# %%
"""# 02 — `fit` / `predict` と学習・テストフェーズ

`fit` は学習ステップを、`predict` はテストステップをモデルに渡します。
どちらも同じモデル関数を使用し、`step.is_train` / `step.is_test` でフェーズを
判別します。学習したパラメータは `ctx.state` に保存します。この状態は呼び出しを
またいで保持され、あとから `model.state` で参照できます。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def symbol_mean(ctx: qm.Context) -> None:
    """`fit` 中に各シンボルの close の平均を学習し、その平均値を予測する。"""
    for step in ctx:
        close = step.inputs["close"]
        if step.is_train:
            ctx.state["total"] = ctx.state.get("total", 0.0) + close
            ctx.state["count"] = ctx.state.get("count", 0) + 1
        if ctx.state.get("count", 0) > 0:
            step.output = ctx.state["total"] / ctx.state["count"]
        else:
            step.output = np.full(close.shape, np.nan)


# %%
# データは終始ワイド形式のままです。qm.split は timestamp 軸を test_from で分割し、
# fit / predict にそのまま渡せる {特徴量名: ワイド形式フレーム} の dict を
# train / test それぞれについて返します。
close = pd.DataFrame(
    [[100.0, 200.0], [110.0, 190.0], [120.0, 180.0], [999.0, 999.0]],
    index=pd.date_range("2024-01-01", periods=4),
    columns=["AAA", "BBB"],
)
train, test = qm.split(close, test_from="2024-01-04")
print("== train['close'](ワイド形式)==")
print(train["close"].to_string())
print("split は {特徴量名: ワイド形式フレーム} の dict を test_from で切って返す")

# %%
model = symbol_mean()
model.fit(train, seed=0)
print("== fit 完了後の学習済み state ==")
print({k: np.round(v, 2) for k, v in model.state.items()})
print("ctx.state は fit の完了後も保持され、model.state で参照できる")

# %%
output = model.predict(test)
print("== テスト日の予測 ==")
print(output.to_wide().to_string())
print("テスト日の極端な close(999)は、テストフェーズなので学習には使われない")

np.testing.assert_allclose(output.array[0], [110.0, 190.0])
