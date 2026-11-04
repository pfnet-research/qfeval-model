# %%
"""# Cookbook — マルチホライズン出力(無名の trailing 軸)

パネルの軸は時刻と銘柄の 2 つに固定されていて、出力はその後ろに任意個の
**無名の trailing 軸**を持てます。この例は複数ホライズンの過去リターンを
一度に予測します —— `(T, S)` の入力から `(T, S, H)` の出力を作ります。

事前の宣言は不要です: 出力の trailing shape は、呼び出しごとに最初の
`step.output` 代入で固定されます。trailing 軸はラベルを持たないので、
各位置の意味(ここではどのホライズンか)は利用者のコードが管理します。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
HORIZONS = (1, 2, 5)  # 遡るステップ数。trailing 軸の並びはこのタプルの順


@qm.model
def multi_horizon_return(ctx: qm.Context) -> None:
    """各ホライズンの過去リターン —— ステップごとに (S, H) を出力する。"""
    history = ctx.state.setdefault("history", [])
    for step in ctx:
        close = np.asarray(step.inputs["close"], dtype=np.float64)  # (S,)
        out = np.full((close.shape[0], len(HORIZONS)), np.nan)  # (S, H)
        for column, horizon in enumerate(HORIZONS):
            if len(history) >= horizon:
                out[:, column] = close / history[-horizon] - 1.0
        step.output = out
        history.append(close.copy())
        del history[: -max(HORIZONS)]


# %%
inputs = pd.DataFrame(
    {
        "AAA": [100.0, 102.0, 101.0, 105.0, 107.0, 104.0, 108.0, 110.0],
        "BBB": [50.0, 49.0, 51.0, 52.0, 50.0, 53.0, 54.0, 52.0],
    },
    index=pd.date_range("2024-01-01", periods=8),
)
pred = multi_horizon_return().predict(inputs)
print(f"array: {pred.array.shape}  # (T, symbol, horizon)")

# %%
# trailing 軸は無名なので、to_wide の columns は整数位置を使う
# (symbol, 位置) の MultiIndex(入力境界が受理する wide 形式と同じ)です。
# trailing 軸の DataFrame 表現はこれだけで、long 形式は 2 次元に限定されます。
# 位置と意味の対応(HORIZONS)は自分のコードで管理してください。
print(pred.to_wide().round(4).to_string())
print(pred.array[-1].round(4))  # 最後の timestamp の (S, H)

# %%
# verify のゲートは trailing 軸を持つモデルもそのまま検証できます。
report = qm.verify.check_all(
    multi_horizon_return,
    inputs,
    test_from=str(inputs.index[len(inputs) // 2].date()),
)
print(report)

# %%
# metrics は (T, S) 平面に意味が結び付いているため、trailing 軸付きの出力を
# 黙って平均せずに拒否します。select_trailing を使うと Features を直接構築せず、
# ホライズンを 1 つずつ選んで評価できます。
one_step = pred.select_trailing(HORIZONS.index(1))
print(f"1 ステップの hit rate: {qm.metrics.hit_rate(one_step, inputs):.3f}")
