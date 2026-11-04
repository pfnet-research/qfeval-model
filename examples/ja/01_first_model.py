# %%
"""# 01 — 最小構成のモデルと `Features` の読み方

モデルは `@qm.model` でデコレートした通常の関数です。時間ステップを順に処理し、
シンボルごとに 1 つの値を出力します。モデルのライフサイクル全体を通して固定される
設定(シンボルユニバース、シード)は、*最初の*
`fit` / `predict` / `run` 呼び出し時に引数で指定します。
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def echo_close(ctx: qm.Context) -> None:
    """当日の close をそのまま出力する恒等モデル。"""
    for step in ctx:
        step.output = step.inputs["close"]


# %%
# 手元のデータはワイド形式(index=timestamp, columns=symbol)です。そして
# パッケージはまさにその形式を受け取ります。フラットなワイド形式のフレームは、
# "close" という名前の単一の特徴量として扱われます。
close = pd.DataFrame(
    [[100.0, 200.0], [101.0, 202.0], [102.0, 198.0]],
    index=pd.date_range("2024-01-01", periods=3),
    columns=["AAA", "BBB"],
)
print("== close(ワイド形式)==")
print(close.to_string())
print("timestamp ごとに 1 行、symbol ごとに 1 列。そのまま predict に渡せる")

# %%
# セッション全体に適用する設定は、最初の呼び出し時に引数で指定します:
#   symbols=(シンボルユニバース)、seed=(乱数シード)。
model = echo_close()
output = model.predict(close, symbols=("AAA", "BBB"), seed=0)

print("== output.to_wide() ==")
print(output.to_wide().to_string())
print("宣言したシンボル軸に沿って、(timestamp, symbol) ごとに 1 つの値を出力")

# %%
# (timestamp, symbol) ごとに 1 行のロング形式(データ受け渡し用の形式)もあり、
# ファイルやデータベースとのやり取りに便利です。`to_frame()` でこの形式を表示でき、
# timestamp / symbol 列を持つロング形式の DataFrame は入力としても受け付けられます。
print("== output.to_frame().head(2) ==")
print(output.to_frame().head(2).to_string(index=False))
print("同じ出力をロング形式で表示。.array は未加工の (T, S) NumPy 配列を返す")

assert output.array.shape == (3, 2)
