# %%
"""# Cookbook — バッチモデル(上級 API)とリーク対策

`@qm.batch_model` は、1 回の呼び出しで受け取るデータ全体を 1 つの `(T, S)`
チャンクとして渡します。ステップ単位の Python 処理が遅すぎる場合(例: 深層学習)
に有用です。その代わり、チャンク内の未来を参照しないよう、実装者が責任を持つ
必要があります。`verify.assert_stream_consistent` は、
チャンクサイズ 1 の実行と比較してリークを検出します。ただし、これは限定的な検査で
あり、安全性を証明するものではありません。バッチモデルは上級者向けの機能です。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.batch_model
def vectorized_return(ctx: qm.Context) -> None:
    """各出力行の計算には、その行以前のデータだけを使用する。"""
    for step in ctx:
        close = step.inputs["close"]
        last = ctx.state.get("last_close", np.full(close.shape[1], np.nan))
        prev = np.vstack([last[None, :], close[:-1]])
        ctx.state["last_close"] = close[-1].copy()
        step.output = close / prev - 1.0


# %%
close = pd.DataFrame(
    [[100.0, 200.0], [110.0, 190.0], [121.0, 180.0]],
    index=pd.date_range("2024-01-01", periods=3),
    columns=["AAA", "BBB"],
)

print("== バッチモデルの出力 ==")
print(vectorized_return().predict(close, seed=0).to_wide().round(3).to_string())

# %%
qm.verify.assert_stream_consistent(vectorized_return, close)
print("== assert_stream_consistent ==")
print("合格: バッチ出力が、1 タイムスタンプずつ処理した場合の出力と一致した")
