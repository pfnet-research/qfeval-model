# %%
"""# 04 — `Model.run`、`qm.split`、`qm.metrics` によるバックテスト

`model.run` は `test_from` より前をすべて fit、`test_from` 以降を predict
して test 期間の `Features` を返します。`test_from` は `range(start, stop)`
の `stop` と同様に学習範囲には含まれず、テスト側に含まれます。他の呼び出しと
同じくモデル自身が前進します(元を残したい場合は clone してから呼ぶ)。
評価指標に必要なフォワードリターンは `qm.metrics` が内部で
計算するため、評価用の情報がモデル側にリークすることはありません。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# AAA は上昇、BBB は下降、CCC は上下を繰り返す、再現可能なサンプルデータ。
# ワイド形式(index=timestamp、columns=symbol)のまま split、run、metrics に
# 渡せます。単一の特徴量 "close" として解釈されます。
days = pd.date_range("2024-01-01", periods=10)
close = pd.DataFrame(
    {
        "AAA": 100.0 * 1.01 ** np.arange(10),
        "BBB": 100.0 * 0.99 ** np.arange(10),
        "CCC": 100.0 + np.tile([0.0, 1.0], 5),
    },
    index=days,
)

# %%
# split はワイド形式の dict に正規化する: フラットなワイド形式は
# {'close': フレーム} になる。
train, test = qm.split(close, test_from="2024-01-06")
print("== qm.split(close, test_from='2024-01-06') — train(ワイド形式)==")
print(train["close"].round(2).to_string())
print("== test(ワイド形式)==")
print(test["close"].round(2).to_string())
print("train は test_from の直前まで、test は test_from から始まる")

# %%
# モメンタム: zoo の部品で構成した前日のリターン。run() は `model` 自身を前進させる。
model = qm.zoo.lag(qm.zoo.simple_return(), periods=1)
output = model.run(close, test_from="2024-01-06", seed=0)
print("== 予測(1 日遅れのリターンをモメンタムシグナルとして使用)==")
print(output.to_wide().round(4).to_string())

# %%
ic = qm.metrics.information_coefficient(output, close)
print("== test 日ごとの information coefficient ==")
print(ic.round(3).to_string())
print(f"平均 IC {ic.mean():+.3f}。モメンタムが翌日のリターンを順位付けできている")

# %%
hit = qm.metrics.hit_rate(output, close)
pnl = qm.metrics.long_short_pnl(output, close)
print("== hit rate とロング・ショート PnL ==")
print(f"hit_rate: {hit:.2f}, 累積 pnl: {pnl.sum():+.4f}")
print("metrics がフォワードリターンを内部で計算するので shift(-1) を書く必要はない")
