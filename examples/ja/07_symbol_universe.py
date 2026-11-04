# %%
"""# 07 — シンボルユニバースの固定

シンボル軸は最初の呼び出しで固定されます。`symbols=` であらかじめ宣言するか、
指定しなければ最初の入力から推定されます。以降の呼び出しでは、一部のシンボルが
入力になくても構いません(対応するセルは NaN になります)。ただし、新しい
シンボルを追加することはできません。出力の形状が暗黙に変わる代わりに、
`ValueError` が送出されます。
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
def day(date: str, closes: dict[str, float]) -> pd.DataFrame:
    """1 日分の close を 1 行のワイド形式(index=timestamp, columns=symbol)で返す。"""
    return pd.DataFrame(closes, index=[pd.Timestamp(date)])


# %%
model = qm.zoo.ffill("close")  # その時点で利用可能な直近の値で欠損行を埋める

# 1 日目のデータは 2 シンボル分しかないが、3 シンボルを宣言しておく。
first = model.predict(
    day("2024-01-01", {"AAA": 100.0, "BBB": 200.0}),
    symbols=("AAA", "BBB", "CCC"),
    seed=0,
)
print("== 1 日目: ユニバースを (AAA, BBB, CCC) と宣言 ==")
print(first.to_wide().to_string())
print("CCC はまだデータがないので、3 シンボルの固定軸上では該当セルが NaN になる")

# %%
second = model.predict(day("2024-01-02", {"AAA": 101.0, "CCC": 50.0}))
print("== 2 日目: BBB が欠け、CCC が現れる ==")
print(second.to_wide().to_string())
print("ffill により BBB は 1 日目の close で補完される。軸の形状は変わらない")

# %%
try:
    model.predict(day("2024-01-03", {"AAA": 102.0, "ZZZ": 1.0}))
except ValueError as error:
    print("== 3 日目: 未知のシンボル ZZZ ==")
    print(f"ValueError: {error}")
    print("新しいシンボルは、新しいモデルを使うか 1 日目に宣言しておく必要がある")
