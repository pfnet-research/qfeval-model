# %%
"""# 06 — 検証ゲート: `verify.check_all`

1 回の呼び出しで、適用可能なすべての検証項目(決定性、ストリーミングの一貫性、
チェックポイント復元の一貫性、バッチモデルの場合はバッチ等価性)を実行し、
レポートを返します。本番投入の可否は `report.all_passed` で判断してください。
`ok` と異なり、検証項目が skip された場合も投入を止められます。
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.batch_model
def leaky(ctx: qm.Context) -> None:
    """意図的に誤った実装: チャンク全体の平均を引く(チャンク内の先読み)。"""
    for step in ctx:
        close = step.inputs["close"]
        step.output = close - close.mean(axis=0)  # 未来の行を参照している!


# %%
# ワイド形式のデータは、変換せずにそのまま検証器に渡せます。
close = pd.DataFrame(
    [[100.0, 200.0], [104.0, 196.0], [96.0, 204.0], [108.0, 192.0]],
    index=pd.date_range("2024-01-01", periods=4),
    columns=["AAA", "BBB"],
)

# %%
report = qm.verify.check_all(qm.zoo.simple_return, close, test_from="2024-01-02")
print("== check_all(qm.zoo.simple_return) ==")
print(report)
print(f"all passed: {report.all_passed} — 本番に投入しても問題ない")
assert report.all_passed

# %%
report = qm.verify.check_all(leaky, close)
print("== check_all(leaky バッチモデル) ==")
print("\n".join(line.split(" — ")[0] for line in str(report).splitlines()))
print(f"all passed: {report.all_passed} — バッチ検証でチャンク内の先読みを検出した")
assert not report.all_passed
