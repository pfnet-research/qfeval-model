# %%
"""# Cookbook — 補助チャネル

タイムスタンプごとの(シンボルごとではない)入力は `side_inputs` として一緒に
渡せます。数値カラムは数値配列のまま、object 型カラムは任意の Python オブジェクト
(文字列・dict など)を運びます。モデルから追加情報を出力するには、
`step.side_outputs`(配列)または `step.info`(call 全体で共有される 1 つの自由形式
dict。`output.info` として読み返せる)を使います。
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def risk_scaled(ctx: qm.Context) -> None:
    for step in ctx:
        scale = (
            0.0 if step.side_inputs["regime"] == "panic" else step.side_inputs["vix"]
        )
        step.output = step.inputs["close"] * 0.0 + scale
        step.side_outputs["raw_vix"] = step.side_inputs["vix"]
        # info は call 全体で共有される 1 つの dict。timestamp をキーにして記録する。
        step.info[str(pd.Timestamp(step.timestamp).date())] = step.side_inputs["regime"]


# %%
# メインの入力はワイドのまま。side_inputs は timestamp ごとのロング形式で、数値の
# `vix` と object 型の `regime` カラムを 1 つのフレームに含める。
days = pd.date_range("2024-01-01", periods=2)
close = pd.DataFrame([[100.0], [101.0]], index=days, columns=["AAA"])
side = pd.DataFrame(
    {"timestamp": days, "vix": [15.0, 80.0], "regime": ["calm", "panic"]}
)

# %%
output = risk_scaled().predict(close, side_data=side, seed=0)
print("== 補助チャネルでスケーリングした出力 ==")
print(output.to_wide().to_string())
print(
    "2 日目は 0。regime チャネルが 'panic' のため、vix によるスケーリングが無効になった"
)

# %%
print("== info を読み返す(call 全体で 1 つの dict)==")
for day, regime in output.info.items():
    print(day, regime)
print("side_outputs には配列を、info には自由形式の診断情報 dict を格納する")
