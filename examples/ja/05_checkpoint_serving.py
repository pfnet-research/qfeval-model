# %%
"""# 05 — チェックポイントとサービングのライフサイクル

API では pickle と同じ用語を使います。ファイルには `save`/`load`、バイト列には
`dumps`/`loads` を使用します。保存のタイミングは*呼び出し側*が決めます。
モデル側でチェックポイントを拒否することはできないため、必要な情報はすべて、
各ステップの終了時点で `ctx.state` に保存されていなければなりません。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def ema_signal(ctx: qm.Context, alpha: float = 0.3) -> None:
    """close の EMA。保持する状態は ctx.state 内の配列 1 つだけ。"""
    for step in ctx:
        close = step.inputs["close"]
        prev = ctx.state.get("ema", close)
        current = alpha * close + (1.0 - alpha) * prev
        ctx.state["ema"] = current  # 各ステップの終了時点でチェックポイントを作成可能
        step.output = current


# %%
# ライブラリは、束縛パラメータやコードの fingerprint を意図的に作らない。
# そのどちらかを変更するたびに更新する構築バージョンを、ホスト側で
# checkpoint と一緒に保存する。
CONFIG_VERSION = "ema-signal-alpha-0.3-v1"


# %%
# ワイド形式のフレームをそのまま入力として使い、split で日付範囲ごとの
# ワイド形式({特徴量名: フレーム} の dict)に切り出す。
close = pd.DataFrame(
    [[100.0], [104.0], [96.0], [108.0], [112.0]],
    index=pd.date_range("2024-01-01", periods=5),
    columns=["AAA"],
)
day4, day5 = qm.split(close, test_from="2024-01-05")
print("== day5(qm.split で切り出したワイド形式)==")
print(day5["close"].to_string())

# %%
server = ema_signal(alpha=0.3)
server.predict(day4, seed=0)
saved = {"config_version": CONFIG_VERSION, "blob": server.dumps()}
print("== 4 日目終了時点のチェックポイント ==")
print(f"blob: {len(saved['blob'])} バイト")
print("保存のタイミングは呼び出し側が決め、モデルはいつでも保存できる状態を保つ")

# %%
# 別プロセスを想定し、同じファクトリからモデルを作成して、状態を復元後に処理を再開する。
# load はモデルの「同一性」——関数の qualified name とモード——を検査する。
# 束縛パラメータの比較や、同じ名前のまま関数本体を書き換えた場合の検出はしない:
# 同じパラメータでモデルを組み立て直すことと、checkpoint を作ったコードの
# バージョン管理はホスト側の責務。
if saved["config_version"] != CONFIG_VERSION:
    raise RuntimeError(f"古い checkpoint: {saved['config_version']}")
restored = ema_signal(alpha=0.3).loads(saved["blob"])
output = restored.predict(day5)
print("== 復元したインスタンスで 5 日目を予測 ==")
print(output.to_wide().round(3).to_string())

# %%
straight = ema_signal(alpha=0.3).predict(close, seed=0)
np.testing.assert_allclose(output.array[0], straight.array[-1], rtol=1e-12)
print("中断せずに実行した結果と完全に一致。状態はすべて state で引き継がれた")
