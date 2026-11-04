# %%
"""# Cookbook — オンライン学習器: 忘却係数付き再帰的最小二乗法(RLS)

`fit` 中に継続して学習するモデルです。未来の情報を使わずに計算できる当日の
リターンを目的変数、前日のリターンを説明変数として回帰し、学習ステップごとに
シンボル別の RLS state を更新します。学習器の state はすべて `ctx.state` に
保存されるため、そのままチェックポイントを作成できます。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def rls_momentum(ctx: qm.Context, forgetting: float = 0.95) -> None:
    ctx.child = qm.zoo.simple_return()
    for step in ctx:
        r = step.child.array  # 当日のリターン(前日の x に対応する目的変数)
        x = ctx.state.get("last_r")  # 前日のリターン(説明変数)
        beta = ctx.state.get("beta", np.zeros(r.shape))
        p = ctx.state.get("p", np.full(r.shape, 100.0))
        if step.is_train and x is not None:
            valid = ~np.isnan(x) & ~np.isnan(r)
            gain = p * x / (forgetting + x * p * x)
            beta = np.where(valid, beta + gain * (r - x * beta), beta)
            p = np.where(valid, (p - gain * x * p) / forgetting, p)
            ctx.state["beta"], ctx.state["p"] = beta, p
        ctx.state["last_r"] = r.copy()
        step.output = beta * np.nan_to_num(r)  # 次の時点のリターンを予測


# %%
days = pd.date_range("2024-01-01", periods=8)
close = pd.DataFrame(
    {"UP": 100.0 * 1.02 ** np.arange(8), "DOWN": 100.0 * 0.98 ** np.arange(8)},
    index=days,
)

# %%
model = rls_momentum()
output = model.run(close, test_from="2024-01-07", seed=0)
print("== 6 日間の学習後の RLS beta ==")
print({k: np.round(v, 3) for k, v in model.state.items() if k == "beta"})
print("beta は正: どちらの系列も対数価格にトレンドがあるため、モメンタムが有効")
print("== テスト期間の予測 ==")
print(output.to_wide().round(5).to_string())
