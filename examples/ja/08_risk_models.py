# %%
"""# 08 — ヒストリカル VaR、期待ショートフォール、ドローダウン

zoo のリスク統計は、符号規約を明示した後方参照の統計量です。リターンは
利益が正、VaR / 期待ショートフォールは損失を正の数として報告し、
ドローダウンは正の資産額・価格水準を読みます。年率化・クランプ・
ポートフォリオ集約は行いません — それらはあなたのモデル側の判断です。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# 急落と回復が見える決定的な価格パス。
days = pd.date_range("2024-01-01", periods=12)
close = pd.DataFrame(
    {
        "AAA": [
            100.0,
            103.0,
            101.0,
            106.0,
            99.0,
            92.0,
            95.0,
            98.0,
            103.0,
            105.0,
            104.0,
            108.0,
        ],
        "BBB": [50.0, 50.5, 51.5, 51.0, 52.0, 52.5, 51.5, 52.0, 53.0, 53.5, 54.0, 54.5],
    },
    index=days,
)

# %%
# 8 ステップ窓・信頼水準 75% のヒストリカル VaR と期待ショートフォール。
# 正の出力 = 損失。confidence は (0, 1) の小数 — 95 ではない。
returns = qm.zoo.simple_return()
var = qm.zoo.rolling_historical_value_at_risk(returns, window=8, confidence=0.75)
es = qm.zoo.rolling_historical_expected_shortfall(returns, window=8, confidence=0.75)
var_out = var.predict(close, seed=0).to_wide()
es_out = es.predict(close, seed=0).to_wide()
print("== 8 ステップ・75% のヒストリカル VaR(正 = 損失) ==")
print(var_out.round(4).to_string())
print("== 8 ステップ・75% の期待ショートフォール ==")
print(es_out.round(4).to_string())
dominates = ((es_out >= var_out - 1e-12) | var_out.isna()).all().all()
print("ES >= VaR が全セルで成立:", bool(dominates))
print("BBB はほとんど損しないので VaR は負になりうる — 0 に切り上げない")

# %%
# ドローダウンはリターンではなく価格水準そのものを読む。
dd = qm.zoo.drawdown("close").predict(close, seed=0).to_wide()
duration = qm.zoo.drawdown_duration("close").predict(close, seed=0).to_wide()
print("== ドローダウン 1 - close / running_max ==")
print(dd.round(4).to_string())
print("== 直近ピークに並んでからのステップ数 ==")
print(duration.to_string())
print("AAA の急落は深いドローダウンとして現れ、ピーク更新で 0 に戻る")

# %%
# 下方偏差: 2 次 lower partial moment の平方根。
downside = qm.zoo.rolling_downside_deviation(returns, window=8)
lpm = qm.zoo.rolling_lower_partial_moment(returns, window=8, order=2)
downside_out = downside.predict(close, seed=0).array
lpm_out = lpm.predict(close, seed=0).array
print("== rolling_downside_deviation == sqrt(2 次 lower partial moment) ==")
print(bool(np.allclose(downside_out, np.sqrt(lpm_out), equal_nan=True)))
print("semivariance の別名は不要 — 2 次 LPM と完全に同じだから")

# %%
# 年率ボラティリティにはリターン系列と頻度の明示が必要。
vol = qm.zoo.annualized_historical_volatility(
    qm.zoo.log_return(), window=8, periods_per_year=252
)
print("== 年率ヒストリカル・ボラティリティ(log リターン、年 252 期間) ==")
print(vol.predict(close, seed=0).to_wide().round(4).to_string())
print("periods_per_year は必須 — バー頻度を勝手に仮定しない")
