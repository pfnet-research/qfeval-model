# %%
"""# 09 — 1 つの OHLCV データで見るテクニカル指標

各指標ラインは独立した zoo ファクトリです — ボリンジャーの上限バンド、
ストキャスティクスの %K、一目均衡表の転換線。どのラインも他モデルの
`source` にできます。指標は市場慣行のスケールを保ち(RSI と
ストキャスティクスは 0-100)、売買判定は出力しません:
閾値判断はあなたのモデルの仕事です。
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm

# %%
# 1 銘柄・16 本の決定的な OHLCV パネル。
days = pd.date_range("2024-01-01", periods=16)
base = 100.0 + np.cumsum(
    [0.0, 1.5, 1.0, -0.5, 2.0, 1.0, -3.0, -2.0, 1.0, 2.5, 1.5, -1.0, 0.5, 2.0, 1.0, 0.5]
)
inputs = {
    "open": pd.DataFrame({"AAA": base + 0.2}, index=days),
    "high": pd.DataFrame({"AAA": base + 1.0}, index=days),
    "low": pd.DataFrame({"AAA": base - 1.0}, index=days),
    "close": pd.DataFrame({"AAA": base + 0.5}, index=days),
    "volume": pd.DataFrame({"AAA": 1000.0 + 10.0 * base}, index=days),
}

# %%
# ボリンジャーバンド: ラインごとに 1 ファクトリ、既定は標本標準偏差 (ddof=1)。
upper = qm.zoo.bollinger_upper_band("close", window=5).predict(inputs, seed=0)
lower = qm.zoo.bollinger_lower_band("close", window=5).predict(inputs, seed=0)
percent_b = qm.zoo.bollinger_percent_b("close", window=5).predict(inputs, seed=0)
bands = pd.DataFrame(
    {
        "close": inputs["close"]["AAA"],
        "upper": upper.to_wide()["AAA"],
        "lower": lower.to_wide()["AAA"],
        "percent_b": percent_b.to_wide()["AAA"],
    }
)
print("== ボリンジャーバンド(窓 5、2 標準偏差) ==")
print(bands.round(3).to_string())
print("percent_b は割合: 中央バンドで 0.5、バンド外でもクランプしない")

# %%
# RSI(単純移動平均方式)とストキャスティクスの組。どちらも 0-100。
rsi = qm.zoo.rolling_relative_strength_index("close", window=5)
percent_k = qm.zoo.stochastic_percent_k(window=5)
percent_d = qm.zoo.stochastic_percent_d(window=5, smooth=3)
oscillators = pd.DataFrame(
    {
        "rsi": rsi.predict(inputs, seed=0).to_wide()["AAA"],
        "percent_k": percent_k.predict(inputs, seed=0).to_wide()["AAA"],
        "percent_d": percent_d.predict(inputs, seed=0).to_wide()["AAA"],
    }
)
print("== RSI とストキャスティクス(0-100 スケール) ==")
print(oscillators.round(2).to_string())
print("名前が方式を語る: rolling mean 方式であり Wilder 平滑化ではない")

# %%
# 一目均衡表の因果的な整列: 時刻 t に表示されるスパン値は
# displacement ステップ前に計算されたもの — 未来には何も書き込まない。
conversion = qm.zoo.ichimoku_conversion_line(window=3).predict(inputs, seed=0)
span_a = qm.zoo.ichimoku_leading_span_a(
    conversion_window=3, base_window=5, displacement=3
).predict(inputs, seed=0)
ichimoku = pd.DataFrame(
    {
        "conversion": conversion.to_wide()["AAA"],
        "span_a": span_a.to_wide()["AAA"],
    }
)
print("== 一目均衡表の転換線と因果的な先行スパン A ==")
print(ichimoku.round(3).to_string())
print("遅行スパンは設計上存在しない: 未来の close が必要になるから")

# %%
# log リターンのヒストリカル・ボラティリティを明示的に年率化。
vol = qm.zoo.annualized_historical_volatility(
    qm.zoo.log_return(), window=5, periods_per_year=252
)
print("== 年率ヒストリカル・ボラティリティ(窓 5) ==")
print(vol.predict(inputs, seed=0).to_wide().round(4).to_string())
print("窓はすべて strict: 必要な本数が揃うまで NaN")
