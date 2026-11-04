# Examples(日本語版)

番号付きスクリプト 7 本と cookbook です。**まずコアの 4 本(01–04)から**
— モデルを書いてバックテストするのに必要なことはすべてここにあります。
モデルを本番に載せる前に運用編の 3 本(05–07)を読んでください。cookbook
はトピック別のリファレンスで、必要になったときに参照するものです。

各スクリプトは `uv run python examples/ja/NN_name.py`(`qfeval-model` が
インストールされた環境なら素の `python`)で実行できます。どれも数秒で
終わり、決定的な出力を表示し、ファイルは一切書きません。すべて CI で
実行されるので(`tests/test_examples.py` を参照)、例が古びて動かなくなることは
ありません。英語版([`examples/en/`](../en/README.md))と同じファイル名・
同じコードで、翻訳されているのは解説(markdown セル・コメント・print 文)
だけです。

すべてのスクリプトは Jupyter notebook としても使えます:
`ipynb-py-convert` のセル形式(`# %%` マーカー。先頭の docstring が
markdown セルになる)で書かれており、リポジトリルートで `make examples`
を実行すると、各スクリプトが隣の `.ipynb` に変換・実行され、出力が
notebook に残ります。生成された notebook は gitignore されており、
`make clean-examples` で削除できます。

入力の契約を 3 行で: `inputs` は **wide 形式**の DataFrame — timestamp
ごとに 1 行、シンボルごとに 1 列 — で、datetime64 の index(tz 付きも可。
UTC に正規化される)を持ち、`"close"` という 1 特徴量として扱われます。
複数の特徴量は `{特徴量名: wide_df}` のマッピングで渡します。long 形式
(timestamp / symbol / 特徴量の列)は転送用フォーマットとして引き続き
どこでも受け付けられます。例 01 が両方の形を表示します。

## コア — まずこの 4 本

| スクリプト | 内容 | 出力例 |
|---|---|---|
| `01_first_model.py` | 最小のモデル、wide 形式の入力、初回呼び出しの引数 `symbols=` / `seed=` | `宣言したシンボル軸上で (timestamp, symbol) ごとに 1 出力` |
| `02_fit_and_predict.py` | `fit` / `predict`、`step.is_train` / `is_test`、学習済み state | `test 日の異常な close (999) は無視される — フェーズが 'test' だから` |
| `03_composition.py` | `ctx.child`、名前付き children、zoo の部品 | `ctx.child = child で登録した名前 'child' を step.child で参照できる` |
| `04_backtest.py` | `qm.split(test_from=)`、`Model.run`、`qm.metrics` | `metrics がフォワードリターンを内部で計算する — 自分で shift(-1) を書くことはない` |

## 運用編 — 本番の前に

| スクリプト | 内容 | 出力例 |
|---|---|---|
| `05_checkpoint_serving.py` | `save`/`load`/`dumps`/`loads`、チェックポイントは外部制御 | `中断なしの実行と完全一致 — 記憶のすべてが state で引き継がれた` |
| `06_verification_gate.py` | 本番昇格のゲートとしての `verify.check_all` と、それに落ちるモデル | `ok: False — バッチチェックがチャンク内の先読みを捕まえた` |
| `07_symbol_universe.py` | `symbols=`、NaN 埋め、新シンボルでの `ValueError` | `新しいシンボルには新しいモデルが必要(または 1 日目に宣言する)` |

## Zoo ツアー — リスクとテクニカル

| スクリプト | 内容 | 出力例 |
|---|---|---|
| `08_risk_models.py` | ヒストリカル VaR / 期待ショートフォール、ドローダウン、下方偏差、年率ボラティリティ | `ES >= VaR が全セルで成立: True` |
| `09_technical_indicators.py` | ボリンジャー、RSI、ストキャスティクス、因果的な一目スパン、ヒストリカル・ボラティリティ | `遅行スパンは設計上存在しない: 未来の close が必要になるから` |

## Cookbook — トピック別(`examples/ja/cookbook/`)

| スクリプト | トピック |
|---|---|
| `missing_data.py` | 欠損: `lag` vs `lag(ffill(source))` vs `ffill` |
| `randomness.py` | `step.rng` と同一シードの再現性 |
| `batch_models.py` | `@qm.batch_model`(上級 API)とリークチェック |
| `side_channels.py` | side 入力(`side_data=`、数値 + object)/ `step.side_outputs` / `info` |
| `online_rls.py` | `ctx.state` に載せるオンライン学習器(忘却付き RLS) |
| `what_if_clone.py` | `clone()` による学習済みモデルの分岐 |
| `multi_horizon.py` | 無名の trailing 軸: `(T, S, H)` のマルチホライズン出力 |
