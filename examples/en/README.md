# Examples

Seven numbered scripts and a cookbook. **Start with the core four (01–04)** —
they cover everything you need to write and backtest a model. Read the
operations three (05–07) before putting a model in production. The cookbook is
a by-topic reference you dip into when a need comes up.

Run any script with `uv run python examples/en/NN_name.py` (or plain `python`
in an environment where `qfeval-model` is installed). Each finishes in a
couple of seconds, prints deterministic output, and writes no files; all are
executed in CI (see `tests/test_examples.py`) so they never rot. A Japanese
translation of every script lives in [`examples/ja/`](../ja/README.md) with
the same file names and code — only the commentary (markdown cells, comments,
print strings) is translated.

Every script doubles as a Jupyter notebook: it is written in the
`ipynb-py-convert` cell format (`# %%` markers; the leading docstring becomes
a markdown cell). `make examples` at the repo root converts each script to a
`.ipynb` next to it and executes it, leaving the outputs in the notebook; the
generated notebooks are gitignored, and `make clean-examples` removes them.

The input contract in three lines: `inputs` is a **wide** DataFrame — one row
per timestamp, one column per symbol — with a datetime64 index (tz-aware
allowed; normalized to UTC), treated as a single feature named `"close"`; pass
a mapping `{feature_name: wide_df}` for several features. Long-format
DataFrames (timestamp / symbol / feature columns) remain the transport format
and are accepted everywhere; example 01 prints both shapes.

## Core — read these four first

| Script | Shows | Sample output line |
|---|---|---|
| `01_first_model.py` | The smallest model; wide inputs; the first-call arguments `symbols=` / `seed=` | `one output per (timestamp, symbol), on the declared symbol axis` |
| `02_fit_and_predict.py` | `fit` / `predict`, `step.is_train` / `is_test`, learned state | `the test day's crazy close (999) is ignored: the phase was 'test'` |
| `03_composition.py` | `ctx.child`, named children, zoo building blocks | `ctx.child = child registered the name 'child', read back as step.child` |
| `04_backtest.py` | `qm.split(test_from=)`, `Model.run`, `qm.metrics` | `metrics computes forward returns internally — never write shift(-1)` |

## Operations — before production

| Script | Shows | Sample output line |
|---|---|---|
| `05_checkpoint_serving.py` | `save`/`load`/`dumps`/`loads`; checkpoints are externally controlled | `identical to an uninterrupted run — state carried the whole memory` |
| `06_verification_gate.py` | `verify.check_all` as a promotion gate, plus a model that fails it | `ok: False — the batch check caught the in-chunk look-ahead` |
| `07_symbol_universe.py` | `symbols=`, NaN filling, `ValueError` on new symbols | `new symbols need a new model (or declare them on day 1)` |

## Zoo tour — risk and technicals

| Script | Shows | Sample output line |
|---|---|---|
| `08_risk_models.py` | Historical VaR / expected shortfall, drawdowns, downside deviation, annualized volatility | `ES >= VaR holds cellwise: True` |
| `09_technical_indicators.py` | Bollinger, RSI, stochastics, causal Ichimoku spans, historical volatility | `the lagging span is absent by design: it would need a future close` |

## Cookbook — by topic (`examples/en/cookbook/`)

| Script | Topic |
|---|---|
| `missing_data.py` | Gaps: `lag` vs `lag(ffill(source))` vs `ffill` |
| `randomness.py` | `step.rng` and same-seed reproducibility |
| `batch_models.py` | `@qm.batch_model` (advanced API) and leak checks |
| `side_channels.py` | Side inputs (`side_data=`, numeric + object) / `step.side_outputs` / `info` |
| `online_rls.py` | An online learner (RLS with forgetting) in `ctx.state` |
| `what_if_clone.py` | Branching a fitted model with `clone()` |
| `multi_horizon.py` | Anonymous trailing axes: a `(T, S, H)` multi-horizon output |
