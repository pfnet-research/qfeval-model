# qfeval-model

[![CI](https://github.com/pfnet-research/qfeval-model/actions/workflows/ci.yml/badge.svg)](https://github.com/pfnet-research/qfeval-model/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qfeval-model.svg)](https://pypi.org/project/qfeval-model/)
[![Python](https://img.shields.io/pypi/pyversions/qfeval-model.svg)](https://pypi.org/project/qfeval-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pfnet-research/qfeval-model/blob/main/LICENSE)

qfeval-model runs prediction models over panel data: values indexed by
timestamp and symbol (for example, a stock ticker). Define a model as a
decorated Python function; the engine advances it one timestamp at a time.
Combine models through child outputs, save their state, and use the
verification functions to compare repeated, streamed, and restored runs.

The example below assumes `prices` is a wide pandas DataFrame with a
`DatetimeIndex` and one column per symbol. Use positive prices spanning the
split date, with at least ten observations before the test period and two
timestamps in it. The model measures each price relative to its rolling mean.
The metric evaluates the signal against later price changes (forward returns).

```python
import qfeval_model as qm


@qm.model
def momentum(ctx: qm.Context, window: int = 20) -> None:
    ctx.child = qm.zoo.rolling_mean("close", window=window)
    for step in ctx:  # one step per timestamp
        step.output = step.inputs["close"] / step.child.array - 1.0


pred = momentum(window=10).run(prices, test_from="2024-01-01", seed=0)
pred.to_wide()    # wide DataFrame: index=timestamp, columns=symbol
pred.to_frame()   # long DataFrame: timestamp, symbol, close

daily_ic = qm.metrics.information_coefficient(pred, prices)

# Match the model parameters and split used above; seed 0 is the default.
report = qm.verify.check_all(
    lambda: momentum(window=10), prices, test_from="2024-01-01"
)
assert report.all_passed, str(report)  # every check ran and passed
```

The main output uses the `"close"` feature name, so `pred.to_wide()` can feed
another model without renaming columns.

## Model behavior

- **Execution.** `@qm.model` yields one step per timestamp. Each step receives
  its current row; retain earlier values in `ctx.state`. `@qm.batch_model`
  processes a whole call in one step for vectorized workloads and requires
  the model author to avoid looking ahead within that call.
- **Composition and evaluation.** `qm.zoo` provides backward-looking models
  such as `lag`, `rolling_mean`, and `cross_sectional_rank`. Register them as
  children to combine their outputs. `qm.metrics` computes forward returns
  and evaluates predictions against them. In the example, `daily_ic` measures
  the rank correlation across symbols at each timestamp.
- **Inputs.** Pass a wide DataFrame, a `{feature: frame}` mapping, or a long
  DataFrame with `timestamp` and `symbol` columns. Timestamps must indicate
  when information became available; the engine cannot detect future
  information already embedded in an input row.
- **Checkpoints.** Persistent model state belongs in `ctx.state`.
  `save`/`load` and `dumps`/`loads` store and restore it together with child
  models and engine state. After restoring into the same model construction,
  `fit` skips already processed rows so a feed can be replayed. Loading across
  library versions is best-effort and warns on a version mismatch; retain
  inputs for rebuilding by replay (see the
  [compatibility policy](https://github.com/pfnet-research/qfeval-model/blob/main/docs/explanation/compatibility.md)).
- **Verification.** `qm.verify.check_all` compares repeated runs, whole-call
  and per-timestamp execution, and checkpoint restoration on supplied inputs.
  `report.all_passed` requires every check to run and pass. These checks can
  reveal state and look-ahead errors; their results apply to the tested inputs.

## Installation

```bash
pip install qfeval-model
```

Requires Python >= 3.10. Runtime dependencies: NumPy, pandas.

## Documentation

- Tutorials, how-to guides, API reference, and explanations live in
  [docs/](https://github.com/pfnet-research/qfeval-model/tree/main/docs)
  (build locally with `uv run --group docs mkdocs serve`).
- [Examples](https://github.com/pfnet-research/qfeval-model/blob/main/examples/README.md):
  nine runnable scripts covering model definition, composition, backtesting,
  checkpoints, verification, symbol universes, risk models, and technical
  indicators, plus a cookbook. English and Japanese versions run in CI.
- [Design](https://github.com/pfnet-research/qfeval-model/blob/main/DESIGN.en.md)
  ([日本語版](https://github.com/pfnet-research/qfeval-model/blob/main/DESIGN.ja.md)):
  execution and data contracts, architecture, and design rationale.

## Development

From a repository checkout, run:

```bash
uv sync --group dev
make lint
make test
```

See [CONTRIBUTING.md](https://github.com/pfnet-research/qfeval-model/blob/main/CONTRIBUTING.md)
for architecture rules, documentation checks, and release steps.

## License

[MIT](https://github.com/pfnet-research/qfeval-model/blob/main/LICENSE).
Copyright (c) 2026 Preferred Networks, Inc.
