# Backtesting with `model.run`

`model.run` drives a backtest: it fits the train period, predicts the test
period, and returns the test-period `Features`. Like `fit` / `predict`, it
advances the model itself — clone first (`model.clone().run(...)`) if you
want to keep the original untouched.

```python
output = model.run(inputs, test_from="2020-01-10", seed=0)
```

`test_from` names the inclusive start of the test side: rows before it are
fit, rows at or after it are predicted — the same half-open convention as
`range(start, stop)`. Pass `test_from=None` to predict everything with no
fit.

To split data yourself, use the same vocabulary. `split` returns the wide
format: wide inputs keep their container (a flat frame splits into flat
frames, a mapping into mappings), and long inputs come back as a
`{feature: wide frame}` mapping.

```python
train, test = qm.split(inputs, test_from="2020-01-10")  # TrainTest named tuple
```

## Evaluating with `qm.metrics`

Forward returns are **not** model inputs (that would leak). `qm.metrics`
owns the forward-return computation — never write the `shift(-1)` yourself:

```python
from qfeval_model import metrics

ic = metrics.information_coefficient(output, inputs)   # per timestamp
hit = metrics.hit_rate(output, inputs)                 # float
pnl = metrics.long_short_pnl(output, inputs)           # per timestamp
fwd = metrics.forward_returns(inputs)                             # long DataFrame
```

`information_coefficient` is the per-timestamp rank correlation between the
output and the next-price-observation return; rows with no price for any
symbol do not consume a forward period. `long_short_pnl` is the PnL of a
rank-weighted, market-neutral book.

See [`examples/en/04_backtest.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/04_backtest.py)
and [`examples/en/cookbook/what_if_clone.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/cookbook/what_if_clone.py).
