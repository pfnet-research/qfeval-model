# qfeval-model

Write prediction models over panel data (timestamp × symbol) as plain
decorated Python functions.

```python
import qfeval_model as qm


@qm.model
def simple_return(ctx: qm.Context) -> None:
    ctx.child = qm.zoo.lag("close")             # yesterday's close
    for step in ctx:
        step.output = step.inputs["close"] / step.child.array - 1.0


model = simple_return()
model.fit(train_inputs, seed=0)
pred = model.predict(test_inputs)
pred.to_wide()  # wide DataFrame: index=timestamp, columns=symbol
```

## Where to go next

- **Tutorials** — start with [your first model](tutorials/01-first-model.md),
  then [composition](tutorials/02-composition.md),
  [batch models](tutorials/03-batch-models.md), and
  [backtesting](tutorials/04-backtest.md).
- **How-to guides** — [checkpointing](howto/checkpointing.md),
  [verification](howto/verification.md), [determinism](howto/determinism.md),
  and [side inputs](howto/side-inputs-metadata.md).
- **Reference** — the [public API](reference/api.md), the
  [zoo](reference/zoo.md), and [metrics](reference/metrics.md).
- **Explanation** — the [execution model](explanation/execution-model.md) is the
  key concept; also [trailing axes](explanation/trailing-axes.md),
  [leak safety](explanation/leak-safety.md),
  [checkpoint semantics](explanation/checkpoint-semantics.md),
  [compatibility](explanation/compatibility.md), and
  [layering](explanation/layering.md).
