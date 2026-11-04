# Your first model

A model is a plain function decorated with `@qm.model`. Its parameters are
declared in the signature, and calling the factory binds them into a runnable
instance.

```python
import qfeval_model as qm


@qm.model
def simple_return(ctx: qm.Context) -> None:
    ctx.child = qm.zoo.lag("close")             # yesterday's close
    for step in ctx:
        step.output = step.inputs["close"] / step.child.array - 1.0
```

The engine injects `ctx`. Inside the loop each `step` exposes the current
timestamp's inputs (`step.inputs["close"]` has shape `(S,)`, one value per
symbol) and an `output` slot to write. `qm.zoo` ships common building blocks
(`lag`, `ffill`, `rolling_mean`, ...) so time alignment never needs hand-rolled
buffers; when a model does need its own memory, it lives in `ctx.state`
(`ctx.state.get(key, default)` is the idiomatic read), which returns the same
mapping on every call.

## Fitting and predicting

```python
model = simple_return()
model.fit(train_inputs, seed=0)
pred = model.predict(test_inputs)
```

`inputs` is the wide `DataFrame` you usually have — index=timestamp,
columns=symbol — treated as a single feature named `"close"`. For several
features, pass a mapping `{feature_name: wide_df}`. A long-format `DataFrame`
with `timestamp`, `symbol`, and feature columns is the transport format and
is accepted everywhere too. `fit` runs the train period; `predict` runs the
test period and returns a `Features`.

Everything fixed for the model's lifetime is an argument of the *first*
call: `symbols=` (the symbol universe) and `seed=` (randomness).

```python
model.predict(inputs, symbols=("AAA", "BBB"), seed=42)
```

## Reading a `Features`

```python
pred.to_frame()     # long-format DataFrame: timestamp, symbol, close (2-D only)
pred.to_wide()      # wide DataFrame: index=timestamp, columns=symbol
pred.array         # the raw (T, S) ndarray
pred.timestamps     # (T,) datetime64[ns]
pred.symbols        # (S,) str
pred.side_data   # {name: (T, *shape)}
pred.info           # one call-wide dict, shared and updated by every step
```

`pred.to_frame()` keeps every cell (including `NaN` rows), so a plain output
has `T × S` rows. Outputs with extra [trailing
axes](../explanation/trailing-axes.md) use `to_wide()`, which renders them as
deeper `MultiIndex` column levels and is their round-trip format; `to_frame()`
rejects them because long format is deliberately 2-D only.

See the runnable version in [`examples/en/01_first_model.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/01_first_model.py).
