# How to use side inputs

Besides the symbol-indexed features, a model can read a per-timestamp auxiliary
channel: **`side_data`** — per-timestamp values that need not have a symbol
axis. Each column keeps its own dtype:

- **numeric / boolean columns** stay numpy arrays. Timestamps a floating-point
  or complex column does not cover become `NaN`; integer and boolean columns
  cannot represent a missing value, so they must cover every input timestamp
  — a sparse `int64` or `bool` column is rejected rather than silently
  zero-filled. For a sparse column, use a pandas nullable dtype (`Int64` /
  `boolean`): nullable columns are converted to `float64` with `NaN` at the
  boundary.
- **object-dtype columns** carry arbitrary per-timestamp Python objects
  (strings, dicts, ...); missing values are `None`.
- **datetime / timedelta columns** keep their numpy dtype; uncovered
  timestamps are `NaT`.

`side_data` is passed as a long-format `DataFrame` with a timestamp column and
is aligned to the call's timestamps. Its timestamps must be a subset of the
inputs' timestamps.

```python
@qm.model
def scaled_by_flag(ctx: qm.Context) -> None:
    ctx.child = qm.zoo.simple_return()
    for step in ctx:
        flag = step.side_inputs["flag"]   # numeric scalar for this timestamp
        note = step.side_inputs["note"]   # arbitrary object (object-dtype column)
        step.output = step.child.array * float(flag)
        # info is a fresh output-only dict for this call; key it yourself to
        # keep per-step records.
        step.info[str(step.timestamp)] = note


pred = scaled_by_flag().predict(inputs, side_data=side_data)
pred.info  # one dict, shared and updated in place by every step of the call
```

In a batch model, `step.side_inputs[name]` gains a leading `(T,)` axis (an
object-dtype column becomes a `(T,)` object array). `step.info` is the same dict
for every step of the call (single or batch), starts empty, and is returned as
`pred.info`; store per-timestamp records under keys you choose.

Side inputs are not stored in checkpoints.

See [`examples/en/cookbook/side_channels.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/cookbook/side_channels.py).
