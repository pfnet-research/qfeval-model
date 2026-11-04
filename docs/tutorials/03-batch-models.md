# Batch models

`@qm.batch_model` is an advanced, performance-oriented API. A batch model
receives a whole chunk (all timestamps of a call) as one step instead of one
step per timestamp. Reach for it only when per-step Python is measurably too
slow (e.g. deep learning); ordinary models should stay single-step.
The checked-in `benchmarks/bench_modes.py` comparison records the material
speedup on a vectorizable 1M-cell backward-return workload that justifies this
extra execution mode.

| | `@qm.model` (single) | `@qm.batch_model` (batch) |
|---|---|---|
| steps per call | `T` (one per timestamp) | 1 (chunk = call) |
| `step.inputs[name]` | `(S,)` | `(T, S)` |
| `step.side_inputs[name]` | scalar value | `(T,)` |
| `step.output` | `(S,)` | `(T, S)` |
| time | `step.timestamp` | `step.timestamps` |

```python
import numpy as np


@qm.batch_model
def batch_return(ctx: qm.Context) -> None:
    for step in ctx:
        close = step.inputs["close"]  # (T, S)
        last = ctx.state.get("last_close", np.full(close.shape[1], np.nan))
        prev = np.vstack([last[None, :], close[:-1]])
        ctx.state["last_close"] = close[-1].copy()
        step.output = close / prev - 1.0
```

## Leak responsibility is yours

Because a batch model sees the whole chunk at once, the engine cannot
structurally prevent in-chunk look-ahead. **Always** verify a batch model
against the per-timestamp semantics:

```python
from qfeval_model import verify

verify.assert_stream_consistent(batch_return, inputs)
```

This runs the model one-shot and one-timestamp-at-a-time and compares the
results (this is the same stream-consistency check used for single models).
It is a spot check, not a proof — order-dependent computations can
pass on one dataset and leak on another, which is exactly why batch models
are positioned as an expert tool.

See [`examples/en/cookbook/batch_models.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/cookbook/batch_models.py).
