# %%
"""# Cookbook — batch models (advanced API) and leak safety

`@qm.batch_model` hands the whole call as one `(T, S)` chunk — useful when
per-step Python is too slow (e.g. deep learning). The price: you are trusted
not to look ahead inside the chunk. `verify.assert_stream_consistent` compares
against chunk-size-1 execution to catch leaks,
but it is a spot check, not a proof — treat batch as an expert tool.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.batch_model
def vectorized_return(ctx: qm.Context) -> None:
    """Row-safe: each output row only uses rows at or before it."""
    for step in ctx:
        close = step.inputs["close"]
        last = ctx.state.get("last_close", np.full(close.shape[1], np.nan))
        prev = np.vstack([last[None, :], close[:-1]])
        ctx.state["last_close"] = close[-1].copy()
        step.output = close / prev - 1.0


# %%
close = pd.DataFrame(
    [[100.0, 200.0], [110.0, 190.0], [121.0, 180.0]],
    index=pd.date_range("2024-01-01", periods=3),
    columns=["AAA", "BBB"],
)

print("== batch model output ==")
print(vectorized_return().predict(close, seed=0).to_wide().round(3).to_string())

# %%
qm.verify.assert_stream_consistent(vectorized_return, close)
print("== assert_stream_consistent ==")
print("passed: batch output matches one-timestamp-at-a-time execution")
