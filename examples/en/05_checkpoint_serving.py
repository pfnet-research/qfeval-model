# %%
"""# 05 — checkpointing and the serving lifecycle

The pickle vocabulary: `save`/`load` for files, `dumps`/`loads` for
bytes. Saving is controlled by the *outside* — a model cannot refuse a
checkpoint, so anything it needs must live in `ctx.state` at every step
boundary.
"""

# %%
import numpy as np
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def ema_signal(ctx: qm.Context, alpha: float = 0.3) -> None:
    """EMA of close — all of its memory is one array in ctx.state."""
    for step in ctx:
        close = step.inputs["close"]
        prev = ctx.state.get("ema", close)
        current = alpha * close + (1.0 - alpha) * prev
        ctx.state["ema"] = current  # checkpoint-ready at every boundary
        step.output = current


# %%
# The library deliberately does not fingerprint bound parameters or code.
# Keep a host-owned construction version beside the checkpoint and bump it
# whenever either changes.
CONFIG_VERSION = "ema-signal-alpha-0.3-v1"


# %%
# The wide frame is the input; split slices it into wide day ranges
# (returned as {feature: frame} dicts).
close = pd.DataFrame(
    [[100.0], [104.0], [96.0], [108.0], [112.0]],
    index=pd.date_range("2024-01-01", periods=5),
    columns=["AAA"],
)
day4, day5 = qm.split(close, test_from="2024-01-05")
print("== day5 (wide slice from qm.split) ==")
print(day5["close"].to_string())

# %%
server = ema_signal(alpha=0.3)
server.predict(day4, seed=0)
saved = {"config_version": CONFIG_VERSION, "blob": server.dumps()}
print("== checkpoint taken after day 4 ==")
print(f"blob: {len(saved['blob'])} bytes")
print("the outside decided when to save; the model had kept state ready")

# %%
# A new process: build the same factory, load, continue where it left off.
# Loading checks *model identity* — the function's qualified name and mode.
# It does not compare bound parameters or detect edits to the function body
# behind the same name: rebuilding the model with the same parameters, and
# versioning the code that produced a checkpoint, is the host application's
# responsibility.
if saved["config_version"] != CONFIG_VERSION:
    raise RuntimeError(f"stale checkpoint: {saved['config_version']}")
restored = ema_signal(alpha=0.3).loads(saved["blob"])
output = restored.predict(day5)
print("== restored instance predicts day 5 ==")
print(output.to_wide().round(3).to_string())

# %%
straight = ema_signal(alpha=0.3).predict(close, seed=0)
np.testing.assert_allclose(output.array[0], straight.array[-1], rtol=1e-12)
print("identical to an uninterrupted run — state carried the whole memory")
