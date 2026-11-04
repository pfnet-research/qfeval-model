# %%
"""# Cookbook — the auxiliary channels

Per-timestamp (not per-symbol) inputs ride along as `side_inputs`: numeric
columns stay numeric arrays, and object-dtype columns carry arbitrary Python
objects (strings, dicts, ...). Models emit extras via `step.side_outputs`
(arrays) and `step.info` (one free-form dict shared by every step of the call,
read back as `output.info`).
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.model
def risk_scaled(ctx: qm.Context) -> None:
    for step in ctx:
        scale = (
            0.0 if step.side_inputs["regime"] == "panic" else step.side_inputs["vix"]
        )
        step.output = step.inputs["close"] * 0.0 + scale
        step.side_outputs["raw_vix"] = step.side_inputs["vix"]
        # info is one call-wide dict; key by timestamp for per-step records.
        step.info[str(pd.Timestamp(step.timestamp).date())] = step.side_inputs["regime"]


# %%
# Main inputs stay wide; side_inputs is one per-timestamp long frame carrying
# both the numeric `vix` and the object-dtype `regime` column.
days = pd.date_range("2024-01-01", periods=2)
close = pd.DataFrame([[100.0], [101.0]], index=days, columns=["AAA"])
side = pd.DataFrame(
    {"timestamp": days, "vix": [15.0, 80.0], "regime": ["calm", "panic"]}
)

# %%
output = risk_scaled().predict(close, side_data=side, seed=0)
print("== outputs scaled by side channels ==")
print(output.to_wide().to_string())
print("day 2 went to 0: the regime channel said 'panic', overriding the vix scale")

# %%
print("== reading info back (one call-wide dict) ==")
for day, regime in output.info.items():
    print(day, regime)
print("side_outputs are arrays; info is a free-form diagnostics dict")
