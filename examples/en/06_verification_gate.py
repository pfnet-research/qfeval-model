# %%
"""# 06 — the verification gate: `verify.check_all`

One call runs every applicable check — determinism, streaming consistency,
checkpoint/restore consistency, and batch equivalence for batch models — and
returns a report. Gate models on `report.all_passed` before promoting them;
unlike `ok`, it does not accept a skipped check.
"""

# %%
import pandas as pd

import qfeval_model as qm


# %%
@qm.batch_model
def leaky(ctx: qm.Context) -> None:
    """BROKEN on purpose: subtracts the chunk mean (in-chunk look-ahead)."""
    for step in ctx:
        close = step.inputs["close"]
        step.output = close - close.mean(axis=0)  # sees future rows!


# %%
# Wide data goes straight into the verifier — no conversion step.
close = pd.DataFrame(
    [[100.0, 200.0], [104.0, 196.0], [96.0, 204.0], [108.0, 192.0]],
    index=pd.date_range("2024-01-01", periods=4),
    columns=["AAA", "BBB"],
)

# %%
report = qm.verify.check_all(qm.zoo.simple_return, close, test_from="2024-01-02")
print("== check_all(qm.zoo.simple_return) ==")
print(report)
print(f"all passed: {report.all_passed} — safe to promote")
assert report.all_passed

# %%
report = qm.verify.check_all(leaky, close)
print("== check_all(leaky batch model) ==")
print("\n".join(line.split(" — ")[0] for line in str(report).splitlines()))
print(
    f"all passed: {report.all_passed} — the batch check caught the in-chunk look-ahead"
)
assert not report.all_passed
