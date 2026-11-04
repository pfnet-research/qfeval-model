# How to verify a model

The `qfeval_model.verify` harness provides three assertions plus a one-call
runner. Each takes a `make_model` factory returning a fresh, unstarted
instance (a decorated model with no required parameters works directly).
Verify under the same conditions the model serves with: the assertions
accept the production symbol universe (`symbols=` on the first call), side
data, and train/test boundary. Put persistent state in `ctx.state`; `info` is
fresh, output-only diagnostics for each public call. Set a non-default seed
inside `make_model` via `model.seed`.

## `check_all` — the gate

Runs every applicable check and returns a `CheckReport`:

```python
report = verify.check_all(make_model, inputs, test_from="2020-01-05")
print(report)            # one PASS/FAIL/SKIP line per check
assert report.all_passed # every check ran and passed
```

It always runs determinism and streaming consistency, adds the
checkpoint/continue check when at least two test timestamps exist
(checkpointing at `checkpoint_at` or the midpoint; with fewer timestamps the
check is reported as `SKIP` with the reason, not silently dropped).
`assert_stream_consistent` also checks batch models against chunk-size-1
execution. `report.ok` is true when no check
failed — a skip stays visible without failing `ok`. `report.all_passed` is
true only when every check actually ran and passed; use it as the promotion
gate in CI so a skipped check cannot slip through.

## `assert_stream_consistent`

Runs a model as one `predict` vs. per-timestamp `predict` and compares. Under
the re-invocation execution model this checks that the prologue is idempotent
and state is handled correctly. For a batch model, the same comparison detects
in-chunk look-ahead and replaces the former batch-specific assertion.

```python
verify.assert_stream_consistent(make_model, inputs)
```

## `assert_deterministic`

Runs the model `runs` times and requires exactly equal results (zero
tolerance; `NaN` / `NaT` must appear at the same positions). Catches use of
external RNG. Every run keeps the seed set by the factory.

```python
verify.assert_deterministic(make_model, inputs, runs=2)
```

## `assert_checkpoint_consistent`

Compares a straight run against a run that saves at `checkpoint_at`, loads into a
fresh instance, and continues.

```python
verify.assert_checkpoint_consistent(
    make_model, inputs, checkpoint_at="2020-01-07"
)
```

At least one row must be processed before `checkpoint_at` (by fit or prediction),
and at least one prediction row must remain after it. A boundary at or before
the first processed timestamp would only compare two fresh full runs, so it is
rejected instead of being reported as a checkpoint PASS.

Models that keep memory anywhere other than `ctx.state` will not agree here
by design — that is the check doing its job.

All assertions raise `VerificationError` on a mismatch. They are not complete
guarantees — they are practical checks over the inputs you supply, worth
wiring into CI. In particular, a PASS is not a runtime schema guarantee:
output dtype, trailing shape, and side-output keys are fixed per call, not
for the model's lifetime, so drift under conditions the checked data never
exercised is not ruled out.

See [`examples/en/06_verification_gate.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/06_verification_gate.py).
