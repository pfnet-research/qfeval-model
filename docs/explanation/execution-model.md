# The execution model

This is the most important concept to understand when writing models.

## Re-invocation, not suspension

The engine calls your model function **once per public call** (`fit` or
`predict`). `for step in ctx:` iterates only the steps of that call and then the
function returns normally. On the next call, the function is called again from
the top.

No suspended execution frame survives across calls. This is a deliberate design
choice: it makes checkpoint and clone trivial, because the only things to
serialize are `ctx.state` and the child models — never a live generator frame.

## Consequences for model authors

- **Persist through `ctx.state`** (or child models). Local variables do not
  survive across calls.
- **Code before the loop (the prologue) and after it (the epilogue) runs on
  every call.** Write it idempotently — reading with a default
  (`ctx.state.get("n", 0)`) and writing back, or `state.setdefault(...)`, are
  the canonical patterns.

```python
@qm.model
def good(ctx: qm.Context) -> None:
    for step in ctx:
        ctx.state["n"] = ctx.state.get("n", 0) + 1   # idempotent read-update
        ...
```

Child models are registered per call, before the loop, by name in the
`ctx.children` dict (`ctx.children["fast"] = m`; `ctx.child = m` is shorthand
for `ctx.children["child"] = m`); a `qm.Model`-typed function parameter is
not registered until you put it into the dict. The engine owns its children:
the first call builds them from the registered candidates' functions and
bound parameters, so the candidate instances themselves never advance.
Registration is idempotent by construction — on later calls it is verified
against the first call's registration (name, function, mode), and the
engine-owned children keep running.

A non-idempotent prologue (e.g. `ctx.state["n"] = 0` on every call) produces
different results depending on how the timeline is split into calls.
`verify.assert_stream_consistent` is the front-line detector for this bug.

## Failure semantics

If the function returns or `break`s before consuming every step, the call
fails with `RuntimeError`. Any failure raised after the model function
started executing is fail-stop: the time cursor does not advance and a first
call's `symbols=` is not fixed, but nothing is rolled back either —
`ctx.state` changes are kept and children (which advance before the parent's
steps) keep their advanced cursors. The instance is therefore marked failed
and refuses further calls and checkpoints; build a fresh instance or restore
a checkpoint into one. Input errors raised before the function runs (a
malformed DataFrame, a repeated `seed=`, `predict` rows at or before the
cursor) leave the instance reusable.
