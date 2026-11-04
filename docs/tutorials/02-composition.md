# Composition with children

A model can consume other models by registering them as named children. The
engine manages children (they are included in checkpoints) and advances them
*before* the parent's steps, so the parent reads each child's output for the
current step by name.

Registration is a plain dict — `ctx.children[name] = m` registers under
`name`, read back as `step.children[name]`. `ctx.child = m` is shorthand for
`ctx.children["child"] = m`, read back as `step.child`.

Registration is always explicit: a model passed as a function parameter is
not registered until you put it into `ctx.children`.

```python
import numpy as np


@qm.model
def moving_average(ctx: qm.Context, child: qm.Model, window: int = 3) -> None:
    # ``child`` is registered under the name "child" and read back via
    # ``step.child``.
    ctx.child = child
    for step in ctx:
        values = [*ctx.state.get("values", []), step.child.array.copy()][-window:]
        ctx.state["values"] = values
        step.output = np.mean(values, axis=0)


model = moving_average(qm.zoo.simple_return(), window=3)
pred = model.predict(inputs, seed=0)
```

The zoo composes the same way: `qm.zoo.rolling_mean(qm.zoo.simple_return(), window=5)`
takes the moving average of a child model's output instead of a column.

## Reading children

`step.child` / `step.children["name"]` is the child's `Features`, row-sliced
to the current step: `.array` and `.side_data` contain that row's child
outputs, while `.info` is empty because call-wide child info is not a safe
per-timestamp channel (in single mode the step's timestamp is squeezed out,
so `.array` is `(S, *trailing)`; in batch mode it is the chunk's `(T, S,
*trailing)`). `step.children` is a plain dict in registration order. Treat
the slices as read-only — mutating them is undefined behaviour. They are
views into the child's whole-call output, so `.copy()` anything you keep in
`ctx.state` (as `moving_average` above does) — a stored view keeps the whole
call's array alive.

## The parent cannot feed the child

Children are *parallel* models that read the same external input as the parent.
The parent can read a child's output, but it **cannot pass computed values into
a child**. If you need a value from the parent inside a child, restructure so
the child computes it from the shared inputs.

## Identity across calls

The first call fixes the registration and builds engine-owned children from
the registered candidates' functions and bound parameters (the candidate
instances themselves never advance). Later calls may pass freshly
constructed instances (building children inside the model is fine): the
engine matches them by name and function identity and keeps running its own
children, so state carries across calls. Changing the names or identity of
children raises `ValueError`.

See [`examples/en/03_composition.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/03_composition.py).
