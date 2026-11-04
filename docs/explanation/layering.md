# Layering

The package has two tiers with one-directional dependencies, enforced in CI
by import-linter:

```
zoo | metrics | verify   external features (loosely coupled, growable)
         core            the minimal core (engine + Model + I/O + split)
```

- **`core`** is everything you must understand to use the library: the
  execution engine (steps, state, children, checkpointing), the public
  `@model` / `@batch_model` decorators with `Model` and `Features`, the
  DataFrame ↔ `Features` input/output boundary, and the workflow helpers
  `split` (train/test) and `Model.run` (one-call backtest). A dedicated
  import-linter contract keeps the leaf modules (`core.constants`,
  `core.errors`) pandas-free; the rest of the core may use pandas.
- **`zoo`, `metrics`, `verify`** are external features: loosely
  coupled packages that depend only on the core's public surface and may not
  import one another. They can grow indefinitely without widening the core —
  every zoo entry is an ordinary `@qm.model`, which is how conveniences land
  in the zoo instead of becoming engine flags. The conventions for adding
  entries are in CONTRIBUTING.md.

There is no supported entry point below `Model`: models interact with the
engine only through the documented `ctx` / `step` / `Features` surface, and
the engine internals behind it are implementation details with no stability
guarantee.
