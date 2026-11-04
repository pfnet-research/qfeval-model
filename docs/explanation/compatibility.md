# Compatibility policy (0.x)

qfeval-model follows Semantic Versioning; during 0.x, breaking changes to
the public surface are allowed in **minor** releases and are always recorded
in the changelog. This page defines what "the public surface" means, per
kind of compatibility.

## Source compatibility

The public source surface is the top-level facade's `__all__` (`import
qfeval_model as qm`) plus the `zoo` / `metrics` / `verify` packages'
`__all__`: their names, function call signatures, *documented* method call
signatures and attributes, and raised exception *types*.
Except for the wire contracts explicitly identified below, this source
surface remains **provisional during 0.x**: a minor release may change it
with a changelog entry. This distinction leaves room to learn from real use
without weakening the DataFrame and reproducibility contracts.
Exceptions may gain common base classes without notice — catch the
documented type (`issubclass` holds), do not depend on exact `__bases__`.

Runtime-managed classes — `Model`, `ModelFactory`, `Context`, `Step`, and
`Features` — are exported for type annotations and `isinstance`; their
constructors are **not** supported API. `Model` / `ModelFactory` construction
would bypass the decorators, `Context` / `Step` take engine internals, and a
directly constructed `Features` would bypass input normalization. Supported
instances come from `@qm.model` / `@qm.batch_model`, calling the resulting
factory, the engine passing `Context` / `Step` to the model function, and
`predict` / `run` returning `Features`. `Model`'s public surface is
`fit` / `predict` / `run`, checkpointing (`dumps` / `loads` / `save` /
`load` / `clone`), and the attributes `state`, `seed`, `symbols`, `cursor`,
`mode`, `started`, and `children`. `children` exposes the engine-owned
child models for inspection; mutating or calling `fit` / `predict` on an
engine-owned child is not supported — children advance only through their
parent.

Of those attributes, the supported writable surface is the *contents* of
`state` and `seed`. The `seed=` call argument is first-call-only; direct
assignment to a top-level `model.seed` remains an expert path. A child seed is
defined only by automatic derivation from the parent seed and registration
name. Directly changing `parent.children["name"].seed` is an undefined expert
operation and is not a compatibility guarantee.
`symbols`, `cursor`, `mode`, `started`, child attributes, and the `children` mapping are
read-only in intent; assigning to them or mutating the `symbols` array or
`children` mapping is undefined behaviour. The `state` dict itself must not
be rebound. These are not supported extension points.
`symbols` never shares memory with a returned result's `symbols`, so editing
a result stays local to that result.

Anything else (`qfeval_model.core.*` submodules, private helpers,
undocumented or underscore attributes, whether satellite packages import
eagerly or lazily) is an implementation detail.

## Concurrency

A `Model` instance is not safe for concurrent or reentrant use: do not run
multiple public operations on one instance concurrently, and do not re-enter
it from its own model function. There is no runtime guard — overlapping calls
can interleave `ctx.state`, and a mid-call checkpoint can capture state that
is not at a call boundary. Give each serving worker its own instance
(`clone()`, or restore the same checkpoint into fresh instances) and
checkpoint only between public calls.

## Data compatibility

The DataFrame contract is a wire format: the `timestamp` / `symbol` key
columns, UTC-naive
`datetime64[ns]` timestamps, last-non-NaN duplicate aggregation, sorted
unique symbol universes, and the dtype rules (uncovered cells require an
inexact dtype; side channels and side outputs use the dtype's missing
value — `NaN` / `NaT` / `None`, else full coverage or `0` defaults).

## Behavioral reproducibility

Three tiers, from strongest to weakest:

- **Frozen across versions and platforms** (a wire contract): the seed
  derivation (`blake2b(seed, payload)` truncated to 64 bits; a step's
  payload is its first timestamp, a child's is its registration name) and
  the Philox bit generator behind `step.rng`. These integer/bit-stream
  values are checked against golden fixtures.
- **Within one environment** (the same Python / NumPy / pandas /
  qfeval-model versions and platform): a same-seed rerun of the same calls
  is bit-identical for deterministic models.
- **Across dependency versions or platforms**: the zoo models keep their
  documented numeric *definitions*, but their floating-point results go
  through NumPy/pandas kernels whose bit patterns may differ — expect
  tolerance-level agreement, not bit-identity. Fixing a numeric bug in a
  zoo model is a changelog-recorded behavior change, not a compatibility
  break.

## Checkpoint compatibility

The envelope carries a `format_version`; only the current version is read,
and there is no migration machinery. Loading checks structure, format
version, and model identity (`__module__` + `__qualname__` + mode) — not the
code or the bound parameters (see
[checkpoint semantics](checkpoint-semantics.md#what-loading-checks-and-what-it-does-not)).
The payload is dominated by your own pickled `state`, so loads across
library versions warn (`library_version` mismatch) and are not actively
guaranteed. This is a deliberate decision, not an oversight: a checkpoint
is a best-effort state blob across library versions, and a host that must
survive library upgrades keeps the inputs needed to construct a fresh model
and refit or replay it rather than relying on cross-version checkpoint reads.

This narrow identity is deliberate: the project prefers a small core and no
false rejection from unstable fingerprints of arbitrary Python bindings.
Consequently it does not prove that code or parameters are compatible; hosts
that require fail-closed reconstruction checks must compare their own stable
construction/version metadata before loading. See
[why the identity is deliberately narrow](checkpoint-semantics.md#why-the-identity-is-deliberately-narrow).

## Deprecation

During 0.x there is no deprecation cycle: a breaking change lands in a
minor release with a changelog entry describing the migration. The wire
contracts above (column names, seed derivation, envelope format) are held
to a higher bar — changing any of them is a redesign, not a routine break.
