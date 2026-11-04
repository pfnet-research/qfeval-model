# Checkpoint semantics

## The outside decides when to save

Checkpointing is controlled by the host, not the model: **a model cannot
refuse a checkpoint**. `save()` (and `clone()`) succeed at any call boundary
and write whatever is in `ctx.state` at that moment (which must be
picklable). There is no defer or heads-up mechanism — the price of that
simplicity is a single contract on the model side: keep `ctx.state`
checkpoint-ready at every step boundary. An instance whose call failed
mid-execution is *not* at a call boundary: it refuses to checkpoint, like it
refuses further calls.

## The pickle vocabulary

The facade follows pickle's naming: `save(f)` / `load(f)` for paths and
binary file objects (read/written at the current position, no seeking), and
`dumps()` / `loads(data)` for bytes. `load` / `loads` populate a fresh,
compatible instance in place and return `self`, so a restored instance is one
expression: `restored = my_model().load(path)`.

## Stable points

A checkpoint (or clone) is valid at a stable point: immediately after a `fit`
or `predict` returns normally. `clone()` is implemented as
`dumps()` → `loads()`, so *clonable* and *savable* are the same
property by construction — and a clone is independent in exactly what a
checkpoint carries (`state`, cursor, seed, universe, and the engine-owned
children). The function and its bound parameters are shared with the
original: mutating a bound list/dict/array affects both instances.

## What is and isn't saved

Saved: `state`, child models (recursively, by registration name), the
symbol universe, the time cursor, and the seed. Not saved: the
input `DataFrame` and the model function itself (your code is matched by its
identity (module + qualname + mode), not serialized).

The engine keeps no input history, so **anything your model needs to remember
must live in `ctx.state`** (or in a child such as `qm.zoo.with_history`, whose
buffer lives in its own state). A model that hides memory in local or global
variables will not reproduce pre-checkpoint values after a `load` —
`verify.assert_checkpoint_consistent` (and `verify.check_all`) detect exactly
this.

## What loading checks — and what it does not

Loading validates the envelope's basic structure, its `format_version`, and
the model identity: the function's `__module__` plus `__qualname__` plus its
mode, *nothing more*. Including the module keeps same-named functions in
different files (a common accident with closures made inside factory
functions) from accepting each other's checkpoints; the cost is that moving
a model function's file orphans its checkpoints. Deliberately not checked:
the implementation (editing code keeps checkpoints loadable — the
PyTorch/scikit-learn stance) and the bound parameters (`scaled(factor=2)`'s
checkpoint loads into `scaled(factor=999)` and runs with 999 from then on).
A checkpoint is a **state blob for a construction you promise is
compatible**: the library fails closed on identity and format, and leaves
code/parameter versioning to the host — if your constructions can drift,
store your own version metadata next to the checkpoint.

## Why the identity is deliberately narrow

The narrow identity is an intentional design constraint, not an incomplete
attempt to infer compatibility. qfeval-model prioritizes a small checkpoint
engine and avoiding false rejection of valid state blobs over automatically
detecting every mismatched reconstruction.

The apparent automatic alternatives do not define a stable rule for arbitrary
Python model parameters:

- `repr` may contain memory addresses or hash-dependent container ordering,
  and a mutable bound object may legitimately have changed since construction;
- pickling/hashing the bindings makes otherwise usable unpicklable or
  identity-bearing objects part of the checkpoint protocol;
- hashing source or bytecode is unreliable for notebooks and closures, and
  rejects checkpoints after compatible refactors or module moves;
- a user-supplied checkpoint ID introduces a second configuration/versioning
  concept even though the host must already version the complete construction.

Accordingly, the library knowingly accepts false negatives (a wrong
same-named construction can pass the identity check) rather than introducing
false positives and more core machinery. This check is not a proof of semantic
compatibility. Applications that need a fail-closed serving boundary must
compare their own stable model/configuration version before calling `load` or
`loads`; the checkpoint remains only the model's state blob.

## Load is a two-stage apply for children

`load` reads into a fresh, unstarted, compatible instance you construct with
the same code. Child state cannot be applied at load time (children are not
registered until the model runs). So child envelopes are stashed and applied
on the first call, matched by position and registration name: the engine
builds a **new** child model from each registered child's function and
parameters and restores the stash into it. The child instances you hold are
never mutated, and — an important consequence — **the child models inside a
restored or cloned parent are different objects from the instances in your
hands**.

## Pickle caveat

Checkpoints are pickled, and unpickling can execute arbitrary code embedded in
the bytes: **never load or restore a checkpoint you do not trust.** The
envelope carries a `format_version` (only the current version is read;
anything else is rejected) and a `library_version` (informational; a mismatch
warns and proceeds). Compatibility across library versions is not actively
guaranteed — the payload is dominated by your own `state`.
