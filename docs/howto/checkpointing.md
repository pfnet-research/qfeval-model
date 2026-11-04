# How to checkpoint and load

Checkpoints save `state`, child models, the symbol universe, seed, and the time cursor
(see [checkpoint semantics](../explanation/checkpoint-semantics.md)). The I/O
follows the pickle vocabulary: `save`/`load` for paths and file objects,
`dumps`/`loads` for bytes. File objects are read/written at their current
position, with no seeking — exactly like `pickle`. Checkpoints *are*
pickles: restoring executes code embedded in the bytes, so load only
checkpoints you produced or otherwise trust. A blob on its own also does
not identify the configuration that produced it — the compatibility check
is deliberately narrow (the model function's `module:qualname` and mode,
not its code or bound parameters; see
[checkpoint semantics](../explanation/checkpoint-semantics.md)) — so store
your model/config version alongside the blob and check it before loading.

```python
model.save(path_or_file)                     # write a checkpoint
restored = simple_return().load(path_or_file)  # load into a fresh instance

blob = model.dumps()                         # -> bytes
restored = simple_return().loads(blob)       # in-place, returns self
```

`load`/`loads` restore **into a freshly constructed, compatible instance**
(same factory, same bound parameters) that has not started yet.
Compatibility is your promise, not the library's check: loading verifies
only the model identity (function `__module__:__qualname__` + mode). Bound
parameters and code are not compared — a checkpoint loads into a construction with
different hyperparameters and silently runs with the new ones, so keep the
construction (and its version) with the checkpoint on the host side (see
[what loading checks](../explanation/checkpoint-semantics.md#what-loading-checks-and-what-it-does-not)).

The minimal host-side pattern is an envelope that pairs the blob with a
construction version you bump whenever the parameters or the code change,
and that you compare before loading:

```python
CONFIG_VERSION = "momentum-w10-v3"  # host-owned construction version
saved = {"config_version": CONFIG_VERSION, "blob": model.dumps()}

# ... later, in the serving process:
if saved["config_version"] != CONFIG_VERSION:
    raise RuntimeError(f"stale checkpoint: {saved['config_version']}")
restored = momentum(window=10).loads(saved["blob"])
```

## The checkpoint contract

Saving is controlled by the outside: **a model cannot refuse a checkpoint**
(there is no hook for it to reject one). The practical consequence:
everything a model needs to resume must be in `ctx.state` at every step
boundary — and it must be picklable, or `save`/`dumps` raise
`CheckpointError`. A path save is atomic: it streams to a temporary file and
replaces the target only on success, so a failed save never destroys an
existing checkpoint. It does not fsync the containing directory, so this is
an atomic-replacement guarantee, not a promise of durability across sudden
power loss.

```python
model.predict(day_inputs)
model.save(path)             # checkpoint at a call boundary
```

## Replaying fit input

After a load you may re-`fit` the same training data; the engine skips rows
at or before the cursor without supplying them to the model function. The
engine does not verify that the replayed rows match what was originally
processed — replaying the same inputs is the caller's responsibility.

`predict` never skips: replayed output timestamps raise `TimeOrderError`.

## Compatibility across versions

A checkpoint is mostly your own data — whatever the model put in `ctx.state`
is pickled as-is — so the library cannot promise that old checkpoints load
into new library versions. qfeval-model keeps the envelope format stable when
it can, but provides no version-migration machinery; treat checkpoints as
tied to the code (and library version) that wrote them.

A load across library versions emits a `QfevalModelWarning`. In CI or a
serving process that should fail closed, escalate it to an error:

```python
import warnings

import qfeval_model as qm

warnings.filterwarnings("error", category=qm.QfevalModelWarning)
```

See [`examples/en/05_checkpoint_serving.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/05_checkpoint_serving.py).
