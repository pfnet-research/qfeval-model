# How to get deterministic randomness

Pass the seed to the first call — it is a model-lifetime setting, like
`symbols=`, and passing it to a later call raises:

```python
model.fit(train_inputs, seed=7)
# or, in a backtest:
model.run(inputs, seed=7)
```

Inside a model, use the step's derived generator rather than a global RNG:

```python
for step in ctx:
    step.output = step.rng.random(step.symbols.shape[0])
```

## How the seed is derived

Each step's seed is derived from the model seed and the step's timestamp:

```
step.seed = blake2b(model_seed, timestamp) mod 2**64
step.rng  = numpy.random.Generator(numpy.random.Philox(key=step.seed))
```

For a single-step model this makes randomness reproducible independent of how
the timeline is split into calls. A batch step keys on its chunk's **first**
timestamp, so its seed *does* change with the chunking: a batch model must not
let `step.rng` affect its per-row output (`verify.check_all` detects the
violation), and one that needs per-timestamp randomness should derive it per
timestamp itself.

A child's seed is derived once, at first registration, from the parent's seed
and the child's registration name:

```
child.seed = blake2b(parent_seed, name) mod 2**64
```

so same-structured siblings — and the parent itself — draw distinct streams.
This automatic derivation is the only stable child-seeding contract. Directly
changing an engine-owned child's seed, or calling `fit` / `predict` on it, is
an undefined expert operation — children advance only through their parent.

## Guarantee boundary

Only library-derived randomness is reproducible. External RNG (`random`,
`numpy.random` globals), non-deterministic BLAS, wall-clock, and external APIs
are **not** covered. Use `verify.assert_deterministic` to catch accidental
non-determinism.

See [`examples/en/cookbook/randomness.py`](https://github.com/pfnet-research/qfeval-model/blob/main/examples/en/cookbook/randomness.py).
