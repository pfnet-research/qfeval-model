# qfeval-model: Design

## 1. Overview

qfeval-model runs prediction models over panel data: values indexed by
timestamp and symbol (for example, a stock ticker). Models are decorated
Python functions. The engine advances them through time, manages child models,
and saves state for restoration after a restart. NumPy and pandas are the only
runtime dependencies.

This document specifies the execution and data contracts and explains their
design. In the example below, `prices` is a wide pandas DataFrame with a
`DatetimeIndex` and one column per symbol. Use positive prices spanning the
split date, with at least ten observations before the test period and two
timestamps in it. The model compares each price with its rolling mean; the
metric evaluates that signal against later price changes (forward returns).

```python
import qfeval_model as qm


@qm.model
def momentum(ctx: qm.Context, window: int = 20) -> None:
    ctx.child = qm.zoo.rolling_mean("close", window=window)
    for step in ctx:  # one step per timestamp
        step.output = step.inputs["close"] / step.child.array - 1.0


pred = momentum(window=10).run(prices, test_from="2024-01-01", seed=0)
daily_ic = qm.metrics.information_coefficient(pred, prices)

# Match the model parameters and split used above; seed 0 is the default.
report = qm.verify.check_all(
    lambda: momentum(window=10), prices, test_from="2024-01-01"
)
assert report.all_passed, str(report)
```

`daily_ic` contains the rank correlation across symbols at each test
timestamp. `report.all_passed` confirms that every verification check ran and
passed on these inputs (§4.8).

## 2. Design philosophy

The design prioritizes explicit state, forward execution, and a small public API.

### 2.1 Function-based models

A model is a function of the form `def f(ctx, params...)` with a decorator.
The decorator returns a factory that accepts the function's parameters,
excluding the injected `ctx`, and creates a model instance. Model authors
implement one loop; the engine supplies `fit` and `predict` without requiring
a base class or registry.

The factory preserves the function's name and docstring, and an instance
prints as the call that created it (`rolling_mean('close', window=2)`).
Persistent model state lives in the plain dictionary `ctx.state`. Defaults,
keyword arguments, and closures over constants work as in ordinary functions.

### 2.2 Time management

Input timestamps represent *available-at* times: the moment when the row's
information became known. With `@qm.model`, the engine delivers exactly one
step per timestamp. Each step receives its current row; models retain earlier
values in state. Retaining steps (`steps = list(ctx)`) can expose later rows
and violates the execution contract. `verify.assert_stream_consistent` can
detect inconsistencies caused by this misuse on the tested inputs.

The engine maintains a cursor at the last processed timestamp. `predict`
rejects rows at or before the cursor, while `fit` skips them to support
replay. Returning to an earlier state requires restoring an earlier
checkpoint into a fresh instance; cloning copies the current state.

Forward returns are computed in `qm.metrics` for evaluation. Zoo transforms
read only current and past data. Callers must ensure their inputs follow the
same availability rule, because the engine cannot detect future information
already embedded in an input row.

`@qm.batch_model` supports vectorized computation: one step receives the
whole call's array, shaped `(T, S, *trailing)` (§3). The author must avoid
look-ahead within that chunk and verify behavior across call boundaries.
The [execution-mode benchmark](benchmarks/README.md#bench_modespy--backward-return-t5000-s200-1m-cells)
records a speedup for a backward-return transform over one million cells.
Performance on other workloads depends on their computation and input costs.

### 2.3 Re-invocation for checkpointing

Long backtests and serving loops need to resume after a restart. A suspended
generator frame cannot be pickled, so each `fit` or `predict` call invokes the
model function afresh. `run` composes these calls (§4.7).

State needed across calls belongs in `ctx.state` or in child models, which
are saved recursively. The engine keeps no separate input history. Mutable
values in closures or bound arguments may survive in memory, but are outside
the checkpoint contract.

The caller chooses when to save between calls; the model cannot refuse or
defer a checkpoint. Models must therefore keep their state picklable at every
step boundary. `clone()` uses the same save-and-restore path, so both operations
preserve the same state (§4.5).

### 2.4 Core and supporting packages

The core (`qfeval_model.core`) defines execution, the `Model` and `Features`
types, DataFrame normalization, and `split`. Built-in models (`zoo`),
evaluation (`metrics`), and verification (`verify`) use the core as separate
packages. They can grow without expanding its public API.

These packages depend on the core independently. CI enforces their import
boundaries with import-linter; §5.1 lists the boundaries and the internal
normalization interface used by verification. Extensions are ordinary models
and functions; the core has no plugin or hook mechanism.

### 2.5 Data ownership and error handling

`Features` is a mutable dataclass backed by engine buffers; `ctx.state` is a
plain dictionary, and step arrays are NumPy views. Accessing results does not
copy them. Treat input and child-output buffers as read-only, and copy arrays
retained in state (§4.2).

The public boundary rejects malformed inputs and shape or dtype mismatches.
A failed call does not roll back state; failures after model execution begins
make the instance unusable (§4.3). Models must keep state picklable, avoid
look-ahead, and replay consistent inputs. Modifying internal attributes or
input buffers is outside the execution contract.

### 2.6 Code size and public concepts

Input and output share one type; children and state use dictionaries.
Configuration uses model parameters and first-call arguments, without a
separate configuration object. When designs meet the same requirements, the
shorter implementation is preferred.

## 3. API and data flow

The public API connects model definition, execution, and results:

| Concept | What it is |
| --- | --- |
| `@qm.model` / `@qm.batch_model` | Decorators that turn a function into a model factory; calling the factory binds parameters and returns a `Model`. |
| `Model` | An executable instance: bound parameters + symbol universe + `state` + time cursor + seed + children. Provides `fit` / `predict` / `run` and checkpointing. |
| `Context` (`ctx`) | The injected first argument of the model function: yields steps, exposes `state` and `children`. |
| `Step` | A single decision point: reads `inputs` / `side_inputs` / `children` and writes `output` / `side_outputs` / `info`. |
| `Features` | The container for normalized inputs and prediction results: `timestamps`, `symbols`, named arrays, side data, and `info`. The engine uses it internally; `fit`, `predict`, and `run` accept DataFrames or mappings (`RawData`). |
| `qm.split` / `Model.run` | Half-open train/test splitting and the fit-then-predict composition. |
| `qm.zoo` / `qm.metrics` / `qm.verify` | Supporting packages for built-in models, evaluation, and verification. |

The engine normalizes inputs, executes the model, and collects outputs in
`Features`. `predict` returns this result; `fit` discards it and returns the
model itself.

```
your DataFrames               wide {name: frame} | flat wide frame | long frame
      │  normalize (core/normalize.py): UTC timestamps, sorted rows,
      │  last-non-NaN dedup, panel expansion, reindex to the fixed
      │  symbol universe — always into fresh copies
      ▼
Features ── timestamps (T,) · symbols (S,) · data {name: (T, S, *)} · side_data · info
      │  model execution (one fit or predict = one run of the function)
      │      for step in ctx:  read step.inputs → write step.output
      ▼
Features ── data {"close": (T, S, *)} (the main output) · side_data · info
      │  .array / .to_wide() / .to_frame()
      ▼
NumPy array / wide DataFrame / long DataFrame
```

To pass predictions to another model, convert them with `to_wide()`. The
execution methods accept DataFrames or mappings, as specified in §4.1.

In each panel array, the first two axes represent `T` timestamps and `S`
symbols. Further dimensions are anonymous trailing axes: they have sizes but
no names or labels. For example, a single-mode model predicting `H` horizons
writes `(S, H)` at each step, producing a `(T, S, H)` result. An ordinary
`(T, S)` panel has no trailing axes.

## 4. Specification

### 4.1 The input boundary

Raw input arguments to `fit`, `predict`, `run`, `split`, verification, and
metrics accept three forms:

1. **A `{name: frame}` mapping of wide DataFrames.** This is the standard form.
   Each frame has a `DatetimeIndex` of timestamps and one column per symbol.
   `MultiIndex` columns encode trailing axes: level 0 is the symbol, and each
   deeper level becomes one anonymous trailing axis. The sorted unique values
   of each level determine that axis's order and size for the call; the labels
   themselves are discarded.
2. **A flat wide DataFrame.** This is shorthand for a single feature, stored
   under the default name `"close"` (`qm.DEFAULT_FEATURE`).
3. **A long DataFrame with `timestamp` and `symbol` columns.** This is the
   transport format for files and databases. It supports 2-D features only.

A DataFrame passed directly with only one of the two reserved columns is
ambiguous and rejected. For `MultiIndex` columns, pass the frame in a mapping.
Use `as_wide` to normalize raw inputs into a wide mapping, or `as_frame` for
a long DataFrame. `RawData` is the type alias for these accepted input types.

Normalization guarantees, applied identically to every shape:

- **Timestamps** are normalized to tz-naive UTC `datetime64[ns]` (tz-aware
  values are converted, while naive values are interpreted as UTC); any
  datetime unit is accepted, but `NaT` is rejected. Wide indexes and long
  `timestamp` columns must already be datetime-typed.
- **Rows are sorted** by timestamp; duplicate `(timestamp, symbol)` cells keep
  the **last non-NaN** value in row order.
- The call's time axis is the **sorted union** of all input features' timestamps.
  Each feature is expanded onto the `(T, S, *trailing)` grid defined by that
  axis, the symbol universe, and its own trailing labels. Cells a feature does
  not cover become NaN, requiring a floating-point or complex dtype. A feature
  of another dtype must cover every cell. For example, object-dtype sector
  labels are accepted with full coverage and reach `step.inputs` unchanged.
- Feature names and symbol labels must be strings; `timestamp` / `symbol` are
  reserved feature names. Side-data column names must also be strings. pandas
  nullable numerics are converted to `float64` at this boundary, exactly once.
- Every produced array is a fresh copy. The engine neither shares memory with
  nor modifies the input DataFrames.
- Inputs must contain at least one feature, one row, and one symbol.

The symbol universe, the set of symbols the model processes, is fixed by the
first call. Pass `symbols=` on that call to declare it explicitly. Later
inputs are reindexed to this universe: missing symbols become NaN columns,
while symbols outside it cause an error. Feature names, dtypes, trailing
shapes, and side-data keys may vary across calls; each model reads the inputs
it needs.

Side data travels alongside the panel. `side_data=` expects a long DataFrame
with a `timestamp` column; each remaining column becomes one `(T,)` series
with a scalar per timestamp. Columns retain their dtype. Object columns can
carry arbitrary Python values, including call-level metadata.

Uncovered timestamps become `None` for object columns, NaN for floating-point
or complex columns, and NaT for datetime or timedelta columns. Integer,
boolean, and other dtypes without missing-value representations require full
coverage of the input timestamps.

Side timestamps must be a subset of the call's input timestamps. Duplicate
timestamps keep the last non-missing value in row order.

### 4.2 The step protocol

The model function processes each call with `for step in ctx:`. The two modes
share one `Step` class and differ only in how they group rows:

| Property | `@qm.model` (single) | `@qm.batch_model` (batch) |
| --- | --- | --- |
| steps per call | one per timestamp | one for the whole call |
| `step.inputs[name]` | `(S, *trailing)` | `(T, S, *trailing)` |
| `step.side_inputs[name]` | scalar | `(T,)` |
| time | `step.timestamp` | `step.timestamps` |
| `step.output` | `(S, *trailing)` | `(T, S, *trailing)` |
| `step.side_outputs[k]` | `(*trailing)` | `(T, *trailing)` |
| `step.children[name]` | child's row for this step | child's whole chunk |

Only the time attribute for the current mode is available; accessing the
other raises an error. Steps also expose `symbols`, `phase` (`"train"` or
`"test"`), the `is_train` and `is_test` shorthands, and a reproducible `seed`
and `rng` (§4.6).

Input, side-input, and child-output arrays are views into buffers spanning the
whole call. Storing a view in `ctx.state` keeps the entire buffer alive. Read
these arrays directly within a step, but use `.copy()` for arrays retained in
state across steps or calls.

The first write to `step.output` (or to a given `step.side_outputs[name]`) in a
call allocates its buffer and fixes the dtype and trailing shape. Later writes
in that call must match. Reading an assigned `step.output` returns a writable
view, allowing updates such as `step.output[mask] = v`.

The main output must have a real floating-point dtype so unwritten cells can
be NaN. An integer buffer would make unwritten rows look like valid zeros.
Side outputs accept any non-object dtype. Their unwritten cells default to
NaN for floating-point or complex buffers, NaT for datetime or timedelta, and
zero otherwise.

If the main output is never written, the result is an all-NaN float64 array
of shape `(T, S)`. A side output that is never written is absent from the
result. This fallback preserves results across chunkings only for models with
the standard output schema: `(S,)` float64 per step and no side outputs.

A sparse model using another main-output dtype or trailing axes must write
NaNs of the correct dtype and shape even on steps with no signal. Models with
side outputs must likewise write every key's default value on no-signal steps,
using that output's dtype and shape. Otherwise, a call with no writes falls
back to the standard schema and differs from a larger chunk containing a write.

The engine checks output schemas within each call. Verification also checks
compatibility when joining results from separate calls (§4.8). Callers that
combine results themselves must handle changes in dtype, shape, or keys.

`step.info` collects output diagnostics in a dictionary, created empty for each
call and shared by its steps. It is returned with predictions.
Model configuration belongs in arguments, per-timestamp auxiliary inputs in
`side_data`, and persistent state in `ctx.state`.

### 4.3 The execution model: re-invocation and monotonic time

Each `fit` or `predict` call that reaches execution invokes the model function
once. The `for step in ctx:` loop yields that call's steps and ends; the
function then returns normally. Local variables do not persist across calls.
Keep state needed for the next call in `ctx.state` or child models.

Code before the loop, the prologue, runs on every call. Make it idempotent:
for example, `ctx.state.setdefault(...)` initializes a value only when absent.
Mutable values in closures or bound arguments are outside the checkpoint
contract, even if they survive in the current process.

A call must consume every step exactly once. Breaking out early raises
`RuntimeError`; the cursor does not advance and the first call's symbol
universe is not committed. State changes already made are kept, and children
that ran before the parent's steps are not rolled back.

Any failure after the model function starts executing marks the instance as
failed. It refuses further calls and checkpoints. Recover by restoring the
last good checkpoint into a fresh instance:
`restored = same_construction().loads(last_good_checkpoint)`. Input errors
raised before execution leave the instance reusable.

Iterating `ctx` twice in one call is an error. Async and generator functions
are rejected at decoration time.

`fit` and `predict` run the same loop. They differ in the phase flag, treatment
of rows at or before the cursor, and return value. `fit` discards output and
returns `self`; `predict` returns `Features`. The engine allows prediction
without prior fitting.

The time cursor records the last processed timestamp. Within a call,
normalization sorts and deduplicates timestamps so they strictly increase.
Across calls:

- `fit` skips rows at or before the cursor without warning. Restore a
  checkpoint and replay the training feed; only new rows reach the model's
  loop. The caller must replay the same data, because the engine does not
  verify skipped rows.
- `predict` raises `TimeOrderError` if any row is at or before the cursor,
  preventing duplicate predictions.

Dividing a timeline into calls must preserve results. For example, processing
a year of 252 trading days in one `predict` call must match 252 daily calls.
Single-mode step seeds also match, because they derive from each timestamp.
Batch seeds depend on chunk boundaries, so batch models must not base per-row
output on these chunk-dependent draws (§4.6). Use
`verify.assert_stream_consistent` to compare whole-call and per-timestamp
execution on supplied inputs.

Pass `symbols=` and `seed=` on the first call to set the universe and seed
explicitly. Passing either argument to a later call is an error. Direct
assignment to `model.seed` is also supported; its effects on child models are
specified in §4.6.

### 4.4 Composition: children

A model composes other models by registering them by name before iteration
starts:

```python
@qm.model
def spread(ctx: qm.Context) -> None:
    ctx.children["fast"] = qm.zoo.rolling_mean("close", window=5)
    ctx.children["slow"] = qm.zoo.rolling_mean("close", window=20)
    for step in ctx:
        step.output = step.children["fast"].array - step.children["slow"].array
```

`ctx.child = m` / `step.child` are shorthands for the name `"child"`.
Registration is explicit: a model passed as a function parameter becomes a
child only when placed in `ctx.children`.

The candidate model must be unstarted. Registration builds an engine-owned
instance from the candidate's function and bound parameters. The candidate
itself never advances; registering it under two names creates separate child
instances. Children and their descendants are included in checkpoints and
clones (§4.5).

Before the parent begins stepping, each child processes the whole call with
the same effective inputs and phase. Its seed derives from the parent's seed
and registration name (§4.6). The parent then reads the child's outputs,
sliced for its current step. Each child follows its own execution mode and
advances only through the parent.

Parent and child read the same inputs separately. The parent cannot pass
computed values into the child; it combines the child's output with its own
computation.

The function registers candidates on every call, but the engine keeps the
instances created on the first call. Names, order, and identity (function
`__module__:__qualname__` and mode) must match that first registration.
Bound parameters are not compared; the host must reconstruct candidates with
the same parameters.

Only main and side outputs flow from child to parent. A child's `info` is
call-wide and could contain values from later timestamps, so it is omitted
from `step.children[...]`. The parent's `info` is also not passed to children;
each child gets a fresh diagnostics dictionary.

### 4.5 Checkpointing

A checkpoint is a versioned, pickled envelope:

```python
{
  "format_version": 1,          # the only version this code reads
  "library_version": "0.1.0",   # mismatch on load → warning, not an error
  "ident": "mymodels:momentum", # the function's __module__:__qualname__
  "mode": "single",             # or "batch"
  "symbols": ...,               # the fixed universe (or None)
  "cursor": ...,                # last processed timestamp (or None)
  "seed": 0,
  "state": {...},               # ctx.state, pickled as-is
  "children": [{"name": ..., "model": ...}, ...],  # envelopes in order
}
```

The engine adds no input history or original DataFrames to the checkpoint.
History retained in `ctx.state` is saved with that state. Model code and bound
parameters are excluded; the host must supply the same function, parameters,
and child structure when reconstructing the model for restoration.

- The API uses pickle's vocabulary: `save(file)` / `load(file)` and `dumps()` /
  `loads(data)`. The file-based methods accept paths or file objects and stream
  the data. `load` / `loads` restore an **unstarted** instance in place and
  return `self`, so `restored = momentum(10).loads(blob)` is a single
  expression.
- Identity consists of the function's `__module__:__qualname__` and mode.
  Functions with the same qualified name in different modules have different
  identities. Changing the defining module or qualified name prevents loading
  existing checkpoints; editing code alone leaves identity unchanged.
  Identity and format mismatches raise `CheckpointError`. A `library_version`
  mismatch warns and proceeds; loading across library versions is best-effort
  (see the [compatibility policy](docs/explanation/compatibility.md)).
- The top-level envelope is validated before anything is committed, so a
  failed top-level restore leaves the model untouched. Child envelopes are
  retained and applied on the first call, after the function registers its
  children. The engine builds fresh children and matches them by position and
  name before restoring their state. A malformed nested envelope therefore
  fails on that first call.
- After a restore, `fit` replay-skips the already-processed rows (§4.3), so
  a serving loop can rebuild, load, replay the feed, and continue.
- `clone()` = `dumps()` → `loads()` into a fresh instance: one code path
  preserves the same state as checkpoint restoration. This allows branching
  from the current state for alternative runs. Bound parameters remain shared
  (§6).
- Unpickling can execute code embedded in the bytes. Load only trusted
  checkpoints.

### 4.6 Determinism

The model seed is an integer, defaulting to 0. Set it with `seed=` on the
first call or by assigning `model.seed` directly.

A child's seed is derived at first registration as the low 64 bits of
`blake2b(parent seed, registration name)`. Mixing in the name gives
same-structured siblings separate random streams. Derivation happens once;
changing the parent's seed later does not propagate. Direct mutation of an
engine-owned child's seed has undefined behavior.

Each step derives `step.seed` as the low 64 bits of
`blake2b(model seed, step's first timestamp)`. `step.rng` is a Philox generator
keyed by that seed and created on first access.

A single-mode step uses its own timestamp. With the model seed unchanged,
its random stream is the same across call boundaries and checkpoint
restoration. A batch step uses its chunk's first timestamp. Its randomness
is reproducible for the same call boundaries; adding a boundary, including
one at a checkpoint, can change the seed and any per-row output that uses it.

This contract covers library-derived randomness. Authors of models using
external random generators, non-deterministic BLAS, wall clocks, or network
calls must ensure their own reproducibility.

### 4.7 Backtesting and evaluation

`Model.run` splits inputs into training and test periods, calls `fit` and
`predict`, and returns predictions:

```python
pred = momentum(10).run(prices, test_from="2024-01-01", seed=0)
```

`test_from` is the inclusive start of the test period. Rows before it are used
for fitting; rows at or after it are used for prediction. Fitting is skipped
if there are no training rows. The call updates the instance, so clone it
first to explore alternatives from the same state. With `test_from=None`,
`run` only predicts.

Splitting uses timestamps and does not infer when information became
available. The standalone function
`qm.split(inputs, test_from=...) -> TrainTest(train, test)` uses the same
half-open intervals. It returns training and test data as normalized wide
`{name: frame}` dictionaries.

`qm.metrics` provides `forward_returns` and evaluates predictions with
`information_coefficient` (rank correlation across symbols at each timestamp),
`hit_rate` (sign agreement, excluding zero predictions), and `long_short_pnl`
(profit and loss of a rank-weighted, market-neutral portfolio).

Forward returns use `price_{t+n} / price_t - 1`, implemented with `shift(-n)`.
Here, `n` counts price-observation rows; rows with no price for any symbol do
not consume a period. This future-dependent computation belongs to evaluation.
Zoo transforms and supplied model features must follow the availability rule
in §2.2.

Metrics accept only `(T, S)` outputs and reject trailing axes. Use
`pred.select_trailing(...)` to select one horizon before evaluation, or
aggregate horizons in the producing model. Prices may contain NaN coverage
gaps; zero and infinite values are rejected. Forward returns are computed on
the price data before being aligned to prediction axes; missing cells remain
NaN. Extra price timestamps and symbols are allowed, including the future
prices needed to evaluate returns.

### 4.8 Verification

`qm.verify` compares executions to detect problems with prologue idempotence,
state handling, and look-ahead.

Each assertion accepts a `make_model` factory returning a fresh instance,
inputs in any supported form, optional `test_from`, `symbols=`, and side data.
Use the production universe and inputs; set a non-default seed inside
`make_model` via `model.seed`. Results apply to the supplied inputs and do not
prove correctness for all inputs.

| Assertion | Runs | Problems it can reveal |
| --- | --- | --- |
| `assert_stream_consistent` | one-shot run vs. one `predict` per timestamp | non-idempotent prologues, broken state handling, and in-chunk look-ahead in batch models |
| `assert_deterministic` | fresh runs with the same seed, compared with zero tolerance | hidden randomness or ambient state affecting the tested outputs |
| `assert_checkpoint_consistent` | uninterrupted run vs. saving before `checkpoint_at`, restoring into a fresh instance, and predicting from that timestamp | state omitted from the checkpoint that affects the tested outputs |

`check_all` runs the applicable assertions and returns a `CheckReport`. It
needs at least two test timestamps for the checkpoint check; otherwise, it
reports that check as skipped with a reason.

`report.ok` means no check failed, including reports with skips.
`report.all_passed` means every check ran and passed; use it as the release
or serving gate. `str(report)` lists PASS, FAIL, or SKIP for each check, and
`raise_if_failed()` raises if any check failed.

All comparisons require matching timestamps and symbols. Stream and checkpoint
checks use configurable numeric tolerances; determinism uses zero tolerance.
Missing values match themselves, and `+0.0` equals `-0.0`. `info` is excluded.
Main-output mismatches report the first differing timestamp and symbol;
side-output mismatches report the timestamp and trailing-axis positions.
Streamed runs also diagnose changes in dtype, trailing shape, and side-output
keys across calls.

CI runs `check_all` for every public zoo model on fixtures with and without
trailing axes. These models also provide examples of state management and
composition.

### 4.9 Errors and warnings

Dedicated exceptions identify failures callers can handle separately.
Other contract violations use standard exceptions.

| Type | Meaning | Typical response |
| --- | --- | --- |
| `TimeOrderError` | `predict` received timestamps at or before the cursor | remove the overlapping or out-of-order rows and retry |
| `CheckpointError` | identity or format mismatch, unpicklable state, restoration into a started model | rebuild and retrain, or restore a compatible checkpoint into a fresh instance |
| `QfevalModelWarning` | operational warning, such as a library version mismatch on restore | review version compatibility |
| `verify.VerificationError` | a verification assertion found a mismatch | fix the model |
| `ValueError` / `RuntimeError` | malformed inputs, shape or dtype mismatches, unconsumed steps, changed child registrations | fix the input or code; recover as described in §4.3 if execution began |

Error messages identify the failed constraint, expected and received values,
and where the constraint was first established.

## 5. Architecture and code map

```
qfeval_model/
├── __init__.py        # facade: re-exports only, no logic
├── core/              # the complete semantics
│   ├── constants.py   #   TIMESTAMP / SYMBOL / DEFAULT_FEATURE
│   ├── errors.py      #   exception & warning classes
│   ├── features.py    #   Features: the unified in/out panel + output buffers
│   ├── context.py     #   Context / Step: the step protocol
│   ├── model.py       #   @model / @batch_model, Model: engine + checkpointing
│   ├── normalize.py   #   the DataFrame boundary (as_wide / as_frame / normalize_call)
│   └── split.py       #   qm.split
├── zoo/               # built-in models, one module per category
├── metrics/           # evaluation functions
└── verify/            # the verification harness
```

### 5.1 Dependencies and public API

`zoo`, `metrics`, and `verify` depend on the public surface of
`qfeval_model.core` and do not depend on one another. Zoo modules may share
internal helpers. Verification also uses the internal `core.normalize`
helpers `to_utc_ns`, `normalize_timestamp_series`, and `split_side_frame` to
share normalization with the engine.

The leaf modules `constants` and `errors` do not depend on pandas. CI enforces
these package boundaries with import-linter. Names not exported through the
top-level API or the supporting packages' `__all__` are implementation details
without a stability guarantee.

Runtime dependencies are NumPy and pandas. Python 3.10 or later is required.

### 5.2 Zoo conventions

Zoo models use `@qm.model`. Most take a first argument named `source`,
accepting a feature name (`str`) or a child model: for example, `lag("close")`
or `lag(momentum())`. `mean_ensemble` takes a sequence of child models.

Argument names follow pandas (`periods`, `window`, `alpha`, `ddof`). Sources
must be real numeric values, normally finite or NaN. `finite_indicator`
also accepts infinities so it can flag them. Transforms may read only current
and past data.

The [zoo reference](docs/reference/zoo.md) covers time alignment, elementwise
transforms, returns, rolling and cross-sectional statistics, ensembles, risk
models, and technical indicators, with definitions for each model.

### 5.3 Development tools

Development uses uv, Ruff, mypy in strict mode, import-linter, and pytest.
Tests include Hypothesis property checks and are grouped by package and
category.

## 6. Selected design decisions

The execution contracts are specified in §4. The following tradeoffs explain
why the library leaves some responsibilities to the caller.

| Decision | Rationale and consequence |
| --- | --- |
| Checkpoint identity contains only `(__module__:__qualname__, mode)` | Code edits preserve identity, including incompatible edits. The host must verify code and parameter compatibility. Automatic fingerprints would require stable rules for mutable bindings, representations containing memory addresses, and hash-dependent container order. |
| `clone()` shares bound parameters | Cloning restores checkpointed state into a fresh instance (§4.5). Deep-copying arbitrary parameters could fail or change object identity, giving clones different semantics from restores. Construct a new model when different parameters are needed. |
| Input history belongs in model state | State survives restoration. Use `ctx.state` directly or `qm.zoo.with_history` as a child; the engine does not maintain a separate history buffer. |
| Input and output share the `"close"` key | An unnamed wide input uses the key for closing prices. Reusing it for the main output lets results feed another model through `to_wide()` without renaming. |
| Trailing axes have no names or allocation limits | Shapes support outputs such as multiple horizons without an axis-label API. Callers manage labels and memory requirements. Cell counts use Python integers; allocation can still fail when a dense cross product is too large. |
| The symbol universe is fixed | A fixed universe keeps output positions consistent across calls. Declare a wider universe on the first call or construct a new instance when it changes. |
| Output schemas are checked per call | The engine fixes dtype and shape on the first write in each call (§4.2). Verification checks compatibility when joining results; callers that join results themselves must handle schema changes. |
| Dedicated exceptions cover recoverable failures | Callers can handle time-order and checkpoint errors separately. Other contract violations use standard exceptions (§4.9); there is no common library exception base class. |

## 7. Non-goals

The host application handles most of the following features, which are outside
the library's scope:

- Data acquisition, feature stores, and scheduling. The library begins once
  you have DataFrames.
- Portfolio construction, transaction costs, and execution simulation.
  `qm.metrics` measures signal quality; portfolio backtesting requires a
  separate system.
- Dynamic symbol universes, feeds that move backward across calls, and data
  revisions. For corrections, replay corrected inputs into a new instance or
  one restored from a checkpoint preceding the affected data. Rows within a
  call are normalized into timestamp order.
- Parallel or distributed execution, caching layers, and plugin/hook systems.
- Engine-side chunk splitting for batch models. A batch call currently yields
  one step for the whole call; callers that need smaller chunks must split the
  data and make separate calls. A `max_chunk` API is not currently defined.
- Checkpoint format migrations and non-pickle formats. The loader accepts
  only the current format version. Support for older versions can be added
  when a second format exists.
- Alternative DataFrame libraries (Polars, PyArrow). The DataFrame boundary is
  a single module (`core/normalize.py`), where an adapter could be added.
  None is currently included.

## 8. Where to go next

- [Documentation](docs/index.md): tutorials, how-to guides, API reference,
  and explanations. The [execution model](docs/explanation/execution-model.md)
  and [leak safety](docs/explanation/leak-safety.md) pages expand §4.3 and §2.2.
- [Runnable examples](examples/en/README.md)
  ([Japanese](examples/ja/README.md)): nine scripts covering model definition
  through risk models and technical indicators, plus a cookbook. All scripts
  run in CI.
- Source code: read [features.py](src/qfeval_model/core/features.py) and
  [context.py](src/qfeval_model/core/context.py) for the data structures and
  step protocol, then [model.py](src/qfeval_model/core/model.py) for execution
  and checkpointing.
