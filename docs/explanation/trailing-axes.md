# Trailing axes

The input in this library has two fixed axes: **timestamp × symbol**. Any
extra structure a feature or an output carries — horizons, order-book levels,
strikes and expiries, factors, tenors — lives in **anonymous trailing axes**:
axes after `(T, S)` that have a shape but no names and no labels. The classic
`(T, S)` input is simply the zero-trailing-axes case, and every existing call
keeps working unchanged.

## Two fixed axes, then shapes

Time keeps its special role: stepping, the monotonic cursor, replay skip, seed
derivation, and checkpoint stable points all work on the timestamp axis and
are untouched by trailing axes. The symbol axis is the fixed second axis: its
labels (sorted, unique, non-empty strings) are fixed on the first call
(`Model.symbols`), and new symbols on a later call raise `ValueError`.

Everything beyond those two axes is *only a shape*. A feature is an array of
shape `(T, S, *trailing)`; the trailing shape is just the array's shape,
nothing more. There is no vocabulary of dimension names or label arrays to
declare and keep in sync — what each trailing position means (which horizon,
which order-book level) lives in your code. This is a deliberate trade: less
machinery in the library, fewer declarations in models, and refusal instead
of guessing wherever a label would be needed (see below).

## Trailing axes at the input boundary

Wide frames — the basic format, passed as a `{feature_name: frame}` mapping —
carry trailing axes in their columns:

- a flat column `Index` is the plain case: columns are symbols, the feature
  is `(T, S)`;
- `MultiIndex` columns add trailing axes: **level 0 is the symbol**, and each
  deeper level is one anonymous trailing axis. The sorted unique values of a
  level fix that axis's order and size for the call; the label values
  themselves are *not retained* (they only order and size the axis), and
  level names are ignored.

```python
columns = pd.MultiIndex.from_product([symbols, ["1d", "5d"]])
iv = pd.DataFrame(values, index=timestamps, columns=columns)
model.fit({"close": close, "iv": iv})   # close: (T, S), iv: (T, S, 2)
```

Features may differ in trailing shape. The symbol universe is the union over
all features (plus any `symbols=` declaration), and each feature is reindexed
to the cross product of the universe and its own trailing labels; missing
cells become NaN (float features only — a non-float feature with holes is
rejected).

Long frames (the transport format) carry 2-D features only, and a *bare*
DataFrame with MultiIndex columns is rejected — pass wide data as a
`{feature_name: frame}` mapping to be explicit.

## Trailing axes in outputs

Nothing is declared up front. A model writes whatever per-symbol shape it
needs, and the first `step.output` assignment of a call locks the output's
trailing shape for that call — the same rule as the dtype lock. (The lock is
per call, not per model lifetime; cross-call drift is diagnosed where results
are joined, by the verify gates.) A multi-horizon model is just:

```python
@qm.model
def multi_horizon_return(ctx: qm.Context, horizons: tuple[int, ...] = (1, 2)) -> None:
    for step in ctx:
        close = step.inputs["close"]                       # (S,)
        out = np.full((len(close), len(horizons)), np.nan)
        ...                                                # one column per horizon
        step.output = out                                  # (S, H) — locks (H,)
```

A later assignment in the same call with a different trailing shape (or
dtype) raises `ValueError`. Side outputs follow the same per-name lock. The
output dtype must be real floating-point, so unwritten cells are always `NaN`.

## Reading the results

`Features` is plain data: `pred.timestamps` `(T,)`, `pred.symbols` `(S,)`,
`pred.array` `(T, S, *trailing)`, plus `side_data` and `info`. For a
plain `(T, S)` output, `pred.to_frame()` renders the long
`timestamp, symbol, close` frame and `pred.to_wide()` the wide
timestamp × symbol frame.

With trailing axes, `pred.to_wide()` renders the anonymous axes by **integer
position** (the axes have no labels, and the library never invents any). Its
columns become a `MultiIndex` — level 0 is the symbol, each deeper level one
trailing axis's positions `0..n-1` — which is exactly the wide layout the
input boundary accepts, so the frame round-trips as a feature of the next
model. `pred.to_frame()` and `as_frame()` reject trailing axes and point to
the wide representation; long format remains a strictly 2-D transport.
What each position means still lives in your code.

`pred.select_trailing(i, ...)` selects one integer position from each
trailing axis and returns a `Features` view with a plain `(T, S)` output. For
example, `pred.select_trailing(0)` selects the first horizon from a `(T, S,
H)` result. This deliberately narrow operation covers evaluation without
adding axis labels or a general-purpose array replacement API.

The returned arrays and dicts are not defensively copied or wrapped
read-only; mutating them is undefined behaviour.

## What refuses trailing axes, and why

The evaluation metrics (`information_coefficient`, `hit_rate`,
`long_short_pnl`) are financially defined over the plain timestamp × symbol
input. An output whose output has trailing axes (`output.ndim != 2`) is
**rejected** rather than silently averaged. Select one position with
`pred.select_trailing(...)` (for example, evaluate one horizon at a time), or
have the model emit an already-aggregated plain output. When in doubt, the
library refuses.

The cross-sectional zoo models need no axis argument:
`qm.zoo.cross_sectional_rank(source)` ranks over the symbol axis,
independently along every trailing axis of the source;
`cross_sectional_demean` works the same way. Time-direction zoo models
(`lag`, `rolling_mean`, `recursive_ewm_mean`, ...) are elementwise over the
cross-section and work with any trailing shape unchanged.

The verify gates normalize inputs into the wide `{feature: frame}` dict and
slice it by the timestamp index, so models with trailing-axis inputs
(MultiIndex columns) run through every gate like any other model.

## Memory: the cross product is dense

Inputs are dense: a feature over symbols × strikes × expiries materializes
`T × S × K × E` cells, NaN-filled where data is missing. That explodes
quickly (10⁴ × 10³ × 50 × 20 = 10¹⁰ cells) — an unallocatable cross product
fails with `MemoryError` at normalization. If you hit it, reduce the symbol
universe or the trailing axes (sparse or ragged inputs are out of scope by
design).

## Checkpoints

Trailing axes add nothing to checkpoints: an envelope stores `ctx.state`,
the children, and the fixed symbol universe, and output shapes are per-call,
so there is nothing about trailing shapes to persist. See
[checkpoint semantics](checkpoint-semantics.md) for the envelope contract.
