# Zoo conventions

Every `qm.zoo` entry is an ordinary `@qm.model`; the zoo has no private
machinery you could not write yourself. What makes the zoo coherent is a
small set of shared conventions, listed here once so each docstring only
has to state its own formula.

## Sources

Most entries take a first argument `source` accepting a feature name
(`str`) or a child model (`qm.Model`), so entries compose freely:
`rolling_mean(simple_return(), 20)` is the moving average of a child
model's output. OHLCV entries accept one source per input
(`high=`, `low=`, `close=`, `volume=`), each a feature name or model.
Ensembles take a list of models instead.

All sources pass one shared boundary:

- Values must be real numeric. Complex inputs raise instead of silently
  dropping the imaginary part.
- Values must be finite or NaN. Infinity raises — `finite_indicator` is
  the one deliberate exception, so it can flag non-finite cells.
- A stateful model locks the source shape on its first step; every later
  step, and every other source of the same model, must match it exactly.
  Implicit NumPy broadcasting across sources is rejected.

## Axes

Axis 0 is the symbol axis; anything after it is trailing axes, processed
independently (see [trailing axes](trailing-axes.md)). Cross-sectional
entries (`cross_sectional_*`) operate over axis 0 at one timestamp;
everything else is elementwise over the cross-section.

## Windows and warm-up

`window` counts observation steps — not elapsed time — and the window
trails, including the current step. Warm-up is strict
(`min_periods=window`): the output is NaN until `window` steps have been
observed. Entries that difference first (`simple_return`, RSI, the
psychological line, DMI) need one extra observation before their first
output.

## NaN

A NaN anywhere in a cell's window makes that cell's output NaN. The
explicit missing-data models are exempt: `rolling_count`, `stale_age`,
the `rolling_valid_*` family, `ffill`, `fill_constant`, `finite_indicator`,
`coalesce`, and the expanding family (which aggregates all finite
observations so far). The `rolling_valid_*` models (`rolling_valid_sum`,
`rolling_valid_mean`, `rolling_valid_var`, `rolling_valid_std`) have a
per-cell observation window: NaN does not advance it, output starts after
that cell has accumulated `window` finite values, and a later NaN repeats
the previous aggregate.
`recursive_ewm_mean` carries its state through missing inputs
(`ignore_na=True` semantics).

## Numeric edges

Zero denominators, zero variance, empty valid sets, and domain errors
(log of a non-positive value, overflow) yield NaN — never infinity, never
a NumPy warning. Variance-like statistics default to `ddof=1` and expose
`ddof` explicitly. Quantile methods are restricted to `"linear"` and
`"inverted_cdf"`; arbitrary method strings are not passed through.

## Output scales

The zoo default is the natural unit of the formula (fractions for rates
and `[0, 1]` ranks). Technical indicators keep their conventional market
scales instead — RSI and stochastics are 0–100, RCI is -100–100 — and
names carry the scale where confusion is likely
(`moving_average_deviation_percent`). No entry outputs buy/sell
decisions; thresholding is the caller's model.

## Multi-line indicators

An indicator with several lines becomes several factories
(`bollinger_upper_band`, `macd_line`, `ichimoku_base_line`, ...), one per
line, each usable as another model's `source`. Nothing is packed into
anonymous trailing axes.

## Checkpoint state

Rolling entries keep bounded buffers of owned copies in `ctx.state` (never
views into a call's input arrays), so every zoo model is checkpoint-ready
at every step boundary and passes `verify.check_all` — the contract test
suite runs every public entry through it.

Reducible rolling statistics (sum/mean, moments, covariance, extrema, and
linear weights) keep prefix/suffix aggregates in a two-stack queue. Each
observation is merged a constant number of times, so total work is O(T), not
O(T * window). Means and variances are locally centered and combined with
Chan's formula; their error therefore does not grow with the full series
length. This is the streaming form of the chunk decomposition described in
[Numerically stable and fast moving mean and variance](https://imoz.jp/scraps/202607_mvar.html).
Order statistics such as median and quantile, plus path-dependent maximum
drawdown, still materialize the bounded window because they do not share that
simple aggregate.
