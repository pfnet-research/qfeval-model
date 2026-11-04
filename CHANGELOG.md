# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(0.x: breaking changes to the public facade are allowed in minor releases and
noted here).

## [0.1.0] - Unreleased

Initial release.

### Added

- **Models as decorated functions.** `@qm.model` (one step per timestamp) and
  `@qm.batch_model` (one step per call) turn a plain function into a
  `ModelFactory`; calling the factory binds parameters and returns an
  executable `Model` with `fit` / `predict` / `run`. Factories are statically
  typed (`ParamSpec`), so parameter typos and type mismatches are caught by
  type checkers.
- **One panel type for input and output.** `Features` carries `timestamps`,
  `symbols`, named `(T, S, *trailing)` `data` arrays, a per-timestamp
  `side_data` channel, and an output-only diagnostics `info` dict; `predict` maps a
  `Features` to a `Features`. `to_wide()` / `to_frame()` render results,
  `as_wide()` / `as_frame()` normalize inputs. Input names, symbol labels,
  and side keys are non-empty strings.
- **DataFrame acceptance.** Wide (`{name: frame}` mappings or a flat wide
  frame stored under `DEFAULT_FEATURE = "close"`) and long
  (`timestamp` / `symbol` / value columns) formats, UTC timestamp
  normalization, last-non-NaN duplicate aggregation, symbol-universe
  reindexing, and anonymous trailing axes via `MultiIndex` columns. Uncovered
  input cells require an inexact dtype; side cells use the dtype's missing
  value where available. Other uncovered cells are rejected rather than
  silently filled. Trailing axes stay in the wide `MultiIndex` format;
  `as_frame()` / `Features.to_frame()` intentionally support 2-D panels only,
  and `Features.select_trailing()` selects a plain result for evaluation.
- **Engine guarantees.** The symbol universe and the model seed are
  model-lifetime settings of the first call (`symbols=` / `seed=` on a later
  call raise `ValueError`); time strictly
  increases across calls (`predict` rejects old rows with `TimeOrderError`,
  `fit` replay-skips them); results are invariant to how the timeline is
  chunked into calls. The main output is real floating-point, so unwritten
  cells are always `NaN` and every metric has one numeric contract. A model
  function must return `None` (an accidentally returned value raises
  `TypeError` instead of being dropped), and a non-string or empty side-output
  name is rejected at write time. An
  instance whose call failed mid-execution is fail-stop:
  nothing is rolled back and it refuses further calls and checkpoints
  (the refusal comes first, before any input or `seed=` handling).
- **Composition.** Child models are registered by name (`ctx.children` /
  `ctx.child`), are engine-owned (registration builds them from the
  candidate's function and bound parameters, so the registered candidate
  never advances), advance in lockstep through their parent, and get their
  seeds derived from the parent's seed and registration name at first
  registration (same-structured siblings draw distinct streams).
- **Checkpointing.** `dumps` / `loads` / `save` / `load` / `clone`; a model
  is checkpoint-ready at every step boundary and can never refuse a
  checkpoint. Model identity is `(__module__:__qualname__, mode)`; envelopes
  are validated at load time; a path save is atomic (a failed save keeps the
  existing file); child state restores at the first call after `loads`.
- **Determinism.** A model seed plus per-step `step.seed` / `step.rng`
  derived from `(seed, timestamp)` only, so single-step randomness is
  independent of call chunking and survives checkpoint/restore (a batch
  step keys on its chunk's first timestamp, so it varies with chunking).
- **`qm.zoo`.** Composable building blocks across time alignment,
  elementwise transforms and algebra, returns, rolling / expanding /
  cross-sectional / pairwise statistics, ensembles, distribution and risk
  statistics (historical VaR and expected shortfall), rolling OLS trends,
  OHLCV primitives, annualized volatility, and classic technical
  indicators (Bollinger bands, envelopes, RSI, stochastics, psychological
  line, RCI, causal Ichimoku lines, rolling-sum DMI). Zoo sources are real
  numeric and finite-or-NaN; complex and infinite inputs are rejected at a
  shared boundary, multiple sources must agree in shape exactly, windows
  are strict (`min_periods=window`), and zero denominators yield NaN
  instead of warnings or infinities. Reducible rolling statistics use
  amortized O(T) prefix/suffix aggregation with locally centered moment
  merges; `rolling_valid_sum` sums each cell's latest finite observations
  without counting NaN; order- and path-dependent statistics retain bounded
  windows.
- **`qm.metrics`.** Evaluation against forward returns: `forward_returns`,
  `information_coefficient`, `hit_rate`, `long_short_pnl`. A forward period
  counts price-observation rows; rows with no price for any symbol do not
  consume the horizon. NaN coverage gaps are allowed, while zero and infinite
  prices are rejected at the shared boundary.
- **`qm.verify`.** Machine-checked model properties: `check_all` runs
  `assert_deterministic`, `assert_stream_consistent`,
  and `assert_checkpoint_consistent`, and reports a check it cannot run on
  the given data as skipped. Stream consistency also checks batch models
  against chunk-size-1 execution.
- **Workflow helpers.** `qm.split` (half-open train/test partition) and
  `Model.run` (fit-then-predict backtest).

Requires Python >= 3.10, numpy >= 1.24, pandas >= 2.1. Fully typed
(`Typing :: Typed`); layer contracts (core vs. satellites) are enforced with
import-linter in CI.
