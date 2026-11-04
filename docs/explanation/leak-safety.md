# Leak safety

Timestamps are **available-at**: an input row at timestamp `t` represents
information that was fully available by `t`. A model may read up to the current
row for `t`, and no further.

## Why `@qm.model` is the main road

With `@qm.model`, each step is handed exactly one timestamp's inputs (`(S,)`),
so the ordinary way to write a model — process each step as it arrives —
leaves no natural place to read a future row. Children advance ahead of the
parent within a call, but the engine runs its own child instances: the
candidates the model function registers never advance, so a child's live
`state` offers no path to future rows either (per-step child outputs are
read via `step.children`). `info` is not a parent/child data path in either
direction: a child's call-wide `info` is never row-sliced, so it is not
exposed to the parent at all (each `step.children[...]` view carries an
empty `info`) — only the row-sliced main/side outputs cross the boundary.
This is a strong default, not a
proof: retaining steps (`steps = list(ctx)`) and reading rows through them
after later steps arrive is outside the contract, and
`verify.assert_stream_consistent` can often, but not always, detect the
resulting inconsistency. Single-step mode is the recommended API for most
models.

## Why `@qm.batch_model` is advanced

`@qm.batch_model` hands the whole chunk to one step (`(T, S)`) for speed. The
engine **cannot** structurally prevent a batch model from reading future rows
within the chunk (`close[1:]`), so leak safety becomes the author's
responsibility. Always pair a batch model with
`verify.assert_stream_consistent` (included in `verify.check_all`), which compares one-shot execution
against per-timestamp execution and flags in-chunk look-ahead. The check is
not a complete guarantee — batch models are an expert tool.

The same caveat extends through composition: a single-mode parent may
register a batch child, and the child processes the parent's whole effective
call *before* the parent's steps run. If that batch child looks ahead within
the chunk, its row-sliced outputs feed the future into the single parent too
— the parent's `@qm.model` decorator alone does not certify the composition.
Compose only batch children that have themselves passed
`verify.check_all` / `assert_stream_consistent`, and verify the composed
model as well.

## Forward returns

Forward returns are a label for evaluation, not an input. Feeding them (or any
future-derived value) into `inputs` / `side_data` leaks. That is why forward
returns are computed only inside `qm.metrics` (`forward_returns`,
`information_coefficient`, `hit_rate`, `long_short_pnl`) — and why `qm.zoo`
deliberately contains no forward-looking model to compose with. Never write
the `shift(-1)` yourself (see the [backtest tutorial](../tutorials/04-backtest.md)).
