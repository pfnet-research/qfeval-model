# Repository Guidance

## Preserve model semantics

- A model function is invoked again for every `fit`, `predict`, or `run` call.
  Its prologue must be idempotent, and all cross-call model state must live in
  `ctx.state`; every step boundary must remain checkpoint-ready.
- State must own arrays retained across calls. Never store a row view from a
  step, because it can retain the entire input buffer.
- Zoo models are single-step and backward-looking by default. Use batch mode
  only as an explicit semantic choice, with no in-chunk look-ahead and with
  chunk-boundary behavior documented and verified.
- Reuse the shared source, state, OHLCV, and window helpers instead of
  duplicating their invariants. Stateful and multi-source zoo models must read
  through the shared source boundary.
- Enforce strict-window readiness with returned counts; do not add per-model
  `None`/warm-up branches.
- Keep arithmetic results finite-or-NaN without NumPy warnings; sanitize owned
  results with the shared helpers.

## Respect compatibility boundaries

- `zoo`, `metrics`, and `verify` may depend only on the public `core` surface;
  they must not depend on one another. Their `__init__.py` files contain only
  re-exports and `__all__`.
- Checkpoint identity intentionally contains only the defining
  `module:qualname` and execution mode. Do not add code or parameter
  fingerprints as a routine hardening change.
- Checkpoint format, seed derivation, the Philox generator, and the rule that
  tz-naive timestamps mean UTC are compatibility surfaces. Changes require an
  explicit migration/versioning decision.
- English is the source of truth for bilingual design documents and examples;
  update the Japanese counterpart in the same change.

## Commits and pull requests

- Follow Conventional Commits for commit messages and PR titles. Write PR
  titles in English.
- Write PR descriptions in English first, followed by a line containing exactly
  `-----`, then the Japanese translation.

## Maintaining AGENTS.md

- Keep only project-specific instructions whose removal could worsen
  implementation decisions. Omit facts evident from code, generic advice, and
  unconditional planning, testing, or documentation-reading requirements.
