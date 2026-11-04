"""``assert_checkpoint_consistent`` (DESIGN §Verification).

Verifies that a straight run agrees with a run that saves at ``checkpoint_at``,
loads into a fresh instance, and continues. Models whose behaviour depends on
anything outside ``ctx.state`` will not agree and are expected to fail.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from qfeval_model.core import Model, RawData, as_wide
from qfeval_model.core.normalize import to_utc_ns
from qfeval_model.verify._compare import (
    compare_outputs,
    execute,
    has_rows,
    partition_call,
    slice_output,
)
from qfeval_model.verify.errors import VerificationError


def assert_checkpoint_consistent(
    make_model: Callable[[], Model],
    inputs: RawData,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    checkpoint_at: Any,
    symbols: Sequence[str] | None = None,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert save/restore continuation matches a straight run.

    Runs the model straight through, then re-runs it over the rows before
    ``checkpoint_at``, saves to bytes, loads into a fresh instance, and
    predicts the remainder; the rows from ``checkpoint_at`` onward of both
    are compared.

    Args:
        make_model: Factory returning a fresh, unstarted model.
        inputs: Inputs, wide or long (any accepted shape).
        side_data: Optional side-input DataFrame.
        test_from: Fit rows before ``test_from``; ``None`` for predict-only.
        checkpoint_at: Timestamp at which to checkpoint and restart (the
            restarted run predicts the rows from ``checkpoint_at`` onward). At
            least one row must have been processed before this boundary,
            either by fit or by the pre-checkpoint prediction.
        symbols: Explicit symbol universe, passed on each run's first call —
            verify with the same universe as production (the restored
            instance takes its universe from the checkpoint).
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.

    Raises:
        VerificationError: If the boundary leaves no processed row before it
            or no prediction row after it, or if the two runs disagree.
            Models that keep their history outside ``ctx.state`` / child state
            are expected to fail here by design (history held *in* state
            checkpoints fine).
    """
    wide_inputs = as_wide(inputs)
    checkpoint_ns = to_utc_ns(checkpoint_at)

    straight = execute(
        make_model(),
        wide_inputs,
        side_data=side_data,
        test_from=test_from,
        symbols=symbols,
    )
    straight_tail = slice_output(straight, straight.timestamps >= checkpoint_ns)

    # ``symbols`` rides the first call only; a started checkpoint carries the
    # universe to the restored instance.
    model = make_model()
    if test_from is not None:
        train, rest = partition_call(wide_inputs, side_data, to_utc_ns(test_from))
        if has_rows(train[0]):
            model.fit(
                train[0],
                side_data=train[1],
                symbols=symbols,
            )
            symbols = None
    else:
        rest = (wide_inputs, side_data)

    if not has_rows(rest[0]):
        raise VerificationError("no rows remain at or after test_from")
    mid, tail = partition_call(*rest, checkpoint_ns)
    if has_rows(mid[0]):
        model.predict(
            mid[0],
            side_data=mid[1],
            symbols=symbols,
        )
        symbols = None

    if not model.started:
        raise VerificationError(
            "no rows were processed before checkpoint_at; choose a checkpoint_at after "
            "the first processed timestamp"
        )

    restored = make_model().loads(model.dumps())

    if not has_rows(tail[0]):
        raise VerificationError("no rows remain after checkpoint_at")
    restored_tail = restored.predict(
        tail[0],
        side_data=tail[1],
        symbols=symbols,
    )

    compare_outputs(straight_tail, restored_tail, rtol=rtol, atol=atol)
