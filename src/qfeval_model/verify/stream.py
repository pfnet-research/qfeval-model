"""``assert_stream_consistent`` (DESIGN §Verification)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from qfeval_model.core import Model, RawData
from qfeval_model.verify._compare import assert_streamed_match


def assert_stream_consistent(
    make_model: Callable[[], Model],
    inputs: RawData,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    symbols: Sequence[str] | None = None,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert one-shot ``predict`` agrees with per-timestamp ``predict``.

    Under the re-invocation execution model this checks that the model's
    prologue is idempotent and its state handling is correct. For a batch
    model it also detects in-chunk look-ahead that changes the output when
    execution is reduced to chunks of one timestamp.

    Args:
        make_model: Factory returning a fresh, unstarted model.
        inputs: Inputs, wide or long (any accepted shape).
        side_data: Optional side-input DataFrame.
        test_from: Fit rows before ``test_from``; ``None`` for predict-only.
        symbols: Explicit symbol universe, passed on each run's first call —
            verify with the same universe as production.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.

    Raises:
        ValueError: If ``make_model`` returns the same or an already-started
            model.
        VerificationError: If the two runs disagree.
    """
    assert_streamed_match(
        make_model,
        inputs,
        side_data=side_data,
        test_from=test_from,
        symbols=symbols,
        rtol=rtol,
        atol=atol,
    )
