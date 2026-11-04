"""``assert_deterministic`` (DESIGN §Verification)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from qfeval_model.core import Model, RawData, as_wide
from qfeval_model.verify._compare import (
    compare_outputs,
    execute,
)


def assert_deterministic(
    make_model: Callable[[], Model],
    inputs: RawData,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    symbols: Sequence[str] | None = None,
    runs: int = 2,
) -> None:
    """Assert ``runs`` fresh runs under the same seed are exactly equal.

    Exact means zero tolerance with missing values matching themselves
    (``NaN`` == ``NaN``, ``NaT`` == ``NaT``); it is a value comparison, so
    ``+0.0`` and ``-0.0`` compare equal.

    Args:
        make_model: Factory returning a fresh, unstarted model.
        inputs: Inputs, wide or long (any accepted shape).
        side_data: Optional side-input DataFrame.
        test_from: Fit rows before ``test_from``; ``None`` for predict-only.
        symbols: Explicit symbol universe, passed on each run's first call —
            verify with the same universe as production.
        runs: Number of runs to compare (must be at least 2).

    Raises:
        ValueError: If ``runs < 2`` or a model is already started.
        VerificationError: If any two runs differ (including NaN positions).
    """
    if runs < 2:
        raise ValueError(f"runs must be at least 2, got {runs}")
    wide_inputs = as_wide(inputs)
    outputs = []
    for _ in range(runs):
        model = make_model()
        if model.started:
            raise ValueError("make_model must return an unstarted model")
        outputs.append(
            execute(
                model,
                wide_inputs,
                side_data=side_data,
                test_from=test_from,
                symbols=symbols,
            )
        )
    for other in outputs[1:]:
        compare_outputs(outputs[0], other, rtol=0.0, atol=0.0)
