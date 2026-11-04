"""``qm.split``: train/test split of model inputs (half-open, wide output).

``test_from`` names the inclusive start of the test side, so the train side is
the half-open interval ``[.., test_from)`` — the same convention as ``range``
and numpy slicing.

``split`` accepts any inputs shape (wide mapping, flat wide DataFrame, or
long DataFrame) and always returns the normalized wide format: a
``{input: wide frame}`` dict per side, directly acceptable by
``fit`` / ``predict``. Only the timestamp index is sliced, so MultiIndex
columns (anonymous trailing axes) pass through untouched.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pandas as pd

from qfeval_model.core.normalize import RawData, as_wide, to_utc_ns


class TrainTest(NamedTuple):
    """The ``(train, test)`` pair of wide input dicts returned by :func:`split`."""

    train: dict[str, pd.DataFrame]
    test: dict[str, pd.DataFrame]


def split(inputs: RawData, *, test_from: Any) -> TrainTest:
    """Split ``inputs`` into wide-format ``(train, test)`` at ``test_from``.

    The test side starts at ``test_from`` (inclusive); the train side is the
    half-open interval before it (``train: < test_from, test: >= test_from``).

    Args:
        inputs: Model inputs in any accepted shape — a wide mapping of
            ``{input: frame}``, a flat wide DataFrame, or a long DataFrame.
        test_from: First timestamp of the test side (inclusive). tz-naive
            values are taken as UTC; tz-aware values are converted to UTC.

    Returns:
        A ``(train, test)`` :class:`TrainTest` named tuple; each side is the
        normalized ``{input: wide frame}`` dict and a non-empty side can be
        passed to ``fit`` / ``predict`` directly. Either side may come back
        empty (zero rows), which is not a valid model input — check before
        calling, the way ``Model.run`` skips the fit when the train side is
        empty. Long inputs are pivoted once (cross-product cells absent from
        the long frame appear as NaN).
    """
    at = to_utc_ns(test_from)
    # ``as_wide`` returns normalized frames (tz-naive ns DatetimeIndex, no NaT,
    # sorted), so the index compares directly with ``at`` and every row lands
    # on exactly one side.
    wide = as_wide(inputs)
    return TrainTest(
        train={name: frame[frame.index < at] for name, frame in wide.items()},
        test={name: frame[frame.index >= at] for name, frame in wide.items()},
    )
