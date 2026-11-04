"""Prediction models over panel data (timestamp x symbol) as decorated functions.

This module is the public facade (re-exports only; no logic lives here).
The internal layering is documented in DESIGN §Architecture and code map:

- ``qfeval_model.core``: the minimal core — execution engine, ``@model`` /
  ``@batch_model``, ``Model``, ``Features``, DataFrame <-> Features I/O, and
  the ``split`` workflow helper.
- ``qfeval_model.zoo`` / ``.metrics`` / ``.verify``: external
  features — loosely coupled packages that depend only on the core's public
  surface and can grow without widening the core.
"""

from qfeval_model import metrics, verify, zoo
from qfeval_model._version import __version__
from qfeval_model.core import (
    DEFAULT_FEATURE,
    SYMBOL,
    TIMESTAMP,
    CheckpointError,
    Context,
    Features,
    Mode,
    Model,
    ModelFactory,
    QfevalModelWarning,
    RawData,
    Step,
    TimeOrderError,
    TrainTest,
    as_frame,
    as_wide,
    batch_model,
    model,
    split,
)

__all__ = [
    "DEFAULT_FEATURE",
    "SYMBOL",
    "TIMESTAMP",
    "CheckpointError",
    "Context",
    "Features",
    "Mode",
    "Model",
    "ModelFactory",
    "QfevalModelWarning",
    "RawData",
    "Step",
    "TimeOrderError",
    "TrainTest",
    "__version__",
    "as_frame",
    "as_wide",
    "batch_model",
    "metrics",
    "model",
    "split",
    "verify",
    "zoo",
]
