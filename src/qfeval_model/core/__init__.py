"""The minimal core: engine, ``Model`` / ``Features``, input I/O, split.

Everything outside ``core`` (the external-feature packages ``zoo`` /
``metrics`` / ``verify``) imports core names from this module, with one
exception: the package-internal SPI of ``qfeval_model.core.normalize``
(``to_utc_ns`` / ``normalize_timestamp_series`` / ``split_side_frame``) is
deliberately not re-exported here and is imported from that module directly.

``as_wide`` / ``as_frame`` are the public RawData conversions (re-exported by
the top-level facade), the counterparts of ``Features.to_wide`` / ``to_frame``.

See DESIGN §The mental model and §Specification.
"""

from qfeval_model.core.constants import (
    DEFAULT_FEATURE,
    SYMBOL,
    TIMESTAMP,
    Mode,
)
from qfeval_model.core.context import Context, Step
from qfeval_model.core.errors import (
    CheckpointError,
    QfevalModelWarning,
    TimeOrderError,
)
from qfeval_model.core.features import Features
from qfeval_model.core.model import (
    Model,
    ModelFactory,
    batch_model,
    model,
)
from qfeval_model.core.normalize import (
    RawData,
    as_frame,
    as_wide,
)
from qfeval_model.core.split import TrainTest, split

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
    "as_frame",
    "as_wide",
    "batch_model",
    "model",
    "split",
]
