"""String literals shared across the library and exposed to users.

These name the fixed axes (``TIMESTAMP`` / ``SYMBOL``) and the default feature
name (``DEFAULT_FEATURE``) — the key an unnamed flat wide frame is stored under
on the input side and the key the main output lives under on the output side.
Referencing them instead of bare strings keeps the library and user code in
sync with the wire contract. ``Mode`` is the model-mode literal type;
``"single"`` (per-step) and ``"batch"`` (whole-input) are written inline.
"""

from __future__ import annotations

from typing import Literal

TIMESTAMP = "timestamp"
"""Reserved column naming the input's time axis in long-format frames."""

SYMBOL = "symbol"
"""Reserved column naming the input's symbol axis in long-format frames."""

Mode = Literal["single", "batch"]
"""A model's execution mode: ``"single"`` (per timestamp) or ``"batch"`` (per call)."""

DEFAULT_FEATURE = "close"
"""Default feature name.

On the input side, a flat wide DataFrame passed without a name is stored
under this key. On the output side, the single main output lives under this
key (read key-free via ``Features.array``) and it names the value column of
``Features.to_frame``.
"""
