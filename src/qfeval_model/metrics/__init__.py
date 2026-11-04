"""``qm.metrics``: standard evaluation metrics for outputs.

Forward-return computation lives here — and only here. Getting the ``shift``
direction wrong turns an evaluation into a leak, so users should never write
it by hand, and the zoo deliberately has no forward-looking model to compose
with (DESIGN principle: the library itself never routes the future into a
model input).

Every metric is a pure function of a :class:`~qfeval_model.core.Features`
(and/or inputs) — metrics are never composable as model inputs. One module per
category; modules depend only on the core's public surface, so the package can
grow without widening the core.
"""

from qfeval_model.metrics.correlation import hit_rate, information_coefficient
from qfeval_model.metrics.pnl import long_short_pnl
from qfeval_model.metrics.returns import forward_returns

__all__ = [
    "forward_returns",
    "hit_rate",
    "information_coefficient",
    "long_short_pnl",
]
