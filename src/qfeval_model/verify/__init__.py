"""Verification harness (a satellite package; DESIGN §Verification).

Assertions:

- ``assert_deterministic``: identical runs under the same seed.
- ``assert_stream_consistent``: one-shot vs. per-timestamp ``predict``
  (including batch/step equivalence for batch models).
- ``assert_checkpoint_consistent``: straight run vs. save/load/continue.

``check_all`` runs every applicable assertion at once and returns a
:class:`CheckReport`.
"""

from qfeval_model.verify.check import CheckReport, CheckResult, check_all
from qfeval_model.verify.checkpoint import assert_checkpoint_consistent
from qfeval_model.verify.determinism import assert_deterministic
from qfeval_model.verify.errors import VerificationError
from qfeval_model.verify.stream import assert_stream_consistent

__all__ = [
    "CheckReport",
    "CheckResult",
    "VerificationError",
    "assert_checkpoint_consistent",
    "assert_deterministic",
    "assert_stream_consistent",
    "check_all",
]
