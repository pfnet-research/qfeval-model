"""Exception and warning hierarchy (DESIGN §Errors and warnings).

Only failures a user may want to catch and recover from get a dedicated
class; contract violations (bad inputs, model-function bugs) raise standard
``ValueError`` / ``RuntimeError``. Error messages should state what went
wrong, the expected and actual values, and where the violated constraint was
first established (e.g. which call fixed the symbol universe).
"""


class TimeOrderError(Exception):
    """A predict call includes timestamps at or before the model cursor.

    Within one call, unsorted or duplicate timestamps are normalized (sorted,
    last-non-NaN aggregated), so this is raised for regressions or replays
    across predict calls. Fit calls replay-skip those timestamps instead.

    Catchable: a caller replaying a feed with out-of-order rows may drop the
    offending call and continue.
    """


class CheckpointError(Exception):
    """Checkpoint save/restore failed.

    Identity mismatch, malformed checkpoint data, or ``ctx.state`` that
    cannot be pickled. Catchable: fall back to retraining.
    """


class QfevalModelWarning(UserWarning):
    """Base class for all qfeval_model warnings."""
