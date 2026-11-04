"""The verification harness's error type.

``VerificationError`` belongs to the ``verify`` package: the core engine
never raises it.
"""


class VerificationError(Exception):
    """A ``qfeval_model.verify`` assertion found mismatching outputs."""
