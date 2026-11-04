"""``verify.check_all``: run every applicable verification gate at once."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import pandas as pd

from qfeval_model.core import Model, RawData, as_wide
from qfeval_model.core.normalize import to_utc_ns
from qfeval_model.verify._compare import all_timestamps
from qfeval_model.verify.checkpoint import assert_checkpoint_consistent
from qfeval_model.verify.determinism import assert_deterministic
from qfeval_model.verify.errors import VerificationError
from qfeval_model.verify.stream import assert_stream_consistent


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one verification check.

    Attributes:
        name: The check's name (e.g. ``"deterministic"``).
        status: ``"pass"``, ``"fail"``, or ``"skip"`` (the check could not
            run on the given data; the message says why).
        message: The failure or skip reason, or ``""`` on success.
    """

    name: str
    status: Literal["pass", "fail", "skip"]
    message: str = ""


@dataclass(frozen=True)
class CheckReport:
    """The result of :func:`check_all`.

    Attributes:
        results: One :class:`CheckResult` per check, skipped checks included.
    """

    results: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether no check failed (skipped checks do not fail the report)."""
        return all(result.status != "fail" for result in self.results)

    @property
    def all_passed(self) -> bool:
        """Whether every check ran and passed (a skipped check fails this).

        Use this as a release gate when the checks must actually have run
        (``ok`` treats a skip — e.g. too little data to checkpoint between
        timestamps — as not-a-failure).
        """
        return all(result.status == "pass" for result in self.results)

    def raise_if_failed(self) -> None:
        """Raise :class:`VerificationError` if any check failed."""
        failed = [result for result in self.results if result.status == "fail"]
        if failed:
            raise VerificationError("; ".join(f"{r.name}: {r.message}" for r in failed))

    def __str__(self) -> str:
        """Render one ``PASS``/``FAIL``/``SKIP`` line per check."""
        lines = []
        for result in self.results:
            suffix = f" — {result.message}" if result.message else ""
            lines.append(f"{result.status.upper()}  {result.name}{suffix}")
        return "\n".join(lines)


def check_all(
    make_model: Callable[[], Model],
    inputs: RawData,
    *,
    side_data: pd.DataFrame | None = None,
    test_from: Any = None,
    checkpoint_at: Any = None,
    symbols: Sequence[str] | None = None,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> CheckReport:
    """Run every applicable verification gate and return a report.

    Executes ``assert_deterministic`` and ``assert_stream_consistent`` always,
    ``assert_checkpoint_consistent`` when at least two test timestamps exist
    (checkpointing at ``checkpoint_at`` or the midpoint; otherwise reported
    as skipped). For batch models, stream consistency also checks chunk-size-1
    equivalence and can detect in-chunk look-ahead. Every check runs the model
    with the seed the factory set (set ``model.seed``
    inside ``make_model`` to check a non-default seed). Model failures inside
    a check become FAIL results; a ``make_model`` that cannot construct a
    model at all violates its contract, and that exception propagates before
    any check runs.

    Args:
        make_model: Factory returning a fresh, unstarted model (a decorated
            model with no required parameters works directly).
        inputs: Inputs, wide or long (any accepted shape).
        side_data: Optional side-input DataFrame.
        test_from: Fit rows before ``test_from``; ``None`` for predict-only.
        checkpoint_at: Timestamp for the checkpoint/restart check (default:
            the middle test timestamp). At least one row must have been
            processed before this boundary.
        symbols: Explicit symbol universe, passed on each run's first call —
            verify with the same universe as production.
        rtol: Relative tolerance for float comparison.
        atol: Absolute tolerance for float comparison.

    Returns:
        A :class:`CheckReport`; inspect ``report.ok`` or print the report.
    """
    wide_inputs = as_wide(inputs)
    # Probe the factory once up front: a factory that cannot even build a
    # model violates the make_model contract, so the exception propagates
    # before any check runs instead of surfacing after a partial report.
    make_model()
    common: dict[str, Any] = {
        "side_data": side_data,
        "test_from": test_from,
        "symbols": symbols,
    }
    checks: list[tuple[str, Callable[[], None]]] = [
        (
            "deterministic",
            partial(assert_deterministic, make_model, wide_inputs, **common),
        ),
        (
            "stream_consistent",
            partial(
                assert_stream_consistent,
                make_model,
                wide_inputs,
                rtol=rtol,
                atol=atol,
                **common,
            ),
        ),
    ]

    results: list[CheckResult] = []
    skipped: list[CheckResult] = []
    test_ts = all_timestamps(wide_inputs)
    if test_from is not None:
        test_ts = test_ts[test_ts >= to_utc_ns(test_from)]
    if len(test_ts) >= 2:
        at = checkpoint_at if checkpoint_at is not None else test_ts[len(test_ts) // 2]
        checks.append(
            (
                "checkpoint_consistent",
                partial(
                    assert_checkpoint_consistent,
                    make_model,
                    wide_inputs,
                    checkpoint_at=at,
                    rtol=rtol,
                    atol=atol,
                    **common,
                ),
            )
        )
    else:
        skipped.append(
            CheckResult(
                name="checkpoint_consistent",
                status="skip",
                message="needs at least two test timestamps to checkpoint between them",
            )
        )

    for name, check in checks:
        try:
            check()
        except Exception as exc:
            # Include the exception type: a bare ``str(KeyError("close"))``
            # renders as just ``'close'``, unreadable in a report.
            message = f"{type(exc).__name__}: {exc}"
            results.append(CheckResult(name=name, status="fail", message=message))
        else:
            results.append(CheckResult(name=name, status="pass"))
    results += skipped

    return CheckReport(results=tuple(results))
