"""Verification harness contracts: each assert_* is tested both ways —
a correct model passes and a model with the targeted defect fails.

Every negative model is defined inline with exactly one defect:

- non_deterministic: seeds nothing (``np.random.default_rng()``).
- reset_prologue: resets ``state["sum"]`` in the prologue on every call.
- global_memory: remembers across calls in a closure, not in ``ctx.state``.
- leaky_batch: reads the next row (t+1) inside a batch chunk.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from conftest import days, wide

import qfeval_model as qm
from qfeval_model.verify import (
    VerificationError,
    assert_checkpoint_consistent,
    assert_deterministic,
    assert_stream_consistent,
    check_all,
)

W4 = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
W1 = wide([[1.0, 2.0]])


def _delta() -> qm.Model:
    """A correct state machine: day-over-day difference kept in ``ctx.state``."""

    @qm.model
    def delta(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"]
            prev = ctx.state.get("prev")
            step.output = close * np.nan if prev is None else close - prev
            ctx.state["prev"] = np.array(close)

    return delta()


def _cumsum_batch() -> qm.Model:
    """A correct batch model: running cumulative sum carried via ``ctx.state``."""

    @qm.batch_model
    def cumsum(ctx: qm.Context) -> None:
        for step in ctx:
            base = ctx.state.get("total", 0.0)
            total = base + np.cumsum(step.inputs["close"], axis=0)
            step.output = total
            ctx.state["total"] = np.array(total[-1])

    return cumsum()


# --- assert_deterministic ----------------------------------------------------


@pytest.mark.parametrize("runs", [2, 3])
def test_deterministic_passes_with_step_rng(runs: int) -> None:
    @qm.model
    def noisy(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"]
            step.output = close + step.rng.standard_normal(close.shape)

    assert_deterministic(noisy, W4, runs=runs)


def test_deterministic_rejects_unseeded_rng() -> None:
    @qm.model
    def non_deterministic(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"]
            step.output = np.random.default_rng().standard_normal(close.shape)

    with pytest.raises(VerificationError, match="output mismatch"):
        assert_deterministic(non_deterministic, W4)


def test_deterministic_rejects_runs_below_two() -> None:
    with pytest.raises(ValueError, match="runs must be at least 2"):
        assert_deterministic(_delta, W4, runs=1)


def test_deterministic_rejects_started_factory() -> None:
    started = _delta()
    started.predict(W4)
    with pytest.raises(ValueError, match="unstarted"):
        assert_deterministic(lambda: started, W4)


def test_deterministic_keeps_the_factory_seed_by_default() -> None:
    # Seeding is controlled by the factory, exactly like the other verify
    # assertions.
    @qm.model
    def record(ctx: qm.Context, out: list[int]) -> None:
        for step in ctx:
            out.append(step.seed)
            step.output = step.inputs["close"]

    factory_seeded: list[int] = []

    def make() -> qm.Model:
        m = record(factory_seeded)
        m.seed = 7
        return m

    assert_deterministic(make, W1)

    default_seeded: list[int] = []
    assert_deterministic(lambda: record(default_seeded), W1)

    assert len(set(factory_seeded)) == 1
    assert len(set(default_seeded)) == 1
    assert factory_seeded[0] != default_seeded[0]


# --- assert_stream_consistent --------------------------------------------------


def test_stream_consistent_passes_for_correct_state_machine() -> None:
    assert_stream_consistent(_delta, W4)


def test_stream_consistent_passes_with_test_from_and_side_data() -> None:
    side_data = pd.DataFrame({"timestamp": days(4), "w": [1.0, 2.0, 3.0, 4.0]})

    @qm.model
    def weighted_delta(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"] * float(step.side_inputs["w"])
            prev = ctx.state.get("prev")
            step.output = close * np.nan if prev is None else close - prev
            ctx.state["prev"] = np.array(close)

    assert_stream_consistent(
        weighted_delta, W4, side_data=side_data, test_from="2020-01-03"
    )


def test_verify_treats_matching_nat_side_outputs_as_equal() -> None:
    # Side inputs represent missing datetimes as NaT, so a model forwarding a
    # datetime side input must verify cleanly; NaT != NaT elementwise, so the
    # comparator has to treat matching NaT (like NaN) as equal.
    side_data = pd.DataFrame(
        {
            "timestamp": days(4),
            "event": pd.DatetimeIndex([days(4)[0], pd.NaT, days(4)[2], pd.NaT]),
        }
    )

    @qm.model
    def forwards_event(ctx: qm.Context) -> None:
        for step in ctx:
            step.side_outputs["event"] = step.side_inputs["event"]
            step.output = step.inputs["close"]

    assert_deterministic(forwards_event, W4, side_data=side_data)
    assert_stream_consistent(forwards_event, W4, side_data=side_data)


def test_stream_consistent_rejects_prologue_reset() -> None:
    @qm.model
    def reset_prologue(ctx: qm.Context) -> None:
        ctx.state["sum"] = 0.0  # the defect: state reset on every call
        for step in ctx:
            ctx.state["sum"] += float(np.sum(step.inputs["close"]))
            step.output = step.inputs["close"] * 0.0 + ctx.state["sum"]

    with pytest.raises(VerificationError, match="output mismatch"):
        assert_stream_consistent(reset_prologue, W4)


def test_stream_consistent_passes_for_sparse_writer_with_explicit_schema() -> None:
    # The never-write shorthand is chunking-invariant only for the standard
    # (S,)/float64 schema (DESIGN §4.2). A sparse model with a custom dtype,
    # a trailing axis, and a side output stays invariant by writing its
    # schema-specific unwritten values explicitly on no-signal steps.
    @qm.model
    def sparse_vector(ctx: qm.Context) -> None:
        for step in ctx:
            signal = step.inputs["close"][0] > 2.0
            value = np.ones((len(step.symbols), 2), dtype=np.float32)
            step.output = value if signal else np.full_like(value, np.nan)
            step.side_outputs["hit"] = np.bool_(signal)

    assert_stream_consistent(sparse_vector, W4)


def test_stream_consistent_rejects_quiet_sparse_writer_with_custom_dtype() -> None:
    # The same model relying on the shorthand instead: a quiet call falls
    # back to (S,)/float64, so per-timestamp streaming drifts from the
    # one-shot call and verify reports it at the join point.
    @qm.model
    def sparse_quiet(ctx: qm.Context) -> None:
        for step in ctx:
            if step.inputs["close"][0] > 2.0:
                step.output = np.ones((len(step.symbols), 2), dtype=np.float32)

    with pytest.raises(VerificationError, match="dtype differs"):
        assert_stream_consistent(sparse_quiet, W4)


# --- batch stream consistency -------------------------------------------------


def test_stream_consistent_passes_for_correct_batch_model() -> None:
    assert_stream_consistent(_cumsum_batch, W4)


def test_stream_consistent_rejects_in_chunk_lookahead() -> None:
    @qm.batch_model
    def leaky_batch(ctx: qm.Context) -> None:
        for step in ctx:
            # The defect: each row reads the next row (t+1) of the chunk.
            step.output = np.roll(step.inputs["close"], -1, axis=0)

    with pytest.raises(VerificationError, match="output mismatch"):
        assert_stream_consistent(leaky_batch, W4)


# --- assert_checkpoint_consistent ----------------------------------------------


def test_checkpoint_consistent_passes_for_state_only_model() -> None:
    assert_checkpoint_consistent(_delta, W4, checkpoint_at="2020-01-03")


def test_checkpoint_consistent_rejects_split_before_any_processed_row() -> None:
    # An unstarted checkpoint only compares two fresh full runs and verifies no
    # continuation state. Treat a head/before-head typo as an invalid check.
    with pytest.raises(VerificationError, match="no rows were processed"):
        assert_checkpoint_consistent(_delta, W4, checkpoint_at="2020-01-01")


def test_checkpoint_consistent_allows_first_test_row_after_fit() -> None:
    # The test-period prefix may be empty when training has already started the
    # model; this still exercises restoring fitted state before prediction.
    assert_checkpoint_consistent(
        _delta,
        W4,
        test_from="2020-01-03",
        checkpoint_at="2020-01-03",
    )


def test_checkpoint_consistent_rejects_memory_outside_state() -> None:
    memory: dict[str, float] = {"sum": 0.0}

    @qm.model
    def global_memory(ctx: qm.Context) -> None:
        for step in ctx:
            # The defect: the running sum lives outside ctx.state.
            step.output = step.inputs["close"] * 0.0 + memory["sum"]
            memory["sum"] += float(np.sum(step.inputs["close"]))

    with pytest.raises(VerificationError, match="output mismatch"):
        assert_checkpoint_consistent(global_memory, W4, checkpoint_at="2020-01-03")


def test_checkpoint_consistent_rejects_checkpoint_after_all_rows() -> None:
    with pytest.raises(VerificationError, match="no rows remain after checkpoint_at"):
        assert_checkpoint_consistent(_delta, W4, checkpoint_at="2021-01-01")


def test_check_all_flags_lookahead_batch_child_under_single_parent() -> None:
    # A single parent may register a batch child, so its leak safety is only
    # as good as the child's (docs/explanation/leak-safety.md). The
    # composition stays checkable: the child's in-chunk look-ahead surfaces
    # through the parent's row-sliced child outputs and fails the stream
    # check on the composed model.
    @qm.batch_model
    def chunk_last(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"]
            step.output = close * 0.0 + close[-1]

    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = chunk_last()
        for step in ctx:
            step.output = step.children["c"].array

    report = check_all(parent, W4)
    assert not report.ok
    streamed = next(r for r in report.results if r.name == "stream_consistent")
    assert streamed.status == "fail"


def test_verify_assertions_accept_the_model_run_settings() -> None:
    # Parity gate: every Model.run execution setting must be accepted by the
    # verify assertions, so a model can be verified under exactly its
    # production conditions (seed is the documented exception — it is set
    # inside make_model via model.seed).
    settings = set(inspect.signature(qm.Model.run).parameters)
    settings -= {"self", "seed"}
    for assertion in (
        assert_deterministic,
        assert_stream_consistent,
        assert_checkpoint_consistent,
        check_all,
    ):
        params = set(inspect.signature(assertion).parameters)
        missing = settings - params
        assert not missing, f"{assertion.__name__} lacks {sorted(missing)}"


# --- make_model returning the same instance -------------------------------------


def test_streamed_match_rejects_shared_instance() -> None:
    shared = _delta()
    with pytest.raises(ValueError, match="fresh, distinct, unstarted"):
        assert_stream_consistent(lambda: shared, W4)


# --- check_all -------------------------------------------------------------------


def test_check_all_passes_single_model_with_three_checks() -> None:
    report = check_all(_delta, W4)
    assert [r.name for r in report.results] == [
        "deterministic",
        "stream_consistent",
        "checkpoint_consistent",
    ]
    assert all(r.status == "pass" for r in report.results)
    assert report.ok
    assert report.all_passed
    report.raise_if_failed()  # no-op on an all-pass report


def test_check_all_batch_model_uses_the_same_three_checks() -> None:
    report = check_all(_cumsum_batch, W4)
    assert [r.name for r in report.results] == [
        "deterministic",
        "stream_consistent",
        "checkpoint_consistent",
    ]
    assert report.ok


def test_check_all_reports_checkpoint_skip_with_one_test_timestamp() -> None:
    report = check_all(_delta, W1)
    assert [(r.name, r.status) for r in report.results] == [
        ("deterministic", "pass"),
        ("stream_consistent", "pass"),
        ("checkpoint_consistent", "skip"),
    ]
    skip = report.results[-1]
    assert "two test timestamps" in skip.message
    assert report.ok  # a skip is visible but does not fail the report
    assert not report.all_passed  # but the strict gate refuses a skip
    report.raise_if_failed()
    assert "SKIP  checkpoint_consistent" in str(report)


def test_check_all_checkpoint_at_is_forwarded() -> None:
    # Boundary observation: a valid split passes; a split beyond the data
    # reaches assert_checkpoint_consistent and fails there.
    assert check_all(_delta, W4, checkpoint_at="2020-01-03").ok
    report = check_all(_delta, W4, checkpoint_at="2021-01-01")
    checkpoint = next(r for r in report.results if r.name == "checkpoint_consistent")
    assert checkpoint.status == "fail"
    assert "no rows remain after checkpoint_at" in checkpoint.message

    before = check_all(_delta, W4, checkpoint_at="2019-01-01")
    checkpoint = next(r for r in before.results if r.name == "checkpoint_consistent")
    assert checkpoint.status == "fail"
    assert "no rows were processed before checkpoint_at" in checkpoint.message


def test_check_all_checkpoint_at_skips_with_one_timestamp() -> None:
    # Pins current behavior: with a single test timestamp the checkpoint
    # check is reported as skipped even when checkpoint_at is explicit.
    report = check_all(_delta, W1, checkpoint_at="2020-01-01")
    checkpoint = next(r for r in report.results if r.name == "checkpoint_consistent")
    assert checkpoint.status == "skip"
    assert report.ok


def test_check_all_reports_prologue_reset_failure() -> None:
    def make() -> qm.Model:
        @qm.model
        def reset_prologue(ctx: qm.Context) -> None:
            ctx.state["sum"] = 0.0  # the defect: state reset on every call
            for step in ctx:
                ctx.state["sum"] += float(np.sum(step.inputs["close"]))
                step.output = step.inputs["close"] * 0.0 + ctx.state["sum"]

        return reset_prologue()

    report = check_all(make, W4)
    assert not report.ok
    failed = [r for r in report.results if r.status == "fail"]
    assert failed and all(r.message for r in failed)
    with pytest.raises(VerificationError, match="stream_consistent: "):
        report.raise_if_failed()
    rendered = str(report)
    assert "PASS" in rendered
    assert "FAIL" in rendered


def test_check_all_absorbs_model_exceptions_as_failures() -> None:
    @qm.model
    def broken(ctx: qm.Context) -> None:
        raise RuntimeError("boom before any comparison")

    report = check_all(broken, W4)
    assert not report.ok
    assert all(r.status == "fail" for r in report.results)
    assert all("boom before any comparison" in r.message for r in report.results)


def test_check_all_fail_message_names_the_exception_type() -> None:
    # str(KeyError("close")) is just "'close'"; the report must say more.
    @qm.model
    def broken(ctx: qm.Context) -> None:
        raise KeyError("close")

    report = check_all(broken, W4)
    failed = [r for r in report.results if r.status == "fail"]
    assert failed
    assert all(r.message == "KeyError: 'close'" for r in failed)


def test_check_all_broken_factory_raises_before_any_check() -> None:
    # A factory that cannot build a model violates the make_model contract;
    # the raw exception propagates up front, not after a partial report.
    def make() -> qm.Model:
        raise RuntimeError("factory is broken")

    with pytest.raises(RuntimeError, match="factory is broken"):
        check_all(make, W4)


# --- first-call settings parity ------------------------------------------------


def test_check_all_passes_the_declared_universe_to_every_run() -> None:
    # Production runs under an explicit universe wider than the data; every
    # verify run (one-shot, streamed, checkpoint restart) must see that same
    # universe, not one inferred from the inputs.
    @qm.model
    def needs_universe(ctx: qm.Context) -> None:
        for step in ctx:
            assert list(step.symbols) == ["A", "B"], "wrong universe"
            step.output = np.zeros(2)

    only_a = {"close": pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=days(4))}
    assert check_all(needs_universe, only_a, symbols=["A", "B"]).ok
    assert not check_all(needs_universe, only_a).ok


def test_verify_does_not_treat_diagnostics_as_state_across_calls() -> None:
    # info is output-only call-wide diagnostics, not persistent model state.
    # The one-shot run increments through its rows, while streamed/checkpoint
    # calls each start from an empty dict. Verification exposes that chunking
    # dependency.
    @qm.model
    def counter(ctx: qm.Context) -> None:
        for step in ctx:
            step.info["calls"] = step.info.get("calls", 0) + 1
            step.output = step.inputs["close"] * step.info["calls"]

    report = check_all(counter, W4)
    assert not report.ok
    assert next(r for r in report.results if r.name == "deterministic").status == "pass"
    assert (
        next(r for r in report.results if r.name == "stream_consistent").status
        == "fail"
    )
    assert (
        next(r for r in report.results if r.name == "checkpoint_consistent").status
        == "fail"
    )


def test_removed_verify_arguments_and_alias_are_not_public() -> None:
    assert not hasattr(qm.verify, "assert_batch_equivalent")
    assert "seed" not in inspect.signature(assert_deterministic).parameters
    for assertion in (
        assert_deterministic,
        assert_stream_consistent,
        assert_checkpoint_consistent,
        check_all,
    ):
        assert "info" not in inspect.signature(assertion).parameters
