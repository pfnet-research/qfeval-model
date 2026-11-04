"""Execution engine: step supply, output buffers, time cursor, fail-closed.

All tests drive the engine through the public fit/predict API with inline
recording models. Covers per-step and batch step supply, the per-call
output buffer (dtype/shape locking, NaN defaults, writable views),
side_outputs, the info channel, time-cursor rules (predict rejects the
past, fit silently skips it), and the fail-closed contract for models
that do not consume their steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, wide

import qfeval_model as qm

# --- step supply (single) ----------------------------------------------------


def test_single_steps_follow_rows() -> None:
    seen: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append((step.timestamp, step.inputs["close"].copy(), step.symbols))
            step.output = step.inputs["close"]

    record(seen).predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert len(seen) == 3
    assert [ts for ts, _, _ in seen] == list(days(3))
    for (_, row, symbols), expected in zip(
        seen, ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]), strict=True
    ):
        assert row.shape == (2,)
        np.testing.assert_array_equal(row, expected)
        assert list(symbols) == ["A", "B"]


def test_single_trailing_input_is_symbols_by_k() -> None:
    columns = pd.MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0), ("B", 1)])
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], index=days(2), columns=columns
    )
    seen: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(step.inputs["x"].copy())
            step.output = step.inputs["x"][:, 0]

    record(seen).predict({"x": frame})
    assert seen[0].shape == (2, 2)
    np.testing.assert_array_equal(seen[0], [[1.0, 2.0], [3.0, 4.0]])


def test_single_step_timestamps_plural_is_attribute_error() -> None:
    @qm.model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            _ = step.timestamps

    with pytest.raises(AttributeError, match="timestamp"):
        bad().predict(wide([[1.0, 2.0]]))


def test_is_train_true_in_fit_false_in_predict() -> None:
    seen: list[bool] = []

    @qm.model
    def record(ctx: qm.Context, out: list[bool]) -> None:
        for step in ctx:
            out.append(step.is_train)

    m = record(seen)
    m.fit(wide([[1.0, 2.0]]))
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert seen == [True, False]


def test_single_side_inputs_are_scalars() -> None:
    side = pd.DataFrame(
        {
            "timestamp": days(2),
            "w": [1.5, 2.5],
            "o": [{"a": 1}, {"b": 2}],
        }
    )
    seen: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append((step.side_inputs["w"], step.side_inputs["o"]))

    record(seen).predict(wide([[1.0, 2.0], [3.0, 4.0]]), side_data=side)
    assert np.ndim(seen[0][0]) == 0
    assert float(seen[0][0]) == 1.5
    assert seen[0][1] == {"a": 1}
    assert seen[1][1] == {"b": 2}


# --- step supply (batch) -----------------------------------------------------


def test_batch_supplies_one_chunk_step() -> None:
    side = pd.DataFrame({"timestamp": days(3), "w": [1.0, 2.0, 3.0]})
    seen: list[Any] = []

    @qm.batch_model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(
                (
                    step.timestamps.copy(),
                    step.inputs["close"].copy(),
                    step.side_inputs["w"].copy(),
                )
            )
            step.output = step.inputs["close"]

    record(seen).predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), side_data=side)
    assert len(seen) == 1
    timestamps, values, w = seen[0]
    assert timestamps.shape == (3,)
    np.testing.assert_array_equal(timestamps, days(3).to_numpy())
    assert values.shape == (3, 2)
    np.testing.assert_array_equal(w, [1.0, 2.0, 3.0])


def test_batch_step_timestamp_singular_is_attribute_error() -> None:
    @qm.batch_model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            _ = step.timestamp

    with pytest.raises(AttributeError, match="timestamps"):
        bad().predict(wide([[1.0, 2.0]]))


# --- output contract (per-call buffer) ----------------------------------------


def test_unassigned_call_yields_all_nan_float64() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    pred = silent().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.array.dtype == np.float64
    assert pred.array.shape == (2, 2)
    assert np.isnan(pred.array).all()


def test_partially_assigned_rows_default_to_nan() -> None:
    @qm.model
    def first_only(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            if i == 0:
                step.output = step.inputs["close"]

    pred = first_only().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.array[0], [1.0, 2.0])
    assert np.isnan(pred.array[1]).all()


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.complex128])
def test_non_float_output_is_value_error(dtype: Any) -> None:
    # An int/bool output buffer would render unwritten rows as plausible
    # 0/False values, and a complex output has no single meaning across the
    # metrics — the main output is restricted to real floating-point
    # (NaN is the one missing value).
    @qm.model
    def ints(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = np.array([1, 0], dtype=dtype)

    with pytest.raises(ValueError, match="real floating-point"):
        ints().predict(wide([[1.0, 2.0]]))


def test_dtype_locked_within_call() -> None:
    @qm.model
    def flip(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            dtype = np.float64 if i == 0 else np.float32
            step.output = np.zeros(2, dtype=dtype)

    with pytest.raises(ValueError, match="dtype mismatch"):
        flip().predict(wide([[1.0, 2.0], [3.0, 4.0]]))


def test_trailing_shape_locked_within_call() -> None:
    @qm.model
    def flip(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            step.output = np.zeros((2, 2 if i == 0 else 3))

    with pytest.raises(ValueError, match="shape mismatch"):
        flip().predict(wide([[1.0, 2.0], [3.0, 4.0]]))


def test_dtype_not_locked_across_calls() -> None:
    @qm.model
    def sized(ctx: qm.Context, dtype_box: list[Any]) -> None:
        for step in ctx:
            step.output = np.zeros(2, dtype=dtype_box[0])

    box: list[Any] = [np.float64]
    m = sized(box)
    m.predict(wide([[1.0, 2.0]]))
    box[0] = np.float32
    pred = m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert pred.array.dtype == np.float32


@pytest.mark.parametrize("value", [1.0, [1.0, 2.0, 3.0]])
def test_bad_single_output_shape_is_value_error(value: Any) -> None:
    @qm.model
    def bad(ctx: qm.Context, v: Any) -> None:
        for step in ctx:
            step.output = v

    with pytest.raises(ValueError, match="shape mismatch"):
        bad(value).predict(wide([[1.0, 2.0]]))


def test_bad_batch_output_shape_names_expected_shape() -> None:
    @qm.batch_model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"][:-1]

    with pytest.raises(ValueError, match="expected"):
        bad().predict(wide([[1.0, 2.0], [3.0, 4.0]]))


def test_object_dtype_output_is_value_error() -> None:
    @qm.model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = np.array(["a", "b"], dtype=object)

    with pytest.raises(ValueError, match="object"):
        bad().predict(wide([[1.0, 2.0]]))


def test_list_output_is_accepted_via_asarray() -> None:
    @qm.model
    def lists(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = [1.0, 2.0]

    pred = lists().predict(wide([[9.0, 9.0]]))
    np.testing.assert_array_equal(pred.array, [[1.0, 2.0]])


def test_output_getter_is_writable_view() -> None:
    @qm.model
    def rmw(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]
            step.output[step.output > 2.0] = 0.0

    pred = rmw().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.array, [[1.0, 2.0], [0.0, 0.0]])


def test_unassigned_output_read_is_runtime_error() -> None:
    @qm.model
    def reads(ctx: qm.Context) -> None:
        for step in ctx:
            _ = step.output

    with pytest.raises(RuntimeError, match="assign it first"):
        reads().predict(wide([[1.0, 2.0]]))


def test_unassigned_side_output_read_is_runtime_error() -> None:
    @qm.model
    def reads(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]
            _ = step.side_outputs["w"]

    with pytest.raises(RuntimeError, match=r"side_outputs\['w'\]"):
        reads().predict(wide([[1.0, 2.0]]))


def test_multi_horizon_single_output() -> None:
    @qm.model
    def horizons(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = np.stack(
                [step.inputs["close"], step.inputs["close"] * 2.0], axis=-1
            )

    pred = horizons().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.array.shape == (2, 2, 2)
    np.testing.assert_array_equal(pred.array[1], [[3.0, 6.0], [4.0, 8.0]])


def test_multi_horizon_batch_output() -> None:
    @qm.batch_model
    def horizons(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = np.stack(
                [step.inputs["close"], step.inputs["close"] * 2.0], axis=-1
            )

    pred = horizons().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.array.shape == (2, 2, 2)
    np.testing.assert_array_equal(pred.array[0], [[1.0, 2.0], [2.0, 4.0]])


# --- side_outputs --------------------------------------------------------------


def test_single_side_output_scalar_becomes_t_vector() -> None:
    @qm.model
    def weights(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            step.side_outputs["w"] = float(i)

    pred = weights().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.side_data["w"].shape == (2,)
    np.testing.assert_array_equal(pred.side_data["w"], [0.0, 1.0])


def test_single_side_output_vector_becomes_t_k() -> None:
    @qm.model
    def weights(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            step.side_outputs["w"] = [float(i), float(i) + 0.5]

    pred = weights().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.side_data["w"].shape == (2, 2)
    np.testing.assert_array_equal(pred.side_data["w"], [[0.0, 0.5], [1.0, 1.5]])


def test_batch_side_output_shapes() -> None:
    @qm.batch_model
    def weights(ctx: qm.Context) -> None:
        for step in ctx:
            step.side_outputs["w"] = np.array([1.0, 2.0])
            step.side_outputs["m"] = np.array([[1.0, 2.0], [3.0, 4.0]])

    pred = weights().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.side_data["w"], [1.0, 2.0])
    assert pred.side_data["m"].shape == (2, 2)


def test_unwritten_side_rows_get_dtype_default() -> None:
    @qm.model
    def sparse(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            if i == 0:
                step.side_outputs["f"] = 1.5
                step.side_outputs["i"] = np.int64(7)

    pred = sparse().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.side_data["f"][0] == 1.5
    assert np.isnan(pred.side_data["f"][1])
    np.testing.assert_array_equal(pred.side_data["i"], [7, 0])


def test_unwritten_temporal_side_rows_default_to_nat() -> None:
    # datetime/timedelta defaults are NaT, not the dtype's zero — a
    # zero-filled datetime64 would render missing rows as 1970-01-01 events.
    @qm.model
    def sparse(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            if i == 0:
                step.side_outputs["at"] = np.datetime64("2024-01-01", "ns")
                step.side_outputs["dt"] = np.timedelta64(5, "m")

    pred = sparse().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert pred.side_data["at"][0] == np.datetime64("2024-01-01")
    assert np.isnat(pred.side_data["at"][1])
    assert pred.side_data["dt"][0] == np.timedelta64(5, "m")
    assert np.isnat(pred.side_data["dt"][1])


def test_never_written_side_name_is_absent() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    pred = silent().predict(wide([[1.0, 2.0]]))
    assert pred.side_data == {}


def test_side_output_dtype_mismatch_within_call_is_value_error() -> None:
    @qm.model
    def flip(ctx: qm.Context) -> None:
        for i, step in enumerate(ctx):
            step.side_outputs["w"] = 1.0 if i == 0 else np.int64(1)

    with pytest.raises(ValueError, match="dtype mismatch"):
        flip().predict(wide([[1.0, 2.0], [3.0, 4.0]]))


def test_side_output_read_back_is_writable_view_and_contains() -> None:
    seen: list[bool] = []

    @qm.model
    def rmw(ctx: qm.Context, out: list[bool]) -> None:
        for step in ctx:
            out.append("w" in step.side_outputs)
            step.side_outputs["w"] = [1.0, 2.0]
            out.append("w" in step.side_outputs)
            step.side_outputs["w"][0] = 9.0

    pred = rmw(seen).predict(wide([[1.0, 2.0]]))
    assert seen == [False, True]
    np.testing.assert_array_equal(pred.side_data["w"], [[9.0, 2.0]])


def test_object_dtype_side_output_is_value_error() -> None:
    @qm.model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            step.side_outputs["w"] = np.array([{"a": 1}], dtype=object)

    with pytest.raises(ValueError, match="object"):
        bad().predict(wide([[1.0, 2.0]]))


# --- info ----------------------------------------------------------------------


def test_info_is_an_output_only_shared_dict() -> None:
    seen: list[Any] = []

    @qm.model
    def annotate(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(step.info)
            step.info["note"] = "hi"

    pred = annotate(seen).predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert seen[0] is seen[1]
    assert pred.info is seen[0]
    assert pred.info == {"note": "hi"}

    next_pred = annotate([]).predict(wide([[5.0, 6.0]], start="2020-01-03"))
    assert next_pred.info == {"note": "hi"}
    assert next_pred.info is not pred.info


def test_info_defaults_to_empty_dict() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    pred = silent().predict(wide([[1.0, 2.0]]))
    assert pred.info == {}


# --- time cursor ---------------------------------------------------------------


@qm.model
def _echo(ctx: qm.Context) -> None:
    for step in ctx:
        step.output = step.inputs["close"]


def test_predict_advances_and_rejects_repeats() -> None:
    m = _echo()
    m.predict(wide([[1.0, 2.0]]))
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    with pytest.raises(qm.TimeOrderError, match="cursor"):
        m.predict(wide([[5.0, 6.0]], start="2020-01-02"))


def test_predict_with_any_past_row_fails_whole_call() -> None:
    m = _echo()
    m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(qm.TimeOrderError, match="cursor"):
        m.predict(wide([[5.0, 6.0], [7.0, 8.0]], start="2020-01-02"))


def test_fit_silently_skips_rows_at_or_before_cursor() -> None:
    steps: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(step.timestamp)

    m = record(steps)
    m.fit(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    m.fit(wide([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], start="2020-01-02"))
    assert steps == [*days(3), days(4)[3]]


def test_fully_skipped_fit_runs_zero_steps_and_keeps_cursor() -> None:
    steps: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(step.timestamp)

    m = record(steps)
    m.fit(wide([[1.0, 2.0], [3.0, 4.0]]))
    before = m.cursor
    m.fit(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert steps == list(days(2))
    assert m.cursor == before


def test_fit_then_predict_ordering_is_free() -> None:
    m = _echo()
    m.fit(wide([[1.0, 2.0]]))
    pred = m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    np.testing.assert_array_equal(pred.array, [[3.0, 4.0]])


def test_cursor_and_started_after_success() -> None:
    m = _echo()
    assert not m.started
    m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert m.started
    assert m.cursor == np.datetime64("2020-01-02")


def test_fit_returns_self_for_chaining() -> None:
    pred = (
        _echo().fit(wide([[1.0, 2.0]])).predict(wide([[3.0, 4.0]], start="2020-01-02"))
    )
    np.testing.assert_array_equal(pred.array, [[3.0, 4.0]])


# --- fail-stop (failed instances) ------------------------------------------------


def test_breaking_model_fails_closed_and_instance_refuses_reuse() -> None:
    @qm.model
    def flaky(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]
            break

    m = flaky()
    with pytest.raises(RuntimeError, match="consumed 1 of 2"):
        m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    # Fail-closed: the cursor and universe did not commit — but ctx.state and
    # children are not rolled back either, so the instance is failed and
    # refuses every further call and checkpoint.
    assert not m.started
    assert m.symbols is None
    for call in [m.predict, m.fit]:
        with pytest.raises(RuntimeError, match="failed an earlier call"):
            call(wide([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(qm.CheckpointError, match="failed an earlier call"):
        m.dumps()
    with pytest.raises(qm.CheckpointError, match="failed an earlier call"):
        m.clone()


def test_finalize_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _echo()

    def fail_finalize(self: qm.Features) -> None:
        raise MemoryError("allocation failed")

    monkeypatch.setattr(qm.Features, "_finalize", fail_finalize)
    with pytest.raises(MemoryError, match="allocation failed"):
        m.predict(wide([[1.0, 2.0]]))
    assert m.cursor is None
    assert m.symbols is None
    with pytest.raises(RuntimeError, match="failed an earlier call"):
        m.predict(wide([[1.0, 2.0]]))


def test_restore_into_failed_unstarted_instance_recovers_it() -> None:
    @qm.model
    def flaky(ctx: qm.Context, broken: list[bool]) -> None:
        ctx.state["dirty"] = True
        for step in ctx:
            step.output = step.inputs["close"]
            if broken:
                break

    blob = flaky([]).dumps()  # an unstarted same-identity checkpoint
    broken = [True]
    m = flaky(broken)
    with pytest.raises(RuntimeError, match="consumed 1 of 2"):
        m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    # A restore replaces the whole state, so it is the sanctioned recovery.
    broken.clear()
    pred = m.loads(blob).predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.array, [[1.0, 2.0], [3.0, 4.0]])


def test_never_iterating_model_fails_closed() -> None:
    @qm.model
    def inert(ctx: qm.Context) -> None:
        pass

    with pytest.raises(RuntimeError, match="consumed 0 of 2"):
        inert().predict(wide([[1.0, 2.0], [3.0, 4.0]]))


def test_second_iteration_is_runtime_error() -> None:
    @qm.model
    def twice(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]
        for step in ctx:  # pragma: no cover
            step.output = step.inputs["close"]

    with pytest.raises(RuntimeError, match="one-shot"):
        twice().predict(wide([[1.0, 2.0]]))


def test_model_exception_propagates_cursor_unchanged_state_kept() -> None:
    @qm.model
    def boom(ctx: qm.Context) -> None:
        ctx.state["touched"] = True
        raise ValueError("boom")

    m = boom()
    with pytest.raises(ValueError, match="boom"):
        m.fit(wide([[1.0, 2.0]]))
    assert not m.started
    assert m.state == {"touched": True}


# --- prologue re-execution -------------------------------------------------------


def test_prologue_runs_every_call_but_results_continue() -> None:
    prologues: list[int] = []

    @qm.model
    def cumulative(ctx: qm.Context, out: list[int]) -> None:
        out.append(1)
        ctx.state.setdefault("sum", 0.0)
        for step in ctx:
            ctx.state["sum"] += float(step.inputs["close"][0])
            step.output = np.full(2, ctx.state["sum"])

    m = cumulative(prologues)
    m.fit(wide([[1.0, 0.0]]))
    p1 = m.predict(wide([[2.0, 0.0]], start="2020-01-02"))
    p2 = m.predict(wide([[3.0, 0.0]], start="2020-01-03"))
    assert len(prologues) == 3
    assert p1.array[0, 0] == 3.0
    assert p2.array[0, 0] == 6.0


# --- symbol universe --------------------------------------------------------------


def test_universe_fixed_at_first_fit_missing_symbols_are_nan() -> None:
    seen: list[Any] = []

    @qm.model
    def record(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            out.append(step.inputs["close"].copy())

    m = record(seen)
    m.fit(wide([[1.0, 2.0]]))
    m.predict(wide([[5.0]], symbols=("A",), start="2020-01-02"))
    assert seen[1].shape == (2,)
    assert seen[1][0] == 5.0
    assert np.isnan(seen[1][1])


def test_out_of_universe_symbol_is_value_error() -> None:
    m = _echo()
    m.fit(wide([[1.0, 2.0]]))
    with pytest.raises(ValueError, match=r"\['C'\] are not in the declared"):
        m.predict(wide([[1.0, 2.0]], symbols=("A", "C"), start="2020-01-02"))


def test_symbols_after_universe_fixed_is_value_error() -> None:
    m = _echo()
    m.fit(wide([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="already-fixed"):
        m.predict(wide([[1.0, 2.0]], start="2020-01-02"), symbols=("A", "B"))


def test_first_predict_also_fixes_universe() -> None:
    m = _echo()
    m.predict(wide([[1.0, 2.0]]))
    assert m.symbols is not None
    assert list(m.symbols) == ["A", "B"]
    with pytest.raises(ValueError, match="already-fixed"):
        m.predict(wide([[1.0, 2.0]], start="2020-01-02"), symbols=("A", "B"))


def test_declared_universe_wins_over_first_input() -> None:
    pred = _echo().predict(wide([[1.0, 2.0]]), symbols=("C", "A", "B"))
    assert pred.array.shape == (1, 3)
    assert list(pred.symbols) == ["A", "B", "C"]
    assert np.isnan(pred.array[0, 2])


# --- public boundary regressions ------------------------------------------------


def test_model_function_returning_a_value_is_type_error() -> None:
    @qm.model
    def accidental_return(ctx: qm.Context) -> None:
        for step in ctx:
            return step.inputs["close"].astype(float)  # type: ignore[return-value]

    # One timestamp: the consumed-steps check alone would pass silently.
    with pytest.raises(TypeError, match="must return None"):
        accidental_return().predict(wide([[1.0, 2.0]]))


def test_result_symbols_are_independent_of_the_model() -> None:
    m = _echo()
    pred = m.predict(wide([[1.0, 2.0]]))
    assert m.symbols is not None
    assert pred.symbols is not m.symbols
    pred.symbols[0] = "Z"
    assert list(m.symbols) == ["A", "B"]
    # Editing the returned result stays local and does not affect the next call.
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))


@pytest.mark.parametrize("name", ["", 123, None])
def test_side_output_name_must_be_a_non_empty_string(name: Any) -> None:
    @qm.model
    def bad(ctx: qm.Context) -> None:
        for step in ctx:
            step.side_outputs[name] = np.array(1.0)
            step.output = step.inputs["close"]

    with pytest.raises(ValueError, match="side output name"):
        bad().predict(wide([[1.0, 2.0]]))


def test_child_must_be_a_model_instance() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = _echo  # type: ignore[assignment]  # forgot to call the factory
        for step in ctx:
            step.output = step.inputs["close"]

    with pytest.raises(TypeError, match="must be a Model instance"):
        parent().predict(wide([[1.0, 2.0]]))


def test_failed_instance_is_rejected_before_input_and_seed_handling() -> None:
    @qm.model
    def flaky(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]
            break

    m = flaky()
    m.seed = 1
    with pytest.raises(RuntimeError, match="consumed 1 of 2"):
        m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    # The failed error wins over input/seed/symbols validation, and the
    # rejected call leaves the instance unchanged.
    with pytest.raises(RuntimeError, match="failed an earlier call"):
        m.predict(wide([[1.0, 2.0]]), seed=99)
    with pytest.raises(RuntimeError, match="failed an earlier call"):
        m.predict(pd.DataFrame({"nonsense": [1]}))
    assert m.seed == 1
