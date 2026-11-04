"""Child models: registration, identity verification, ordering, reads.

Children advance through the whole call before the parent, so a parent
step can read the child's output for its own timestamp. Registration
builds engine-owned children from the candidates' functions and bound
parameters (the candidates themselves never advance); re-registration is
verified by name, module-qualified function name, and mode only, and the
engine-owned child keeps its state across calls.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
from conftest import wide

import qfeval_model as qm


def test_ctx_child_is_shorthand_for_children_child() -> None:
    seen: dict[str, Any] = {}

    @qm.model
    def wraps(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            seen["same"] = step.child is step.children["child"]
            step.output = step.child.array

    m = wraps()
    pred = m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert seen["same"] is True
    assert m.children is not None
    assert list(m.children) == ["child"]
    np.testing.assert_array_equal(pred.array, [[np.nan, np.nan], [1.0, 2.0]])


def test_registered_candidate_never_advances() -> None:
    # The candidate instance assigned to ctx.children stays untouched: the
    # engine runs its own copy, so the candidate's live state cannot reveal
    # the engine's progress (in a one-shot call the whole call's future used
    # to be readable through it before any parent step ran).
    seen: list[tuple[bool, dict[str, Any], Any, int]] = []

    @qm.model
    def latest(ctx: qm.Context) -> None:
        for step in ctx:
            ctx.state["latest"] = float(step.inputs["close"][0])
            step.output = step.inputs["close"]

    @qm.model
    def parent(ctx: qm.Context) -> None:
        candidate = latest()
        ctx.child = candidate
        for step in ctx:
            seen.append(
                (
                    candidate.started,
                    dict(candidate.state),
                    candidate.cursor,
                    candidate.seed,
                )
            )
            step.output = step.child.array

    parent().predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert seen == [(False, {}, None, 0)] * 3


def test_ctx_child_read_is_attribute_error() -> None:
    @qm.model
    def reads(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        _ = ctx.child  # registration is write-only; results are step.child
        for step in ctx:  # pragma: no cover
            step.output = step.inputs["close"]

    with pytest.raises(AttributeError, match="write-only"):
        reads().predict(wide([[1.0, 2.0]]))


def test_step_child_unregistered_is_attribute_error() -> None:
    @qm.model
    def orphan(ctx: qm.Context) -> None:
        for step in ctx:
            _ = step.child

    with pytest.raises(AttributeError, match="no child registered under the name"):
        orphan().predict(wide([[1.0, 2.0]]))


def test_children_advance_before_parent_and_single_steps_are_sliced() -> None:
    order: list[str] = []
    seen: list[tuple[Any, Any, Any]] = []

    @qm.model
    def child(ctx: qm.Context) -> None:
        for step in ctx:
            order.append("child")
            step.output = step.inputs["close"] + 1.0
            step.side_outputs["w"] = 7.0

    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = child()
        for step in ctx:
            order.append("parent")
            c = step.children["c"]
            seen.append((c.array.copy(), c.side_data["w"].copy(), c.timestamps))
            step.output = c.array

    w = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pred = parent().predict(w)
    # The child advances over the whole call before any parent step runs.
    assert order == ["child"] * 3 + ["parent"] * 3
    for i, (arr, side, ts) in enumerate(seen):
        assert arr.shape == (2,)  # (S,), T axis squeezed
        np.testing.assert_array_equal(arr, w.iloc[i].to_numpy() + 1.0)
        assert side.shape == ()  # side_data row-sliced to a scalar
        assert float(side) == 7.0
        assert ts == w.index.values[i]
    np.testing.assert_array_equal(pred.array, w.to_numpy() + 1.0)


def test_batch_parent_sees_child_row_range() -> None:
    seen: list[Any] = []

    @qm.model
    def child(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"] + 1.0

    @qm.batch_model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = child()
        for step in ctx:
            seen.append(step.children["c"].array.copy())
            step.output = step.children["c"].array

    w = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    parent().predict(w)
    assert len(seen) == 1
    assert seen[0].shape == (3, 2)  # (T, S) row range, not squeezed
    np.testing.assert_array_equal(seen[0], w.to_numpy() + 1.0)


def test_inline_construction_reuses_first_registered_instance() -> None:
    @qm.model
    def wraps(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    m = wraps()
    first = m.predict(wide([[1.0, 2.0]]))
    assert m.children is not None
    registered = m.children["child"]
    # Second call constructs a fresh lag(), but the first instance keeps
    # being used: its warmup carries over across calls.
    second = m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert m.children["child"] is registered
    np.testing.assert_array_equal(first.array, [[np.nan, np.nan]])
    np.testing.assert_array_equal(second.array, [[1.0, 2.0]])


def test_arg_passed_child_registration() -> None:
    @qm.model
    def wraps(ctx: qm.Context, c: qm.Model) -> None:
        ctx.child = c
        for step in ctx:
            step.output = step.child.array

    m = wraps(qm.zoo.lag("close", 1))
    m.predict(wide([[1.0, 2.0]]))
    second = m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    np.testing.assert_array_equal(second.array, [[1.0, 2.0]])


def test_bound_model_argument_alone_is_not_registered() -> None:
    child = qm.zoo.lag("close", 1)

    @qm.model
    def ignores(ctx: qm.Context, c: qm.Model) -> None:
        for step in ctx:
            step.output = step.inputs["close"]

    m = ignores(child)
    m.predict(wide([[1.0, 2.0]]))
    assert m.children == {}
    assert child.started is False


def test_child_name_change_across_calls_is_value_error() -> None:
    calls: list[int] = []

    @qm.model
    def parent(ctx: qm.Context) -> None:
        calls.append(0)
        name = "a" if len(calls) == 1 else "b"
        ctx.children[name] = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.inputs["close"]

    m = parent()
    m.predict(wide([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="child names changed"):
        m.predict(wide([[3.0, 4.0]], start="2020-01-02"))


def test_child_qualname_change_across_calls_is_value_error() -> None:
    calls: list[int] = []

    @qm.model
    def parent(ctx: qm.Context) -> None:
        calls.append(0)
        if len(calls) == 1:
            ctx.children["c"] = qm.zoo.lag("close", 1)
        else:
            ctx.children["c"] = qm.zoo.ffill("close")
        for step in ctx:
            step.output = step.inputs["close"]

    m = parent()
    m.predict(wide([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="changed across calls: registered"):
        m.predict(wide([[3.0, 4.0]], start="2020-01-02"))


def test_child_mode_change_across_calls_is_value_error() -> None:
    def _variant(mode: str) -> qm.Model:
        def shared_child(ctx: qm.Context) -> None:
            for step in ctx:
                step.output = step.inputs["close"]

        shared_child.__qualname__ = "shared_child"
        factory = (
            qm.model(shared_child) if mode == "single" else qm.batch_model(shared_child)
        )
        return factory()

    calls: list[int] = []

    @qm.model
    def parent(ctx: qm.Context) -> None:
        calls.append(0)
        ctx.children["c"] = _variant("single" if len(calls) == 1 else "batch")
        for step in ctx:
            step.output = step.inputs["close"]

    m = parent()
    m.predict(wide([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="changed across calls: registered"):
        m.predict(wide([[3.0, 4.0]], start="2020-01-02"))


def test_child_bound_argument_change_is_accepted() -> None:
    calls: list[int] = []

    @qm.model
    def parent(ctx: qm.Context) -> None:
        calls.append(0)
        ctx.children["c"] = qm.zoo.lag("close", 1 if len(calls) == 1 else 2)
        for step in ctx:
            step.output = step.children["c"].array

    m = parent()
    m.predict(wide([[1.0, 2.0]]))
    # Verification covers only (name, module:qualname, mode): the periods=1
    # instance registered first keeps being used, so the second call sees the lag-1
    # value (periods=2 would still be NaN).
    second = m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    np.testing.assert_array_equal(second.array, [[1.0, 2.0]])


def test_started_child_rejected_at_first_registration() -> None:
    child = qm.zoo.lag("close", 1)
    child.predict(wide([[1.0, 2.0]]))

    @qm.model
    def wraps(ctx: qm.Context, c: qm.Model) -> None:
        ctx.child = c
        for step in ctx:
            step.output = step.child.array

    with pytest.raises(ValueError, match="already started"):
        wraps(child).predict(wide([[3.0, 4.0]], start="2020-01-02"))


def test_child_seed_is_derived_from_parent_seed_and_name() -> None:
    # The derivation is a wire contract (checkpoints store per-child seeds),
    # pinned by an independent reimplementation and a golden literal:
    # blake2b(digest_size=8, parent seed as 8B big-endian || name UTF-8).
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.children["c"].array

    m = parent()
    m.predict(wide([[1.0, 2.0]]), seed=5)
    assert m.children is not None

    h = hashlib.blake2b(digest_size=8)
    h.update((5).to_bytes(8, "big"))
    h.update(b"c")
    assert m.children["c"].seed == int.from_bytes(h.digest(), "big")
    assert m.children["c"].seed == 11115230998399133854


def test_parent_seed_change_after_start_does_not_propagate_to_children() -> None:
    # A child seed is derived once at registration; assigning the parent's
    # seed afterwards leaves it untouched, and a checkpoint round-trip
    # restores both sides of that relationship unchanged.
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.children["c"].array

    m = parent()
    m.predict(wide([[1.0, 2.0]]), seed=5)
    assert m.children is not None
    child_seed = m.children["c"].seed
    m.seed = 99
    assert m.children["c"].seed == child_seed
    restored = parent().loads(m.dumps())
    assert restored.seed == 99
    # Children are rebuilt on the restored instance's next call; the child
    # keeps its checkpointed seed rather than re-deriving from the new 99.
    restored.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert restored.children is not None
    assert restored.children["c"].seed == child_seed


def test_same_structured_siblings_draw_distinct_streams() -> None:
    # Two children of the same function under one parent (and the parent
    # itself) must not share a random stream, or duplicating a stochastic
    # model for an ensemble would collapse to the single model.
    draws: dict[str, list[float]] = {"a": [], "b": [], "parent": []}

    @qm.model
    def noisy(ctx: qm.Context, out: list[float]) -> None:
        for step in ctx:
            out.append(float(step.rng.random()))
            step.output = step.inputs["close"]

    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["a"] = noisy(draws["a"])
        ctx.children["b"] = noisy(draws["b"])
        for step in ctx:
            draws["parent"].append(float(step.rng.random()))
            step.output = step.children["a"].array

    parent().predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), seed=5)
    assert len(draws["a"]) == 3
    assert draws["a"] != draws["b"]
    assert draws["a"] != draws["parent"]
    assert draws["b"] != draws["parent"]


def test_child_streams_are_reproducible_across_runs() -> None:
    @qm.model
    def noisy(ctx: qm.Context, out: list[float]) -> None:
        for step in ctx:
            out.append(float(step.rng.random()))
            step.output = step.inputs["close"]

    @qm.model
    def parent(ctx: qm.Context, out: list[float]) -> None:
        ctx.children["c"] = noisy(out)
        for step in ctx:
            step.output = step.children["c"].array

    w = wide([[1.0, 2.0], [3.0, 4.0]])
    runs: list[list[float]] = []
    for _ in range(2):
        seen: list[float] = []
        parent(seen).predict(w, seed=5)
        runs.append(seen)
    assert runs[0] == runs[1]


def test_child_info_is_isolated_from_parent_diagnostics() -> None:
    @qm.model
    def leaky_child(ctx: qm.Context) -> None:
        for step in ctx:
            step.info["from_child"] = 1
            step.output = step.inputs["close"]

    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.children["c"] = leaky_child()
        for step in ctx:
            step.output = step.children["c"].array

    pred = parent().predict(wide([[1.0, 2.0]]))
    assert pred.info == {}
    assert "from_child" not in pred.info


def test_child_call_wide_info_is_not_exposed_to_parent_steps() -> None:
    # The child completes the whole call before the parent's steps run, and
    # its ``info`` is call-wide (never row-sliced) — exposing it would let a
    # single-mode parent step read values the child wrote at later
    # timestamps. Only the row-sliced outputs cross the boundary.
    @qm.model
    def teller(ctx: qm.Context) -> None:
        for step in ctx:
            step.info["last"] = float(step.inputs["close"][0])
            step.side_outputs["mark"] = float(step.inputs["close"][0])
            step.output = step.inputs["close"]

    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = teller()
        for step in ctx:
            assert step.child.info == {}
            # The row-sliced side output sees this step's row only.
            step.output = step.child.array * 0.0 + step.child.side_data["mark"]

    pred = parent().predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    np.testing.assert_array_equal(pred.array, [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])


def test_info_is_not_a_channel_at_any_composition_depth() -> None:
    # The same rule applies at every parent/child boundary of a multi-level
    # composition: each level sees an empty ``info`` on its child's view.
    @qm.model
    def leaf(ctx: qm.Context) -> None:
        for step in ctx:
            step.info["secret"] = 100.0
            step.output = step.inputs["close"]

    @qm.model
    def mid(ctx: qm.Context) -> None:
        ctx.child = leaf()
        for step in ctx:
            step.info["secret"] = 10.0
            step.output = step.child.array + step.child.info.get("secret", 0.0)

    @qm.model
    def top(ctx: qm.Context) -> None:
        ctx.child = mid()
        for step in ctx:
            step.output = step.child.array + step.child.info.get("secret", 0.0)

    pred = top().predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.array, [[1.0, 2.0], [3.0, 4.0]])


def test_grandchild_composition_rolling_mean_of_lag() -> None:
    m = qm.zoo.rolling_mean(qm.zoo.lag("close", 1), 2)
    w = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    pred = m.predict(w)
    # lag(1): NaN, [1,2], [3,4], [5,6]; rolling_mean(2): NaN, NaN, [2,3], [4,5]
    np.testing.assert_array_equal(
        pred.array,
        [[np.nan, np.nan], [np.nan, np.nan], [2.0, 3.0], [4.0, 5.0]],
    )
    assert m.children is not None
    lag_child = m.children["source"]
    assert lag_child.children is not None
    assert lag_child.children == {}  # string source: no grandchild


def test_same_instance_under_two_names_becomes_two_independent_children() -> None:
    # Registration builds an engine-owned child per name, so registering one
    # candidate under two names yields two independent children (and the
    # candidate itself never advances).
    shared = qm.zoo.lag("close", 1)

    @qm.model
    def two_names(ctx: qm.Context) -> None:
        ctx.children["x"] = shared
        ctx.children["y"] = shared
        for step in ctx:
            step.output = step.children["x"].array + step.children["y"].array

    m = two_names()
    pred = m.predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(pred.array, [[np.nan, np.nan], [2.0, 4.0]])
    assert shared.started is False
    assert m.children is not None
    assert m.children["x"] is not m.children["y"]


def test_advancing_an_engine_owned_child_directly_is_runtime_error() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    m = parent()
    m.predict(wide([[1.0, 2.0]]))
    assert m.children is not None
    m.children["child"].predict(wide([[3.0, 4.0]], start="2020-01-02"))
    with pytest.raises(RuntimeError, match="lockstep"):
        m.predict(wide([[5.0, 6.0]], start="2020-01-03"))


def test_failed_call_with_children_rejects_reuse_with_clear_error() -> None:
    # Children advance before the parent's steps and are not rolled back, so
    # a mid-call failure marks the instance failed; re-passing the same
    # inputs must be refused, for predict and fit alike.
    def _flaky(fail: list[bool]) -> qm.Model:
        @qm.model
        def flaky(ctx: qm.Context) -> None:
            ctx.child = qm.zoo.lag("close", 1)
            for step in ctx:
                if fail[0]:
                    raise ValueError("boom")
                step.output = step.child.array

        return flaky()

    w = wide([[1.0, 2.0], [3.0, 4.0]])
    for call in ["predict", "fit"]:
        fail = [True]
        m = _flaky(fail)
        with pytest.raises(ValueError, match="boom"):
            getattr(m, call)(w)
        fail[0] = False
        with pytest.raises(RuntimeError, match="failed an earlier call"):
            getattr(m, call)(w)
