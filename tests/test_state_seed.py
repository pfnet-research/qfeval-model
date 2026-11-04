"""``ctx.state``, per-step seed derivation, ``step.rng``, determinism.

The seed derivation is a wire contract (checkpointed runs must replay
bit-identically across versions and platforms), so it is pinned both by
an independent reimplementation of the formula and by one hard-coded
golden value as a canary.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, wide

import qfeval_model as qm

# --- state ---------------------------------------------------------------------


def test_ctx_state_is_the_same_dict_across_calls() -> None:
    ids: list[int] = []

    @qm.model
    def record(ctx: qm.Context, out: list[int]) -> None:
        out.append(id(ctx.state))
        ctx.state["n"] = ctx.state.get("n", 0) + 1
        for _step in ctx:
            pass

    m = record(ids)
    m.fit(wide([[1.0, 2.0]]))
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert len(set(ids)) == 1
    # Model.state's contract is content equality only; its identity and
    # writability are deliberately left unpinned.
    assert m.state == {"n": 2}


def test_state_written_in_fit_is_visible_in_predict() -> None:
    seen: list[Any] = []

    @qm.model
    def carry(ctx: qm.Context, out: list[Any]) -> None:
        for step in ctx:
            if step.is_train:
                ctx.state["mark"] = 42
            else:
                out.append(ctx.state["mark"])

    m = carry(seen)
    m.fit(wide([[1.0, 2.0]]))
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert seen == [42]


def test_bound_mutable_argument_is_writable_from_the_model() -> None:
    log: list[float] = []

    @qm.model
    def record(ctx: qm.Context, out: list[float]) -> None:
        for step in ctx:
            out.append(float(step.inputs["close"][0]))

    record(log).predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert log == [1.0, 3.0]


# --- seed derivation goldens ------------------------------------------------------


def _derived_seed(model_seed: int, ts: np.datetime64) -> int:
    """Independent reimplementation of the step-seed derivation."""
    h = hashlib.blake2b(digest_size=8)
    h.update((model_seed % 2**64).to_bytes(8, "big"))
    ns = int(ts.astype("datetime64[ns]").astype(np.int64))
    h.update(ns.to_bytes(8, "big", signed=True))
    return int.from_bytes(h.digest(), "big")


@qm.model
def _seed_recorder(ctx: qm.Context, out: list[tuple[np.datetime64, int]]) -> None:
    for step in ctx:
        out.append((step.timestamp, step.seed))


@pytest.mark.parametrize("model_seed", [0, 3, -1, 2**64 + 5])
def test_step_seed_matches_independent_derivation(model_seed: int) -> None:
    seen: list[tuple[np.datetime64, int]] = []
    m = _seed_recorder(seen)
    m.predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), seed=model_seed)
    assert len(seen) == 3
    for ts, seed in seen:
        assert seed == _derived_seed(model_seed, np.datetime64(ts))


def test_step_seed_golden_literal() -> None:
    # blake2b(digest_size=8, 0 as 8B big-endian || 2020-01-01T00:00 ns as 8B
    # big-endian signed) — a cross-platform / cross-version stability canary.
    seen: list[tuple[np.datetime64, int]] = []
    _seed_recorder(seen).predict(wide([[1.0, 2.0]]))
    assert seen[0][1] == 16874120039426919206


def test_step_seed_is_independent_of_call_chunking() -> None:
    seen_bulk: list[tuple[np.datetime64, int]] = []
    _seed_recorder(seen_bulk).predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    seen_by_row: list[tuple[np.datetime64, int]] = []
    m = _seed_recorder(seen_by_row)
    for i in range(3):
        m.predict(wide([[1.0, 2.0]], start=str(days(3)[i].date())))
    assert seen_bulk == seen_by_row


def test_batch_step_seed_uses_chunk_first_timestamp() -> None:
    batch_seeds: list[int] = []

    @qm.batch_model
    def record(ctx: qm.Context, out: list[int]) -> None:
        for step in ctx:
            out.append(step.seed)

    record(batch_seeds).predict(wide([[1.0, 2.0], [3.0, 4.0]]))

    single_seen: list[tuple[np.datetime64, int]] = []
    _seed_recorder(single_seen).predict(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert batch_seeds == [single_seen[0][1]]


def test_batch_step_seed_depends_on_call_chunking() -> None:
    # The documented guarantee boundary: a batch step keys on its chunk's
    # first timestamp, so re-chunking the timeline re-seeds it — a batch
    # model must not let step.rng affect per-row output (verify detects it).
    @qm.batch_model
    def record(ctx: qm.Context, out: list[int]) -> None:
        for step in ctx:
            out.append(step.seed)

    one_shot: list[int] = []
    record(one_shot).predict(wide([[1.0, 2.0], [3.0, 4.0]]))

    streamed: list[int] = []
    m = record(streamed)
    m.predict(wide([[1.0, 2.0]]))
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))

    assert len(one_shot) == 1 and len(streamed) == 2
    assert streamed[0] == one_shot[0]  # same chunk head, same seed
    assert streamed[1] != one_shot[0]  # a new chunk head derives a new seed


def test_distinct_model_seeds_give_distinct_step_seeds() -> None:
    seeds: list[int] = []
    for model_seed in (0, 1):
        seen: list[tuple[np.datetime64, int]] = []
        _seed_recorder(seen).predict(wide([[1.0, 2.0]]), seed=model_seed)
        seeds.append(seen[0][1])
    assert seeds[0] != seeds[1]


# --- rng --------------------------------------------------------------------------


def test_rng_is_cached_within_a_step() -> None:
    same: list[bool] = []

    @qm.model
    def record(ctx: qm.Context, out: list[bool]) -> None:
        for step in ctx:
            out.append(step.rng is step.rng)

    record(same).predict(wide([[1.0, 2.0]]))
    assert same == [True]


def test_same_seed_runs_are_bit_identical() -> None:
    @qm.model
    def noisy(ctx: qm.Context, out: list[float]) -> None:
        for step in ctx:
            out.append(step.rng.random())

    runs: list[list[float]] = []
    for _ in range(2):
        draws: list[float] = []
        noisy(draws).predict(wide([[1.0, 2.0], [3.0, 4.0]]), seed=7)
        runs.append(draws)
    assert runs[0] == runs[1]


def test_rng_is_philox_based() -> None:
    kinds: list[type] = []

    @qm.model
    def record(ctx: qm.Context, out: list[type]) -> None:
        for step in ctx:
            out.append(type(step.rng.bit_generator))

    record(kinds).predict(wide([[1.0, 2.0]]))
    assert kinds == [np.random.Philox]


# --- seed settings ----------------------------------------------------------------


def test_first_fit_seed_is_recorded_on_the_model() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    m.fit(wide([[1.0, 2.0]]), seed=42)
    assert m.seed == 42


def test_seed_on_a_later_call_is_value_error() -> None:
    # seed= is a model-lifetime setting of the first call, exactly like
    # symbols=; direct assignment (model.seed = ...) stays the expert path.
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    m.fit(wide([[1.0, 2.0]]), seed=1)
    with pytest.raises(ValueError, match="first call"):
        m.predict(wide([[3.0, 4.0]], start="2020-01-02"), seed=2)
    assert m.seed == 1


def test_seed_assignment_after_start_applies_to_future_steps() -> None:
    # Direct assignment is the documented expert path for a started model:
    # step seeds always derive from the model seed at execution time, so the
    # new value governs every later call (registered children keep their
    # registration-time seeds; see test_children).
    seen: list[tuple[np.datetime64, int]] = []
    m = _seed_recorder(seen)
    m.predict(wide([[1.0, 2.0]]))
    m.seed = 7
    m.predict(wide([[3.0, 4.0]], start="2020-01-02"))
    assert seen[0][1] == _derived_seed(0, np.datetime64(seen[0][0]))
    assert seen[1][1] == _derived_seed(7, np.datetime64(seen[1][0]))


def test_input_error_before_the_run_leaves_seed_unchanged() -> None:
    # An input error is raised before the model function runs and leaves the
    # instance reusable *as if the call never happened* — including the seed,
    # which commits only once the inputs are accepted.
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    with pytest.raises(ValueError):
        m.predict(pd.DataFrame(), seed=123)
    assert m.started is False
    assert m.seed == 0
    m.fit(wide([[1.0, 2.0]]), seed=7)  # still accepts the first-call seed
    assert m.seed == 7


def test_checkpoint_restores_the_seed_value() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    m.fit(wide([[1.0, 2.0]]), seed=9)
    restored = silent().loads(m.dumps())
    assert restored.seed == 9


def test_numpy_integer_seed_is_normalized_and_roundtrips() -> None:
    # NumPy integers are realistic inputs in a NumPy-first library; the
    # public boundary normalizes them to a plain int so the checkpoint
    # invariant (a successful dumps() loads back) holds.
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    m.fit(wide([[1.0, 2.0]]), seed=np.int64(7))
    assert type(m.seed) is int
    assert m.seed == 7
    assert silent().loads(m.dumps()).seed == 7


@pytest.mark.parametrize("bad", [1.5, "7"])
def test_non_integer_seed_is_value_error_and_instance_reusable(bad: Any) -> None:
    # A bad seed is an input error: it is raised before the model function
    # runs and leaves the instance reusable as if the call never happened.
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    with pytest.raises(ValueError, match="seed= must be an integer"):
        m.fit(wide([[1.0, 2.0]]), seed=bad)
    assert m.started is False
    assert m.seed == 0
    m.fit(wide([[1.0, 2.0]]), seed=7)
    assert m.seed == 7


def test_huge_and_negative_int_seeds_roundtrip() -> None:
    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    for seed in (-1, 2**200, -(2**200)):
        m = silent()
        m.fit(wide([[1.0, 2.0]]), seed=seed)
        assert silent().loads(m.dumps()).seed == seed


def test_directly_assigned_numpy_seed_works_and_is_normalized_at_dumps() -> None:
    # Direct assignment (model.seed = ...) is the expert path. NumPy integer
    # scalars must work during derivation, and the envelope normalizes one so
    # a dumps() that succeeds always loads back.
    @qm.model
    def stochastic(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.rng.random(len(step.symbols))

    m = stochastic()
    m.seed = np.int64(5)  # type: ignore[assignment]
    result = m.predict(wide([[1.0, 2.0]]))
    assert np.isfinite(result.array).all()
    assert stochastic().loads(m.dumps()).seed == 5


def test_directly_assigned_bad_seed_is_checkpoint_error_at_dumps() -> None:
    from qfeval_model.core.errors import CheckpointError

    @qm.model
    def silent(ctx: qm.Context) -> None:
        for _step in ctx:
            pass

    m = silent()
    m.fit(wide([[1.0, 2.0]]))
    m.seed = "oops"  # type: ignore[assignment]
    with pytest.raises(CheckpointError, match="seed must be an integer"):
        m.dumps()
