"""Checkpointing: dumps/loads/save/load/clone, replay skip, envelope.

Covers round-trips (a restored model continues bit-identically to an
uninterrupted run), identity validation before restore and its atomicity
(a failed loads leaves the target untouched), two-phase child restore,
fit-time replay skip, and the pickle envelope's key set and
format_version as a compatibility canary.
"""

from __future__ import annotations

import io
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import days, wide

import qfeval_model as qm
from qfeval_model.core.errors import (
    CheckpointError,
    QfevalModelWarning,
    TimeOrderError,
)


def _noisy_factory() -> Any:
    """A model factory using state + step.rng (the strictest roundtrip subject)."""

    @qm.model
    def noisy(ctx: qm.Context) -> None:
        for step in ctx:
            base = step.rng.random(step.inputs["close"].shape)
            prev = ctx.state.get("prev")
            out = base if prev is None else base + prev
            ctx.state["prev"] = out
            step.output = out

    return noisy


TRAIN = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
TEST = wide([[7.0, 8.0], [9.0, 10.0]], start="2020-01-04")


# --- roundtrip ---------------------------------------------------------------


def test_unstarted_roundtrip() -> None:
    noisy = _noisy_factory()
    blob = noisy().dumps()
    restored = noisy().loads(blob)
    assert restored.started is False
    assert restored.symbols is None
    assert restored.cursor is None
    # The restored instance behaves like a brand-new one.
    a = restored.fit(TRAIN).predict(TEST)
    b = noisy().fit(TRAIN).predict(TEST)
    np.testing.assert_array_equal(a.array, b.array)


def test_roundtrip_emits_no_warnings() -> None:
    noisy = _noisy_factory()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = noisy().fit(TRAIN)
        restored = noisy().loads(m.dumps())
        restored.predict(TEST)


def test_roundtrip_continuation_is_bit_identical() -> None:
    noisy = _noisy_factory()
    straight = noisy().fit(TRAIN)
    expected = straight.predict(TEST)

    saved = noisy().fit(TRAIN)
    restored = noisy().loads(saved.dumps())
    got = restored.predict(TEST)

    np.testing.assert_array_equal(got.array, expected.array)
    np.testing.assert_array_equal(got.timestamps, expected.timestamps)
    np.testing.assert_array_equal(restored.state["prev"], straight.state["prev"])
    assert restored.cursor == straight.cursor
    assert restored.seed == straight.seed
    assert straight.symbols is not None and restored.symbols is not None
    np.testing.assert_array_equal(restored.symbols, straight.symbols)


@pytest.mark.parametrize("kind", ["str", "path", "file"])
def test_save_load_routes_match_dumps_loads(kind: str, tmp_path: Path) -> None:
    noisy = _noisy_factory()
    m = noisy().fit(TRAIN)
    blob = m.dumps()
    if kind == "file":
        buffer = io.BytesIO()
        m.save(buffer)
        data = buffer.getvalue()
        buffer.seek(0)
        restored = noisy().load(buffer)
    else:
        path = tmp_path / "model.ckpt"
        target: str | Path = str(path) if kind == "str" else path
        m.save(target)
        data = path.read_bytes()
        restored = noisy().load(target)
    assert data == blob
    expected = noisy().loads(blob).predict(TEST)
    np.testing.assert_array_equal(restored.predict(TEST).array, expected.array)


def test_save_load_accepts_long_unicode_path(tmp_path: Path) -> None:
    noisy = _noisy_factory()
    path = tmp_path / "長いチェックポイント用ディレクトリ" / "モデル状態.ckpt"
    path.parent.mkdir()
    noisy().fit(TRAIN).save(path)
    restored = noisy().load(path)
    np.testing.assert_array_equal(
        restored.predict(TEST).array, noisy().fit(TRAIN).predict(TEST).array
    )


def test_file_like_streams_two_models_without_seeking() -> None:
    noisy = _noisy_factory()
    m1 = noisy().fit(TRAIN, seed=1)
    m2 = noisy().fit(TRAIN, seed=2)
    buffer = io.BytesIO()
    m1.save(buffer)
    m2.save(buffer)
    buffer.seek(0)
    r1 = noisy().load(buffer)
    r2 = noisy().load(buffer)
    assert (r1.seed, r2.seed) == (1, 2)
    np.testing.assert_array_equal(r1.state["prev"], m1.state["prev"])
    np.testing.assert_array_equal(r2.state["prev"], m2.state["prev"])


def test_loads_and_load_return_self() -> None:
    noisy = _noisy_factory()
    blob = noisy().fit(TRAIN).dumps()
    m = noisy()
    assert m.loads(blob) is m
    buffer = io.BytesIO(blob)
    m2 = noisy()
    assert m2.load(buffer) is m2


def test_clone_matches_dumps_loads_and_is_independent() -> None:
    noisy = _noisy_factory()
    m = noisy().fit(TRAIN)
    clone = m.clone()
    via_blob = noisy().loads(m.dumps())
    np.testing.assert_array_equal(clone.state["prev"], via_blob.state["prev"])
    assert clone.cursor == m.cursor

    # Independence: mutating the original's state does not touch the clone.
    m.state["prev"] = m.state["prev"] + 100.0
    expected = via_blob.predict(TEST)
    got = clone.predict(TEST)
    np.testing.assert_array_equal(got.array, expected.array)


# --- identity and validation -------------------------------------------------


def test_loads_into_different_qualname_is_checkpoint_error() -> None:
    @qm.model
    def alpha(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]

    @qm.model
    def beta(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]

    blob = alpha().fit(TRAIN).dumps()
    with pytest.raises(CheckpointError, match="identity mismatch") as excinfo:
        beta().loads(blob)
    assert "alpha" in str(excinfo.value)
    assert "beta" in str(excinfo.value)
    assert "single" in str(excinfo.value)


def test_loads_same_qualname_different_mode_is_checkpoint_error() -> None:
    def _variant(mode: str) -> qm.Model:
        def shared(ctx: qm.Context) -> None:
            for step in ctx:
                step.output = step.inputs["close"]

        shared.__qualname__ = "shared"
        factory = qm.model(shared) if mode == "single" else qm.batch_model(shared)
        return factory()

    blob = _variant("single").fit(TRAIN).dumps()
    with pytest.raises(CheckpointError, match="identity mismatch"):
        _variant("batch").loads(blob)


def test_loads_same_qualname_different_module_is_checkpoint_error() -> None:
    # Same-named functions in different files (easy to produce with closures
    # made inside factory functions) must not accept each other's checkpoints.
    def _variant(module: str) -> qm.Model:
        def shared(ctx: qm.Context) -> None:
            for step in ctx:
                step.output = step.inputs["close"]

        shared.__qualname__ = "shared"
        shared.__module__ = module
        return qm.model(shared)()

    blob = _variant("pkg_a.models").fit(TRAIN).dumps()
    with pytest.raises(CheckpointError, match="identity mismatch"):
        _variant("pkg_b.models").loads(blob)


def test_loads_same_qualname_different_implementation_is_accepted() -> None:
    def _variant(scale: float) -> qm.Model:
        def shared(ctx: qm.Context) -> None:
            for step in ctx:
                step.output = step.inputs["close"] * scale

        shared.__qualname__ = "shared"
        return qm.model(shared)()

    blob = _variant(1.0).fit(TRAIN).dumps()
    restored = _variant(2.0).loads(blob)
    pred = restored.predict(TEST)
    np.testing.assert_array_equal(pred.array, TEST.to_numpy() * 2.0)


def test_loads_different_bound_arguments_is_accepted() -> None:
    blob = qm.zoo.lag("close", 1).fit(TRAIN).dumps()
    restored = qm.zoo.lag("close", 2).loads(blob)
    assert restored.cursor == days(3).values[-1]


def test_loads_into_started_model_is_checkpoint_error() -> None:
    noisy = _noisy_factory()
    blob = noisy().fit(TRAIN).dumps()
    started = noisy().fit(TRAIN)
    with pytest.raises(CheckpointError, match="already started"):
        started.loads(blob)
    buffer = io.BytesIO(blob)
    with pytest.raises(CheckpointError, match="already started"):
        started.load(buffer)


def test_corrupted_bytes_is_checkpoint_error() -> None:
    with pytest.raises(CheckpointError, match="failed to deserialize"):
        _noisy_factory()().loads(b"not a pickle")


def _valid_envelope() -> dict[str, Any]:
    envelope = pickle.loads(_noisy_factory()().fit(TRAIN).dumps())
    assert isinstance(envelope, dict)
    return envelope


def test_non_dict_envelope_is_malformed_checkpoint_error() -> None:
    with pytest.raises(CheckpointError, match="malformed"):
        _noisy_factory()().loads(pickle.dumps([1, 2, 3]))


def test_missing_required_key_is_malformed_checkpoint_error() -> None:
    envelope = _valid_envelope()
    del envelope["seed"]
    with pytest.raises(CheckpointError, match="malformed"):
        _noisy_factory()().loads(pickle.dumps(envelope))


def test_format_version_mismatch_is_checkpoint_error() -> None:
    envelope = _valid_envelope()
    envelope["format_version"] = 0
    with pytest.raises(CheckpointError, match="unsupported format_version"):
        _noisy_factory()().loads(pickle.dumps(envelope))


@pytest.mark.parametrize(
    "field, value",
    [
        ("seed", "bad"),
        ("cursor", "bad"),
        ("symbols", ["A"]),
        ("state", {1: "non-str key"}),
        ("children", ["bad"]),
        ("children", [{"name": 1, "model": {}}]),
        ("children", [{"name": "c", "model": "bad"}]),
    ],
)
def test_wrongly_typed_envelope_field_is_checkpoint_error(
    field: str, value: Any
) -> None:
    # The engine relies on these field types; a corrupted envelope must fail
    # at load time as CheckpointError, not later as a raw TypeError/KeyError.
    envelope = _valid_envelope()
    envelope[field] = value
    with pytest.raises(CheckpointError, match="malformed"):
        _noisy_factory()().loads(pickle.dumps(envelope))


def test_failed_loads_leaves_model_unchanged() -> None:
    @qm.model
    def other(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]

    # Give the target model non-default state via a valid same-identity
    # checkpoint (kept unstarted: cursor is None).
    envelope = _valid_envelope()
    envelope.update(cursor=None, symbols=None, seed=7, state={"k": 1}, children=[])
    target = _noisy_factory()().loads(pickle.dumps(envelope))

    blob = other().fit(TRAIN).dumps()
    with pytest.raises(CheckpointError, match="identity mismatch"):
        target.loads(blob)
    assert target.state == {"k": 1}
    assert target.seed == 7
    assert target.cursor is None
    assert target.symbols is None


def test_library_version_mismatch_warns_once_and_restores() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    envelope = pickle.loads(parent().fit(TRAIN).dumps())
    envelope["library_version"] = "0.0.0-not-installed"
    envelope["children"][0]["model"]["library_version"] = "0.0.0-not-installed"
    blob = pickle.dumps(envelope)

    restored = parent()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored.loads(blob)
    assert len(caught) == 1
    assert issubclass(caught[0].category, QfevalModelWarning)
    assert "0.0.0-not-installed" in str(caught[0].message)
    # The warning points at the caller, not a library-internal line.
    assert caught[0].filename == __file__

    # The most common load path (a filesystem path) also points here.
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/checkpoint.pkl"
        with open(path, "wb") as handle:
            handle.write(blob)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = parent().load(path)
    assert [w.filename for w in caught] == [__file__]

    # The child restore at the first call does not warn again.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = restored.predict(TEST)
    assert caught == []
    np.testing.assert_array_equal(pred.array[0], TRAIN.to_numpy()[-1])


# --- state pickle failure ------------------------------------------------------


def test_unpicklable_state_fails_dumps_and_save_but_model_survives() -> None:
    @qm.model
    def bad_state(ctx: qm.Context) -> None:
        ctx.state["fn"] = lambda: None
        for step in ctx:
            step.output = step.inputs["close"]

    m = bad_state().fit(TRAIN)
    with pytest.raises(CheckpointError, match="failed to serialize"):
        m.dumps()
    with pytest.raises(CheckpointError, match="failed to serialize"):
        m.save(io.BytesIO())
    # The model keeps working after the failed checkpoint.
    pred = m.predict(TEST)
    np.testing.assert_array_equal(pred.array, TEST.to_numpy())


def test_non_str_state_key_fails_at_dumps_not_at_loads() -> None:
    # A dumps() that succeeds must yield a checkpoint loads() accepts, so
    # the str-key contract is enforced at save time (a save-then-reject
    # would surface the violation only after the state looked safely stored).
    @qm.model
    def bad_key(ctx: qm.Context) -> None:
        ctx.state[1] = "x"  # type: ignore[index]
        for step in ctx:
            step.output = step.inputs["close"]

    m = bad_key().fit(TRAIN)
    with pytest.raises(CheckpointError, match="state keys"):
        m.dumps()
    with pytest.raises(CheckpointError, match="state keys"):
        m.save(io.BytesIO())


def test_failed_save_to_path_keeps_the_existing_checkpoint(tmp_path: Path) -> None:
    # A path save streams to a temporary sibling and os.replace()s it only on
    # success, so a failed save never truncates the previous checkpoint.
    @qm.model
    def flip_state(ctx: qm.Context, bad: bool) -> None:
        ctx.state["fn"] = (lambda: None) if bad else None
        for step in ctx:
            step.output = step.inputs["close"]

    path = tmp_path / "model.ckpt"
    flip_state(False).fit(TRAIN).save(path)
    good = path.read_bytes()

    with pytest.raises(CheckpointError, match="failed to serialize"):
        flip_state(True).fit(TRAIN).save(path)
    assert path.read_bytes() == good
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left behind


# --- children: two-stage restore ----------------------------------------------


def test_child_state_restored_at_first_call() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    straight = parent().fit(TRAIN)
    expected = straight.predict(TEST)

    blob = parent().fit(TRAIN).dumps()
    restored = parent().loads(blob)
    # Children are not rebuilt at load time; the first call applies the stash.
    assert restored.children is None
    got = restored.predict(TEST)
    assert restored.children is not None
    np.testing.assert_array_equal(got.array, expected.array)
    # The lag warmup survived: the first test row sees the last train row.
    np.testing.assert_array_equal(got.array[0], TRAIN.to_numpy()[-1])


def test_dumps_right_after_loads_preserves_stashed_children() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    expected = parent().fit(TRAIN).predict(TEST)
    blob = parent().fit(TRAIN).dumps()
    # Before the first call the children exist only as stashed envelopes; a
    # re-save must carry them verbatim (byte equality does not hold because
    # pickle memoizes shared arrays, so assert the continuation instead).
    resaved = parent().loads(blob).dumps()
    got = parent().loads(resaved).predict(TEST)
    np.testing.assert_array_equal(got.array, expected.array)
    # The lag warmup survived the loads -> dumps -> loads round-trip.
    np.testing.assert_array_equal(got.array[0], TRAIN.to_numpy()[-1])


def test_clone_right_after_loads_keeps_child_state() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    expected = parent().fit(TRAIN).predict(TEST)
    blob = parent().fit(TRAIN).dumps()
    got = parent().loads(blob).clone().predict(TEST)
    np.testing.assert_array_equal(got.array, expected.array)
    # The lag warmup survived the loads -> clone hop.
    np.testing.assert_array_equal(got.array[0], TRAIN.to_numpy()[-1])


def test_grandchild_checkpoint_recurses() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.rolling_mean(qm.zoo.lag("close", 1), 2)
        for step in ctx:
            step.output = step.child.array

    expected = parent().fit(TRAIN).predict(TEST)
    blob = parent().fit(TRAIN).dumps()
    got = parent().loads(blob).predict(TEST)
    np.testing.assert_array_equal(got.array, expected.array)


def _variant_parent(child_factory: Any, name: str = "c") -> qm.Model:
    """Parents sharing one qualname but registering different children."""

    def vparent(ctx: qm.Context) -> None:
        if child_factory is not None:
            ctx.children[name] = child_factory()
        for step in ctx:
            step.output = step.inputs["close"]

    vparent.__qualname__ = "vparent"
    return qm.model(vparent)()


def test_stash_count_mismatch_is_value_error_at_first_call() -> None:
    blob = _variant_parent(lambda: qm.zoo.lag("close", 1)).fit(TRAIN).dumps()
    restored = _variant_parent(None).loads(blob)
    with pytest.raises(ValueError, match="children but the model registered"):
        restored.predict(TEST)


def test_stash_name_mismatch_is_value_error_at_first_call() -> None:
    blob = _variant_parent(lambda: qm.zoo.lag("close", 1), name="c").fit(TRAIN).dumps()
    restored = _variant_parent(lambda: qm.zoo.lag("close", 1), name="d").loads(blob)
    with pytest.raises(ValueError, match="at position"):
        restored.predict(TEST)


def test_stash_child_identity_mismatch_is_checkpoint_error_at_first_call() -> None:
    blob = _variant_parent(lambda: qm.zoo.lag("close", 1)).fit(TRAIN).dumps()
    restored = _variant_parent(lambda: qm.zoo.ffill("close")).loads(blob)  # no raise
    with pytest.raises(CheckpointError, match="identity mismatch"):
        restored.predict(TEST)


def test_restore_with_started_arg_child_leaves_user_instance_alone() -> None:
    @qm.model
    def wraps(ctx: qm.Context, c: qm.Model) -> None:
        ctx.child = c
        for step in ctx:
            step.output = step.child.array

    child = qm.zoo.lag("close", 1)
    blob = wraps(child).fit(TRAIN).dumps()
    cursor_before = child.cursor

    restored = wraps(child).loads(blob)
    pred = restored.predict(TEST)
    assert restored.children is not None
    assert restored.children["child"] is not child
    assert child.cursor == cursor_before
    np.testing.assert_array_equal(pred.array[0], TRAIN.to_numpy()[-1])


# --- replay skip (fit only) -----------------------------------------------------


def test_refit_of_saved_train_is_fully_skipped() -> None:
    @qm.model
    def rec(ctx: qm.Context, seen: list[np.datetime64]) -> None:
        for step in ctx:
            seen.append(step.timestamp)
            step.output = step.inputs["close"]

    first: list[np.datetime64] = []
    blob = rec(first).fit(TRAIN).dumps()
    assert len(first) == 3

    replayed: list[np.datetime64] = []
    rec(replayed).loads(blob).fit(TRAIN)
    assert replayed == []

    extended: list[np.datetime64] = []
    train5 = wide([[float(i), float(i)] for i in range(5)])
    rec(extended).loads(blob).fit(train5)
    assert extended == list(days(5).values[3:])


def test_batch_fit_replay_skip_partial_and_full() -> None:
    # The replay skip is mode-independent: a batch refit of already-processed
    # rows supplies no chunk at all, and an extended refit supplies exactly
    # the new rows as one chunk.
    @qm.batch_model
    def rec(ctx: qm.Context, seen: list[list[np.datetime64]]) -> None:
        for step in ctx:
            seen.append(list(step.timestamps))
            step.output = step.inputs["close"]

    first: list[list[np.datetime64]] = []
    blob = rec(first).fit(TRAIN).dumps()
    assert first == [list(days(3).values)]

    replayed: list[list[np.datetime64]] = []
    rec(replayed).loads(blob).fit(TRAIN)  # full skip: zero steps supplied
    assert replayed == []

    extended: list[list[np.datetime64]] = []
    train5 = wide([[float(i), float(i)] for i in range(5)])
    rec(extended).loads(blob).fit(train5)  # partial skip: new rows only
    assert extended == [list(days(5).values[3:])]


def test_replay_skip_with_children_keeps_continuation_intact() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    expected = parent().fit(TRAIN).predict(TEST)
    blob = parent().fit(TRAIN).dumps()
    restored = parent().loads(blob)
    restored.fit(TRAIN)  # effective T=0: fully replay-skipped
    got = restored.predict(TEST)
    np.testing.assert_array_equal(got.array, expected.array)


def test_predict_below_cursor_after_restore_is_time_order_error() -> None:
    noisy = _noisy_factory()
    blob = noisy().fit(TRAIN).dumps()
    restored = noisy().loads(blob)
    with pytest.raises(TimeOrderError):
        restored.predict(TRAIN)


# --- envelope canary -------------------------------------------------------------


def test_envelope_key_set_and_format_version() -> None:
    @qm.model
    def parent(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close", 1)
        for step in ctx:
            step.output = step.child.array

    m = parent().fit(TRAIN)
    envelope = pickle.loads(m.dumps())
    expected_keys = {
        "format_version",
        "library_version",
        "ident",
        "mode",
        "symbols",
        "cursor",
        "seed",
        "state",
        "children",
    }
    assert set(envelope) == expected_keys
    assert envelope["format_version"] == 1
    assert envelope["mode"] == "single"
    assert isinstance(envelope["children"], list)
    (entry,) = envelope["children"]
    assert set(entry) == {"name", "model"}
    assert entry["name"] == "child"
    assert set(entry["model"]) == expected_keys
    assert entry["model"]["format_version"] == 1
