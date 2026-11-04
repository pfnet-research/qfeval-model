"""Public entry points and the execution engine: ``@model``, ``Model``.

A decorated function becomes a :class:`ModelFactory`; calling the factory binds
parameters and returns a :class:`Model`. The factory preserves ``__name__`` /
``__doc__`` / ``__signature__`` (with ``ctx`` removed) so decorated models look
like plain functions (DESIGN §Models are functions, not frameworks).

The model function is re-invoked once per public call (``fit`` / ``predict``);
no suspended execution frame survives across calls (DESIGN §The execution
model). This
module implements symbol-universe fixing, monotonic time handling,
output-buffer management, step supply, named children, and checkpointing.

Model identity is the function's ``__module__`` plus ``__qualname__`` plus its
mode; checkpoint restores and child re-registration check exactly that.
Including the module keeps same-named functions in different files (a common
accident with closures made inside factory functions) from accepting each
other's checkpoints; the cost is that moving a model function's file orphans
its checkpoints. Bound parameters are not part of the identity: a model
is assumed to be reconstructed with the same parameters (children under the
same name, checkpoints into the same construction). Code changes are
deliberately not detected either: a checkpoint saved by one implementation of a
function is accepted by any other implementation with the same qualified name
(the same stance as PyTorch / scikit-learn). Providing a compatible model
structure — and versioning the code and parameters that produced a checkpoint
— is the host application's responsibility.

:class:`Features` (the plain dataclass ``predict`` returns) lives in
:mod:`qfeval_model.core.features` and is re-exported here.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import operator
import os
import pickle
import tempfile
import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Concatenate,
    Generic,
    Literal,
    ParamSpec,
    SupportsIndex,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from qfeval_model._version import __version__
from qfeval_model.core.constants import Mode
from qfeval_model.core.context import Context, _derive_seed
from qfeval_model.core.errors import (
    CheckpointError,
    QfevalModelWarning,
    TimeOrderError,
)
from qfeval_model.core.features import Features
from qfeval_model.core.normalize import (
    RawData,
    normalize_call,
    split_side_frame,
    to_utc_ns,
)
from qfeval_model.core.split import split

__all__ = ["Features", "Model", "ModelFactory", "batch_model", "model"]

FORMAT_VERSION = 1
"""The current checkpoint format version (the only version this code reads)."""

P = ParamSpec("P")
"""The bound parameters of a decorated model function (``ctx`` removed)."""


def _ident(fn: Callable[..., None]) -> str:
    """The model-identity name: ``fn``'s ``__module__`` and ``__qualname__``
    (see the module docstring for the full identity contract)."""
    return f"{fn.__module__}:{fn.__qualname__}"


def _read_envelope(file: BinaryIO) -> Any:
    """Read one pickled envelope, normalizing failures to ``CheckpointError``."""
    try:
        return pickle.load(file)
    except Exception as exc:
        raise CheckpointError(f"failed to deserialize checkpoint: {exc}") from exc


class ModelFactory(Generic[P]):
    """A decorated model function; calling it constructs a :class:`Model`.

    Generic over the model function's parameters (``ctx`` removed), so a
    factory call is statically checked against the function's signature.

    Attributes:
        fn: The wrapped model function.
        mode: ``"single"`` or ``"batch"``.
    """

    # Set by ``functools.update_wrapper`` to mirror the wrapped function.
    __name__: str
    __qualname__: str
    __wrapped__: Callable[Concatenate[Context, P], None]

    def __init__(self, fn: Callable[Concatenate[Context, P], None], mode: Mode) -> None:
        """Wrap ``fn`` as a factory.

        Args:
            fn: The model function whose first parameter is ``ctx``. It must
                be a plain synchronous function (it is called once per public
                call and consumes its :class:`Context` exactly once) taking
                ``ctx`` as a positional first parameter; both are checked
                here, at decoration time.
            mode: ``"single"`` or ``"batch"``.

        Raises:
            TypeError: If ``fn`` is an async or generator function, or its
                first parameter cannot receive ``ctx`` positionally.
        """
        if (
            inspect.iscoroutinefunction(fn)
            or inspect.isgeneratorfunction(fn)
            or inspect.isasyncgenfunction(fn)
        ):
            raise TypeError(
                f"model function {fn.__qualname__!r} must be a plain "
                "synchronous function, not an async or generator function "
                "(the engine calls it once per public call and it must "
                "consume its steps itself)"
            )
        self.fn = fn
        self.mode = mode
        functools.update_wrapper(self, fn)
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        if not params or params[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"model function {fn.__qualname__!r} must take the Context "
                f"as its positional first parameter; got signature {signature}"
            )
        # Drop the injected ``ctx`` parameter from the public signature and
        # replace the model function's ``-> None`` with what calling the
        # factory actually returns.
        self.__signature__ = signature.replace(
            parameters=params[1:], return_annotation=Model
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Model:
        """Bind parameters and return an executable :class:`Model`.

        Raises:
            TypeError: If ``args`` / ``kwargs`` do not match the signature.
        """
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        return Model(self.fn, tuple(bound.args), dict(bound.kwargs), mode=self.mode)


class Model:
    """An executable model instance bound to concrete parameters.

    Holds the fixed symbol universe, persistent state, time cursor, seed, and
    named children. The public surface is ``fit`` / ``predict`` / ``run``
    plus checkpointing (``save`` / ``load`` / ``dumps`` / ``loads`` /
    ``clone``); ``_run_features`` is the engine-level entry the public calls
    delegate to. ``children`` exposes the engine-owned child models for
    inspection; do not mutate or call ``fit`` / ``predict`` on them directly —
    children advance only through their parent.
    """

    def __init__(
        self,
        fn: Callable[..., None],
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
        *,
        mode: Mode,
    ) -> None:
        """Create a model (usually via :meth:`ModelFactory.__call__`).

        Args:
            fn: The model function (takes ``ctx`` plus bound parameters).
            call_args: Positional parameters bound after ``ctx``.
            call_kwargs: Keyword parameters bound after ``ctx``.
            mode: ``"single"`` or ``"batch"``.
        """
        self.fn = fn
        self.call_args = tuple(call_args)
        self.call_kwargs = dict(call_kwargs or {})
        self.mode = mode
        self.symbols: npt.NDArray[np.str_] | None = None
        self.state: dict[str, Any] = {}
        self.cursor: np.datetime64 | None = None
        self.seed: int = 0
        # Set when a call failed after the model function started executing;
        # the instance then refuses further calls and checkpoints (state and
        # children are not rolled back). A restore replaces the whole state
        # and clears the latch.
        self._failed = False
        # Child envelopes (plain dicts) held from a restore until the first
        # call rebuilds and re-registers the children (see
        # ``_apply_stashed_children``).
        self._stashed_children: list[dict[str, Any]] = []
        # Children (composition). ``children`` is None until the first call
        # fixes the registration; afterwards it maps name -> child in
        # registration order. Children are owned by the engine: registration
        # builds them from the registered candidate's function and bound
        # parameters, and they advance only through the parent (their seeds
        # derived from the parent's seed and registration name).
        self.children: dict[str, Model] | None = None

    def __repr__(self) -> str:
        """Return ``fn(args...)`` built from the bound parameters' own reprs
        (only as deterministic and address-free as those reprs are)."""
        parts = [repr(v) for v in self.call_args]
        parts += [f"{k}={v!r}" for k, v in self.call_kwargs.items()]
        return f"{self.fn.__qualname__}({', '.join(parts)})"

    @property
    def started(self) -> bool:
        """Whether at least one call has advanced the time cursor."""
        return self.cursor is not None

    def _check_not_failed(self) -> None:
        if self._failed:
            raise RuntimeError(
                "this model instance failed an earlier call and refuses "
                "further calls (ctx.state and children are not rolled back "
                "on failure); build a fresh instance or restore a checkpoint "
                "into one"
            )

    # --- public calls --------------------------------------------------------

    def _call(
        self,
        phase: Literal["train", "test"],
        inputs: RawData,
        side_data: pd.DataFrame | None,
        symbols: Sequence[str] | None,
        seed: SupportsIndex | None,
    ) -> Features:
        """Apply the call settings, normalize the inputs, and run the input."""
        # Refuse first, before any validation or normalization, so a failed
        # instance never reports a different error or changes state (its seed)
        # on a re-call.
        self._check_not_failed()
        if seed is not None:
            if self.started:
                raise ValueError(
                    "seed= is a model-lifetime setting of the first call; "
                    "the model has already started"
                )
            # Normalize integer-compatible values (np.int64, ...) to a plain
            # int here, before the model runs, so the checkpoint invariant
            # (loads() requires an int seed) holds for every started model.
            try:
                seed = operator.index(seed)
            except TypeError:
                raise ValueError(
                    f"seed= must be an integer, got {type(seed).__name__}: {seed!r}"
                ) from None
        if symbols is not None and self.symbols is not None:
            raise ValueError(
                "symbols= cannot be combined with an already-fixed symbol universe"
            )
        input = normalize_call(
            inputs,
            side_data,
            symbols=self.symbols if self.symbols is not None else symbols,
        )
        # Commit the seed only after the inputs are accepted, so an input
        # error leaves the instance fully unchanged (reusable as if the call
        # never happened).
        if seed is not None:
            self.seed = seed
        return self._run_features(input, phase)

    def fit(
        self,
        inputs: RawData,
        *,
        side_data: pd.DataFrame | None = None,
        symbols: Sequence[str] | None = None,
        seed: SupportsIndex | None = None,
    ) -> Model:
        """Fit the model over ``inputs`` (train phase) and return ``self``.

        ``inputs`` may be wide (a ``{name: frame}`` mapping or a flat wide
        DataFrame) or long; wide is the basic format and long the transport
        format. Wide MultiIndex columns carry anonymous trailing axes (level 0
        is the symbol).

        ``symbols=`` / ``seed=`` are model-lifetime settings passed on the
        first call (design principle: everything fixed at model start is an
        argument of the first call); passing either to a later call raises
        ``ValueError``. ``symbols=`` pre-declares the symbol universe.

        An input error (malformed inputs, a repeated ``seed=``, ...) is raised
        before the model runs and leaves the instance reusable. Once the
        model function has started executing, a failure marks the instance
        as failed: nothing is rolled back (``ctx.state`` changes are kept
        and children keep their advanced cursors), so the instance refuses
        further calls and checkpoints — build a fresh instance or restore a
        checkpoint into one.
        """
        self._call("train", inputs, side_data, symbols, seed)
        return self

    def predict(
        self,
        inputs: RawData,
        *,
        side_data: pd.DataFrame | None = None,
        symbols: Sequence[str] | None = None,
        seed: SupportsIndex | None = None,
    ) -> Features:
        """Predict over ``inputs`` (test phase) and return a :class:`Features`.

        Accepts the same input shapes and ``symbols=`` / ``seed=`` settings
        as :meth:`fit`.
        """
        return self._call("test", inputs, side_data, symbols, seed)

    def run(
        self,
        inputs: RawData,
        *,
        side_data: pd.DataFrame | None = None,
        test_from: Any = None,
        symbols: Sequence[str] | None = None,
        seed: SupportsIndex | None = None,
    ) -> Features:
        """Backtest: fit the rows before ``test_from``, predict the rest.

        A convenience composition of :meth:`fit` and :meth:`predict`: the
        inputs are partitioned with :func:`~qfeval_model.core.split.split`
        (side data by their ``timestamp`` column), the train
        side is fitted (skipped when empty or ``test_from`` is ``None``), and
        the test side's :class:`Features` is returned. Advances ``self``
        like any other call — clone first (``model.clone().run(...)``) to keep
        the original untouched.

        Args:
            inputs: Inputs in any accepted shape — a wide
                ``{name: frame}`` mapping, a flat wide DataFrame, or a long
                DataFrame.
            side_data: Optional side-input DataFrame.
            test_from: First test timestamp (inclusive); rows before it are
                fitted. ``None`` for predict-only.
            symbols: Optional explicit symbol universe (first call only).
            seed: Model seed (first call only, see :meth:`fit`).
        Returns:
            The test-period :class:`Features`.

        Raises:
            ValueError: If the test period is empty.
        """
        if test_from is not None:
            at = to_utc_ns(test_from)
            train_inputs, test_inputs = split(inputs, test_from=at)
            if not any(len(frame) > 0 for frame in test_inputs.values()):
                raise ValueError("run() found no test rows at or after test_from")
            train_side, test_side = split_side_frame(side_data, at)
            if any(len(frame) > 0 for frame in train_inputs.values()):
                self.fit(
                    train_inputs,
                    side_data=train_side,
                    symbols=symbols,
                    seed=seed,
                )
                # The first call consumes symbols=/seed=; passing them to the
                # predict below would raise (the model has started).
                symbols = seed = None
            inputs, side_data = test_inputs, test_side

        return self.predict(
            inputs,
            side_data=side_data,
            symbols=symbols,
            seed=seed,
        )

    # --- children (composition) --------------------------------------------

    def _run_children(
        self,
        entries: list[tuple[str, Model]],
        input: Features,
        phase: Literal["train", "test"],
    ) -> dict[str, Features]:
        """Register/verify the named children and advance them over ``input``.

        The single per-call children entry point. On the first call the
        children are registered; later calls verify the registration is
        unchanged (a restored checkpoint applies its stashed child envelopes
        first). Each registered child is then advanced over the parent's
        effective ``input`` before the parent's own steps, and its result is
        returned by name (an empty mapping when there are no children).

        Children are owned by the engine and advance in lockstep with the
        parent, so each child's cursor must equal the parent's before it
        runs; a diverged cursor means the child was advanced outside the
        parent (children advance only through their parent). Diagnostics are
        not a parent/child channel in either direction: each child gets its own
        fresh ``info``, and the child's resulting ``info`` is dropped here —
        it is call-wide (never row-sliced), so exposing it would let a
        single-mode parent step read the child's later timestamps. The
        row-sliced main/side outputs are the only child -> parent data paths.
        """
        for name, child in entries:
            if not isinstance(child, Model):
                raise TypeError(
                    f"child {name!r} must be a Model instance, got "
                    f"{type(child).__name__} (a decorated factory must be "
                    "called first: ctx.children[name] = my_model())"
                )
        if self._stashed_children:
            self._apply_stashed_children(entries)
        elif self.children is None:
            self._register_children_first(entries)
        else:
            self._verify_children(entries)
        results: dict[str, Features] = {}
        for name, child in (self.children or {}).items():
            if child.cursor != self.cursor:
                raise RuntimeError(
                    f"child {name!r} is at cursor {child.cursor!r} but the "
                    f"parent is at {self.cursor!r}; children advance in "
                    "lockstep through their parent — do not advance "
                    "parent.children[...] directly"
                )
            result = child._run_features(replace(input, info={}), phase)
            results[name] = replace(result, info={})
        return results

    def _apply_stashed_children(self, entries: list[tuple[str, Model]]) -> None:
        """Apply restored child envelopes onto freshly built child models.

        The stashed child envelopes were kept verbatim at restore time, so a
        structurally broken child envelope fails here (at the first call), not
        at load time.
        """
        stash = self._stashed_children
        if len(entries) != len(stash):
            raise ValueError(
                f"restored checkpoint has {len(stash)} children but the model "
                f"registered {len(entries)}"
            )
        children: dict[str, Model] = {}
        for position, (name, child) in enumerate(entries):
            entry = stash[position]
            if entry["name"] != name:
                raise ValueError(
                    f"restored checkpoint has child {entry['name']!r} at position "
                    f"{position} but the model registered {name!r}"
                )
            restored = Model(
                child.fn, child.call_args, child.call_kwargs, mode=child.mode
            )
            restored._restore_envelope(entry["model"], warn_version=False)
            children[name] = restored
        self.children = children
        self._stashed_children = []

    def _register_children_first(self, entries: list[tuple[str, Model]]) -> None:
        # Names are unique here: the only caller feeds the items of the
        # Context's children dict.
        children: dict[str, Model] = {}
        for name, child in entries:
            if child.started:
                raise ValueError(
                    f"child {name!r} has already started (its cursor is at "
                    f"{child.cursor!r}); children must be unstarted when "
                    "first registered and are fitted only through their "
                    "parent, which derives their seeds"
                )
            # The engine owns its children: registration builds a fresh
            # instance from the candidate's function and bound parameters —
            # the same reconstruction as a checkpoint restore — so the
            # candidate the model function assigned never advances and its
            # ``state`` cannot leak the engine's progress back into the call.
            owned = Model(child.fn, child.call_args, child.call_kwargs, mode=child.mode)
            # Mixing in the registration name keeps same-structured siblings
            # (and the parent itself) on distinct random streams.
            owned.seed = _derive_seed(self.seed, name.encode())
            children[name] = owned
        self.children = children

    def _verify_children(self, entries: list[tuple[str, Model]]) -> None:
        assert self.children is not None
        if [name for name, _ in entries] != list(self.children):
            raise ValueError(
                f"child names changed across calls: registered "
                f"{list(self.children)!r}, got {[name for name, _ in entries]!r}"
            )
        for name, child in entries:
            registered = self.children[name]
            if (_ident(child.fn), child.mode) != (
                _ident(registered.fn),
                registered.mode,
            ):
                raise ValueError(
                    f"child {name!r} changed across calls: registered "
                    f"({_ident(registered.fn)}, {registered.mode}), got "
                    f"({_ident(child.fn)}, {child.mode})"
                )

    # --- checkpoint / clone ------------------------------------------------

    def dumps(self) -> bytes:
        """Serialize the model state to a checkpoint byte string.

        Saving succeeds at any call boundary; the model cannot refuse a
        checkpoint (models keep their state checkpoint-ready at every step
        boundary). The one exception is an instance whose call failed
        mid-execution: it is not at a call boundary and refuses to
        checkpoint.

        The checkpoint records the *model identity* (see the module docstring
        for what identity covers and what loading does and does not check).

        Raises:
            CheckpointError: If the state cannot be pickled or the instance
                failed an earlier call.
        """
        envelope = self._build_envelope()
        try:
            return pickle.dumps(envelope)
        except Exception as exc:
            raise CheckpointError(f"failed to serialize model state: {exc}") from exc

    def save(self, file: str | Path | BinaryIO) -> None:
        """Serialize the model to a path or binary file object.

        A path is written atomically: the envelope is streamed to a temporary
        file in the same directory and moved over ``file`` only on success,
        so a failed save never destroys an existing checkpoint. File objects
        are written at their current position (no seeking) and streamed like
        :func:`pickle.dump` (no intermediate byte string); a failure
        mid-write may leave partial bytes in ``file``. See :meth:`dumps` for
        the checkpoint contract.

        Raises:
            CheckpointError: If the state cannot be pickled or the instance
                failed an earlier call.
        """
        if isinstance(file, str | Path):
            path = Path(file)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
            try:
                with os.fdopen(fd, "wb") as handle:
                    self.save(handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return
        envelope = self._build_envelope()
        try:
            pickle.dump(envelope, file)
        except Exception as exc:
            raise CheckpointError(f"failed to serialize model state: {exc}") from exc

    def _build_envelope(self) -> dict[str, Any]:
        """Build the current-format envelope (plain Python values, pickled as-is).

        Between a restore and the first call the children exist only as
        stashed envelopes; writing them back verbatim keeps a
        restore-then-save (or clone) lossless.
        """
        if self._failed:
            raise CheckpointError(
                "cannot checkpoint a model instance that failed an earlier "
                "call (its state is not at a call boundary); build a fresh "
                "instance or restore a checkpoint into one"
            )
        # Enforce the state-key and seed contracts at save time, so a dumps()
        # that succeeds always produces a checkpoint loads() accepts.
        bad_keys = [key for key in self.state if not isinstance(key, str)]
        if bad_keys:
            raise CheckpointError(
                f"ctx.state keys must be str to checkpoint, got {bad_keys!r}"
            )
        try:
            seed = operator.index(self.seed)
        except TypeError:
            raise CheckpointError(
                f"model seed must be an integer to checkpoint, got {self.seed!r}"
            ) from None
        children = list(self._stashed_children)
        if self.children:
            children = [
                {"name": name, "model": child._build_envelope()}
                for name, child in self.children.items()
            ]
        return {
            "format_version": FORMAT_VERSION,
            "library_version": __version__,
            "ident": _ident(self.fn),
            "mode": self.mode,
            "symbols": self.symbols,
            "cursor": self.cursor,
            "seed": seed,
            "state": dict(self.state),
            "children": children,
        }

    def loads(self, data: bytes) -> Model:
        """Restore model state from a checkpoint (before the model starts).

        Identity covers the model function and mode, not the bound parameters
        or code: restoring into a compatibly constructed instance (the same
        bound parameters) is the caller's guarantee — see
        ``docs/explanation/checkpoint-semantics.md``.

        The envelope is validated (format version, identity) before this model
        is modified, so a malformed or mismatching top-level checkpoint leaves
        the model untouched. Nested child envelopes are kept verbatim and only
        rebuilt at the first call, so a structurally broken child surfaces
        then. Returns ``self`` so a restored instance can be built in one
        expression: ``restored = my_model().loads(blob)``.

        A ``library_version`` differing from the installed qfeval-model
        version emits a :class:`QfevalModelWarning` (the restore proceeds).

        Checkpoints are pickles: restoring executes arbitrary code embedded in
        the bytes, so never restore untrusted data.

        Raises:
            CheckpointError: If the model already started, the checkpoint is
                malformed, the format version is unsupported, or the model
                identity does not match.
        """
        self._check_unstarted()
        try:
            envelope = pickle.loads(data)
        except Exception as exc:
            raise CheckpointError(f"failed to deserialize checkpoint: {exc}") from exc
        self._restore_envelope(envelope)
        return self

    def load(self, file: str | Path | BinaryIO) -> Model:
        """Load model state in-place from a path or binary file object.

        Reads from the file object's current position (no seeking) and
        streams like :func:`pickle.load` (no intermediate byte string); see
        :meth:`loads` for the contract and the trust warning. Returns
        ``self``.
        """
        self._check_unstarted()
        # Both branches restore from this frame, so the version warning's
        # stacklevel points at the caller for paths and file objects alike.
        if isinstance(file, str | Path):
            with open(file, "rb") as handle:
                envelope = _read_envelope(handle)
        else:
            envelope = _read_envelope(file)
        self._restore_envelope(envelope)
        return self

    def _check_unstarted(self) -> None:
        if self.started:
            raise CheckpointError(
                "cannot restore into a model that has already started"
            )

    def _restore_envelope(self, envelope: Any, *, warn_version: bool = True) -> None:
        """Validate a deserialized envelope and commit it to this (unstarted) model.

        A checkpoint is self-produced, so validation only needs to keep a
        corrupted envelope from surfacing as a confusing failure later: the
        fields the engine relies on are type-checked here and any violation
        is a :class:`CheckpointError` at load time. The envelope's own
        ``children`` dicts are kept verbatim in :attr:`_stashed_children` and
        rebuilt at the first call, so a broken nested child surfaces then,
        not here.
        """
        try:
            version = envelope["format_version"]
            if version != FORMAT_VERSION:
                raise CheckpointError(
                    f"malformed checkpoint: unsupported format_version "
                    f"{version!r} (expected {FORMAT_VERSION})"
                )
            ident = envelope["ident"]
            mode = envelope["mode"]
            library_version = envelope["library_version"]
            symbols = envelope["symbols"]
            cursor = envelope["cursor"]
            seed = envelope["seed"]
            state = dict(envelope["state"])
            children = list(envelope["children"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"malformed checkpoint: {exc}") from exc

        for what, value, ok in [
            ("seed", seed, isinstance(seed, int)),
            ("cursor", cursor, cursor is None or isinstance(cursor, np.datetime64)),
            (
                "symbols",
                symbols,
                symbols is None
                or (isinstance(symbols, np.ndarray) and symbols.dtype.kind == "U"),
            ),
            ("state keys", list(state), all(isinstance(k, str) for k in state)),
            (
                "children",
                children,
                all(
                    isinstance(entry, dict)
                    and isinstance(entry.get("name"), str)
                    and isinstance(entry.get("model"), dict)
                    for entry in children
                ),
            ),
        ]:
            if not ok:
                raise CheckpointError(f"malformed checkpoint: bad {what}: {value!r}")

        if (ident, mode) != (_ident(self.fn), self.mode):
            raise CheckpointError(
                f"model identity mismatch: checkpoint is for "
                f"({ident}, {mode}), this model is "
                f"({_ident(self.fn)}, {self.mode})"
            )
        if warn_version and library_version != __version__:
            warnings.warn(
                f"checkpoint was saved with qfeval-model "
                f"{library_version} but {__version__} is installed; the "
                "restore proceeds, but behaviour may differ across versions",
                QfevalModelWarning,
                stacklevel=3,
            )
        # All validation is done; the assignments below cannot fail. The
        # restore replaces the whole model state, so any previously registered
        # children give way to the stashed envelopes and a failed instance
        # becomes usable again.
        self.symbols = symbols
        self.cursor = cursor
        self.seed = seed
        self.state.clear()
        self.state.update(state)
        self.children = None
        self._stashed_children = children
        self._failed = False

    def clone(self) -> Model:
        """Return a copy via ``dumps`` -> ``loads``.

        Exactly the checkpointed state is independent: ``state``, the
        cursor, the seed, the symbol universe, and the engine-owned children
        (recursively). The function and the bound parameters are shared with
        the original, exactly as they would be with a restored checkpoint —
        mutating a bound list/dict/array affects both instances.
        """
        new = Model(self.fn, self.call_args, self.call_kwargs, mode=self.mode)
        return new.loads(self.dumps())

    # --- core loop ---------------------------------------------------------

    def _run_features(
        self, input: Features, phase: Literal["train", "test"]
    ) -> Features:
        """Advance the model over ``input`` and return the call's output.

        ``"train"`` calls skip rows at or before the model cursor without
        supplying them to the model function (fit replay skip); the caller is
        responsible for replaying the same inputs that were originally
        processed. ``"test"`` calls reject such rows with
        :class:`TimeOrderError` instead. ``fit`` discards the output and
        ``predict`` returns it. The symbol universe and the cursor commit
        only when the call succeeds (fail-closed). A failure raised after
        the model function started executing marks the instance as failed:
        ``ctx.state`` and children are not rolled back, so the instance
        refuses further calls.
        """
        self._check_not_failed()
        idx = 0
        if self.cursor is not None:
            idx = int(np.searchsorted(input.timestamps, self.cursor, side="right"))
        if phase == "test" and idx > 0:
            raise TimeOrderError(
                "predict received timestamps at or before the model cursor "
                f"{self.cursor!r}; timestamps must strictly increase across calls"
            )

        eff_input = input[idx:] if idx > 0 else input

        result = Features(
            timestamps=eff_input.timestamps,
            symbols=eff_input.symbols,
        )
        ctx = Context(self, eff_input, phase, result)
        n = len(eff_input.timestamps)
        expected = n if self.mode == "single" else (1 if n > 0 else 0)
        try:
            returned = self.fn(ctx, *self.call_args, **self.call_kwargs)
            if returned is not None:
                # A returned value would be silently dropped (with a 1-row
                # call even the consumed-steps check would pass), so an
                # accidental ``return step.inputs[...]`` must fail loudly.
                raise TypeError(
                    f"model function returned {type(returned).__name__}; "
                    "model functions must return None and write results "
                    "via step.output"
                )
            if ctx._consumed != expected:
                raise RuntimeError(
                    f"model consumed {ctx._consumed} of {expected} steps; a "
                    "model function must consume every step of a call "
                    "exactly once"
                )
            # Finalization can allocate the default output. It is part of the
            # call transaction and must finish before cursor/universe commit.
            result._finalize()
        except BaseException:
            self._failed = True
            raise

        # The call succeeded: commit the symbol universe and the cursor.
        # ``eff_input`` dropped everything at or before the cursor, so its
        # last timestamp (if any) is strictly beyond it — advance
        # unconditionally.
        # Copy so the committed universe never shares memory with the
        # returned result: editing ``pred.symbols`` must stay local to that
        # result and not rewrite the model's fixed universe.
        if self.symbols is None:
            self.symbols = input.symbols.copy()
        if n:
            self.cursor = eff_input.timestamps[-1]

        return result


def model(fn: Callable[Concatenate[Context, P], None]) -> ModelFactory[P]:
    """Decorator: turn ``fn`` into a single-step (``@qm.model``) factory."""
    return ModelFactory(fn, "single")


def batch_model(fn: Callable[Concatenate[Context, P], None]) -> ModelFactory[P]:
    """Decorator: turn ``fn`` into a batch (``@qm.batch_model``) factory.

    A model's own mode does not certify its composition: a ``@qm.model``
    parent may register a batch child, whose in-chunk look-ahead would reach
    the parent through the row-sliced child outputs — verify the composed
    model (see ``docs/explanation/leak-safety.md``).
    """
    return ModelFactory(fn, "batch")
