"""``Context`` / ``Step``: the step iteration protocol.

The model function is called once per public call and consumes steps via
``for step in ctx:``. A :class:`Step` is a lightweight view over the model's
input arrays and over the call's result ``Features``; only the
output's buffers are writable (DESIGN §The step protocol).
``single`` and ``batch`` modes share one ``Step`` class and differ only in how
rows are grouped and which of ``timestamp`` / ``timestamps`` exists. The dicts a
step hands out are plain and shared; mutating them is undefined behaviour.

Children are registered by name in the ``ctx.children`` dict (see
:class:`Context`).
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from qfeval_model.core.constants import Mode
from qfeval_model.core.features import Features

if TYPE_CHECKING:
    from qfeval_model.core.model import Model


def _derive_seed(seed: int, data: bytes) -> int:
    """Derive a 64-bit seed from a base seed and a distinguishing payload.

    The library's one seed-derivation rule: ``blake2b(seed, data)`` truncated
    to 64 bits. ``Step.seed`` mixes in the step's first timestamp and child
    registration mixes in the child's name (DESIGN §Determinism).
    """
    # ``model.seed`` is intentionally assignable as an expert path.  Keep the
    # one derivation boundary compatible with NumPy integer scalars too, just
    # like the public ``seed=`` argument.
    seed = operator.index(seed)
    h = hashlib.blake2b(digest_size=8)
    h.update((seed % 2**64).to_bytes(8, "big"))
    h.update(data)
    return int.from_bytes(h.digest(), "big")


class SideOutputs:
    """Dict-like write/read accessor for a step's side outputs.

    The write counterpart of ``step.side_inputs`` (which is read-only):
    ``step.side_outputs[name] = value`` writes a side output and
    ``step.side_outputs[name]`` returns the writable view of one already
    written this call. Assignment validates dtype/shape against the call's
    output buffers (stored in the result's ``side_data``); in single mode the
    logical shape is ``(*trailing)``, reshaped to ``(1, *trailing)``.
    """

    def __init__(self, step: Step) -> None:
        """Bind to the owning step."""
        self._step = step

    def __setitem__(self, name: str, value: Any) -> None:
        """Write side output ``name`` for this step."""
        if not isinstance(name, str) or not name:
            raise ValueError("side output name must be a non-empty string")
        self._step._write(name, value)

    def __getitem__(self, name: str) -> Any:
        """Return this step's view of side output ``name`` (written this call).

        A writable ``(T, *trailing)`` view in batch mode and a writable
        ``(*trailing)`` view in single mode — except that a single-mode side
        output with no trailing axes reads back as a NumPy scalar (assign via
        ``step.side_outputs[name] = ...`` to change it).
        """
        return self._step._read(name)

    def __contains__(self, name: str) -> bool:
        """Return whether ``name`` has been written in this call."""
        return name in self._step._result.side_data


class Step:
    """A single decision step (single mode) or a whole chunk (batch mode).

    Attributes:
        inputs: Inputs: each input's per-timestamp array
            (``(S, *trailing)`` in single mode, ``(T, S, *trailing)`` in
            batch mode). The arrays are input views.
        side_inputs: Side inputs: a scalar value (single mode — a NumPy
            scalar or an arbitrary Python object) or a ``(T,)`` array (batch
            mode) per name. Object-dtype side inputs carry arbitrary Python
            objects (the transport for what were once ``metadata`` values).
        side_outputs: Dict-like accessor (:class:`SideOutputs`) for this step's
            side outputs (written into the result's ``side_data``).
        children: Child results for this step: name to a row-sliced
            :class:`~qfeval_model.core.features.Features` (the child's
            per-timestamp row squeezed at this step's index in single mode,
            the whole-chunk row range in batch mode). Only the row-sliced
            main/side outputs cross the child -> parent boundary; the
            child's call-wide ``info`` is not exposed (each view's ``info``
            is empty).
    """

    def __init__(self, ctx: Context, rows: slice) -> None:
        """Create a step view (constructed by :class:`Context`).

        Everything the step needs is reachable from ``ctx``: the owning model
        (``ctx._model``), the call's effective input (``ctx._input``), and the
        per-call child outputs (``ctx._child_outputs``).
        """
        self._ctx = ctx
        self._rows = rows
        self._mode: Mode = ctx._model.mode
        # The single row selector: an ``int`` in single mode (which squeezes the
        # ``T`` axis on read) or the row ``slice`` in batch mode. Decided once
        # here so no method re-tests the mode to pick between ``_i`` and rows.
        self._key: int | slice = self._i if self._mode == "single" else rows
        self._rng: np.random.Generator | None = None
        input = ctx._input
        self.inputs: dict[str, npt.NDArray[Any]] = self._view(input.data)
        # ``dict[str, Any]``: a single-mode side input is a scalar (a NumPy
        # scalar or an arbitrary Python object), not an ndarray.
        self.side_inputs: dict[str, Any] = self._view(input.side_data)
        self.side_outputs = SideOutputs(self)
        self.children: dict[str, Features] = {
            name: pred[self._key] for name, pred in ctx._child_outputs.items()
        }

    @property
    def _i(self) -> int:
        return int(self._rows.start)

    def _view(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        """Slice each array to this step (a row in single mode, rows in batch)."""
        return {name: array[self._key] for name, array in mapping.items()}

    def _write(self, name: str | None, value: Any) -> None:
        """Write ``value`` into the result slot ``name`` (``None`` = output).

        Step is the sole ndarray-izer of written values and the owner of the
        single-mode ``T`` axis: a single step's value is ``(S, *trailing)`` and
        gains a length-1 ``T`` axis so the buffer write is uniform.
        """
        arr = np.asarray(value)
        self._result._assign(
            name, self._rows, arr[None] if self._mode == "single" else arr
        )

    def _read(self, name: str | None) -> Any:
        """Return the writable view of result slot ``name`` (``None`` = output).

        The row selector ``_key`` squeezes the ``T`` axis in single mode, so a
        single step reads back ``(S, *trailing)`` and a batch step ``(T, ...)``
        — a trailing-less single side output therefore reads back as a NumPy
        scalar, not a view.
        """
        return self._result._view(name, self._key)

    # --- reads -------------------------------------------------------------

    @property
    def timestamp(self) -> np.datetime64:
        """The step's timestamp (single mode only)."""
        if self._mode != "single":
            raise AttributeError("batch steps expose 'timestamps', not 'timestamp'")
        return cast(np.datetime64, self._ctx._input.timestamps[self._i])

    @property
    def timestamps(self) -> npt.NDArray[np.datetime64]:
        """The chunk's timestamps ``(T,)`` (batch mode only)."""
        if self._mode != "batch":
            raise AttributeError("single steps expose 'timestamp', not 'timestamps'")
        return cast(npt.NDArray[np.datetime64], self._ctx._input.timestamps[self._rows])

    @property
    def symbols(self) -> npt.NDArray[np.str_]:
        """The fixed symbol labels ``(S,)``."""
        return self._ctx._input.symbols

    @property
    def phase(self) -> Literal["train", "test"]:
        """``"train"`` (fit) or ``"test"`` (predict)."""
        return self._ctx.phase

    @property
    def is_train(self) -> bool:
        """Whether this step belongs to a ``fit`` (train) call."""
        return self._ctx.phase == "train"

    @property
    def is_test(self) -> bool:
        """Whether this step belongs to a ``predict`` (test) call."""
        return self._ctx.phase == "test"

    @property
    def seed(self) -> int:
        """Deterministic step seed derived from the model seed and timestamp.

        The same ``(model seed, timestamp)`` always yields the same seed, so
        single-mode steps are independent of how the timeline is chunked into
        calls (DESIGN §Determinism). A batch step keys on the chunk's *first*
        timestamp, so its seed depends on how the caller chunks calls; a batch
        model must not let ``step.rng`` affect per-row output (the verify
        harness detects the violation).
        """
        base = self._ctx._input.timestamps[self._rows.start]
        ns = int(base.astype("datetime64[ns]").astype(np.int64))
        return _derive_seed(self._ctx._model.seed, ns.to_bytes(8, "big", signed=True))

    @property
    def rng(self) -> np.random.Generator:
        """A lazily created, deterministic random generator for this step."""
        if self._rng is None:
            self._rng = np.random.Generator(np.random.Philox(key=self.seed))
        return self._rng

    @property
    def child(self) -> Features:
        """The child registered as ``ctx.child`` (name ``"child"``)."""
        try:
            return self.children["child"]
        except KeyError:
            raise AttributeError(
                "no child registered under the name 'child'; assign one with "
                "ctx.child = model (named children are registered via "
                "ctx.children['name'] and read via step.children['name'])"
            ) from None

    # --- writes ------------------------------------------------------------

    @property
    def output(self) -> npt.NDArray[Any]:
        """Writable view of this step's output slot (``(S, *trailing)`` in
        single mode, ``(T, S, *trailing)`` in batch mode)."""
        # Unlike a side output, the main output always keeps its S axis, so
        # the read is always an ndarray view.
        return cast(npt.NDArray[Any], self._read(None))

    @output.setter
    def output(self, value: Any) -> None:
        self._write(None, value)

    @property
    def info(self) -> dict[str, Any]:
        """The call's free-form diagnostic dict (shared by every step)."""
        return self._ctx._result.info

    @property
    def _result(self) -> Features:
        return self._ctx._result


class Context:
    """Injected first argument of a model function; supplies steps and state.

    Children are registered by putting models into the ``ctx.children`` dict
    before iterating (``ctx.children["fast"] = m``); ``ctx.child = m`` is
    shorthand for ``ctx.children["child"] = m``. The dict is read when
    iteration starts; the engine owns its children and builds them from each
    registered candidate's function and bound parameters, so the candidate
    instances themselves never advance. Registration is always explicit;
    passing a model as a bound parameter does not register it.
    """

    def __init__(
        self,
        model: Model,
        input: Features,
        phase: Literal["train", "test"],
        result: Features,
    ) -> None:
        """Create a context for one public call.

        Args:
            model: The owning model.
            input: The call's effective input.
            phase: ``"train"`` or ``"test"``.
            result: The call's result :class:`Features` (its output buffers).
        """
        self._model = model
        self._input = input
        self.phase = phase
        self._result = result
        self.children: dict[str, Model] = {}
        # The call's child outputs by name; populated when iteration starts
        # (see :meth:`__iter__`) and read by each :class:`Step`.
        self._child_outputs: dict[str, Features] = {}
        self._started = False
        self._consumed = 0

    @property
    def child(self) -> Model:
        """Write-only shorthand: ``ctx.child = m`` is ``ctx.children["child"] = m``."""
        raise AttributeError(
            "ctx.child is write-only registration shorthand; read a child's "
            "per-step output via step.child (or step.children['name'])"
        )

    @child.setter
    def child(self, value: Model) -> None:
        self.children["child"] = value

    @property
    def state(self) -> dict[str, Any]:
        """The model's persistent state dict (always the same object).

        Keys must be ``str`` and values must be picklable for checkpointing
        to succeed. The library performs no deep-copy defence or rollback:
        if a call fails mid-way, changes already made to ``ctx.state`` are
        kept (DESIGN §The execution model).
        """
        return self._model.state

    def _make_step(self, rows: slice) -> Step:
        """Create a step over the covered rows."""
        step = Step(self, rows)
        self._consumed += 1
        return step

    def __iter__(self) -> Iterator[Step]:
        """Yield this call's steps exactly once (the context is one-shot).

        Raises:
            RuntimeError: On a second iteration attempt; re-iterating would
                re-supply already-processed timestamps within one call.
        """
        if self._started:
            raise RuntimeError(
                "Context is one-shot; iterate it exactly once per call "
                "(a second 'for step in ctx' would re-process the same "
                "timestamps)"
            )
        self._started = True
        self._child_outputs = self._model._run_children(
            list(self.children.items()), self._input, self.phase
        )
        n = len(self._input.timestamps)
        if self._model.mode == "single":
            for i in range(n):
                yield self._make_step(slice(i, i + 1))
        elif n > 0:
            yield self._make_step(slice(0, n))
