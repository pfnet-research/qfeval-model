"""``@model`` / ``@batch_model`` decorators: ModelFactory, binding, repr.

Calling a decorated factory binds arguments eagerly (signature errors
surface at construction, not at fit), preserves the wrapped function's
metadata, and yields independent instances with no shared state.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from conftest import wide

import qfeval_model as qm


@qm.model
def documented(ctx: qm.Context, source: str) -> None:
    """A documented model."""
    for step in ctx:
        step.output = step.inputs[source]


def test_factory_preserves_function_metadata() -> None:
    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A documented model."
    assert documented.__qualname__ == "documented"


def test_factory_signature_drops_ctx() -> None:
    assert list(inspect.signature(documented).parameters) == ["source"]


def test_factory_signature_returns_model() -> None:
    # Calling a factory returns a Model, not the model function's None.
    assert inspect.signature(documented).return_annotation is qm.Model


def test_factory_wrapped_is_original_function() -> None:
    assert documented.__wrapped__.__qualname__ == "documented"
    assert list(inspect.signature(documented.__wrapped__).parameters) == [
        "ctx",
        "source",
    ]


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((), {}),  # missing required argument
        (("close", "extra"), {}),  # too many positional arguments
        (("close",), {"unknown": 1}),  # unknown keyword argument
    ],
)
def test_call_binds_eagerly_and_raises_type_error(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(TypeError):
        documented(*args, **kwargs)


@qm.model
def moving_average(ctx: qm.Context, source: Any, *, window: int = 10) -> None:
    for step in ctx:
        step.output = step.inputs["close"]


@qm.model
def backward_return(ctx: qm.Context, source: str = "close") -> None:
    for step in ctx:
        step.output = step.inputs[source]


def test_defaults_are_applied_and_visible_in_repr() -> None:
    assert repr(moving_average("close")) == "moving_average('close', window=10)"


def test_repr_is_nested_and_address_free() -> None:
    m = moving_average(backward_return(), window=10)
    assert repr(m) == "moving_average(backward_return('close'), window=10)"
    assert "0x" not in repr(m)


def test_two_instances_do_not_share_state() -> None:
    @qm.model
    def counter(ctx: qm.Context) -> None:
        ctx.state.setdefault("n", 0)
        for step in ctx:
            ctx.state["n"] += 1
            step.output = step.inputs["close"]

    first = counter()
    second = counter()
    first.fit(wide([[1.0, 2.0], [3.0, 4.0]]))
    assert first.state == {"n": 2}
    assert second.state == {}


def test_model_decorator_makes_single_mode() -> None:
    assert documented("close").mode == "single"


def test_batch_model_decorator_makes_batch_mode() -> None:
    @qm.batch_model
    def whole(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"]

    assert whole().mode == "batch"


def test_generator_function_is_rejected_at_decoration() -> None:
    with pytest.raises(TypeError, match="plain synchronous function"):

        @qm.model
        def oops(ctx: qm.Context) -> Any:
            yield  # pragma: no cover


def test_async_function_is_rejected_at_decoration() -> None:
    # The decorator also rejects these statically; the runtime check is for
    # untyped callers.
    with pytest.raises(TypeError, match="plain synchronous function"):

        @qm.model  # type: ignore[arg-type]
        async def oops(ctx: qm.Context) -> None:
            pass  # pragma: no cover


def test_missing_ctx_parameter_is_rejected_at_decoration() -> None:
    with pytest.raises(TypeError, match="positional first parameter"):

        @qm.model  # type: ignore[arg-type]
        def oops() -> None:
            pass  # pragma: no cover


def test_keyword_only_first_parameter_is_rejected_at_decoration() -> None:
    with pytest.raises(TypeError, match="positional first parameter"):

        @qm.model  # type: ignore[arg-type]
        def oops(*, ctx: qm.Context) -> None:
            pass  # pragma: no cover
