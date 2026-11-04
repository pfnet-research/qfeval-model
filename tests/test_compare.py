"""Diagnostics of ``qfeval_model.verify._compare``: comparison, concat,
and cross-call drift reporting.

The diagnostic wording is user-facing output, hence part of the contract;
this private module is tested directly because the messages are hard to
reach from the public surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import days, wide

import qfeval_model as qm
from qfeval_model.core import Features, TimeOrderError
from qfeval_model.verify import VerificationError, assert_stream_consistent
from qfeval_model.verify._compare import compare_outputs, concat_outputs


def feat(
    values: Any,
    *,
    start: str = "2020-01-01",
    symbols: tuple[str, ...] = ("A", "B"),
    side: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
) -> Features:
    """Build a Features from an array literal (timestamps from ``days``)."""
    array = np.asarray(values)
    return Features(
        timestamps=days(array.shape[0], start=start).to_numpy(),
        symbols=np.array(list(symbols)),
        data={"close": array},
        side_data={name: np.asarray(v) for name, v in (side or {}).items()},
        info=dict(info or {}),
    )


# --- compare_outputs -----------------------------------------------------------


def test_compare_equal_outputs_pass_including_nan_positions() -> None:
    a = feat([[1.0, np.nan], [3.0, 4.0]], side={"w": [np.nan, 2.0]})
    b = feat([[1.0, np.nan], [3.0, 4.0]], side={"w": [np.nan, 2.0]})
    compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_rejects_differing_timestamps() -> None:
    a = feat([[1.0, 2.0]])
    b = feat([[1.0, 2.0]], start="2020-01-02")
    with pytest.raises(VerificationError, match="timestamps differ"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_rejects_differing_symbols() -> None:
    a = feat([[1.0, 2.0]])
    b = feat([[1.0, 2.0]], symbols=("A", "C"))
    with pytest.raises(VerificationError, match="symbols differ"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_reports_broadcastable_shape_mismatch_before_dtype() -> None:
    # (2, 2, 1) vs (2, 2, 3) broadcast silently under np.isclose; the shape
    # check must fire first — even when the dtype differs too.
    a = feat(np.zeros((2, 2, 1), dtype=np.float64))
    b = feat(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(
        VerificationError,
        match=r"output shape mismatch: expected \(2, 2, 1\), got \(2, 2, 3\)",
    ):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_rejects_broadcastable_side_data_shapes() -> None:
    a = feat(np.zeros((2, 2)), side={"w": np.zeros((2, 2, 1))})
    b = feat(np.zeros((2, 2)), side={"w": np.zeros((2, 2, 3))})
    with pytest.raises(VerificationError, match=r"side_data\['w'\] shape mismatch"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_rejects_differing_dtypes() -> None:
    a = feat(np.zeros((2, 2), dtype=np.float64))
    b = feat(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(
        VerificationError, match="output dtype mismatch: expected float64"
    ):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_reports_value_mismatch_with_timestamp_and_symbol() -> None:
    a = feat([[1.0, 2.0], [3.0, 4.0]])
    b = feat([[1.0, 2.0], [3.0, 9.0]])
    with pytest.raises(VerificationError, match=r"timestamp 2020-01-02.*symbol B"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_reports_trailing_axis_position() -> None:
    a = feat(np.zeros((2, 2, 3)))
    b = feat(np.zeros((2, 2, 3)))
    b.array[0, 1, 2] = 1.0
    with pytest.raises(VerificationError, match=r"symbol B, axis 2 index 2"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_reports_side_data_name_and_axis_position() -> None:
    a = feat(np.zeros((2, 2)), side={"w": np.zeros((2, 3))})
    b = feat(np.zeros((2, 2)), side={"w": np.zeros((2, 3))})
    b.side_data["w"][0, 2] = 1.0
    with pytest.raises(
        VerificationError,
        match=r"side_data\['w'\] mismatch at timestamp 2020-01-01.*axis 1 index 2",
    ):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_rejects_nan_against_value() -> None:
    a = feat([[np.nan, 2.0]])
    b = feat([[1.0, 2.0]])
    with pytest.raises(VerificationError, match="output mismatch"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_tolerance_pass_and_fail() -> None:
    a = feat([[1.0, 2.0]])
    b = feat([[1.0 + 1e-8, 2.0]])
    compare_outputs(a, b, rtol=1e-7, atol=0.0)
    with pytest.raises(VerificationError, match="output mismatch"):
        compare_outputs(a, b, rtol=0.0, atol=0.0)


def test_compare_int_arrays_exactly() -> None:
    compare_outputs(feat([[1, 2]]), feat([[1, 2]]), rtol=1.0, atol=1.0)
    with pytest.raises(VerificationError, match="output mismatch"):
        compare_outputs(feat([[1, 2]]), feat([[1, 3]]), rtol=1.0, atol=1.0)


def test_compare_ignores_info() -> None:
    a = feat([[1.0, 2.0]], info={"note": "a"})
    b = feat([[1.0, 2.0]], info={"note": "b", "extra": 1})
    compare_outputs(a, b, rtol=0.0, atol=0.0)


# --- concat_outputs ------------------------------------------------------------


def test_concat_joins_data_and_side_data_along_axis0() -> None:
    a = feat([[1.0, 2.0], [3.0, 4.0]], side={"w": [1.0, 2.0]})
    b = feat([[5.0, 6.0]], start="2020-01-03", side={"w": [3.0]})
    joined = concat_outputs([a, b])
    assert np.array_equal(joined.timestamps, days(3).to_numpy())
    assert np.array_equal(joined.array, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.array_equal(joined.side_data["w"], [1.0, 2.0, 3.0])


def test_concat_rejects_empty_list() -> None:
    with pytest.raises(VerificationError, match="empty list"):
        concat_outputs([])


def test_concat_rejects_differing_symbols() -> None:
    a = feat([[1.0, 2.0]])
    b = feat([[3.0, 4.0]], start="2020-01-02", symbols=("A", "C"))
    with pytest.raises(VerificationError, match="differing symbols"):
        concat_outputs([a, b])


def test_concat_rejects_non_increasing_timestamps() -> None:
    a = feat([[1.0, 2.0]])
    duplicate = feat([[3.0, 4.0]])
    with pytest.raises(TimeOrderError, match="non-increasing"):
        concat_outputs([a, duplicate])


def test_concat_diagnoses_output_dtype_drift_from_real_calls() -> None:
    # Induce the drift by actually running a model that changes its output
    # dtype on the third call (the lock-free engine no longer rejects this
    # per call; concat_outputs is where it must surface).
    @qm.model
    def drifting(ctx: qm.Context) -> None:
        for step in ctx:
            calls = ctx.state.get("calls", 0)
            out = step.inputs["close"] * 1.0
            step.output = out.astype(np.float32) if calls >= 2 else out
            ctx.state["calls"] = calls + 1

    w = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model = drifting()
    preds = [model.predict(w.iloc[[i]]) for i in range(3)]
    with pytest.raises(
        VerificationError,
        match=r"outputs\[0\] has float64, outputs\[2\] has float32",
    ):
        concat_outputs(preds)


def test_concat_diagnoses_output_trailing_shape_drift() -> None:
    a = feat(np.zeros((1, 2)))
    b = feat(np.zeros((1, 2, 3)), start="2020-01-02")
    with pytest.raises(VerificationError, match="output trailing shape differs"):
        concat_outputs([a, b])


def test_concat_diagnoses_side_data_key_drift() -> None:
    a = feat([[1.0, 2.0]], side={"w": [1.0]})
    b = feat([[3.0, 4.0]], start="2020-01-02")
    with pytest.raises(
        VerificationError, match=r"side_data keys differ: outputs\[1\] lacks"
    ):
        concat_outputs([a, b])


def test_concat_diagnoses_side_data_dtype_drift() -> None:
    a = feat([[1.0, 2.0]], side={"w": np.zeros(1, dtype=np.float64)})
    b = feat(
        [[3.0, 4.0]],
        start="2020-01-02",
        side={"w": np.zeros(1, dtype=np.float32)},
    )
    with pytest.raises(VerificationError, match=r"side_data\['w'\] dtype differs"):
        concat_outputs([a, b])


def test_concat_diagnoses_side_data_trailing_shape_drift() -> None:
    a = feat([[1.0, 2.0]], side={"w": np.zeros((1,))})
    b = feat([[3.0, 4.0]], start="2020-01-02", side={"w": np.zeros((1, 2))})
    with pytest.raises(
        VerificationError, match=r"side_data\['w'\] trailing shape differs"
    ):
        concat_outputs([a, b])


# --- end-to-end: the drift diagnosis reaches assert_stream_consistent -----------


def test_stream_consistent_surfaces_side_data_key_drift() -> None:
    @qm.model
    def sometimes_side(ctx: qm.Context) -> None:
        for step in ctx:
            close = step.inputs["close"]
            step.output = close * 1.0
            if float(close[0]) > 2.0:
                step.side_outputs["w"] = 1.0

    w = wide([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(VerificationError, match="side_data keys differ"):
        assert_stream_consistent(sometimes_side, w)
