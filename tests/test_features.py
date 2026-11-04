"""``Features`` behaviour: array / to_wide / to_frame / row slicing.

Includes the round-trip contract that ``to_wide()`` output is accepted
back as model input, so one model's predictions can feed the next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, wide

import qfeval_model as qm

if TYPE_CHECKING:
    from typing import assert_type

    def _typing_contract_for_row_scalars(pred: qm.Features) -> None:
        """Keep public annotations honest for normal integer row indexing."""
        row = pred[0]
        assert_type(row.timestamps, Any)
        assert_type(row.side_data["value"], Any)


def make_features(
    data: dict[str, Any] | None = None,
    side_data: dict[str, Any] | None = None,
    n: int = 3,
) -> qm.Features:
    return qm.Features(
        timestamps=days(n).to_numpy(),
        symbols=np.array(["A", "B"], dtype=str),
        data={} if data is None else data,
        side_data={} if side_data is None else side_data,
    )


def test_features_is_a_plain_dataclass_with_empty_defaults() -> None:
    f = qm.Features(timestamps=days(2).to_numpy(), symbols=np.array(["A"], dtype=str))
    assert f.data == {}
    assert f.side_data == {}
    assert f.info == {}


def test_features_equality_uses_container_identity() -> None:
    # ndarray value equality has no single boolean truth value. Features is a
    # mutable runtime container; value comparison belongs to verify helpers.
    first = make_features({"close": np.zeros((3, 2))})
    second = make_features({"close": np.zeros((3, 2))})
    assert first == first
    assert first != second


def test_array_returns_the_sole_entry_itself() -> None:
    values = np.zeros((3, 2))
    f = make_features({"close": values})
    assert f.array is values


def test_array_requires_exactly_one_entry() -> None:
    with pytest.raises(ValueError, match=r"exactly one entry.*\[\]"):
        _ = make_features().array
    two = make_features({"a": np.zeros((3, 2)), "b": np.zeros((3, 2))})
    with pytest.raises(ValueError, match=r"exactly one entry.*\['a', 'b'\]"):
        _ = two.array


def test_array_writes_through_to_data() -> None:
    f = make_features({"close": np.zeros((3, 2))})
    f.array[1, 0] = 7.0
    assert f.data["close"][1, 0] == 7.0


def test_to_wide_flat() -> None:
    values = np.arange(6, dtype=np.int64).reshape(3, 2)
    f = make_features({"close": values})
    w = f.to_wide()
    assert w.index.name == "timestamp"
    assert w.columns.name == "symbol"
    assert list(w.columns) == ["A", "B"]
    assert np.array_equal(w.index.to_numpy(), days(3).to_numpy())
    assert np.array_equal(w.to_numpy(), values)
    assert all(dtype == np.int64 for dtype in w.dtypes)


def test_to_wide_trailing_axes_use_position_multiindex() -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3)
    f = make_features({"close": values}, n=2)
    w = f.to_wide()
    expected = pd.MultiIndex.from_product(
        [np.array(["A", "B"], dtype=str), range(2), range(3)],
        names=["symbol", None, None],
    )
    assert w.columns.equals(expected)
    # Row-major reshape: the cell for (symbol, axis2, axis3) matches the array.
    assert w.loc[days(2)[0], ("A", 1, 2)] == values[0, 0, 1, 2]
    assert w.loc[days(2)[1], ("B", 0, 1)] == values[1, 1, 0, 1]


def test_to_wide_output_round_trips_as_wide_input() -> None:
    rec: dict[str, Any] = {}

    @qm.batch_model
    def probe_model(ctx: qm.Context, rec: dict[str, Any]) -> None:
        for step in ctx:
            rec["inputs"] = step.inputs

    flat = make_features({"close": np.arange(6, dtype=np.float64).reshape(3, 2)})
    probe_model(rec).predict(flat.to_wide())
    assert np.array_equal(rec["inputs"]["close"], flat.data["close"])

    trailing = make_features(
        {"close": np.arange(12, dtype=np.float64).reshape(3, 2, 2)}
    )
    probe_model(rec).predict({"close": trailing.to_wide()})
    assert np.array_equal(rec["inputs"]["close"], trailing.data["close"])


def test_to_frame_flat() -> None:
    values = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    f = make_features({"close": values})
    frame = f.to_frame()
    assert list(frame.columns) == ["timestamp", "symbol", "close"]
    assert len(frame) == 6  # T*S, NaN rows kept
    assert np.isnan(frame["close"][1])
    assert list(frame["symbol"][:2]) == ["A", "B"]


def test_to_frame_rejects_trailing_axes_and_points_to_wide() -> None:
    values = np.arange(12, dtype=np.int64).reshape(3, 2, 2)
    f = make_features({"close": values})
    with pytest.raises(ValueError, match=r"trailing axes.*to_wide\(\)"):
        f.to_frame()


def test_select_trailing_returns_plain_output_view() -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3)
    side = np.array([10.0, 20.0])
    f = make_features({"close": values}, side_data={"weight": side}, n=2)
    f.info["note"] = "shared diagnostic"

    selected = f.select_trailing(1, -1)

    np.testing.assert_array_equal(selected.array, values[:, :, 1, -1])
    assert selected.array.shape == (2, 2)
    assert np.shares_memory(selected.array, values)
    assert selected.timestamps is f.timestamps
    assert selected.symbols is f.symbols
    assert selected.side_data is f.side_data
    assert selected.info is f.info
    assert not isinstance(selected.to_wide().columns, pd.MultiIndex)


def test_select_trailing_requires_one_index_per_trailing_axis() -> None:
    flat = make_features({"close": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="at least one trailing axis"):
        flat.select_trailing()

    trailing = make_features({"close": np.zeros((3, 2, 2, 3))})
    with pytest.raises(ValueError, match=r"needs 2 index\(es\), got 1"):
        trailing.select_trailing(0)


def test_int_index_squeezes_to_one_step() -> None:
    f = make_features(
        {"close": np.arange(12, dtype=np.float64).reshape(3, 2, 2)},
        side_data={"w": np.arange(6, dtype=np.float64).reshape(3, 2)},
    )
    row = f[1]
    assert np.shape(row.timestamps) == ()
    assert row.timestamps == days(3).to_numpy()[1]
    assert row.data["close"].shape == (2, 2)
    assert np.array_equal(row.data["close"], f.data["close"][1])
    assert row.side_data["w"].shape == (2,)
    assert row.symbols is f.symbols


def test_int_index_can_produce_scalar_side_data() -> None:
    f = make_features(
        {"close": np.arange(6, dtype=np.float64).reshape(3, 2)},
        side_data={
            "number": np.arange(3, dtype=np.float64),
            "object": np.array([{"row": i} for i in range(3)], dtype=object),
        },
    )
    row = f[1]
    assert isinstance(row.timestamps, np.datetime64)
    assert isinstance(row.side_data["number"], np.float64)
    assert row.side_data["object"] == {"row": 1}


def test_trailing_step_view_renders_only_to_wide() -> None:
    # ``pred[i]`` is the same public type as a whole call, so its documented
    # conversions work too: a step view renders as a length-1 call.
    f = make_features({"close": np.arange(12, dtype=np.float64).reshape(3, 2, 2)})
    row = f[1]
    w = row.to_wide()
    assert w.shape == (1, 4)
    assert w.index[0] == days(3)[1]
    np.testing.assert_array_equal(w.to_numpy().reshape(2, 2), f.data["close"][1])
    with pytest.raises(ValueError, match=r"trailing axes.*to_wide\(\)"):
        row.to_frame()


def test_slice_index_returns_views_and_shares_symbols_info() -> None:
    f = make_features(
        {"close": np.arange(6, dtype=np.float64).reshape(3, 2)},
        side_data={"w": np.arange(3, dtype=np.float64)},
    )
    part = f[1:3]
    assert np.shares_memory(part.data["close"], f.data["close"])
    assert np.shares_memory(part.side_data["w"], f.side_data["w"])
    assert np.shares_memory(part.timestamps, f.timestamps)
    assert part.symbols is f.symbols
    assert part.info is f.info
    assert len(part.timestamps) == 2


def test_engine_produced_features_behave_the_same() -> None:
    @qm.model
    def emit(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"] * 2.0
            step.side_outputs["w"] = 0.5

    pred = emit().predict(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert np.array_equal(pred.array, [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    row = pred[1]
    assert np.shape(row.timestamps) == ()
    assert row.data["close"].shape == (2,)
    assert row.side_data["w"] == 0.5
    part = pred[0:2]
    assert np.shares_memory(part.array, pred.array)
    assert part.symbols is pred.symbols
    assert part.info is pred.info
    frame = pred.to_frame()
    assert list(frame.columns) == ["timestamp", "symbol", "close"]
    assert len(frame) == 6
