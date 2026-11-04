"""Input contract: as_wide / as_frame / normalization.

Covers the accepted input shapes (flat wide frame, wide mapping, long
frame), timestamp normalization (timezone, unit, NaT, sorting), symbol
validation and ordering, duplicate aggregation (later non-NaN wins),
panel expansion and dtype rules, trailing axes via MultiIndex columns,
declared symbol universes (``symbols=``), and side_data normalization —
each observed through the public fit/predict surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, long_of, wide

import qfeval_model as qm
from qfeval_model.core.normalize import to_utc_ns


def probe(
    data: qm.RawData,
    side_data: pd.DataFrame | None = None,
    symbols: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Record what a predict call delivers to the model (the observable side)."""
    rec: dict[str, Any] = {}

    @qm.batch_model
    def probe_model(ctx: qm.Context, rec: dict[str, Any]) -> None:
        for step in ctx:
            rec["timestamps"] = step.timestamps
            rec["symbols"] = step.symbols
            rec["inputs"] = step.inputs
            rec["side_inputs"] = step.side_inputs

    probe_model(rec).predict(data, side_data=side_data, symbols=symbols)
    return rec


# --- accepted shapes (as_wide branches) ---------------------------------------


def test_flat_wide_frame_normalizes_to_close() -> None:
    w = wide([[1.0, 2.0], [3.0, 4.0]])
    result = qm.as_wide(w)
    assert list(result) == ["close"]
    frame = result["close"]
    assert frame.index.name == "timestamp"
    assert frame.columns.name == "symbol"
    assert np.array_equal(frame.to_numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_wide_mapping_passes_through_canonicalized() -> None:
    data = {"a": wide([[1.0, 2.0]]), "b": wide([[3.0, 4.0]])}
    result = qm.as_wide(data)
    assert list(result) == ["a", "b"]
    assert np.array_equal(result["b"].to_numpy(), [[3.0, 4.0]])
    assert result["a"].index.name == "timestamp"


def test_long_frame_pivots_per_input_column() -> None:
    long = pd.DataFrame(
        {
            "timestamp": list(days(2)) * 2,
            "symbol": ["A", "A", "B", "B"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [5.0, 6.0, 7.0, 8.0],
        }
    )
    result = qm.as_wide(long)
    assert sorted(result) == ["f1", "f2"]
    assert np.array_equal(result["f1"].to_numpy(), [[1.0, 3.0], [2.0, 4.0]])
    assert np.array_equal(result["f2"].to_numpy(), [[5.0, 7.0], [6.0, 8.0]])


def test_bare_multiindex_columns_frame_is_rejected() -> None:
    frame = pd.DataFrame(
        [[1.0, 2.0]],
        index=days(1),
        columns=pd.MultiIndex.from_tuples([("A", 0), ("A", 1)]),
    )
    with pytest.raises(ValueError, match="MultiIndex columns are not accepted"):
        qm.as_wide(frame)


@pytest.mark.parametrize("present", ["timestamp", "symbol"])
def test_single_key_column_is_ambiguous(present: str) -> None:
    frame = pd.DataFrame({present: [1], "x": [2.0]})
    with pytest.raises(ValueError, match="ambiguous inputs"):
        qm.as_wide(frame)


def test_non_frame_non_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a DataFrame or a mapping"):
        qm.as_wide(np.zeros((2, 2)))  # type: ignore[arg-type]


def test_empty_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        qm.as_wide({})


def test_mapping_value_must_be_frame() -> None:
    with pytest.raises(ValueError, match="must be a DataFrame"):
        qm.as_wide({"x": [1.0, 2.0]})  # type: ignore[dict-item]


def test_mapping_key_must_be_str() -> None:
    with pytest.raises(ValueError, match="must be str"):
        qm.as_wide({1: wide([[1.0, 2.0]])})  # type: ignore[dict-item]


@pytest.mark.parametrize("name", ["timestamp", "symbol"])
def test_mapping_key_must_not_be_reserved(name: str) -> None:
    # timestamp/symbol are the long-format key columns.
    with pytest.raises(ValueError, match="is reserved"):
        qm.as_wide({name: wide([[1.0, 2.0]])})


@pytest.mark.parametrize("name", ["axis2", "axis10"])
def test_axis_n_feature_names_are_accepted(name: str) -> None:
    long = long_of(wide([[1.0, 2.0]])).rename(columns={"close": name})
    assert list(qm.as_wide(long)) == [name]
    assert list(qm.as_wide({name: wide([[1.0, 2.0]])})) == [name]


def test_empty_names_are_rejected_on_every_path() -> None:
    # The name contract (non-empty str) is one definition shared by input
    # names, symbol labels, and side keys.
    with pytest.raises(ValueError, match="non-empty"):
        qm.as_wide({"": wide([[1.0, 2.0]])})
    with pytest.raises(ValueError, match="non-empty"):
        probe(wide([[1.0, 2.0]], symbols=("A", "")))
    with pytest.raises(ValueError, match="non-empty"):
        probe(
            wide([[1.0, 2.0]]),
            side_data=pd.DataFrame({"timestamp": days(1), "": [1.0]}),
        )


def test_long_without_input_columns_is_rejected() -> None:
    frame = pd.DataFrame({"timestamp": days(1), "symbol": ["A"]})
    with pytest.raises(ValueError, match="at least one feature"):
        qm.as_wide(frame)


def test_long_duplicate_column_labels_are_rejected() -> None:
    frame = pd.DataFrame(
        [[days(1)[0], "A", 1.0, 2.0]], columns=["timestamp", "symbol", "x", "x"]
    )
    with pytest.raises(ValueError, match="duplicate"):
        qm.as_wide(frame)


def test_long_non_str_column_label_is_rejected() -> None:
    frame = pd.DataFrame([[days(1)[0], "A", 1.0]], columns=["timestamp", "symbol", 1])
    with pytest.raises(ValueError, match="must be str"):
        qm.as_wide(frame)


def test_zero_row_wide_input_is_rejected() -> None:
    empty = pd.DataFrame(
        np.empty((0, 2)), index=pd.DatetimeIndex([]), columns=["A", "B"]
    )
    with pytest.raises(ValueError, match="at least one row"):
        probe(empty)


def test_zero_row_long_input_is_rejected() -> None:
    empty = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([]),
            "symbol": pd.Series([], dtype=str),
            "close": pd.Series([], dtype=float),
        }
    )
    with pytest.raises(ValueError, match="at least one row"):
        probe(empty)


def test_fit_wide_then_predict_long_is_accepted() -> None:
    rec: dict[str, Any] = {}

    @qm.batch_model
    def probe_model(ctx: qm.Context, rec: dict[str, Any]) -> None:
        for step in ctx:
            rec["inputs"] = step.inputs

    m = probe_model(rec)
    m.fit(wide([[1.0, 2.0], [3.0, 4.0]]))
    test_wide = wide([[5.0, 6.0], [7.0, 8.0]], start="2020-01-03")
    long = pd.DataFrame(
        {
            "timestamp": list(days(2, start="2020-01-03")) * 2,
            "symbol": ["A", "A", "B", "B"],
            "close": [5.0, 7.0, 6.0, 8.0],
        }
    )
    m.predict(long)
    from_long = rec["inputs"]["close"].copy()
    m2 = probe_model(rec)
    m2.fit(wide([[1.0, 2.0], [3.0, 4.0]]))
    m2.predict(test_wide)
    assert np.array_equal(rec["inputs"]["close"], from_long)


# --- timestamp normalization ---------------------------------------------------


def test_wide_tz_aware_index_converts_to_utc_naive() -> None:
    index = pd.DatetimeIndex(["2020-01-01 09:00"], tz="Asia/Tokyo")
    frame = pd.DataFrame([[1.0, 2.0]], index=index, columns=["A", "B"])
    result = qm.as_wide(frame)["close"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index[0] == pd.Timestamp("2020-01-01 00:00")


def test_long_tz_aware_column_converts_to_utc_naive() -> None:
    long = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(["2020-01-01 09:00"], tz="Asia/Tokyo"),
            "symbol": ["A"],
            "close": [1.0],
        }
    )
    result = qm.as_wide(long)["close"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is None
    assert result.index[0] == pd.Timestamp("2020-01-01 00:00")


def test_to_utc_ns_scalar_paths() -> None:
    tokyo = pd.Timestamp("2020-01-01 09:00", tz="Asia/Tokyo")
    assert to_utc_ns(tokyo) == np.datetime64("2020-01-01T00:00:00", "ns")
    # naive is taken as UTC (value unchanged), strings are parsed.
    assert to_utc_ns("2020-01-01") == np.datetime64("2020-01-01", "ns")
    with pytest.raises(ValueError, match="missing value"):
        to_utc_ns(None)
    with pytest.raises(ValueError, match="missing value"):
        to_utc_ns(pd.NaT)
    with pytest.raises(ValueError, match="timestamp-like"):
        to_utc_ns("not a timestamp")


def test_naive_timestamps_are_taken_as_utc() -> None:
    frame = wide([[1.0, 2.0]])
    result = qm.as_wide(frame)["close"]
    assert result.index[0] == pd.Timestamp("2020-01-01")


@pytest.mark.parametrize("unit", ["s", "us"])
def test_coarser_datetime_units_normalize_to_ns(unit: str) -> None:
    index = pd.DatetimeIndex(np.array(["2020-01-01"], dtype=f"datetime64[{unit}]"))
    frame = pd.DataFrame([[1.0, 2.0]], index=index, columns=["A", "B"])
    result = qm.as_wide(frame)["close"]
    assert result.index.dtype == np.dtype("datetime64[ns]")


def test_nat_in_wide_index_is_rejected() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2020-01-01"), pd.NaT])
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=["A", "B"])
    with pytest.raises(ValueError, match="NaT"):
        qm.as_wide(frame)


def test_nat_in_long_column_is_rejected() -> None:
    long = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([pd.NaT]),
            "symbol": ["A"],
            "close": [1.0],
        }
    )
    with pytest.raises(ValueError, match="NaT"):
        qm.as_wide(long)


def test_non_datetime_wide_index_is_rejected() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=[0], columns=["A", "B"])
    with pytest.raises(ValueError, match="datetime index"):
        qm.as_wide(frame)


def test_object_dtype_long_timestamp_column_is_rejected() -> None:
    long = pd.DataFrame({"timestamp": ["2020-01-01"], "symbol": ["A"], "close": [1.0]})
    with pytest.raises(ValueError, match="must be datetime64"):
        qm.as_wide(long)


def test_unsorted_rows_come_out_sorted() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-01"])
    frame = pd.DataFrame([[2.0, 20.0], [1.0, 10.0]], index=index, columns=["A", "B"])
    rec = probe(frame)
    assert np.array_equal(rec["timestamps"], days(2).to_numpy())
    assert np.array_equal(rec["inputs"]["close"], [[1.0, 10.0], [2.0, 20.0]])
    long = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(["2020-01-02", "2020-01-01"]),
            "symbol": ["A", "A"],
            "close": [2.0, 1.0],
        }
    )
    rec = probe(long)
    assert np.array_equal(rec["timestamps"], days(2).to_numpy())
    assert np.array_equal(rec["inputs"]["close"], [[1.0], [2.0]])


# --- symbol / name validation --------------------------------------------------


def test_int_wide_column_labels_are_rejected() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=days(1), columns=[1, 2])
    with pytest.raises(ValueError, match="must be str"):
        probe(frame)


def test_duplicate_wide_columns_are_rejected() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=days(1), columns=["A", "A"])
    with pytest.raises(ValueError, match="duplicates"):
        probe(frame)


def test_non_str_long_symbols_are_rejected() -> None:
    long = pd.DataFrame({"timestamp": days(1), "symbol": [1], "close": [1.0]})
    with pytest.raises(ValueError, match="must be str"):
        probe(long)


def test_symbol_axis_is_sorted_not_input_order() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=days(1), columns=["B", "A"])
    rec = probe(frame)
    assert list(rec["symbols"]) == ["A", "B"]
    assert np.array_equal(rec["inputs"]["close"], [[2.0, 1.0]])


# --- duplicate aggregation (last non-NaN in row order) ---------------------------


def test_long_duplicates_keep_last_non_nan() -> None:
    long = pd.DataFrame(
        {
            "timestamp": [days(1)[0]] * 3,
            "symbol": ["A"] * 3,
            "close": [1.0, np.nan, 3.0],
        }
    )
    assert probe(long)["inputs"]["close"][0, 0] == 3.0


def test_long_duplicates_all_nan_stay_nan() -> None:
    long = pd.DataFrame(
        {
            "timestamp": [days(1)[0]] * 2,
            "symbol": ["A"] * 2,
            "close": [np.nan, np.nan],
        }
    )
    assert np.isnan(probe(long)["inputs"]["close"][0, 0])


def test_long_duplicates_aggregate_per_input_independently() -> None:
    long = pd.DataFrame(
        {
            "timestamp": [days(1)[0]] * 2,
            "symbol": ["A"] * 2,
            "f1": [1.0, np.nan],
            "f2": [np.nan, 2.0],
        }
    )
    rec = probe(long)
    assert rec["inputs"]["f1"][0, 0] == 1.0
    assert rec["inputs"]["f2"][0, 0] == 2.0


def test_wide_duplicate_index_keeps_last_non_nan_per_cell() -> None:
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    frame = pd.DataFrame([[1.0, 5.0], [np.nan, 6.0]], index=index, columns=["A", "B"])
    rec = probe(frame)
    assert np.array_equal(rec["inputs"]["close"], [[1.0, 6.0]])


# --- panel expansion & dtype -----------------------------------------------------


def test_timestamp_axis_is_union_across_inputs() -> None:
    data = {
        "f1": wide([[1.0, 2.0], [3.0, 4.0]]),
        "f2": wide([[5.0, 6.0], [7.0, 8.0]], start="2020-01-02"),
    }
    rec = probe(data)
    assert np.array_equal(rec["timestamps"], days(3).to_numpy())
    assert np.isnan(rec["inputs"]["f1"][2]).all()
    assert np.isnan(rec["inputs"]["f2"][0]).all()
    assert np.array_equal(rec["inputs"]["f2"][1:], [[5.0, 6.0], [7.0, 8.0]])


def test_symbol_axis_is_union_across_inputs() -> None:
    data = {
        "f1": wide([[1.0]], symbols=("A",)),
        "f2": wide([[2.0]], symbols=("B",)),
    }
    rec = probe(data)
    assert list(rec["symbols"]) == ["A", "B"]
    assert rec["inputs"]["f1"][0, 0] == 1.0 and np.isnan(rec["inputs"]["f1"][0, 1])
    assert np.isnan(rec["inputs"]["f2"][0, 0]) and rec["inputs"]["f2"][0, 1] == 2.0


def test_int_input_with_missing_cells_is_rejected() -> None:
    data = {
        "f1": wide([[1.0, 2.0], [3.0, 4.0]]),
        "f2": pd.DataFrame([[1, 2]], index=days(1), columns=["A", "B"], dtype=np.int64),
    }
    with pytest.raises(ValueError, match="NaN-filled"):
        probe(data)


def test_fully_covered_int_and_bool_inputs_keep_dtype() -> None:
    data = {
        "i": pd.DataFrame(
            [[1, 2], [3, 4]], index=days(2), columns=["A", "B"], dtype=np.int64
        ),
        "b": pd.DataFrame(
            [[True, False], [False, True]], index=days(2), columns=["A", "B"]
        ),
    }
    rec = probe(data)
    assert rec["inputs"]["i"].dtype == np.int64
    assert np.array_equal(rec["inputs"]["i"], [[1, 2], [3, 4]])
    assert rec["inputs"]["b"].dtype == np.bool_
    assert np.array_equal(rec["inputs"]["b"], [[True, False], [False, True]])


def test_fully_covered_object_input_keeps_dtype() -> None:
    # A (T, S) categorical input such as sector labels rides the general
    # dtype rule (DESIGN §The input boundary): full coverage passes through
    # with the dtype preserved, missing cells are rejected below.
    labels = pd.DataFrame(
        [["tech", "fin"], ["tech", "fin"]],
        index=days(2),
        columns=["A", "B"],
        dtype=object,
    )
    rec = probe({"close": wide([[1.0, 2.0], [3.0, 4.0]]), "sector": labels})
    assert rec["inputs"]["sector"].dtype == object
    assert rec["inputs"]["sector"][1].tolist() == ["tech", "fin"]


def test_one_named_input_rejects_mixed_object_and_numeric_columns() -> None:
    # A normal CSV ingestion mistake can turn just one symbol column into
    # strings. Silently promoting the entire named feature to object moves the
    # error into model arithmetic, far away from the input boundary.
    mixed = pd.DataFrame(
        {"A": [1.0, 2.0], "B": ["3", "4"]},
        index=days(2),
    )
    with pytest.raises(ValueError, match="mix object and non-object dtypes"):
        probe(mixed)


def test_object_input_with_missing_cells_is_rejected() -> None:
    labels = pd.DataFrame(
        [["tech", "fin"]], index=days(1), columns=["A", "B"], dtype=object
    )
    data = {"close": wide([[1.0, 2.0], [3.0, 4.0]]), "sector": labels}
    with pytest.raises(ValueError, match="non-float feature 'sector' has missing"):
        probe(data)


@pytest.mark.parametrize(
    "column",
    [
        pd.Series(["tech", "fin", "tech"], dtype=object),
        pd.Series([1, 2, 3], dtype=np.int64),
    ],
    ids=["object", "int64"],
)
def test_long_input_with_missing_cells_matches_the_wide_rule(
    column: pd.Series,
) -> None:
    # The same logical data as the wide rejections above, in long form: the
    # (day 2, "B") cell is absent, so the coverage rule must match — an
    # object column's pivot NaN must not slip through as a value.
    ts = days(2)
    df = pd.DataFrame(
        {"timestamp": [ts[0], ts[0], ts[1]], "symbol": ["A", "B", "A"], "x": column}
    )
    with pytest.raises(ValueError, match="non-float feature 'x' has missing"):
        probe(df)


def test_fully_covered_long_object_input_is_accepted() -> None:
    # The long counterpart of the covered wide object input: same acceptance.
    ts = days(2)
    df = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[0], ts[1], ts[1]],
            "symbol": ["A", "B", "A", "B"],
            "sector": pd.Series(["tech", "fin", "tech", "fin"], dtype=object),
        }
    )
    rec = probe(df)
    assert rec["inputs"]["sector"].dtype == object
    assert rec["inputs"]["sector"].tolist() == [["tech", "fin"], ["tech", "fin"]]


@pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean"])
def test_pandas_nullable_dtypes_become_float64_with_nan(dtype: str) -> None:
    frame = pd.DataFrame({"A": pd.Series([1, None], dtype=dtype, index=days(2))})
    rec = probe(frame)
    values = rec["inputs"]["close"]
    assert values.dtype == np.float64
    assert values[0, 0] == 1.0
    assert np.isnan(values[1, 0])


def test_normalized_arrays_do_not_share_memory_with_input() -> None:
    w = wide([[1.0, 2.0]])
    rec = probe(w)
    w.iloc[0, 0] = 999.0
    assert rec["inputs"]["close"][0, 0] == 1.0


# --- trailing axes (MultiIndex columns) ------------------------------------------


def test_trailing_axis_positions_are_sorted_labels() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("A", "z"), ("A", "a"), ("B", "z"), ("B", "a")]
    )
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=days(1), columns=columns)
    rec = probe({"x": frame})
    values = rec["inputs"]["x"]
    assert values.shape == (1, 2, 2)
    # Labels are sorted ("a" -> 0, "z" -> 1) and not retained.
    assert np.array_equal(values[0], [[2.0, 1.0], [4.0, 3.0]])


def test_trailing_level_names_are_ignored() -> None:
    tuples = [("A", 0), ("A", 1), ("B", 0), ("B", 1)]
    named = pd.MultiIndex.from_tuples(tuples, names=["foo", "bar"])
    unnamed = pd.MultiIndex.from_tuples(tuples)
    row = [[1.0, 2.0, 3.0, 4.0]]
    rec_named = probe({"x": pd.DataFrame(row, index=days(1), columns=named)})
    rec_unnamed = probe({"x": pd.DataFrame(row, index=days(1), columns=unnamed)})
    assert np.array_equal(rec_named["inputs"]["x"], rec_unnamed["inputs"]["x"])


def test_inputs_may_mix_trailing_shapes() -> None:
    columns = pd.MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0), ("B", 1)])
    data = {
        "flat": wide([[1.0, 2.0]]),
        "tr": pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=days(1), columns=columns),
    }
    rec = probe(data)
    assert rec["inputs"]["flat"].shape == (1, 2)
    assert rec["inputs"]["tr"].shape == (1, 2, 2)


def test_missing_trailing_product_cells_become_nan() -> None:
    columns = pd.MultiIndex.from_tuples([("A", 0), ("A", 1), ("B", 0)])
    frame = pd.DataFrame([[1.0, 2.0, 3.0]], index=days(1), columns=columns)
    rec = probe({"x": frame})
    values = rec["inputs"]["x"]
    assert values.shape == (1, 2, 2)
    assert values[0, 1, 0] == 3.0
    assert np.isnan(values[0, 1, 1])


def test_unsortable_trailing_labels_are_rejected() -> None:
    columns = pd.MultiIndex.from_tuples([("A", 1), ("A", "x")])
    frame = pd.DataFrame([[1.0, 2.0]], index=days(1), columns=columns)
    with pytest.raises(ValueError, match="cannot be sorted"):
        probe({"x": frame})


# --- symbols= (declared universe) -------------------------------------------------


def test_declared_superset_yields_nan_columns() -> None:
    rec = probe(wide([[1.0]], symbols=("A",)), symbols=("A", "C"))
    assert list(rec["symbols"]) == ["A", "C"]
    assert rec["inputs"]["close"][0, 0] == 1.0
    assert np.isnan(rec["inputs"]["close"][0, 1])


def test_empty_declared_universe_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one symbol"):
        probe(wide([[1.0, 2.0]]), symbols=())


def test_scalar_string_declared_universe_is_rejected() -> None:
    # symbols="ABC" would otherwise iterate per character (['A', 'B', 'C']),
    # silently mangling a single multi-character symbol.
    with pytest.raises(ValueError, match="sequence of symbol labels"):
        probe(wide([[1.0, 2.0]]), symbols="AB")  # type: ignore[arg-type]


def test_non_str_declared_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be str"):
        probe(wide([[1.0, 2.0]]), symbols=("A", 1))  # type: ignore[arg-type]


def test_duplicate_declared_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        probe(wide([[1.0, 2.0]]), symbols=("A", "A"))


def test_declared_order_is_normalized_to_sorted() -> None:
    rec = probe(wide([[1.0, 2.0]]), symbols=("B", "A"))
    assert list(rec["symbols"]) == ["A", "B"]
    assert np.array_equal(rec["inputs"]["close"], [[1.0, 2.0]])


def test_input_symbol_outside_declared_universe_is_value_error() -> None:
    with pytest.raises(ValueError, match=r"\['B'\] are not in the declared"):
        probe(wide([[1.0, 2.0]], symbols=("A", "B")), symbols=("A",))


# --- side_data (normalize_side_frame via predict) ----------------------------------


def test_side_data_arrives_as_per_name_arrays() -> None:
    side = pd.DataFrame({"timestamp": days(2), "vol": [0.1, 0.2]})
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)
    assert np.allclose(rec["side_inputs"]["vol"], [0.1, 0.2])
    assert rec["side_inputs"]["vol"].shape == (2,)


def test_side_data_timestamps_must_be_subset() -> None:
    side = pd.DataFrame({"timestamp": days(3), "vol": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="subset"):
        probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)


def test_side_data_missing_timestamps_fill_float_and_object() -> None:
    side = pd.DataFrame(
        {
            "timestamp": days(1),
            "f": [1.5],
            "o": pd.Series([{"k": 1}], dtype=object),
        }
    )
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)
    assert np.isnan(rec["side_inputs"]["f"][1])
    assert rec["side_inputs"]["o"][1] is None
    assert rec["side_inputs"]["o"][0] == {"k": 1}


@pytest.mark.parametrize(
    "column, expected",
    [
        (pd.Series([pd.Timestamp("2020-02-01")]), np.datetime64("2020-02-01")),
        (pd.Series([pd.Timedelta(days=1)]), np.timedelta64(1, "D")),
    ],
    ids=["datetime64", "timedelta64"],
)
def test_side_data_missing_timestamps_fill_temporal_nat(
    column: pd.Series, expected: np.datetime64 | np.timedelta64
) -> None:
    side = pd.DataFrame({"timestamp": days(1), "x": column})
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)
    values = rec["side_inputs"]["x"]
    assert values.dtype == column.dtype
    assert values[0] == expected
    assert np.isnat(values[1])


@pytest.mark.parametrize(
    "column",
    [pd.Series([7], dtype=np.int64), pd.Series([True])],
    ids=["int64", "bool"],
)
def test_side_data_missing_timestamps_reject_non_nan_dtypes(column: pd.Series) -> None:
    # Mirrors the input-side rule: a dtype that cannot represent a missing
    # value must cover every timestamp (no silent zero/False fill).
    side = pd.DataFrame({"timestamp": days(1), "x": column})
    with pytest.raises(ValueError, match="cannot be NaN-filled"):
        probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)


def test_side_data_full_coverage_int_and_bool_keep_dtype() -> None:
    side = pd.DataFrame(
        {
            "timestamp": days(2),
            "i": pd.Series([7, 8], dtype=np.int64),
            "b": [True, False],
        }
    )
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)
    assert rec["side_inputs"]["i"].dtype == np.int64
    assert list(rec["side_inputs"]["i"]) == [7, 8]
    assert rec["side_inputs"]["b"].dtype == np.bool_
    assert list(rec["side_inputs"]["b"]) == [True, False]


def test_side_data_without_timestamp_column_is_value_error() -> None:
    side = pd.DataFrame({"vol": [0.1]})
    with pytest.raises(ValueError, match="'timestamp' column"):
        probe(wide([[1.0, 2.0]]), side)


def test_side_data_object_missing_values_become_none() -> None:
    side = pd.DataFrame(
        {
            "timestamp": days(3),
            "o": pd.Series([{"a": 1}, None, np.nan], dtype=object),
        }
    )
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), side)
    assert rec["side_inputs"]["o"][0] == {"a": 1}
    assert rec["side_inputs"]["o"][1] is None
    assert rec["side_inputs"]["o"][2] is None


def test_side_data_duplicate_timestamps_keep_last() -> None:
    side = pd.DataFrame({"timestamp": [days(1)[0], days(1)[0]], "vol": [0.1, 0.9]})
    rec = probe(wide([[1.0, 2.0]]), side)
    assert rec["side_inputs"]["vol"][0] == 0.9


def test_side_data_nullable_float_becomes_float64() -> None:
    side = pd.DataFrame(
        {"timestamp": days(2), "vol": pd.array([1.5, pd.NA], dtype="Float64")}
    )
    rec = probe(wide([[1.0, 2.0], [3.0, 4.0]]), side)
    values = rec["side_inputs"]["vol"]
    assert values.dtype == np.float64
    assert values[0] == 1.5
    assert np.isnan(values[1])


def test_side_data_non_datetime_timestamp_column_is_rejected() -> None:
    side = pd.DataFrame({"timestamp": ["2020-01-01"], "vol": [0.1]})
    with pytest.raises(ValueError, match="must be datetime64"):
        probe(wide([[1.0, 2.0]]), side)


def test_side_data_non_str_column_name_is_rejected() -> None:
    side = pd.DataFrame([[days(1)[0], 0.1]], columns=["timestamp", 1])
    with pytest.raises(ValueError, match="must be str"):
        probe(wide([[1.0, 2.0]]), side)


def test_omitted_side_data_is_empty_mapping() -> None:
    assert probe(wide([[1.0, 2.0]]))["side_inputs"] == {}


# --- as_frame (canonical long) -------------------------------------------------


def test_as_frame_emits_every_cell_including_nan() -> None:
    frame = qm.as_frame({"close": wide([[1.0, np.nan], [3.0, 4.0]])})
    assert list(frame.columns) == ["timestamp", "symbol", "close"]
    assert len(frame) == 4
    assert np.isnan(frame["close"][1])
    assert list(frame["symbol"]) == ["A", "B", "A", "B"]


def test_as_frame_outer_merges_multiple_inputs() -> None:
    data = {
        "f1": wide([[1.0, 2.0]]),
        "f2": wide([[3.0, 4.0]], start="2020-01-02"),
    }
    frame = qm.as_frame(data)
    assert list(frame.columns) == ["timestamp", "symbol", "f1", "f2"]
    assert len(frame) == 4
    day1 = frame[frame["timestamp"] == pd.Timestamp("2020-01-01")]
    assert np.array_equal(day1["f1"].to_numpy(), [1.0, 2.0])
    assert np.isnan(day1["f2"]).all()


def test_as_frame_rejects_trailing_axes_and_points_to_wide_format() -> None:
    columns = pd.MultiIndex.from_tuples(
        [("A", "y"), ("A", "z"), ("B", "y"), ("B", "z")]
    )
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=days(1), columns=columns)
    with pytest.raises(ValueError, match=r"trailing axes.*to_wide\(\)"):
        qm.as_frame({"x": frame})


def test_as_wide_validates_symbols_and_duplicates() -> None:
    # as_wide itself applies the string-symbol / duplicate validation its
    # contract documents, on every accepted shape (not only at a model call).
    with pytest.raises(ValueError, match="must be str"):
        qm.as_wide(pd.DataFrame([[1.0, 2.0]], index=days(1), columns=[1, 2]))
    with pytest.raises(ValueError, match="duplicates"):
        qm.as_wide(pd.DataFrame([[1.0, 2.0]], index=days(1), columns=["A", "A"]))
    with pytest.raises(ValueError, match="must be str"):
        qm.as_wide(pd.DataFrame({"timestamp": days(1), "symbol": [1], "close": [1.0]}))


def test_as_wide_and_as_frame_are_idempotent() -> None:
    data = {"f1": wide([[1.0, np.nan], [3.0, 4.0]])}
    once = qm.as_wide(data)
    twice = qm.as_wide(once)
    assert list(once) == list(twice)
    pd.testing.assert_frame_equal(once["f1"], twice["f1"])
    long_once = qm.as_frame(data)
    long_twice = qm.as_frame(long_once)
    pd.testing.assert_frame_equal(long_once, long_twice)


def test_as_frame_is_idempotent_with_disjoint_timestamps() -> None:
    # Inputs covering different timestamps land on the shared dense grid on
    # the first conversion, so re-converting the long frame adds no rows.
    a = pd.DataFrame({"A": [1.0, 2.0]}, index=days(2))
    b = pd.DataFrame({"B": [3.0]}, index=days(1, start="2020-01-03"))
    f1 = qm.as_frame({"a": a, "b": b})
    assert len(f1) == 6  # 3 timestamps x 2 symbols
    assert f1["timestamp"].is_monotonic_increasing
    pd.testing.assert_frame_equal(qm.as_frame(f1), f1)


def test_as_frame_is_idempotent_with_disjoint_symbols() -> None:
    a = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=days(1))
    b = pd.DataFrame({"C": [3.0]}, index=days(1))
    f1 = qm.as_frame({"a": a, "b": b})
    assert len(f1) == 3  # 1 timestamp x 3 symbols
    assert list(f1["symbol"]) == ["A", "B", "C"]
    pd.testing.assert_frame_equal(qm.as_frame(f1), f1)


def test_as_frame_keeps_covered_non_float_dtypes() -> None:
    # A non-float input that covers the whole grid keeps its dtype, and the
    # long frame converts back without change.
    data = {
        "n": pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=days(2)),
        "flag": pd.DataFrame({"A": [True, False], "B": [False, True]}, index=days(2)),
    }
    f1 = qm.as_frame(data)
    assert f1["n"].dtype == np.int64
    assert f1["flag"].dtype == np.bool_
    pd.testing.assert_frame_equal(qm.as_frame(f1), f1)


def test_mixed_column_dtypes_are_a_value_error() -> None:
    # A datetime column next to floats in one wide input has no common
    # dtype; the boundary reports the usual ValueError instead of leaking
    # NumPy's promotion error.
    frame = pd.DataFrame(
        {"A": [1.0], "B": pd.to_datetime(["2020-01-01"])}, index=days(1)
    )
    with pytest.raises(ValueError, match="no common dtype"):
        probe({"x": frame})


def test_as_frame_rejects_non_float_input_with_missing_grid_cells() -> None:
    # Aligning onto the dense grid would need NaN cells the int dtype cannot
    # represent — the same coverage rule as fit/predict.
    a = pd.DataFrame({"A": [1, 2]}, index=days(2))
    b = pd.DataFrame({"B": [3.0]}, index=days(1, start="2020-01-03"))
    with pytest.raises(ValueError, match="non-float feature 'a'"):
        qm.as_frame({"a": a, "b": b})
