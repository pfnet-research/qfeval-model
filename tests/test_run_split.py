"""``Model.run`` and ``qm.split``.

``split`` cuts a panel at ``test_from`` into a half-open train/test pair
of normalized wide dicts; ``run`` is the fit-train-then-predict-test
one-liner and must match the manually composed equivalent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import days, wide

import qfeval_model as qm

W4 = wide([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


# --- split ---------------------------------------------------------------------


def test_split_is_half_open_at_test_from() -> None:
    train, test = qm.split({"close": W4}, test_from="2020-01-03")
    # The row exactly at test_from belongs to the test side.
    np.testing.assert_array_equal(train["close"].index.values, days(2).values)
    np.testing.assert_array_equal(test["close"].index.values, days(4).values[2:])
    np.testing.assert_array_equal(train["close"].to_numpy(), W4.to_numpy()[:2])
    np.testing.assert_array_equal(test["close"].to_numpy(), W4.to_numpy()[2:])


def test_split_normalizes_flat_wide_to_close_dict() -> None:
    train, test = qm.split(W4, test_from="2020-01-03")
    assert set(train) == {"close"}
    assert set(test) == {"close"}
    np.testing.assert_array_equal(train["close"].to_numpy(), W4.to_numpy()[:2])


def test_split_pivots_long_input_with_nan_for_missing_cells() -> None:
    long = pd.DataFrame(
        {
            "timestamp": [days(2)[0], days(2)[0], days(2)[1]],
            "symbol": ["A", "B", "A"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    train, test = qm.split(long, test_from="2020-01-02")
    np.testing.assert_array_equal(train["close"].to_numpy(), [[1.0, 2.0]])
    # The (2020-01-02, B) cell is absent from the long frame -> NaN.
    np.testing.assert_array_equal(test["close"].to_numpy(), [[3.0, np.nan]])
    assert list(test["close"].columns) == ["A", "B"]


def test_split_empty_side_keeps_frame_and_columns() -> None:
    train, test = qm.split(W4, test_from="2020-02-01")
    assert len(test["close"]) == 0
    assert list(test["close"].columns) == ["A", "B"]
    np.testing.assert_array_equal(train["close"].to_numpy(), W4.to_numpy())
    _, early_test = qm.split(W4, test_from="2019-01-01")
    np.testing.assert_array_equal(early_test["close"].to_numpy(), W4.to_numpy())


def test_split_passes_multiindex_trailing_columns_through() -> None:
    columns = pd.MultiIndex.from_product([["A", "B"], [0, 1]])
    trailing = pd.DataFrame(
        np.arange(12, dtype=np.float64).reshape(3, 4), index=days(3), columns=columns
    )
    train, test = qm.split({"close": trailing}, test_from="2020-01-03")
    assert train["close"].columns.equals(columns)
    assert test["close"].columns.equals(columns)
    np.testing.assert_array_equal(train["close"].to_numpy(), trailing.to_numpy()[:2])
    np.testing.assert_array_equal(test["close"].to_numpy(), trailing.to_numpy()[2:])


@pytest.mark.parametrize(
    "test_from, train_rows",
    [
        ("2020-01-03", 2),
        # Tokyo 09:00 is exactly midnight UTC: the boundary row goes to test.
        (pd.Timestamp("2020-01-03 09:00", tz="Asia/Tokyo"), 2),
        # New York midnight is 05:00 UTC: 2020-01-03 00:00 lands in train.
        (pd.Timestamp("2020-01-03 00:00", tz="America/New_York"), 3),
    ],
)
def test_split_test_from_accepts_string_and_tz_aware(
    test_from: Any, train_rows: int
) -> None:
    train, test = qm.split(W4, test_from=test_from)
    assert len(train["close"]) == train_rows
    assert len(test["close"]) == 4 - train_rows


def test_split_returns_named_tuple() -> None:
    result = qm.split(W4, test_from="2020-01-03")
    train, test = result
    assert isinstance(result, tuple)
    assert result.train is train
    assert result.test is test


@pytest.mark.parametrize("test_from", [20200103, 20200103.0, np.int64(20200103)])
def test_split_rejects_numeric_test_from(test_from: Any) -> None:
    # A bare number would parse as epoch *nanoseconds* (1970-01-01 ...), so
    # the realistic YYYYMMDD input mistake would silently put every row on
    # the test side; the boundary rejects numbers outright.
    with pytest.raises(ValueError, match="epoch nanoseconds"):
        qm.split(W4, test_from=test_from)


# --- run -------------------------------------------------------------------------


def _cumsum_factory() -> Any:
    """A stateful model so run vs manual composition actually exercises state."""

    @qm.model
    def cumsum(ctx: qm.Context) -> None:
        for step in ctx:
            total = ctx.state.get("total", 0.0) + step.inputs["close"]
            ctx.state["total"] = total
            step.output = total

    return cumsum


def test_run_equals_manual_split_fit_predict() -> None:
    cumsum = _cumsum_factory()
    got = cumsum().run(W4, test_from="2020-01-03")

    train, test = qm.split(W4, test_from="2020-01-03")
    expected = cumsum().fit(train).predict(test)

    np.testing.assert_array_equal(got.array, expected.array)
    np.testing.assert_array_equal(got.timestamps, expected.timestamps)


def test_run_returns_test_period_features_only() -> None:
    got = _cumsum_factory()().run(W4, test_from="2020-01-03")
    np.testing.assert_array_equal(got.timestamps, days(4).values[2:])
    # Cumulative totals continue from the fitted train rows.
    np.testing.assert_array_equal(got.array, [[9.0, 12.0], [16.0, 20.0]])


def test_run_without_test_rows_is_value_error() -> None:
    with pytest.raises(ValueError, match="no test rows"):
        _cumsum_factory()().run(W4, test_from="2020-02-01")


def test_run_with_empty_train_skips_fit() -> None:
    phases: list[bool] = []

    @qm.model
    def probe(ctx: qm.Context) -> None:
        for step in ctx:
            phases.append(step.is_train)
            step.output = step.inputs["close"]

    got = probe().run(W4, test_from="2020-01-01")
    assert phases == [False] * 4  # no fit happened
    np.testing.assert_array_equal(got.timestamps, days(4).values)


def test_run_with_test_from_none_is_pure_predict() -> None:
    phases: list[bool] = []

    @qm.model
    def probe(ctx: qm.Context) -> None:
        for step in ctx:
            phases.append(step.is_train)
            step.output = step.inputs["close"]

    got = probe().run(W4)
    assert phases == [False] * 4
    np.testing.assert_array_equal(got.timestamps, days(4).values)


def test_run_splits_side_data_by_timestamp() -> None:
    seen: list[tuple[bool, float]] = []

    @qm.model
    def probe(ctx: qm.Context) -> None:
        for step in ctx:
            seen.append((step.is_train, float(step.side_inputs["w"])))
            step.output = step.inputs["close"]

    side = pd.DataFrame({"timestamp": days(4), "w": [1.0, 2.0, 3.0, 4.0]})
    probe().run(W4, side_data=side, test_from="2020-01-03")
    assert seen == [(True, 1.0), (True, 2.0), (False, 3.0), (False, 4.0)]


def test_run_side_data_without_timestamp_column_is_value_error() -> None:
    side = pd.DataFrame({"w": [1.0]})
    with pytest.raises(ValueError, match="'timestamp' column"):
        _cumsum_factory()().run(W4, side_data=side, test_from="2020-01-03")


def test_run_rejects_string_side_timestamps_at_the_split() -> None:
    # The split applies the same strict rule as side-frame normalization
    # (datetime dtype required), so a bad side frame fails before any fit.
    side = pd.DataFrame({"timestamp": ["2020-01-01"], "w": [1.0]})
    m = _cumsum_factory()()
    with pytest.raises(ValueError, match="datetime64"):
        m.run(W4, side_data=side, test_from="2020-01-03")
    assert not m.started


def test_run_consumes_symbols_and_seed_on_the_fit_call() -> None:
    m = _cumsum_factory()()
    # symbols=/seed= go to the first (fit) call only; the internal predict
    # must not receive them again (that would raise: universe already fixed).
    got = m.run(W4, test_from="2020-01-03", symbols=("C", "A", "B"), seed=3)
    assert m.symbols is not None
    assert list(m.symbols) == ["A", "B", "C"]
    assert m.seed == 3
    assert got.array.shape == (2, 3)
    assert np.isnan(got.array[:, 2]).all()  # declared-only symbol C is NaN


def test_run_advances_self_and_clone_idiom_keeps_original() -> None:
    m = _cumsum_factory()()
    m.clone().run(W4, test_from="2020-01-03")
    assert m.started is False  # the clone ran, not the original

    m.run(W4, test_from="2020-01-03")
    assert m.cursor == days(4).values[-1]


def test_run_then_continue_predicting() -> None:
    m = _cumsum_factory()()
    m.run(W4, test_from="2020-01-03")
    later = wide([[9.0, 10.0]], start="2020-01-05")
    pred = m.predict(later)
    np.testing.assert_array_equal(pred.array, [[25.0, 30.0]])
    assert m.cursor == days(5).values[-1]
