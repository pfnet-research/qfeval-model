"""Property-based tests (hypothesis) over random panels.

Example-based tests pin boundaries; these properties cover the surface:
normalization idempotence, wide ≡ long, duplicate-aggregation rules,
call-splitting and checkpoint-splitting invariance, single ≡ batch, and
rolling-window models against pandas as an independent reference. Each
property compares the library against an independent implementation or a
different code path of the library itself — never the same code twice.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from conftest import days, long_of, panels
from hypothesis import given, settings
from hypothesis import strategies as st

import qfeval_model as qm


@qm.model
def _echo_f0(ctx: qm.Context) -> None:
    for step in ctx:
        step.output = step.inputs["f0"]


@qm.model
def _echo_close(ctx: qm.Context) -> None:
    for step in ctx:
        step.output = step.inputs["close"]


@qm.model
def _rng_walk(ctx: qm.Context) -> None:
    """State + rng model: the subject of the call-splitting invariances."""
    state = ctx.state
    state.setdefault("prev", None)
    for step in ctx:
        x = step.inputs["f0"]
        prev = state["prev"]
        noise = step.rng.random(x.shape)
        step.output = (x if prev is None else x - prev) + noise
        state["prev"] = x.copy()


def _rows(p: dict[str, pd.DataFrame], a: int, b: int) -> dict[str, pd.DataFrame]:
    return {name: frame.iloc[a:b] for name, frame in p.items()}


# --- 1. normalization is idempotent ----------------------------------------


@settings(deadline=None)
@given(p=panels())
def test_as_wide_and_as_frame_idempotent(p: dict[str, pd.DataFrame]) -> None:
    once = qm.as_wide(p)
    twice = qm.as_wide(once)
    assert once.keys() == twice.keys()
    for name, frame in once.items():
        pd.testing.assert_frame_equal(frame, twice[name])
    frame_once = qm.as_frame(p)
    pd.testing.assert_frame_equal(frame_once, qm.as_frame(frame_once))


# --- 2. wide == long ---------------------------------------------------------


@settings(deadline=None)
@given(p=panels())
def test_wide_equals_long(p: dict[str, pd.DataFrame]) -> None:
    long = long_of(p)
    wide_norm = qm.as_wide(p)
    long_norm = qm.as_wide(long)
    assert wide_norm.keys() == long_norm.keys()
    for name, frame in wide_norm.items():
        pd.testing.assert_frame_equal(frame, long_norm[name])
    a = _echo_f0().predict(p, seed=0)
    b = _echo_f0().predict(long, seed=0)
    assert a.timestamps.tobytes() == b.timestamps.tobytes()
    assert list(a.symbols) == list(b.symbols)
    assert a.array.tobytes() == b.array.tobytes()


# --- 3. duplicate aggregation (backward non-NaN wins) ------------------------


@settings(deadline=None)
@given(p=panels(), data=st.data())
def test_duplicate_aggregation(p: dict[str, pd.DataFrame], data: st.DataObject) -> None:
    base = long_of(p)
    n = len(base)
    extra = data.draw(st.lists(st.integers(0, n - 1), max_size=5))
    frame = pd.concat([base, base.iloc[extra]], ignore_index=True)
    features = [c for c in frame.columns if c not in ("timestamp", "symbol")]
    for name in features:
        mask = data.draw(
            st.lists(st.booleans(), min_size=len(frame), max_size=len(frame))
        )
        frame.loc[np.asarray(mask), name] = np.nan

    result = qm.as_wide(frame)
    for name in features:
        # Independent pure-Python scan: per cell, the last non-NaN row wins.
        expected: dict[tuple[Any, Any], float] = {}
        for _, row in frame.iterrows():
            value = row[name]
            if not np.isnan(value):
                expected[(row["timestamp"], row["symbol"])] = float(value)
        got = result[name]
        arr = got.to_numpy()
        for i, ts in enumerate(got.index):
            for j, symbol in enumerate(got.columns):
                want = expected.get((ts, symbol), float("nan"))
                cell = float(arr[i, j])
                assert (np.isnan(want) and np.isnan(cell)) or cell == want


# --- 4. to_wide / to_frame round-trip ----------------------------------------


@settings(deadline=None)
@given(p=panels())
def test_output_roundtrips_as_input(p: dict[str, pd.DataFrame]) -> None:
    pred = _echo_f0().predict(p, seed=0)
    via_wide = qm.as_wide(pred.to_wide())["close"]
    assert via_wide.to_numpy().tobytes() == pred.array.tobytes()
    via_long = qm.as_wide(pred.to_frame())["close"]
    assert via_long.to_numpy().tobytes() == pred.array.tobytes()


# --- 5. split partitions the rows ---------------------------------------------


@settings(deadline=None)
@given(p=panels(), data=st.data())
def test_split_partition(p: dict[str, pd.DataFrame], data: st.DataObject) -> None:
    whole = qm.as_wide(p)
    test_from = data.draw(st.sampled_from(list(whole["f0"].index)))
    train, test = qm.split(p, test_from=test_from)
    for name, frame in whole.items():
        assert (train[name].index < test_from).all()
        assert (test[name].index >= test_from).all()
        assert test_from in test[name].index  # the boundary row is test-side
        pd.testing.assert_frame_equal(pd.concat([train[name], test[name]]), frame)


# --- 6. call-splitting invariance ---------------------------------------------


@settings(deadline=None)
@given(p=panels(), data=st.data())
def test_call_split_invariance(p: dict[str, pd.DataFrame], data: st.DataObject) -> None:
    whole = _rng_walk().predict(p, seed=7).array
    t = whole.shape[0]
    cuts = sorted(data.draw(st.sets(st.integers(1, max(t - 1, 1)), max_size=3)))
    bounds = [0, *(c for c in cuts if c < t), t]
    canonical = qm.as_wide(p)
    model = _rng_walk()
    parts = [
        model.predict(_rows(canonical, a, b), seed=7 if a == 0 else None).array
        for a, b in itertools.pairwise(bounds)
    ]
    assert np.concatenate(parts).tobytes() == whole.tobytes()


# --- 7. checkpoint-splitting invariance ----------------------------------------


@settings(deadline=None)
@given(p=panels(), data=st.data())
def test_checkpoint_split_invariance(
    p: dict[str, pd.DataFrame], data: st.DataObject
) -> None:
    whole = _rng_walk().predict(p, seed=3).array
    t = whole.shape[0]
    cut = data.draw(st.integers(0, t))
    canonical = qm.as_wide(p)
    model = _rng_walk()
    parts = []
    if cut > 0:
        parts.append(model.predict(_rows(canonical, 0, cut), seed=3).array)
    restored = _rng_walk().loads(model.dumps())
    if cut < t:
        parts.append(
            restored.predict(
                _rows(canonical, cut, t), seed=3 if cut == 0 else None
            ).array
        )
    assert np.concatenate(parts).tobytes() == whole.tobytes()


# --- 8. single == batch ----------------------------------------------------------


@qm.model
def _diff_single(ctx: qm.Context) -> None:
    state = ctx.state
    state.setdefault("prev", None)
    for step in ctx:
        x = step.inputs["f0"]
        prev = state["prev"]
        step.output = np.full_like(x, np.nan) if prev is None else x - prev
        state["prev"] = x.copy()


@qm.batch_model
def _diff_batch(ctx: qm.Context) -> None:
    state = ctx.state
    state.setdefault("prev", None)
    for step in ctx:
        x = step.inputs["f0"]  # (T, S)
        prev = state["prev"]
        first = np.full(x.shape[1:], np.nan) if prev is None else prev
        step.output = x - np.concatenate([first[None], x[:-1]], axis=0)
        state["prev"] = x[-1].copy()


@settings(deadline=None)
@given(p=panels(), data=st.data())
def test_single_equals_batch(p: dict[str, pd.DataFrame], data: st.DataObject) -> None:
    # Two calls each, so the batch model's cross-call state is exercised too.
    canonical = qm.as_wide(p)
    t = len(canonical["f0"].index)
    cut = data.draw(st.integers(1, t))
    single, batch = _diff_single(), _diff_batch()
    parts_s = [single.predict(_rows(canonical, 0, cut), seed=0).array]
    parts_b = [batch.predict(_rows(canonical, 0, cut), seed=0).array]
    if cut < t:
        parts_s.append(single.predict(_rows(canonical, cut, t)).array)
        parts_b.append(batch.predict(_rows(canonical, cut, t)).array)
    assert np.array_equal(
        np.concatenate(parts_s), np.concatenate(parts_b), equal_nan=True
    )


# --- 9. cross-sectional rank properties -----------------------------------------------


@settings(deadline=None)
@given(p=panels())
def test_cross_sectional_rank_properties(p: dict[str, pd.DataFrame]) -> None:
    source = qm.as_wide(p)["f0"]
    x = source.to_numpy()
    r = qm.zoo.cross_sectional_rank("f0").predict(p, seed=0).array
    valid = ~np.isnan(r)
    assert np.array_equal(np.isnan(r), np.isnan(x))  # NaN positions preserved
    assert ((r[valid] >= 0.0) & (r[valid] <= 1.0)).all()
    # Monotone-transform invariance: doubling is exact and order-preserving.
    r2 = qm.zoo.cross_sectional_rank("f0").predict({"f0": source * 2.0}, seed=0).array
    assert np.array_equal(r, r2, equal_nan=True)
    # Tie-free rows (unique valid values) average to exactly 0.5.
    for i in range(x.shape[0]):
        row = x[i][~np.isnan(x[i])]
        if len(row) > 0 and len(np.unique(row)) == len(row):
            assert float(np.nanmean(r[i])) == pytest.approx(0.5)


# --- 10. rolling family == pandas -----------------------------------------------------


@settings(deadline=None)
@given(p=panels(), window=st.integers(2, 4), alpha=st.floats(0.01, 1.0))
def test_rolling_matches_pandas(
    p: dict[str, pd.DataFrame], window: int, alpha: float
) -> None:
    frame = qm.as_wide(p)["f0"]
    rolling = frame.rolling(window, min_periods=window)
    mean = rolling.mean().to_numpy()
    std = rolling.std(ddof=1).to_numpy()

    rmean = qm.zoo.rolling_mean("f0", window).predict(p, seed=0).array
    assert np.allclose(rmean, mean, equal_nan=True)

    # pandas computes the rolling variance incrementally, so a window whose
    # std is far below the magnitudes seen before it keeps their rounding
    # error (and a near-constant window can round to exactly-zero std under
    # one algorithm but not the other). Comparing the two algorithms is only
    # meaningful on well-conditioned cells; blank the rest on both sides.
    scale = frame.abs().expanding().max().to_numpy()
    ill = std <= 1e-4 * scale

    def blank_ill(values: Any) -> Any:
        return np.where(ill, np.nan, values)

    rstd = qm.zoo.rolling_std("f0", window).predict(p, seed=0).array
    assert np.allclose(blank_ill(rstd), blank_ill(std), equal_nan=True)

    zscore = qm.zoo.rolling_zscore("f0", window).predict(p, seed=0).array
    with np.errstate(divide="ignore", invalid="ignore"):
        z_ref = np.where(std > 0, (frame.to_numpy() - mean) / std, np.nan)
    assert np.allclose(blank_ill(zscore), blank_ill(z_ref), equal_nan=True)

    ewm = qm.zoo.recursive_ewm_mean("f0", alpha).predict(p, seed=0).array
    ewm_ref = frame.ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()
    assert np.allclose(ewm, ewm_ref, equal_nan=True)

    rvar = qm.zoo.rolling_var("f0", window).predict(p, seed=0).array
    assert np.allclose(
        blank_ill(rvar), blank_ill(rolling.var(ddof=1).to_numpy()), equal_nan=True
    )

    for name, reference in (
        ("rolling_sum", rolling.sum()),
        ("rolling_min", rolling.min()),
        ("rolling_max", rolling.max()),
        ("rolling_median", rolling.median()),
        ("rolling_range", rolling.max() - rolling.min()),
    ):
        got = getattr(qm.zoo, name)("f0", window).predict(p, seed=0).array
        assert np.allclose(got, reference.to_numpy(), equal_nan=True), name


@settings(deadline=None)
@given(p=panels(), periods=st.integers(1, 3))
def test_diff_matches_pandas(p: dict[str, pd.DataFrame], periods: int) -> None:
    frame = qm.as_wide(p)["f0"]
    got = qm.zoo.diff("f0", periods=periods).predict(p, seed=0).array
    assert np.array_equal(got, frame.diff(periods).to_numpy(), equal_nan=True)


@settings(deadline=None)
@given(p=panels())
def test_expanding_matches_pandas(p: dict[str, pd.DataFrame]) -> None:
    frame = qm.as_wide(p)["f0"]
    expanding = frame.expanding()
    for name, reference in (
        ("expanding_sum", expanding.sum()),
        ("expanding_mean", expanding.mean()),
        ("expanding_min", expanding.min()),
        ("expanding_max", expanding.max()),
    ):
        got = getattr(qm.zoo, name)("f0").predict(p, seed=0).array
        assert np.allclose(got, reference.to_numpy(), equal_nan=True), name

    # Welford vs pandas' incremental variance: compare well-conditioned cells
    # only, mirroring the rolling test above.
    std_ref = expanding.std(ddof=1).to_numpy()
    scale = frame.abs().expanding().max().to_numpy()
    ill = std_ref <= 1e-4 * scale

    def blank_ill(values: Any) -> Any:
        return np.where(ill, np.nan, values)

    evar = qm.zoo.expanding_var("f0").predict(p, seed=0).array
    estd = qm.zoo.expanding_std("f0").predict(p, seed=0).array
    assert np.allclose(
        blank_ill(evar), blank_ill(expanding.var(ddof=1).to_numpy()), equal_nan=True
    )
    assert np.allclose(blank_ill(estd), blank_ill(std_ref), equal_nan=True)


@settings(deadline=None)
@given(p=panels(), window=st.integers(2, 4))
def test_rolling_cov_matches_pandas(p: dict[str, pd.DataFrame], window: int) -> None:
    frame = qm.as_wide(p)["f0"]
    lagged = frame.shift(1)
    got = qm.zoo.rolling_cov("f0", qm.zoo.lag("f0"), window).predict(p, seed=0).array
    reference = frame.rolling(window, min_periods=window).cov(lagged).to_numpy()
    # Blank poorly conditioned cells: pandas' incremental covariance keeps
    # rounding error from earlier large magnitudes.
    scale = frame.abs().expanding().max().to_numpy()
    ill = np.abs(reference) <= 1e-4 * scale**2
    got = np.where(ill, np.nan, got)
    reference = np.where(ill, np.nan, reference)
    assert np.allclose(got, reference, equal_nan=True, rtol=1e-5)


# --- 11. trailing-axis normalization == naive dense construction ----------------------


@st.composite
def _trailing_frame(draw: st.DrawFn) -> pd.DataFrame:
    """Wide frame with 2-level (symbol, k) columns: duplicated/unsorted rows,
    NaN cells, and random holes in the (symbol, k) product."""
    t = draw(st.integers(1, 5))
    n_symbols = draw(st.integers(1, 3))
    n_k = draw(st.integers(1, 3))
    symbols = [f"S{i}" for i in range(n_symbols)]
    ks = sorted(draw(st.sets(st.integers(0, 5), min_size=n_k, max_size=n_k)))
    product = [(symbol, k) for symbol in symbols for k in ks]
    columns = draw(
        st.lists(
            st.sampled_from(product), min_size=1, max_size=len(product), unique=True
        )
    )
    index = draw(st.lists(st.sampled_from(list(days(t))), min_size=1, max_size=t + 2))
    values = st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ) | st.just(float("nan"))
    rows = draw(
        st.lists(
            st.lists(values, min_size=len(columns), max_size=len(columns)),
            min_size=len(index),
            max_size=len(index),
        )
    )
    return pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(index),
        columns=pd.MultiIndex.from_tuples(columns),
    )


def _naive_dense(
    frame: pd.DataFrame,
) -> tuple[list[Any], list[str], npt.NDArray[np.float64]]:
    """Independent pure-Python dense (T, S, K) construction."""
    timestamps = sorted(set(frame.index))
    symbols = sorted({str(c[0]) for c in frame.columns})
    ks = sorted({c[1] for c in frame.columns})
    ti = {v: i for i, v in enumerate(timestamps)}
    si = {v: i for i, v in enumerate(symbols)}
    ki = {v: i for i, v in enumerate(ks)}
    out = np.full((len(timestamps), len(symbols), len(ks)), np.nan)
    for pos in range(len(frame)):
        ts = frame.index[pos]
        for raw_col, value in zip(frame.columns, frame.iloc[pos], strict=True):
            symbol, k = cast("tuple[Any, Any]", raw_col)
            if not np.isnan(value):  # later non-NaN rows win; NaN never overwrites
                out[ti[ts], si[str(symbol)], ki[k]] = float(value)
    return timestamps, symbols, out


@settings(deadline=None)
@given(frame=_trailing_frame())
def test_trailing_normalization_matches_naive(frame: pd.DataFrame) -> None:
    timestamps, symbols, expected = _naive_dense(frame)
    pred = _echo_close().predict({"close": frame}, seed=0)
    assert [pd.Timestamp(ts) for ts in pred.timestamps] == timestamps
    assert list(pred.symbols) == symbols
    assert pred.array.shape == expected.shape
    assert np.array_equal(pred.array, expected, equal_nan=True)
