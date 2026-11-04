"""Scaling benchmark for the paths that used to be quadratic.

Run as a regression reference (see benchmarks/README.md):

    uv run python benchmarks/bench_scaling.py

Each section doubles one input dimension and prints the wall-time ratio; a
healthy (near-linear, up to log factors) path stays close to 2x, a quadratic
path approaches 4x.

- union:    wide normalization with per-input disjoint timestamps
- streamed: streamed verification execution (per-timestamp slicing)
- children: many named children read per step (child lookup)
- lag:      ``zoo.lag`` with a large window (queue update)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

import qfeval_model as qm
from qfeval_model.core.normalize import normalize_wide_inputs
from qfeval_model.verify._compare import execute_streamed


def _time(fn: Callable[[], object]) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def _report(label: str, base: float, doubled: float) -> None:
    print(
        f"{label:32s} base={base:8.4f}s doubled={doubled:8.4f}s "
        f"ratio={doubled / base:5.2f}x"
    )


def _make_inputs(n_timestamps: int) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=n_timestamps, freq="1D")
    values = np.random.default_rng(0).standard_normal((n_timestamps, 4)) + 100.0
    return {"close": pd.DataFrame(values, index=index, columns=list("ABCD"))}


@qm.model
def _backward_return(ctx: qm.Context) -> None:
    for step in ctx:
        close = step.inputs["close"]
        prev = ctx.state.get("prev")
        step.output = close * np.nan if prev is None else close / prev - 1.0
        ctx.state["prev"] = np.array(close)


def bench_union(n_inputs: int) -> float:
    # Each input has its own disjoint timestamp range.
    inputs = {}
    for i in range(n_inputs):
        index = pd.date_range("2020-01-01", periods=50, freq="1D") + pd.Timedelta(
            minutes=i
        )
        inputs[f"f{i}"] = pd.DataFrame(
            np.random.default_rng(i).standard_normal((50, 2)),
            index=index,
            columns=["A", "B"],
        )
    return _time(lambda: normalize_wide_inputs(inputs))


def bench_streamed(n_timestamps: int) -> float:
    inputs = _make_inputs(n_timestamps)
    return _time(lambda: execute_streamed(_backward_return(), inputs))


def bench_children(n_children: int) -> float:
    @qm.model
    def wide_ensemble(ctx: qm.Context) -> None:
        for i in range(n_children):
            ctx.children[f"c{i}"] = qm.zoo.lag("close", 1)
        for step in ctx:
            total = step.inputs["close"] * 0.0
            for i in range(n_children):
                total = total + np.nan_to_num(step.children[f"c{i}"].array)
            step.output = total

    inputs = _make_inputs(40)
    return _time(lambda: wide_ensemble().predict(inputs))


def bench_lag_window(periods: int) -> float:
    inputs = _make_inputs(4 * periods)
    return _time(lambda: qm.zoo.lag("close", periods=periods).predict(inputs))


def main() -> None:
    _report("union (F inputs)", bench_union(100), bench_union(200))
    _report("streamed (T timestamps)", bench_streamed(200), bench_streamed(400))
    _report("children (C children)", bench_children(20), bench_children(40))
    _report("lag window (W periods)", bench_lag_window(200), bench_lag_window(400))


if __name__ == "__main__":
    main()
