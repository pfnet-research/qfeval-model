"""Single-step vs. batch execution on a representative panel transform.

Run as the evidence for retaining the advanced ``@batch_model`` API (see
benchmarks/README.md):

    uv run python benchmarks/bench_modes.py

Both implementations compute the same backward return over a 1M-cell panel.
The benchmark includes the shared public input-normalization and output paths;
only model execution mode differs.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

import qfeval_model as qm


@qm.model
def _single_return(ctx: qm.Context) -> None:
    for step in ctx:
        close = step.inputs["close"]
        previous = ctx.state.get("previous")
        step.output = close * np.nan if previous is None else close / previous - 1.0
        ctx.state["previous"] = close.copy()


@qm.batch_model
def _batch_return(ctx: qm.Context) -> None:
    for step in ctx:
        close = step.inputs["close"]
        previous = ctx.state.get("previous", np.full(close.shape[1], np.nan))
        shifted = np.vstack([previous[None], close[:-1]])
        step.output = close / shifted - 1.0
        ctx.state["previous"] = close[-1].copy()


def _median_time(fn: Callable[[], object], repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    n_timestamps = 5_000
    n_symbols = 200
    values = np.random.default_rng(0).lognormal(
        mean=4.0, sigma=0.2, size=(n_timestamps, n_symbols)
    )
    inputs = pd.DataFrame(
        values,
        index=pd.date_range("2020-01-01", periods=n_timestamps, freq="1min"),
        columns=[f"S{i:03d}" for i in range(n_symbols)],
    )

    single_output = _single_return().predict(inputs).array
    batch_output = _batch_return().predict(inputs).array
    np.testing.assert_allclose(single_output, batch_output, equal_nan=True)

    single = _median_time(lambda: _single_return().predict(inputs))
    batch = _median_time(lambda: _batch_return().predict(inputs))
    print(f"single  {single:.4f}s")
    print(f"batch   {batch:.4f}s")
    print(f"speedup {single / batch:.2f}x")


if __name__ == "__main__":
    main()
