"""Input-path benchmark: the single wide -> ``Features`` normalization path.

Run outside CI as a regression reference (see benchmarks/README.md):

    uv run python benchmarks/bench_inputs.py

Measures wall time and tracemalloc peak for ``normalize_wide_inputs`` on
dense / sparse wide inputs (single and multiple features, plus a trailing
axis), for ``as_frame`` long rendering, and for a ``Model.run`` backtest pass.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import qfeval_model as qm
from qfeval_model.core.normalize import as_frame, normalize_wide_inputs

T, S = 2_000, 500  # 1M cells per feature


def _wide_frame(rng: np.random.Generator, sparse: bool) -> pd.DataFrame:
    values = rng.standard_normal((T, S))
    if sparse:
        values[rng.random((T, S)) < 0.5] = np.nan
    index = pd.date_range("2000-01-03", periods=T, freq="B")
    return pd.DataFrame(values, index=index, columns=[f"S{i:04d}" for i in range(S)])


def _trailing_frame(rng: np.random.Generator, depth: int = 2) -> pd.DataFrame:
    """A wide frame with one trailing axis of ``depth`` labels (level 0=symbol)."""
    columns = pd.MultiIndex.from_product(
        [[f"S{i:04d}" for i in range(S)], range(depth)]
    )
    values = rng.standard_normal((T, S * depth))
    index = pd.date_range("2000-01-03", periods=T, freq="B")
    return pd.DataFrame(values, index=index, columns=columns)


def _measure(label: str, fn: Callable[[], Any]) -> None:
    tracemalloc.start()
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:<48} {elapsed * 1e3:9.1f} ms {peak / 2**20:9.1f} MiB peak")


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"input: T={T}, S={S} ({T * S / 1e6:.1f}M cells per feature)\n")

    for sparse in (False, True):
        density = "sparse" if sparse else "dense"
        for n_inputs in (1, 5):
            inputs = {f"f{i}": _wide_frame(rng, sparse) for i in range(n_inputs)}
            _measure(
                f"normalize_wide_inputs ({density}, F={n_inputs})",
                lambda inputs=inputs: normalize_wide_inputs(inputs),
            )

    trailing = {"f0": _trailing_frame(rng)}
    _measure(
        "normalize_wide_inputs (dense, F=1, trailing depth=2)",
        lambda: normalize_wide_inputs(trailing),
    )

    dense = {"f0": _wide_frame(rng, sparse=False)}
    _measure("as_frame (long render, dense, F=1)", lambda: as_frame(dense))

    @qm.model
    def zero(ctx: qm.Context) -> None:
        for step in ctx:
            step.output = step.inputs["close"] * 0.0

    frame = _wide_frame(rng, sparse=False)
    test_from = frame.index[T // 2]
    _measure(
        "Model.run (dense, F=1, fit half + predict half)",
        lambda: zero().run(frame, test_from=test_from),
    )


if __name__ == "__main__":
    main()
