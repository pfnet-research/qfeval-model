"""Checkpoint benchmark: save / load / clone with a large ``ctx.state``.

Run outside CI as a regression reference (see benchmarks/README.md):

    uv run python benchmarks/bench_checkpoint.py

Measures wall time and tracemalloc peak for ``save`` (path, streamed via
``pickle.dump``), ``dumps`` (bytes), ``load``, and ``clone`` with ~256 MiB of
state.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

import qfeval_model as qm

STATE_MIB = 256


@qm.model
def big_state(ctx: qm.Context) -> None:
    if "blob" not in ctx.state:
        ctx.state["blob"] = np.zeros(STATE_MIB * 2**17, dtype=np.float64)
    for step in ctx:
        step.output = step.inputs["close"] * 0.0


def _measure(label: str, fn: Callable[[], Any]) -> None:
    tracemalloc.start()
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:<28} {elapsed * 1e3:9.1f} ms {peak / 2**20:9.1f} MiB peak")


def main() -> None:
    inputs = pd.DataFrame(
        {"A": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"])
    )
    model = big_state()
    model.fit(inputs)
    print(f"state size: {STATE_MIB} MiB\n")
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.qm"
        _measure("save (path, streamed)", lambda: model.save(path))
        _measure("dumps (bytes)", model.dumps)
        _measure("load (path)", lambda: big_state().load(path))
        _measure("clone", model.clone)


if __name__ == "__main__":
    main()
