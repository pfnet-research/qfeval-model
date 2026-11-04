"""Documentation contract tests: the README's and tutorial's examples, executed.

Transcribing the examples keeps the documentation and the implementation
from drifting apart; a failure here means one of them must change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import wide

import qfeval_model as qm


def test_readme_momentum_example() -> None:
    # README's opening example, transcribed verbatim (data supplied here).
    @qm.model
    def momentum(ctx: qm.Context, window: int = 20) -> None:
        ctx.child = qm.zoo.rolling_mean(
            "close", window=window
        )  # compose a building block
        for step in ctx:  # one step per timestamp
            step.output = step.inputs["close"] / step.child.array - 1.0

    prices = wide(
        [[float(100 + t), float(200 - t)] for t in range(45)],
        start="2023-12-01",
    )

    pred = momentum(window=10).run(prices, test_from="2024-01-01", seed=0)
    w = pred.to_wide()  # wide DataFrame: index=timestamp, columns=symbol
    frame = pred.to_frame()  # long DataFrame: timestamp, symbol, close

    daily_ic = qm.metrics.information_coefficient(pred, prices)

    # Verify the same construction that was run above (window=10).
    report = qm.verify.check_all(
        lambda: momentum(window=10), prices, test_from="2024-01-01"
    )
    assert report.all_passed, str(report)  # every check ran and passed

    # The prediction covers exactly the test period.
    assert w.index[0] == pd.Timestamp("2024-01-01")
    assert len(w) == 14
    # to_frame's value column is DEFAULT_FEATURE ("close"), as documented.
    assert list(frame.columns) == ["timestamp", "symbol", "close"]
    assert len(daily_ic) == len(w)
    # Hand-checked golden: on 2024-01-01 (t=31 of the linear series 100+t),
    # sma(10) = 100 + mean(22..31) = 126.5, so momentum = 131/126.5 - 1.
    # The sma warmup crossed the fit -> predict boundary inside run().
    assert w.iloc[0, 0] == pytest.approx(4.5 / 126.5)


def test_tutorial_simple_return_example() -> None:
    # docs/tutorials/01-first-model.md, transcribed verbatim.
    @qm.model
    def simple_return(ctx: qm.Context) -> None:
        ctx.child = qm.zoo.lag("close")  # yesterday's close
        for step in ctx:
            step.output = step.inputs["close"] / step.child.array - 1.0

    train_inputs = wide([[100.0, 200.0], [110.0, 100.0]])
    test_inputs = wide([[121.0, 50.0], [242.0, 25.0]], start="2020-01-03")

    model = simple_return()
    model.fit(train_inputs, seed=0)
    pred = model.predict(test_inputs)
    frame = pred.to_frame()  # long-format DataFrame: timestamp, symbol, close
    w = pred.to_wide()  # wide DataFrame: index=timestamp, columns=symbol

    assert list(frame.columns) == ["timestamp", "symbol", "close"]
    assert frame.shape == (4, 3)
    assert w.index.name == "timestamp"
    assert w.columns.name == "symbol"
    assert list(w.columns) == ["A", "B"]
    # lag's state crosses the fit -> predict boundary: the first test row is
    # priced against the last train close.
    assert w.to_numpy() == pytest.approx(
        np.array([[121.0 / 110.0 - 1.0, -0.5], [1.0, -0.5]])
    )
