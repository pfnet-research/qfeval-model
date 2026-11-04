# Metrics (evaluation)

`qm.metrics` owns forward-return computation — getting the `shift` direction
wrong turns an evaluation into a leak, so users should never write it by hand.
Price panels may contain `NaN` coverage gaps, but zero and infinite prices are
rejected at the shared metric boundary because they do not define consistent
forward-return semantics across the metrics.

::: qfeval_model.metrics.returns

::: qfeval_model.metrics.correlation

::: qfeval_model.metrics.pnl
