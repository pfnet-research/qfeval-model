# Zoo (built-in models)

`qm.zoo` is the landing place for conveniences that would otherwise grow the
core: every entry is an ordinary `@qm.model`. Most entries take a first
argument `source` that accepts a feature name (`str`) or a child model
(`qm.Model`), so zoo models compose freely; the exception is
`mean_ensemble`, which takes a list of models. All zoo models are
deterministic, checkpoint-ready, and pass `verify.check_all`.

::: qfeval_model.zoo.fill

::: qfeval_model.zoo.transforms

::: qfeval_model.zoo.returns

::: qfeval_model.zoo.rolling

::: qfeval_model.zoo.expanding

::: qfeval_model.zoo.cross_section

::: qfeval_model.zoo.pairwise

::: qfeval_model.zoo.ensemble

::: qfeval_model.zoo.distribution

::: qfeval_model.zoo.risk

::: qfeval_model.zoo.regression

::: qfeval_model.zoo.ohlcv

::: qfeval_model.zoo.volatility

::: qfeval_model.zoo.technical_trend

::: qfeval_model.zoo.technical_oscillator
