# Public API

The public source surface is the top-level facade — `import qfeval_model as qm` —
whose `__all__` re-exports exactly the names below, plus `qm.__version__`
(the installed package version string) and the `zoo`, `metrics`, and
`verify` packages documented on their own pages. It is provisional during
0.x except for the wire contracts called out in the
[compatibility policy](../explanation/compatibility.md). Anything not listed
here is an implementation detail with no stability guarantee (see [Core
internals](core.md)).

::: qfeval_model.core.model
    options:
      members: [model, batch_model, ModelFactory, Model]

::: qfeval_model.core.features
    options:
      members: [Features]

::: qfeval_model.core.context
    options:
      members: [Context, Step]

::: qfeval_model.core.normalize
    options:
      members: [as_wide, as_frame, RawData]

::: qfeval_model.core.split
    options:
      members: [split, TrainTest]

::: qfeval_model.core.constants
    options:
      members: [TIMESTAMP, SYMBOL, DEFAULT_FEATURE, Mode]

::: qfeval_model.core.errors
