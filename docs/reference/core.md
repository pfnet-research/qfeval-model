# Core internals

These names are the internal execution engine behind the
[public API](api.md). They are documented for contributors and the curious;
they carry no stability guarantee, and the supported entry point is the
top-level facade. The `normalize` names below are the package-internal SPI
the satellite packages (`zoo` / `metrics` / `verify`) build on.

::: qfeval_model.core.context
    options:
      members: [SideOutputs]

::: qfeval_model.core.normalize
    options:
      members:
        - normalize_call
        - normalize_wide_inputs
        - normalize_side_frame
        - normalize_timestamp_series
        - to_utc_ns
        - split_side_frame
