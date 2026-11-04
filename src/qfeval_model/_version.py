"""The single source of the package version.

Read by hatch at build time (``[tool.hatch.version]``), re-exported by the
facade as ``qfeval_model.__version__``, and stamped into checkpoint envelopes
by ``core.model``. A leaf module so the core can import it without touching
the facade (which would invert the layer contract).
"""

__version__ = "0.1.0"
