# Examples

The example scripts are provided in two languages with identical file names
and code:

- [`en/`](en/README.md) — English (the source of truth)
- [`ja/`](ja/README.md) — 日本語版

`make examples` at the repo root converts every script in both trees to a
Jupyter notebook next to it and executes it.

CI also compares the executable AST of each English/Japanese pair while
ignoring translated docstrings, comments, display text, and exception text.
Code, names, literals, and call structure must otherwise stay identical.
