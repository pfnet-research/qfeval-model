# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub private vulnerability reporting](https://github.com/pfnet-research/qfeval-model/security/advisories/new)
rather than a public issue. We will confirm receipt and keep you informed of
the fix's progress.

## Checkpoints are pickles

Model checkpoints are Python pickles: restoring one executes code embedded in
the bytes, so loading an untrusted checkpoint is arbitrary code execution.
This is a documented design decision (see DESIGN §Checkpointing and the
`Model.loads` docstring), not a vulnerability — only restore checkpoints you
produced and stored in trusted locations.
