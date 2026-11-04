# Benchmarks (regression reference)

Micro-benchmarks for the input path, execution modes, scaling, checkpointing,
and rolling-kernel implementation selection. They are not part of the test suite,
but CI runs all eight scripts so they cannot silently rot; re-run the relevant one
when changing the corresponding implementation and compare against the reference
numbers below.

```
uv run python benchmarks/bench_inputs.py
uv run python benchmarks/bench_scaling.py
uv run python benchmarks/bench_modes.py
uv run python benchmarks/bench_checkpoint.py
uv run python benchmarks/bench_rolling_valid_sum_candidates.py
uv run python benchmarks/bench_rolling_sum_candidates.py
uv run python benchmarks/bench_window_kernels.py
uv run python benchmarks/bench_zoo_hotspots.py
```

All eight print wall time; `bench_inputs.py`, `bench_checkpoint.py`, and
`bench_zoo_hotspots.py` also print `tracemalloc` peak (Python + numpy
allocations; the resident-set peak is higher but moves proportionally).
`bench_scaling.py` doubles one input dimension and
prints the wall-time ratio (a healthy near-linear path stays close to 2x).
The rolling-sum candidate benchmarks additionally check error against
`math.fsum`, recovery after overflow, and persistent NumPy state size.

## Reference results

Recorded 2026-08-13 on an Apple Silicon laptop (macOS, CPython 3.12.12,
NumPy 2.5.1, pandas 3.0.3); treat relative differences, not absolute numbers,
as the regression signal.

### `bench_inputs.py` — `normalize_wide_inputs` (T=2000, S=500; 1M cells/feature)

| case | time | tracemalloc peak |
|---|---|---|
| dense, F=1 | 41 ms | 15 MiB |
| dense, F=5 | 233 ms | 77 MiB |
| sparse, F=1 | 36 ms | 15 MiB |
| sparse, F=5 | 183 ms | 77 MiB |
| dense, F=1, trailing depth=2 | 81 ms | 31 MiB |
| `as_frame` (long render, dense, F=1) | 26 ms | 46 MiB |
| `Model.run` (dense, F=1, fit half + predict half) | 112 ms | 23 MiB |

`normalize_wide_inputs` is the single wide -> `Features` path used by `fit` /
`predict` / `run`; `as_frame` is the long-rendering counterpart. Changes to the
input boundary must not regress the dense case.

### `bench_scaling.py` — doubling one input dimension (ratio ~2x = near-linear)

| section | base | doubled | ratio |
|---|---|---|---|
| union (F disjoint-timestamp inputs) | 0.027 s | 0.035 s | 1.32x |
| streamed (T timestamps) | 0.055 s | 0.098 s | 1.77x |
| children (C named children) | 0.007 s | 0.012 s | 1.77x |
| lag window (W periods) | 0.003 s | 0.005 s | 1.85x |

### `bench_modes.py` — backward return (T=5000, S=200; 1M cells)

This representative transform exercises the same public normalization and
output paths in both modes. The result arrays are checked for equality before
timing. A material speedup is the reason the advanced `@batch_model` escape
hatch remains in the 0.1 API.

| mode | time |
|---|---|
| single | 0.028 s |
| batch | 0.009 s (3.17x faster) |

### `bench_checkpoint.py` — save / load / clone (256 MiB state)

| operation | time | tracemalloc peak |
|---|---|---|
| `save(path)` (streamed `pickle.dump`, atomic replace) | 99 ms | 256 MiB |
| `dumps()` (bytes) | 53 ms | 640 MiB |
| `load(path)` | 37 ms | 256 MiB |
| `clone()` | 58 ms | 640 MiB |

`save(path)` streams the envelope with `pickle.dump` into a temporary file
and atomically replaces the target, so its peak stays at one state copy;
`dumps()` / `clone()` necessarily hold the byte string as well. Checkpoint
changes must keep `save(path)` streaming (no `dumps()`-based rewrite).

### `bench_rolling_valid_sum_candidates.py` — stable streaming sum selection

All three candidates keep fixed-capacity state and avoid subtracting an expired
value from a long-lived total. On a large-level input with asynchronous NaNs,
their maximum relative errors against `math.fsum` were `5.0e-16` (full ring
resum), `1.3e-16` (segment tree), and `7.5e-16` (chunk prefix/suffix). All
recovered after an overflowing window left the state.

| case | full ring resum | segment tree | chunk prefix/suffix |
|---|---:|---:|---:|
| T=2000, S=64, W=16, dense | 19.6 ms | 56.4 ms | **15.8 ms** |
| T=2000, S=512, W=64, dense | 52.7 ms | 164.9 ms | **23.8 ms** |
| T=2000, S=512, W=64, 20% NaN | **49.1 ms** | 144.1 ms | 54.6 ms |
| T=1000, S=1024, W=256, 20% NaN | 72.5 ms | 142.0 ms | **36.5 ms** |

The chunk implementation is selected: it is amortized O(1), wins the dense and
long-window cases, and keeps its error local to at most two `window`-length
cumulative sums. It keeps two `S x W` buffers in the aligned dense case (0.52
MiB for S=512, W=64); asynchronous NaNs lazily allocate one more same-size
scratch buffer (0.77 MiB total), which is then reused rather than reallocated.

### `bench_rolling_sum_candidates.py` — strict-window sum selection

The three fixed-capacity candidates were compared with the previous two-stack
queue. On a `1e12`-level input with NaNs, maximum relative errors against
`math.fsum` were `4.6e-16` (full ring resum), `1.8e-16` (segment tree), and
`8.7e-16` (chunk prefix/suffix), all below `window * eps`. All three candidates
recovered immediately after an overflowing value left; the previous queue
retained ten extra NaN cells for one timestamp in this test.

| case | previous queue | ring resum | segment tree | chunk prefix/suffix |
|---|---:|---:|---:|---:|
| T=2000, S=64, W=16, dense | 39.4 ms | 12.2 ms | 14.6 ms | **11.3 ms** |
| T=2000, S=512, W=64, dense | 60.6 ms | 53.9 ms | 23.5 ms | **14.6 ms** |
| T=2000, S=512, W=64, 20% NaN | 59.6 ms | 54.5 ms | 26.3 ms | **21.9 ms** |
| T=1000, S=1024, W=256, 20% NaN | 32.9 ms | 92.4 ms | 19.8 ms | **13.7 ms** |

The chunk implementation is selected. At each chunk boundary it builds the
right cumulative sums with contiguous symbol-row additions; amortized work is
O(1) per timestamp. Dense S=512, W=64 state is two fixed sum buffers totaling
0.52 MiB. The first NaN lazily adds fixed boolean and compact-count buffers,
bringing the state to 0.58 MiB; these buffers are then reused.

### `bench_window_kernels.py` — shared rolling-kernel fixed buffers

The prefix/suffix decomposition was extended to locally centered moments,
bivariate moments, logarithmic products, centered time weights, and value-only
extrema. Order-dependent full blocks use a duplicated ring, while argmin/argmax
retain the two-stack path: maintaining chunk index matrices lost on wide short
windows, so that candidate was rejected.

| kernel (T=1536, S=128, W=63) | time | persistent state | speedup vs previous |
|---|---:|---:|---:|
| moments | 42.4 ms | 0.14 MiB | 1.5x |
| bivariate moments | 77.9 ms | 0.33 MiB | 1.4x |
| product | 26.9 ms | 0.09 MiB | 1.5x |
| extrema (values only) | 9.5 ms | 0.14 MiB | 2.8x |
| linear stats | 41.6 ms | 0.15 MiB | 1.7x |
| full block | 1.0 ms | 0.12 MiB | 24x |

The speedup column is the implementation-selection run against the prior
two-stack/deque kernels on the same machine. Across the wider grid S=1..512,
W=5..252, moments, bivariate moments, product, and linear stats remained faster;
the duplicated ring traded exactly 2x persistent block memory for eliminating a
full-window allocation and copy on every warm timestamp.

### `bench_zoo_hotspots.py` — model-level algorithm and allocation canary

This benchmark covers the model-level costs omitted by the kernel benchmark:
axis-wise ranks, rolling maximum drawdown, Expected Shortfall selection,
integrated trend R², count-only windows, cross-sectional ranks, state updates,
OHLCV reads, coalescing, and ensembles. It runs the review grid
(`S=1,64,512`, `W=5,63,252`, dense/20% NaN) and prints wall time, persistent
state, and `tracemalloc` peak for each case.
