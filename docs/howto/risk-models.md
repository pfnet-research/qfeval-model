# How to compute risk statistics

The `qm.zoo` risk entries are backward-looking statistics over a return,
P&L, or wealth source. They stop at the per-cell statistic: no portfolio
aggregation, no position sizing, no capital charge — those belong to your
own models and reporting.

## Sign conventions

- Return/P&L sources are **positive for profit, negative for loss**.
- VaR and Expected Shortfall convert internally to the loss sample
  `L = -x` and report **positive numbers for losses**. A profit-only
  window has a negative VaR; nothing is clamped to zero.
- Drawdown sources are **positive wealth or price levels**. Returns are
  never implicitly compounded into wealth — compose explicitly:

```python
wealth = qm.zoo.rolling_compounded_return(my_returns, window=250)
dd = qm.zoo.drawdown("close")             # on a price/wealth level
```

## Historical VaR

```python
var = qm.zoo.rolling_historical_value_at_risk(
    qm.zoo.simple_return(), window=250, confidence=0.95
)
```

- `confidence` is a fraction in `(0, 1)`. `95` raises.
- The quantile is the empirical `inverted_cdf`:
  `VaR_c = inf {l | F_n(l) >= c}`. There is no interpolation option.
- No horizon scaling, sqrt-time rule, or annualization is applied.

## Historical Expected Shortfall (CVaR)

```python
es = qm.zoo.rolling_historical_expected_shortfall(
    qm.zoo.simple_return(), window=250, confidence=0.95
)
```

`ES_c = ∫_c^1 VaR_u du / (1 - c)` on the empirical distribution: the mean
of the largest `window * (1 - confidence)` losses, with the boundary
sample entering at fractional weight. This tail-average definition — not
`mean(L[L >= VaR])`, which over-weights ties at the quantile — guarantees
`ES >= VaR`, with equality on a constant loss. "CVaR" is a synonym; the
canonical name here is expected shortfall.

## Drawdowns and downside risk

| Entry | Use |
|---|---|
| `drawdown(source)` | `1 - x / running_max` over all history |
| `rolling_drawdown(source, window)` | drawdown from the trailing window peak |
| `rolling_max_drawdown(source, window)` | worst peak-to-trough inside the window |
| `drawdown_duration(source)` | steps since the running peak was last matched |
| `rolling_worst_loss(source, window)` | `-min(x)` of a return/P&L window |
| `rolling_downside_deviation(source, window, threshold)` | `sqrt(mean(max(threshold - x, 0)²))` |

`rolling_max_drawdown` validates the positive-level contract: a finite zero
or negative source cell raises `ValueError` (NaN remains the missing value).

A rolling semivariance is not a separate entry — it is exactly
`rolling_lower_partial_moment(source, window, threshold, order=2)`.

## Annualized volatility

```python
vol = qm.zoo.annualized_historical_volatility(
    qm.zoo.log_return(), window=20, periods_per_year=252
)
```

The source is an already-computed return series and `periods_per_year` is
required — the library never assumes your bar frequency.
