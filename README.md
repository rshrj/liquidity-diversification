# Liquidity diversification — crypto replication

Replicates the method of Pennesi & Senkevich, *Liquidity diversification in FX*
(HSBC FX eRisk, Nov 2020,
[link](https://www.business.hsbc.co.nz/en-gb/insights/managing-cash-flow/liquidity-diversification-fx))
on BTC spot venues, using free public data. The HSBC article used proprietary
quote streams from 11 anonymised FX venues. Here the 11 venues are crypto
exchanges, with Binance BTC/USDT as the primary-venue reference.

```bash
uv run liqdiv                      # Aug/Sep/Oct 2026 (1st of each month), ~220 MB download, ~40 s
uv run liqdiv --help               # grid, timestamps, reference, detrend window, spread, dates
uv run pytest
```

Outputs go to `results/`:

- `fig1_correlation.png`: analogue of the article's Figure 1
- `fig2_inside_spread.png`: analogue of the article's Figure 2
- `fig3_out_of_sample.png`
- `venue_stats.csv`, `correlation.csv`, `summary.json`

## Data

[Tardis.dev](https://docs.tardis.dev/downloadable-csv-files) `quotes` (top of
book, tick by tick). The full day for the **1st of each month** is free, with no
API key. All venues are recorded by the same collector, which gives one common
receive clock (`local_timestamp`). Venues are listed in `src/liqdiv/config.py`.

## Method

Venue sets, grids and the reference price are configurable.

1. **Common grid.** Each venue's top of book is put on a 1 s grid using the last
   quote at or before each grid point (previous tick). A feed silent for more than
   120 s counts as down. Only instants where all venues are quoting are kept;
   coverage is 99.5% on the default days.
2. **Residuals.** For each venue *i*: `r_i = 1e4·(log mid_i − log mid_ref)` in bp.
3. **Detrending (not in the article; essential for crypto).** A centred 15-minute
   rolling mean of `r_i` is subtracted from mid, bid and ask. This removes the
   slowly varying USDT/USD basis and persistent venue premia. Without it the
   USD-vs-USDT offset (~6–11 bp) dominates everything.
4. **Correlation (Figure 1).** Pearson correlation of residuals between venues.
5. **Ordering.** The article's greedy rule: start with the venue that has the
   lowest average correlation with all others, then repeatedly add the venue with
   the lowest average correlation with those **already in the aggregator**. The
   descending order is the mirror image. Random order is the mean of 200
   permutations, with a 5–95% band.
6. **Inside spread (Figure 2).** `IS_S = min_{i∈S} ask_i − max_{i∈S} bid_i`,
   averaged over time. This can only fall as venues are added, so the orderings
   differ only in *how fast* it falls. Two versions:
   - **Constant spread** (the article's setup): every venue gets spread *s*, so
     `IS_S = s − range_S(r)`. The default *s* is the median quoted spread of the
     venues (`--spread-bp` overrides it; the article's FX value was 0.64 bp).
   - **Quoted spreads**: actual bids and asks.
7. **Additions not in the article:**
   - an **oracle greedy** ordering that minimises IS directly, giving an in-sample
     upper bound
   - an **out-of-sample** test: the order is chosen on all days but the last and
     evaluated on the last day

## Findings (default run: 2026-08-01, 09-01, 10-01)

Average inside spread in bp, in-sample, quoted spreads:

| venues | 1 | 2 | 3 | 5 | 11 |
|---|---|---|---|---|---|
| correlation ascending | 0.014 | −0.757 | −0.892 | −1.252 | −1.519 |
| random (mean) | 0.220 | −0.417 | −0.716 | −1.051 | −1.519 |
| correlation descending | 0.014 | −0.347 | −0.515 | −0.738 | −1.519 |
| oracle greedy | 0.002 | −0.757 | −1.018 | −1.291 | −1.519 |

- **The article's qualitative claim holds.** Ascending order beats random by
  about 0.15–0.35 bp at 2–5 venues, and descending order is the worst. Ascending
  matches the oracle bound at 2 venues, trails it by about 0.1 bp at 3–4, and
  converges with it from about 6.
- **It also holds out of sample.** With the order fixed on Aug/Sep and evaluated
  on Oct, ascending stays within 0.07 bp of the test-day oracle and well ahead of
  random.
- **Crypto venues are less correlated than the article's FX venues.** Residual
  correlations range from about −0.09 to 0.71, against 0.5–0.96 for the FX venues
  in the article's matrix (excluding its two outlier venues). The least correlated
  venue here is KuCoin; the most redundant cluster is OKX / Bybit / Crypto.com.
- **The aggregated crypto book is crossed on average (about −1.5 bp).** Venue mids
  sit about 1–2 bp apart for seconds to minutes. That is well inside taker fees of
  roughly 1–10 bp, so no one arbitrages it away. In the article's FX data the
  11-venue mid range was only about 0.31 bp. Two consequences:
  - The "25% improvement" benchmark is degenerate here, because quoted BTC spreads
    are about 0.01–0.07 bp (Binance.US about 0.5 bp, Bitfinex about 1.5 bp).
  - **A negative inside spread is not free money.** The binding execution cost in
    crypto is fees, not spreads.

### Sensitivity (in-sample, quoted, 3 venues)

| setting | ascending | random | descending | 11-venue IS |
|---|---|---|---|---|
Random order here is the mean of 50 permutations; the oracle at 3 venues is
−1.018 bp in every row except the last two (2-min detrend −0.766, no detrend
−11.028).

| setting | ascending | random | descending | 11-venue IS |
|---|---|---|---|---|
| default (local ts, 1 s, primary ref, 15 min detrend) | −0.892 | −0.725 | −0.515 | −1.519 |
| exchange timestamps | −0.889 | −0.724 | −0.514 | −1.518 |
| 100 ms grid | −0.891 | −0.725 | −0.515 | −1.519 |
| reference = LP mean mid | −1.018 | −0.725 | −0.515 | −1.519 |
| reference = LP median mid | −0.950 | −0.725 | −0.562 | −1.519 |
| detrend 2 min | −0.681 | −0.540 | −0.404 | −1.131 |
| no detrend | −11.028 | −6.490 | −1.592 | −11.356 |

- **Timestamps and grid don't matter.** The dislocations are persistent, not
  latency artifacts.
- **The detrend window sets the scale but not the ranking.**
- **The reference price changes which venues are picked early, not the ranking of
  the orderings.** Ascending beats random under every reference; with the LP mean
  mid it hits the oracle at 3 venues.

## Data licence

This repo contains **no market data**. `uv run liqdiv` downloads Tardis.dev's
free datasets to `data/` (gitignored). Whoever runs it accepts the
[Tardis.dev Terms of Service](https://docs.tardis.dev/legal/terms-of-service)
themselves.

The committed `results/` are derived charts and statistics. Clause 9.3 of the
terms allows publishing these in "charts, statistics, research reports".

**Coinbase is deliberately excluded from the venue set.** Clause 23 of the terms
forbids redistributing data or derived works from Coinbase. If you add it back
for your own analysis, don't publish the outputs.

Market data courtesy of [Tardis.dev](https://tardis.dev).

## Code licence

MIT (see `LICENSE`). This covers the code only, not any market data.

## Caveats

- **Few days.** Only free 1st-of-month days are used: 3 days, about 259k
  one-second points. More days need a Tardis subscription, or quotes recorded
  yourself from the exchanges' public websockets.
- **No fees.** Inside spreads ignore fees. Adding a uniform taker fee shifts every
  curve equally; venue-specific fee tiers would change the ranking.
- **Top of book only.** Size is ignored, as in the article. A book that is crossed
  for 0.001 BTC means little.
- **Greedy is myopic.** It compares a candidate only with venues already
  included, so after one independent venue the next pick is close to a tie (see
  `tests/test_analysis.py`).
