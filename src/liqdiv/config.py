"""Venue universe and defaults.

Data comes from Tardis.dev's free tier: the full day of the 1st of every month is
downloadable without an API key (https://docs.tardis.dev/downloadable-csv-files).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    exchange: str  # Tardis exchange id
    symbol: str  # Tardis dataset symbol (as it appears in the download URL)

    @property
    def name(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    @property
    def quote_ccy(self) -> str:
        return "USDT" if self.symbol.upper().endswith("USDT") else "USD"

    @property
    def label(self) -> str:
        return f"{self.exchange} ({self.quote_ccy})"


# Reference ("primary venue") - the deepest BTC book and main locus of price
# discovery, playing the role EBS/Refinitiv Matching play in spot FX.
PRIMARY = Venue("binance", "BTCUSDT")

# The 11 liquidity providers that make up the candidate aggregator (the article
# also uses 11). Coinbase is deliberately excluded: Tardis.dev's licence (Clause
# 23) forbids publishing anything derived from Coinbase data. Mix of USD- and
# USDT-quoted books; the slowly varying USDT/USD basis is removed by detrending
# residuals (see analysis.residuals).
LPS: tuple[Venue, ...] = (
    Venue("binance-us", "BTCUSD"),
    Venue("kraken", "BTC-USD"),
    Venue("bitstamp", "BTCUSD"),
    Venue("okex", "BTC-USDT"),
    Venue("bybit-spot", "BTCUSDT"),
    Venue("bitfinex", "BTCUSD"),
    Venue("gemini", "BTCUSD"),
    Venue("huobi", "BTCUSDT"),
    Venue("kucoin", "BTC-USDT"),
    Venue("gate-io", "BTC_USDT"),
    Venue("crypto-com", "BTC_USD"),
)

# Free days (1st of the month). Kraken's BTC/USD symbol exists from 2026-07-10.
DEFAULT_DATES: tuple[str, ...] = ("2026-08-01", "2026-09-01", "2026-10-01")
