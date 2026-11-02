"""Download Tardis top-of-book quotes and sample them onto a common time grid."""

from __future__ import annotations

import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Venue

log = logging.getLogger(__name__)

URL = "https://datasets.tardis.dev/v1/{ex}/quotes/{d:%Y}/{d:%m}/{d:%d}/{sym}.csv.gz"
US = 1_000_000  # Tardis timestamps are integer microseconds since epoch
DAY_US = 86_400 * US


# ----------------------------------------------------------------------------- download


def raw_path(raw_dir: Path, venue: Venue, day: date) -> Path:
    return raw_dir / f"{venue.exchange}_{venue.symbol}_{day:%Y-%m-%d}.csv.gz"


def download(venue: Venue, day: date, raw_dir: Path, retries: int = 3) -> Path | None:
    """Fetch one venue-day of quotes (cached). Returns None if unavailable."""
    path = raw_path(raw_dir, venue, day)
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = URL.format(ex=venue.exchange, d=day, sym=venue.symbol)
    tmp = path.with_name(path.name + ".part")
    err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "liqdiv/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(path)
            log.info("downloaded %s %s (%.1f MB)", venue.name, day, path.stat().st_size / 1e6)
            return path
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                log.warning("%s %s not available (HTTP %d)", venue.name, day, e.code)
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2**attempt)
    log.warning("%s %s download failed: %s", venue.name, day, err)
    tmp.unlink(missing_ok=True)
    return None


def download_all(venues: list[Venue], days: list[date], raw_dir: Path, workers: int = 6) -> None:
    jobs = [(v, d) for d in days for v in venues]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda vd: download(vd[0], vd[1], raw_dir), jobs))


# ----------------------------------------------------------------------------- parsing


def load_quotes(path: Path) -> pd.DataFrame:
    """Raw quotes with sane top of book (both sides present, not crossed)."""
    df = pd.read_csv(
        path,
        usecols=["timestamp", "local_timestamp", "bid_price", "ask_price"],
        engine="pyarrow",
    )
    df = df.dropna()
    df = df[(df.bid_price > 0) & (df.ask_price >= df.bid_price)]
    return df.astype(
        {"timestamp": "int64", "local_timestamp": "int64", "bid_price": "float64", "ask_price": "float64"}
    )


def day_start_us(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * US


def to_grid(
    ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, grid: np.ndarray, max_stale_us: int
) -> tuple[np.ndarray, np.ndarray]:
    """Previous-tick sampling: the quote in force at each grid instant.

    Grid points before the first quote, or whose last quote is older than
    ``max_stale_us`` (feed outage), are NaN. Ties in ``ts`` resolve to the last row.
    """
    order = np.argsort(ts, kind="stable")
    ts, bid, ask = ts[order], bid[order], ask[order]
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    idx = np.clip(idx, 0, None)
    ok &= (grid - ts[idx]) <= max_stale_us
    return np.where(ok, bid[idx], np.nan), np.where(ok, ask[idx], np.nan)


def raw_stats(q: pd.DataFrame) -> dict:
    lat_ms = (q.local_timestamp - q.timestamp).to_numpy() / 1e3
    return {
        "updates": int(len(q)),
        "latency_ms_p50": float(np.median(lat_ms)),
        "latency_ms_p99": float(np.percentile(lat_ms, 99)),
    }


def grid_venue_day(
    venue: Venue,
    day: date,
    raw_dir: Path,
    cache_dir: Path,
    ts_col: str,
    dt_us: int,
    max_stale_us: int,
) -> tuple[pd.DataFrame | None, dict | None]:
    """Gridded bid/ask for one venue-day (cached as parquet)."""
    key = f"{venue.exchange}_{venue.symbol}_{day:%Y-%m-%d}_{ts_col}_{dt_us}us_{max_stale_us}us"
    pq, js = cache_dir / f"{key}.parquet", cache_dir / f"{key}.json"
    if pq.exists() and js.exists():
        return pd.read_parquet(pq), json.loads(js.read_text())

    path = raw_path(raw_dir, venue, day)
    if not path.exists():
        return None, None
    q = load_quotes(path)
    if q.empty:
        log.warning("%s %s: no valid quotes", venue.name, day)
        return None, None

    start = day_start_us(day)
    grid = start + dt_us * np.arange(DAY_US // dt_us, dtype=np.int64)
    bid, ask = to_grid(
        q[ts_col].to_numpy(), q.bid_price.to_numpy(), q.ask_price.to_numpy(), grid, max_stale_us
    )
    out = pd.DataFrame({"ts": grid, "bid": bid, "ask": ask})
    stats = raw_stats(q)

    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq, index=False)
    js.write_text(json.dumps(stats))
    return out, stats


def load_panel(
    venues: list[Venue],
    days: list[date],
    data_dir: Path,
    ts_col: str = "local_timestamp",
    dt_ms: int = 1000,
    max_stale_s: float = 120.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (bid, ask, stats). bid/ask are T x V frames on a UTC DatetimeIndex."""
    raw_dir, cache_dir = data_dir / "raw", data_dir / "grid"
    download_all(venues, days, raw_dir)

    dt_us, max_stale_us = dt_ms * 1000, int(max_stale_s * US)
    bids, asks, stats = [], [], []
    for day in days:
        grid = day_start_us(day) + dt_us * np.arange(DAY_US // dt_us, dtype=np.int64)
        idx = pd.to_datetime(grid, unit="us", utc=True)
        b = pd.DataFrame(np.nan, index=idx, columns=[v.name for v in venues])
        a = b.copy()
        for v in venues:
            g, s = grid_venue_day(v, day, raw_dir, cache_dir, ts_col, dt_us, max_stale_us)
            if g is None:
                continue
            b[v.name] = g.bid.to_numpy()
            a[v.name] = g.ask.to_numpy()
            stats.append({"venue": v.name, "date": f"{day:%Y-%m-%d}", **s})
        bids.append(b)
        asks.append(a)
    return pd.concat(bids), pd.concat(asks), pd.DataFrame(stats)
