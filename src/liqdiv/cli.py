"""Run the liquidity-diversification study end to end.

    uv run liqdiv                       # default venues & dates
    uv run liqdiv --dates 2026-10-01 --dt-ms 200 --reference mean
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .analysis import (
    constant_spread_book,
    correlation_greedy,
    residuals,
    run_experiment,
)
from .data import load_panel
from .plots import plot_correlation, plot_curves

log = logging.getLogger("liqdiv")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dates", nargs="+", default=list(config.DEFAULT_DATES),
                   help="UTC days (Tardis free tier: 1st of each month)")
    p.add_argument("--dt-ms", type=int, default=1000, help="sampling grid step")
    p.add_argument("--max-stale-s", type=float, default=120.0,
                   help="treat a venue as down if its last quote is older than this")
    p.add_argument("--timestamp", choices=["local", "exchange"], default="local",
                   help="local = Tardis receive time (one common clock); exchange = venue clock")
    p.add_argument("--reference", choices=["primary", "mean", "median"], default="primary")
    p.add_argument("--detrend-min", type=float, default=15.0,
                   help="rolling window (minutes) removing slow basis from residuals; 0 disables")
    p.add_argument("--spread-bp", type=float, default=None,
                   help="constant spread for the article's experiment (default: median quoted LP spread)")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="drop LPs quoting on less than this share of grid points")
    p.add_argument("--n-random", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _fmt(curve: np.ndarray) -> str:
    return " ".join(f"{x:6.3f}" for x in curve)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    days = sorted(date.fromisoformat(d) for d in args.dates)
    venues = [config.PRIMARY, *config.LPS]
    labels_by_name = {v.name: v.label for v in venues}
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ts_col = "local_timestamp" if args.timestamp == "local" else "timestamp"
    bid, ask, raw = load_panel(venues, days, args.data_dir, ts_col, args.dt_ms, args.max_stale_s)

    # --- universe & common support -------------------------------------------------
    lps = [v.name for v in config.LPS]
    cover = (bid[lps].notna() & ask[lps].notna()).mean()
    dropped = [c for c in lps if cover[c] < args.min_coverage]
    if dropped:
        print(f"Dropping LPs with coverage < {args.min_coverage:.0%}: "
              + ", ".join(f"{c} ({cover[c]:.0%})" for c in dropped))
    lps = [c for c in lps if c not in dropped]
    if len(lps) < 3:
        raise SystemExit("fewer than 3 usable LPs - nothing to analyse")

    detrend_points = int(args.detrend_min * 60_000 / args.dt_ms) if args.detrend_min > 0 else 0
    res = residuals(bid, ask, lps, args.reference, config.PRIMARY.name, detrend_points)
    common = res.r.notna().all(axis=1) & res.b.notna().all(axis=1) & res.a.notna().all(axis=1)
    R, B, A = (x.loc[common].to_numpy() for x in (res.r, res.b, res.a))
    day_of = res.r.index[common].normalize()
    print(f"{len(lps)} LPs, {len(days)} day(s), {common.sum():,} common grid points "
          f"({common.mean():.1%} of grid) at {args.dt_ms} ms")

    # --- per-venue stats ------------------------------------------------------------
    corr = np.corrcoef(R.T)
    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    lat = raw.groupby("venue")[["latency_ms_p50", "latency_ms_p99", "updates"]].mean() if len(raw) else None
    stats = pd.DataFrame(
        {
            "label": [labels_by_name[c] for c in lps],
            "coverage": cover[lps].to_numpy(),
            "quoted_spread_bp": (A - B).mean(axis=0),
            "resid_std_bp": R.std(axis=0),
            "resid_mean_abs_bp": np.abs(R).mean(axis=0),
            "avg_corr": np.nanmean(off, axis=1),
        },
        index=lps,
    )
    if lat is not None:
        stats = stats.join(lat)
    s = args.spread_bp if args.spread_bp is not None else float(np.median(stats.quoted_spread_bp))

    # --- in-sample experiments (the article's setting) ------------------------------
    orders = {"ascending": correlation_greedy(corr, True), "descending": correlation_greedy(corr, False)}
    asc_rank = {lps[i]: r + 1 for r, i in enumerate(orders["ascending"])}
    stats["ascending_rank"] = [asc_rank[c] for c in lps]

    exp_const = run_experiment("constant", *constant_spread_book(R, s), orders, args.n_random, rng)
    exp_quoted = run_experiment("quoted", B, A, orders, args.n_random, rng)

    tick = [f"{labels_by_name[c]} #{asc_rank[c]}" for c in lps]
    ref_txt = {"primary": f"vs {config.PRIMARY.label}", "mean": "vs LP mean mid",
               "median": "vs LP median mid"}[args.reference]
    span = f"{days[0]:%Y-%m-%d}" + (f" … {days[-1]:%Y-%m-%d} ({len(days)} days)" if len(days) > 1 else "")
    plot_correlation(corr, tick, args.out / "fig1_correlation.png",
                     f"BTC mid-price residual correlation, {ref_txt}\n"
                     f"{span} · #k = rank in correlation-ascending order")
    plot_curves([exp_const, exp_quoted],
                [f"Constant spread s = {s:.3f} bp (article's setup)", "Quoted spreads (actual bid/ask)"],
                args.out / "fig2_inside_spread.png",
                f"Inside spread vs number of BTC venues — in-sample, {span}")

    # --- out-of-sample: order on earlier days, evaluate on the last -----------------
    exp_oos = None
    if len(days) > 1:
        test_day = pd.Timestamp(days[-1], tz="UTC")
        tr, te = (day_of != test_day), (day_of == test_day)
        if tr.sum() and te.sum():
            c_tr = np.corrcoef(R[tr].T)
            o_tr = {"ascending": correlation_greedy(c_tr, True), "descending": correlation_greedy(c_tr, False)}
            exp_oos_c = run_experiment("constant-oos", *constant_spread_book(R[te], s), o_tr, args.n_random, rng)
            exp_oos = run_experiment("quoted-oos", B[te], A[te], o_tr, args.n_random, rng)
            plot_curves([exp_oos_c, exp_oos],
                        [f"Constant spread s = {s:.3f} bp", "Quoted spreads"],
                        args.out / "fig3_out_of_sample.png",
                        f"Out-of-sample: order chosen on {days[0]:%Y-%m-%d}…{days[-2]:%Y-%m-%d}, "
                        f"evaluated on {days[-1]:%Y-%m-%d} (oracle is in-sample on the test day)")

    # --- report ------------------------------------------------------------------------
    stats.sort_values("ascending_rank").to_csv(args.out / "venue_stats.csv", float_format="%.4f")
    pd.DataFrame(corr, index=lps, columns=lps).to_csv(args.out / "correlation.csv", float_format="%.4f")

    def exp_dict(e):
        return {
            "target_bp": e.target,
            "curves_bp": {k: v.tolist() for k, v in e.curves.items()},
            "random_p5_p95": [e.random_band[0].tolist(), e.random_band[1].tolist()],
            "orders": {k: [lps[i] for i in o] for k, o in e.orders.items()},
            "venues_to_25pct": e.hits,
        }

    summary = {
        "settings": {k: str(v) for k, v in vars(args).items()},
        "constant_spread_bp": s,
        "lps": lps,
        "dropped": dropped,
        "common_points": int(common.sum()),
        "in_sample": {"constant": exp_dict(exp_const), "quoted": exp_dict(exp_quoted)},
        "out_of_sample": ({"constant": exp_dict(exp_oos_c), "quoted": exp_dict(exp_oos)}
                          if exp_oos is not None else None),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("\nVenues (sorted by correlation-ascending rank):")
    cols = ["label", "ascending_rank", "avg_corr", "quoted_spread_bp", "resid_std_bp", "coverage"]
    cols += [c for c in ("latency_ms_p50", "latency_ms_p99") if c in stats]
    print(stats.sort_values("ascending_rank")[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    for e in (exp_const, exp_quoted) + ((exp_oos_c, exp_oos) if exp_oos is not None else ()):
        print(f"\n[{e.name}] 25% target = {e.target:.3f} bp; venues needed: {e.hits}")
        for k, c in e.curves.items():
            print(f"  {k:<11}{_fmt(c)}")
    print(f"\nWrote figures and tables to {args.out.resolve()}")


if __name__ == "__main__":
    main()
