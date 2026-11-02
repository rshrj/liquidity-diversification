"""Liquidity-diversification analysis (after Pennesi & Senkevich, HSBC, 2020).

Everything is expressed in basis points relative to a reference mid. For each LP i
at grid time t:

    r_i(t) = 1e4 * (log mid_i(t) - log ref(t))       residual mid
    b_i(t), a_i(t)                                     bid / ask on the same scale

Slowly varying offsets (USDT/USD basis, persistent fee-driven premia) are removed
by subtracting a centred rolling mean of r_i from r_i, b_i and a_i.

The aggregator inside spread over a venue set S is

    IS_S(t) = min_{i in S} a_i(t) - max_{i in S} b_i(t)

and is non-increasing as S grows. The article's "constant spread" experiment is
the special case b_i = r_i - s/2, a_i = r_i + s/2, giving IS_S = s - range_S(r).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- residuals


@dataclass
class Residuals:
    r: pd.DataFrame  # mid residual, bp
    b: pd.DataFrame  # bid, bp
    a: pd.DataFrame  # ask, bp


def residuals(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    lps: list[str],
    reference: str = "primary",
    primary: str | None = None,
    detrend_points: int = 0,
) -> Residuals:
    """Residuals of each LP vs a reference mid, optionally detrended per day."""
    lb, la = np.log(bid), np.log(ask)
    lmid = 0.5 * (lb + la)
    if reference == "primary":
        if primary is None:
            raise ValueError("reference='primary' needs a primary venue")
        ref = lmid[primary]
    elif reference == "mean":
        ref = lmid[lps].mean(axis=1)
    elif reference == "median":
        ref = lmid[lps].median(axis=1)
    else:
        raise ValueError(f"unknown reference {reference!r}")

    r = lmid[lps].sub(ref, axis=0) * 1e4
    b = lb[lps].sub(ref, axis=0) * 1e4
    a = la[lps].sub(ref, axis=0) * 1e4

    if detrend_points > 1:
        minp = max(1, detrend_points // 4)
        trend = r.groupby(r.index.normalize()).transform(
            lambda x: x.rolling(detrend_points, center=True, min_periods=minp).mean()
        )
        r, b, a = r - trend, b - trend, a - trend
    return Residuals(r, b, a)


# ----------------------------------------------------------------------------- orderings


def correlation_greedy(corr: np.ndarray, ascending: bool = True) -> list[int]:
    """The article's procedure: start with the LP least (most) correlated on
    average with all others, then repeatedly add the LP whose average correlation
    with those already in the aggregator is lowest (highest)."""
    n = len(corr)
    off = corr.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    pick = np.argmin if ascending else np.argmax
    order = [int(pick(np.nanmean(off, axis=1)))]
    while len(order) < n:
        rem = [i for i in range(n) if i not in order]
        score = corr[np.ix_(rem, order)].mean(axis=1)
        order.append(rem[int(pick(score))])
    return order


def oracle_greedy(B: np.ndarray, A: np.ndarray) -> list[int]:
    """Greedy ordering that directly minimises the mean inside spread.

    Starts from the best pair (exhaustive), so ties at k=1 under constant spreads
    don't make the start arbitrary. In-sample upper bound for any ordering rule.
    """
    n = B.shape[1]
    best, best_val = (0, 1), np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = np.mean(np.minimum(A[:, i], A[:, j]) - np.maximum(B[:, i], B[:, j]))
            if v < best_val:
                best, best_val = (i, j), v
    i, j = best
    own = (A - B).mean(axis=0)
    order = [i, j] if own[i] <= own[j] else [j, i]
    amin, bmax = np.minimum(A[:, i], A[:, j]), np.maximum(B[:, i], B[:, j])
    while len(order) < n:
        rem = [k for k in range(n) if k not in order]
        vals = [np.mean(np.minimum(amin, A[:, k]) - np.maximum(bmax, B[:, k])) for k in rem]
        k = rem[int(np.argmin(vals))]
        order.append(k)
        amin, bmax = np.minimum(amin, A[:, k]), np.maximum(bmax, B[:, k])
    return order


# ----------------------------------------------------------------------------- curves


def inside_spread_curve(B: np.ndarray, A: np.ndarray, order: list[int]) -> np.ndarray:
    """Mean inside spread after adding venues 1..k of ``order`` (k = 1..n)."""
    amin = np.minimum.accumulate(A[:, order], axis=1)
    bmax = np.maximum.accumulate(B[:, order], axis=1)
    return (amin - bmax).mean(axis=0)


def random_curves(B: np.ndarray, A: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    n = B.shape[1]
    return np.stack([inside_spread_curve(B, A, list(rng.permutation(n))) for _ in range(n_perm)])


def venues_to_target(curve: np.ndarray, target: float) -> int | None:
    """Smallest k (1-based) whose inside spread is <= target."""
    hit = np.nonzero(curve <= target)[0]
    return int(hit[0]) + 1 if hit.size else None


@dataclass
class Experiment:
    """Curves for one spread model ('constant' or 'quoted') on one sample."""

    name: str
    curves: dict[str, np.ndarray]
    random_band: tuple[np.ndarray, np.ndarray]  # 5th / 95th percentile
    orders: dict[str, list[int]]
    target: float  # 25% improvement on the average single-venue spread
    hits: dict[str, int | None] = field(default_factory=dict)


def run_experiment(
    name: str,
    B: np.ndarray,
    A: np.ndarray,
    corr_orders: dict[str, list[int]],
    n_perm: int,
    rng: np.random.Generator,
) -> Experiment:
    curves = {k: inside_spread_curve(B, A, o) for k, o in corr_orders.items()}
    rnd = random_curves(B, A, n_perm, rng)
    curves["random"] = rnd.mean(axis=0)
    oracle = oracle_greedy(B, A)
    curves["oracle"] = inside_spread_curve(B, A, oracle)
    target = 0.75 * float((A - B).mean())  # avg single-venue spread == random curve at k=1
    exp = Experiment(
        name=name,
        curves=curves,
        random_band=(np.percentile(rnd, 5, axis=0), np.percentile(rnd, 95, axis=0)),
        orders={**corr_orders, "oracle": oracle},
        target=target,
    )
    exp.hits = {k: venues_to_target(c, target) for k, c in curves.items()}
    return exp


def constant_spread_book(R: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    return R - s / 2, R + s / 2
